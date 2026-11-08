#include "hd_projection.hpp"

#include "hd_parallel.hpp"
#include "hd_sha256.hpp"

#include <zstd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace visium_hd::projection {

namespace fs = std::filesystem;
using json::Value;

ZarrArray::ZarrArray(const std::string& path) : path_(path) {
    const Value meta = json::read_file((fs::path(path) / "zarr.json").string());
    if (meta.at("zarr_format").as_int() != 3 || meta.at("node_type").as_string() != "array") {
        throw std::runtime_error("not a zarr v3 array: " + path);
    }
    if (meta.at("data_type").as_string() != "uint32") throw std::runtime_error("zarr array is not uint32: " + path);
    const Value& shape = meta.at("shape");
    const Value& grid = meta.at("chunk_grid");
    if (shape.size() != 2 || grid.at("name").as_string() != "regular") throw std::runtime_error("unsupported zarr array: " + path);
    const Value& chunk = grid.at("configuration").at("chunk_shape");
    for (int i = 0; i < 2; ++i) {
        shape_[i] = shape[i].as_int();
        chunk_[i] = chunk[i].as_int();
        if (shape_[i] < 0 || chunk_[i] < 1) throw std::runtime_error("bad zarr shape: " + path);
    }
    const Value& keys = meta.at("chunk_key_encoding");
    if (keys.at("name").as_string() != "default") throw std::runtime_error("unsupported zarr chunk keys: " + path);
    if (const Value* config = keys.find("configuration")) {
        if (const Value* separator = config->find("separator")) separator_ = separator->as_string();
    }
    fill_ = static_cast<std::uint32_t>(meta.at("fill_value").as_int());
    const Value& codecs = meta.at("codecs");
    for (std::size_t i = 0; i < codecs.size(); ++i) {
        const std::string name = codecs[i].at("name").as_string();
        if (i == 0 && name == "bytes") {
            const Value* config = codecs[i].find("configuration");
            if (config && config->find("endian") && config->at("endian").as_string() != "little") {
                throw std::runtime_error("big-endian zarr array: " + path);
            }
        } else if (i == 1 && name == "zstd") {
            zstd_ = true;
        } else {
            throw std::runtime_error("unsupported zarr codec " + name + ": " + path);
        }
    }
    if (const Value* transformers = meta.find("storage_transformers")) {
        if (transformers->size() != 0) throw std::runtime_error("zarr storage transformers are not supported: " + path);
    }
}

std::vector<std::uint32_t> ZarrArray::chunk(std::int64_t row, std::int64_t col) const {
    const std::size_t count = static_cast<std::size_t>(chunk_[0] * chunk_[1]);
    std::vector<std::uint32_t> out(count, fill_);
    const fs::path file = fs::path(path_) / ("c" + separator_ + std::to_string(row) + separator_ + std::to_string(col));
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) return out;
    const std::streamsize size = input.tellg();
    input.seekg(0);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (size > 0 && !input.read(bytes.data(), size)) throw std::runtime_error("cannot read " + file.string());
    const std::size_t expected = count * sizeof(std::uint32_t);
    if (zstd_) {
        const unsigned long long content = ZSTD_getFrameContentSize(bytes.data(), bytes.size());
        if (content != ZSTD_CONTENTSIZE_UNKNOWN && content != expected) throw std::runtime_error("bad zarr chunk size: " + file.string());
        const std::size_t got = ZSTD_decompress(out.data(), expected, bytes.data(), bytes.size());
        if (ZSTD_isError(got) || got != expected) throw std::runtime_error("cannot decompress " + file.string());
    } else {
        if (bytes.size() != expected) throw std::runtime_error("bad zarr chunk size: " + file.string());
        std::memcpy(out.data(), bytes.data(), expected);
    }
    return out;
}

namespace {

Matrix3 matrix_from(const Value& value, const std::string& what) {
    Matrix3 m{};
    if (value.size() != 3) throw std::runtime_error("expected a 3 x 3 matrix: " + what);
    for (int i = 0; i < 3; ++i) {
        if (value[i].size() != 3) throw std::runtime_error("expected a 3 x 3 matrix: " + what);
        for (int j = 0; j < 3; ++j) m[i][j] = value[i][j].as_double();
    }
    return m;
}

// score_hd_flex_registration_roi.project: (x, y) through a 3 x 3 homography.
void project_point(const Matrix3& m, double x, double y, double& out_x, double& out_y) {
    const double a = m[0][0] * x + m[0][1] * y + m[0][2];
    const double b = m[1][0] * x + m[1][1] * y + m[1][2];
    const double c = m[2][0] * x + m[2][1] * y + m[2][2];
    if (std::abs(c) < 1e-12) throw std::runtime_error("transform projects an expression-grid point to infinity");
    out_x = a / c;
    out_y = b / c;
}

// The labels > 0 of `array` at pixels (py, px) of the points marked inside, read chunk by chunk.
std::vector<std::uint8_t> labelled(const ZarrArray& array, const std::vector<std::int64_t>& py,
                                   const std::vector<std::int64_t>& px, const std::vector<std::uint8_t>& inside,
                                   int threads) {
    const std::int64_t rows = (array.height() + array.chunk_height() - 1) / array.chunk_height();
    const std::int64_t cols = (array.width() + array.chunk_width() - 1) / array.chunk_width();
    const std::size_t chunks = static_cast<std::size_t>(rows * cols);
    std::vector<std::size_t> start(chunks + 1, 0);
    for (std::size_t i = 0; i < py.size(); ++i) {
        if (inside[i]) ++start[static_cast<std::size_t>((py[i] / array.chunk_height()) * cols + px[i] / array.chunk_width()) + 1];
    }
    for (std::size_t k = 0; k < chunks; ++k) start[k + 1] += start[k];
    std::vector<std::size_t> order(start[chunks]);
    std::vector<std::size_t> fill(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < py.size(); ++i) {
        if (inside[i]) order[fill[static_cast<std::size_t>((py[i] / array.chunk_height()) * cols + px[i] / array.chunk_width())]++] = i;
    }
    std::vector<std::uint8_t> out(py.size(), 0);
    parallel_for(chunks, threads, [&](std::size_t k) {
        if (start[k] == start[k + 1]) return;
        const std::int64_t row = static_cast<std::int64_t>(k) / cols;
        const std::int64_t col = static_cast<std::int64_t>(k) % cols;
        const std::vector<std::uint32_t> values = array.chunk(row, col);
        for (std::size_t j = start[k]; j < start[k + 1]; ++j) {
            const std::size_t i = order[j];
            const std::int64_t y = py[i] - row * array.chunk_height();
            const std::int64_t x = px[i] - col * array.chunk_width();
            out[i] = values[static_cast<std::size_t>(y * array.chunk_width() + x)] > 0;
        }
    });
    return out;
}

Value int_list(std::initializer_list<long long> values) {
    Value out = Value::array();
    for (long long v : values) out.push(v);
    return out;
}

}  // namespace

Result project(const std::string& segmentation, const std::string& registration_path, const std::string& capture_path,
               int width, int threads) {
    const Value registration = json::read_file(registration_path);
    const Matrix3 native = matrix_from(registration.at("moving_to_fixed"), "moving_to_fixed");
    const Matrix3 capture = matrix_from(json::read_file(capture_path).at("spot_colrow_to_cytassist"), "spot_colrow_to_cytassist");
    const double moving_height = registration.at("moving_shape_yx")[0].as_double();
    const double moving_width = registration.at("moving_shape_yx")[1].as_double();
    if (!(moving_width > 0) || !(moving_height > 0)) throw std::runtime_error("ROI geometry is empty");
    const Matrix3 inverse_native = count_registration::inverse(native);

    // The grid centres that project into the registered microscope image (the native ROI).
    const std::size_t total = static_cast<std::size_t>(width) * static_cast<std::size_t>(width);
    std::vector<double> moving_x(total), moving_y(total);
    std::vector<std::uint8_t> in_roi(total);
    const std::size_t block = 1 << 16;
    parallel_for((total + block - 1) / block, threads, [&](std::size_t k) {
        for (std::size_t flat = k * block; flat < std::min(total, (k + 1) * block); ++flat) {
            const double row = static_cast<double>(flat / static_cast<std::size_t>(width));
            const double col = static_cast<double>(flat % static_cast<std::size_t>(width));
            double cx = 0, cy = 0;
            project_point(capture, col, row, cx, cy);
            project_point(inverse_native, cx - 0.0, cy - 0.0, moving_x[flat], moving_y[flat]);
            in_roi[flat] = moving_x[flat] >= 0.0 && moving_x[flat] < moving_width && moving_y[flat] >= 0.0
                && moving_y[flat] < moving_height;
        }
    });
    std::vector<std::size_t> indices;
    for (std::size_t flat = 0; flat < total; ++flat) {
        if (in_roi[flat]) indices.push_back(flat);
    }
    if (indices.empty()) throw std::runtime_error("native capture-grid ROI selection is empty or not strictly ordered");
    long long col_min = width, col_max = -1;
    for (std::size_t flat : indices) {
        const long long col = static_cast<long long>(flat % static_cast<std::size_t>(width));
        col_min = std::min(col_min, col);
        col_max = std::max(col_max, col);
    }
    Value audit = Value::object();
    audit.set("native_roi_2um_bins", static_cast<unsigned long long>(indices.size()));
    audit.set("selection_scope", "full_registered_extent");
    audit.set("geometry_source", "independent_capture_grid");
    audit.set("native_row_min", static_cast<long long>(indices.front() / static_cast<std::size_t>(width)));
    audit.set("native_row_max", static_cast<long long>(indices.back() / static_cast<std::size_t>(width)));
    audit.set("native_col_min", col_min);
    audit.set("native_col_max", col_max);

    // The Cellpose labels at the pixel of each location of the ROI.
    const fs::path root(segmentation);
    const fs::path summary_path = root / "summary.json";
    const fs::path nucleus_path = root / "masks" / "nucleus_labels.zarr";
    const fs::path cell_path = root / "masks" / "cell_labels.zarr";
    for (const fs::path& path : {summary_path, nucleus_path, cell_path}) {
        if (!fs::exists(path)) throw std::runtime_error("incomplete Cellpose segmentation: " + path.string());
    }
    const Value summary = json::read_file(summary_path.string());
    const Value& contract = summary.at("input_contract");
    if (const Value* loaded = contract.find("expression_or_vendor_segmentation_loaded")) {
        if (loaded->kind() == Value::Kind::Boolean ? loaded->as_bool() : !loaded->is_null()) {
            throw std::runtime_error("Cellpose segmentation declares expression or vendor-segmentation ancestry");
        }
    }
    const Value* outside = summary.at("counts").find("nucleus_outside_cell_pixels");
    if (!outside || outside->as_int() != 0) throw std::runtime_error("Cellpose summary declares a nucleus/cell hierarchy failure");
    const double roi_x0 = contract.at("source_roi_xyxy")[0].as_double();
    const double roi_y0 = contract.at("source_roi_xyxy")[1].as_double();
    const long long downsample = contract.at("downsample").as_int();
    const long long seg_height = contract.at("segmentation_shape_yx")[0].as_int();
    const long long seg_width = contract.at("segmentation_shape_yx")[1].as_int();
    const std::string nucleus_hash = tree_sha256(nucleus_path.string());
    const std::string cell_hash = tree_sha256(cell_path.string());
    if (nucleus_hash != summary.at("outputs").at("nucleus_labels.zarr").as_string()) {
        throw std::runtime_error("Cellpose nucleus label tree differs from its declared hash");
    }
    if (cell_hash != summary.at("outputs").at("cell_labels.zarr").as_string()) {
        throw std::runtime_error("Cellpose cell label tree differs from its declared hash");
    }
    const ZarrArray nucleus(nucleus_path.string());
    const ZarrArray cell(cell_path.string());
    if (nucleus.height() != seg_height || nucleus.width() != seg_width || cell.height() != seg_height || cell.width() != seg_width) {
        throw std::runtime_error("Cellpose label shapes disagree with their declared segmentation geometry");
    }
    const int moving_downsample = 16;   // the registration's moving image is source[::16]
    std::vector<std::int64_t> py(indices.size()), px(indices.size());
    std::vector<std::uint8_t> inside(indices.size());
    parallel_for((indices.size() + block - 1) / block, threads, [&](std::size_t k) {
        for (std::size_t i = k * block; i < std::min(indices.size(), (k + 1) * block); ++i) {
            const std::size_t flat = indices[i];
            // registration_to_segmentation_coordinate, 'decimate' sampling, origins at zero.
            const double raw_x = 0.0 + moving_x[flat] * moving_downsample;
            const double raw_y = 0.0 + moving_y[flat] * moving_downsample;
            const double sx = (raw_x - roi_x0 + 0.5) / static_cast<double>(downsample) - 0.5;
            const double sy = (raw_y - roi_y0 + 0.5) / static_cast<double>(downsample) - 0.5;
            px[i] = static_cast<std::int64_t>(std::floor(sx));
            py[i] = static_cast<std::int64_t>(std::floor(sy));
            inside[i] = px[i] >= 0 && px[i] < seg_width && py[i] >= 0 && py[i] < seg_height;
        }
    });
    const std::vector<std::uint8_t> in_nucleus = labelled(nucleus, py, px, inside, threads);
    const std::vector<std::uint8_t> in_cell = labelled(cell, py, px, inside, threads);
    Result result;
    result.nucleus.assign(total, 0);
    result.support.assign(total, 0);
    long long unsupported = 0, supported = 0, nuclei = 0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (in_nucleus[i] && !in_cell[i]) throw std::runtime_error("Cellpose nucleus bin is outside its expanded cell domain");
        if (!inside[i]) {
            ++unsupported;
            continue;
        }
        result.support[indices[i]] = 1;
        result.nucleus[indices[i]] = in_nucleus[i];
        ++supported;
        nuclei += in_nucleus[i];
    }
    audit.set("segmentation_root", fs::canonical(root).string());
    audit.set("segmentation_summary_sha256", sha256_file(summary_path.string()));
    audit.set("nucleus_labels_tree_sha256", nucleus_hash);
    audit.set("cell_labels_tree_sha256", cell_hash);
    audit.set("segmentation_shape_yx", int_list({seg_height, seg_width}));
    audit.set("segmentation_downsample", downsample);
    audit.set("registration_moving_source_downsample", moving_downsample);
    audit.set("registration_moving_sampling", "decimate");
    audit.set("coordinate_resampling", "explicit source-pixel and pixel-centre transform");
    audit.set("grid_geometry", "independent_capture_grid");
    audit.set("native_roi_bins_outside_cellpose_pixel_support", unsupported);
    audit.set("support_kind", "native_registered_image_coverage");
    audit.set("support_is_histological_tissue_classification", false);
    audit.set("supported_grid_locations", supported);
    audit.set("nucleus_grid_locations", nuclei);
    // The frozen registration's moving image is source[::16], not an area resize.
    const Matrix3 decimation{{{16.0, 0.0, 0.0}, {0.0, 16.0, 0.0}, {0.0, 0.0, 1.0}}};
    Matrix3 grid_to_image = count_registration::multiply(count_registration::multiply(decimation, inverse_native), capture);
    const double scale = grid_to_image[2][2];
    Value matrix = Value::array();
    for (auto& row : grid_to_image) {
        Value r = Value::array();
        for (double& v : row) {
            v /= scale;
            r.push(v);
        }
        matrix.push(r);
    }
    audit.set("grid_to_full_resolution_image", matrix);
    result.audit = audit;
    return result;
}

}  // namespace visium_hd::projection
