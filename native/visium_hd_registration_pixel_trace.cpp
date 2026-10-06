#include "hd_registration_core.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tiffio.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using visium_hd::registration::pixel_center_resize_transform;
using visium_hd::registration::project_point;

namespace {

struct Args {
    fs::path microscope_original;
    fs::path cytassist_original;
    fs::path microscope_roi_ds4;
    fs::path cytassist_roi;
    fs::path reference_json;
    fs::path out_dir;
    int roi_x0 = 4608;
    int roi_y0 = 12800;
    int roi_size = 8704;
    int roi_downsample = 4;
    int cytassist_x0 = 1272;
    int cytassist_y0 = 1253;
    int work_size = 607;
};

std::string usage() {
    return R"USAGE(usage: visium_hd_registration_pixel_trace [options]

Required:
  --microscope-original FILE
  --cytassist-original FILE
  --microscope-roi-ds4 FILE
  --cytassist-roi FILE
  --reference-json FILE
  --out-dir DIR

Geometry defaults reproduce the H1-GMHFWPH/D1 CRC ROI fixture. Optional:
  --roi-x0 INT --roi-y0 INT --roi-size INT --roi-downsample INT
  --cytassist-x0 INT --cytassist-y0 INT --work-size INT
)USAGE";
}

int parse_int(const std::string& value, const std::string& option) {
    size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) throw std::runtime_error("invalid integer for " + option);
    return parsed;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") {
            std::cout << usage();
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++i];
        if (option == "--microscope-original") args.microscope_original = value;
        else if (option == "--cytassist-original") args.cytassist_original = value;
        else if (option == "--microscope-roi-ds4") args.microscope_roi_ds4 = value;
        else if (option == "--cytassist-roi") args.cytassist_roi = value;
        else if (option == "--reference-json") args.reference_json = value;
        else if (option == "--out-dir") args.out_dir = value;
        else if (option == "--roi-x0") args.roi_x0 = parse_int(value, option);
        else if (option == "--roi-y0") args.roi_y0 = parse_int(value, option);
        else if (option == "--roi-size") args.roi_size = parse_int(value, option);
        else if (option == "--roi-downsample") args.roi_downsample = parse_int(value, option);
        else if (option == "--cytassist-x0") args.cytassist_x0 = parse_int(value, option);
        else if (option == "--cytassist-y0") args.cytassist_y0 = parse_int(value, option);
        else if (option == "--work-size") args.work_size = parse_int(value, option);
        else throw std::runtime_error("unknown option: " + option);
    }
    if (args.microscope_original.empty() || args.cytassist_original.empty() ||
        args.microscope_roi_ds4.empty() || args.cytassist_roi.empty() ||
        args.reference_json.empty() || args.out_dir.empty()) {
        throw std::runtime_error("all required paths must be supplied\n" + usage());
    }
    if (args.roi_size <= 0 || args.roi_downsample <= 0 ||
        args.roi_size % args.roi_downsample != 0 || args.work_size <= 0) {
        throw std::runtime_error("invalid ROI/downsample/work geometry");
    }
    return args;
}

cv::Matx33d read_named_matrix(const fs::path& path, const std::string& key) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read reference JSON: " + path.string());
    const std::string text(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()
    );
    const std::string token = "\"" + key + "\"";
    const size_t key_at = text.find(token);
    if (key_at == std::string::npos) throw std::runtime_error("missing matrix key: " + key);
    size_t cursor = text.find('[', key_at + token.size());
    if (cursor == std::string::npos) throw std::runtime_error("missing matrix array: " + key);
    cv::Matx33d matrix;
    int count = 0;
    while (cursor < text.size() && count < 9) {
        while (cursor < text.size() &&
               !(text[cursor] == '-' || text[cursor] == '+' || text[cursor] == '.' ||
                 std::isdigit(static_cast<unsigned char>(text[cursor])))) {
            ++cursor;
        }
        if (cursor == text.size()) break;
        char* end = nullptr;
        const double value = std::strtod(text.c_str() + cursor, &end);
        if (end == text.c_str() + cursor) throw std::runtime_error("invalid matrix number");
        matrix(count / 3, count % 3) = value;
        ++count;
        cursor = static_cast<size_t>(end - text.c_str());
    }
    if (count != 9) throw std::runtime_error("matrix does not contain nine numbers: " + key);
    return matrix;
}

cv::Matx33d inverse_matrix(const cv::Matx33d& matrix) {
    cv::Mat source(3, 3, CV_64F, const_cast<double*>(matrix.val));
    cv::Mat inverted;
    if (cv::invert(source, inverted, cv::DECOMP_LU) == 0.0) {
        throw std::runtime_error("singular trace transform");
    }
    cv::Matx33d result;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result(row, column) = inverted.at<double>(row, column);
        }
    }
    return result;
}

class TiledRgb8 {
  public:
    explicit TiledRgb8(const fs::path& path) {
        handle_ = TIFFOpen(path.c_str(), "r");
        if (handle_ == nullptr) throw std::runtime_error("cannot open TIFF: " + path.string());
        uint16_t bits = 0;
        uint16_t samples = 0;
        uint16_t planar = PLANARCONFIG_CONTIG;
        uint16_t orientation = ORIENTATION_TOPLEFT;
        TIFFGetField(handle_, TIFFTAG_IMAGEWIDTH, &width_);
        TIFFGetField(handle_, TIFFTAG_IMAGELENGTH, &height_);
        TIFFGetField(handle_, TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetField(handle_, TIFFTAG_SAMPLESPERPIXEL, &samples);
        TIFFGetFieldDefaulted(handle_, TIFFTAG_PLANARCONFIG, &planar);
        TIFFGetFieldDefaulted(handle_, TIFFTAG_ORIENTATION, &orientation);
        if (!TIFFIsTiled(handle_) || bits != 8 || samples != 3 ||
            planar != PLANARCONFIG_CONTIG || orientation != ORIENTATION_TOPLEFT) {
            throw std::runtime_error(
                "original microscope must be tiled, RGB8 contiguous, top-left"
            );
        }
        TIFFGetField(handle_, TIFFTAG_TILEWIDTH, &tile_width_);
        TIFFGetField(handle_, TIFFTAG_TILELENGTH, &tile_height_);
        tile_bytes_ = TIFFTileSize(handle_);
    }

    ~TiledRgb8() {
        if (handle_ != nullptr) TIFFClose(handle_);
    }

    TiledRgb8(const TiledRgb8&) = delete;
    TiledRgb8& operator=(const TiledRgb8&) = delete;

    std::array<unsigned char, 3> rgb(uint32_t x, uint32_t y) {
        if (x >= width_ || y >= height_) throw std::runtime_error("TIFF pixel is out of bounds");
        const ttile_t tile_id = TIFFComputeTile(handle_, x, y, 0, 0);
        auto found = tiles_.find(tile_id);
        if (found == tiles_.end()) {
            std::vector<unsigned char> tile(static_cast<size_t>(tile_bytes_));
            const tmsize_t read = TIFFReadEncodedTile(
                handle_, tile_id, tile.data(), tile_bytes_
            );
            if (read < 0) throw std::runtime_error("failed reading original microscope tile");
            found = tiles_.emplace(tile_id, std::move(tile)).first;
        }
        const uint32_t local_x = x % tile_width_;
        const uint32_t local_y = y % tile_height_;
        const size_t offset =
            (static_cast<size_t>(local_y) * tile_width_ + local_x) * 3;
        const auto& tile = found->second;
        return {tile[offset], tile[offset + 1], tile[offset + 2]};
    }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t tile_width() const { return tile_width_; }
    uint32_t tile_height() const { return tile_height_; }

  private:
    TIFF* handle_ = nullptr;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t tile_width_ = 0;
    uint32_t tile_height_ = 0;
    tmsize_t tile_bytes_ = 0;
    std::map<ttile_t, std::vector<unsigned char>> tiles_;
};

double matrix_max_abs_difference(const cv::Matx33d& left, const cv::Matx33d& right) {
    double result = 0.0;
    for (int i = 0; i < 9; ++i) result = std::max(result, std::abs(left.val[i] - right.val[i]));
    return result;
}

struct GridError {
    double rmse = 0.0;
    double mean_x = 0.0;
    double mean_y = 0.0;
    double maximum = 0.0;
};

GridError compare_grid(
    const cv::Matx33d& candidate,
    const cv::Matx33d& reference,
    int width,
    int height
) {
    GridError result;
    double squared = 0.0;
    int count = 0;
    for (int gy = 0; gy < 9; ++gy) {
        for (int gx = 0; gx < 9; ++gx) {
            const cv::Point2d point(
                gx * (width - 1.0) / 8.0,
                gy * (height - 1.0) / 8.0
            );
            const cv::Point2d delta =
                project_point(candidate, point) - project_point(reference, point);
            const double norm = cv::norm(delta);
            squared += norm * norm;
            result.mean_x += delta.x;
            result.mean_y += delta.y;
            result.maximum = std::max(result.maximum, norm);
            ++count;
        }
    }
    result.rmse = std::sqrt(squared / count);
    result.mean_x /= count;
    result.mean_y /= count;
    return result;
}

void write_matrix(std::ostream& out, const std::string& name, const cv::Matx33d& matrix) {
    out << name;
    for (double value : matrix.val) out << '\t' << value;
    out << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (fs::exists(args.out_dir)) {
            throw std::runtime_error("output directory already exists: " + args.out_dir.string());
        }
        fs::create_directories(args.out_dir);

        TiledRgb8 microscope(args.microscope_original);
        const cv::Mat microscope_ds = cv::imread(args.microscope_roi_ds4, cv::IMREAD_COLOR);
        const cv::Mat cytassist = cv::imread(args.cytassist_original, cv::IMREAD_COLOR);
        const cv::Mat cytassist_roi = cv::imread(args.cytassist_roi, cv::IMREAD_COLOR);
        if (microscope_ds.empty() || cytassist.empty() || cytassist_roi.empty()) {
            throw std::runtime_error("OpenCV could not read one or more fixture images");
        }
        const int roi_ds_size = args.roi_size / args.roi_downsample;
        if (microscope_ds.cols != roi_ds_size || microscope_ds.rows != roi_ds_size) {
            throw std::runtime_error("microscope ROI fixture has unexpected dimensions");
        }
        const cv::Rect cytassist_bounds(
            args.cytassist_x0,
            args.cytassist_y0,
            cytassist_roi.cols,
            cytassist_roi.rows
        );
        if ((cytassist_bounds & cv::Rect(0, 0, cytassist.cols, cytassist.rows)) !=
            cytassist_bounds) {
            throw std::runtime_error("CytAssist crop lies outside original image");
        }
        cv::Mat cytassist_difference;
        cv::absdiff(cytassist(cytassist_bounds), cytassist_roi, cytassist_difference);
        double cytassist_max_difference = 0.0;
        cv::minMaxLoc(cytassist_difference.reshape(1), nullptr, &cytassist_max_difference);

        const cv::Matx33d sr_full = read_named_matrix(
            args.reference_json, "microscope_fullres_to_cytassist_fullres"
        );
        const cv::Matx33d declared_local = read_named_matrix(
            args.reference_json, "microscope_roi_ds4_to_cytassist_roi"
        );
        const double center_offset = 0.5 * (args.roi_downsample - 1.0);
        const cv::Matx33d ds_to_raw_center(
            args.roi_downsample, 0.0, args.roi_x0 + center_offset,
            0.0, args.roi_downsample, args.roi_y0 + center_offset,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d ds_to_raw_declared(
            args.roi_downsample, 0.0, args.roi_x0,
            0.0, args.roi_downsample, args.roi_y0,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d cytassist_global_to_local(
            1.0, 0.0, -args.cytassist_x0,
            0.0, 1.0, -args.cytassist_y0,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d center_corrected_local =
            cytassist_global_to_local * sr_full * ds_to_raw_center;
        const cv::Matx33d reconstructed_declared_local =
            cytassist_global_to_local * sr_full * ds_to_raw_declared;
        const cv::Matx33d ds_to_work = pixel_center_resize_transform(
            roi_ds_size, roi_ds_size, args.work_size, args.work_size
        );
        const double naive_scale = static_cast<double>(args.work_size) / roi_ds_size;
        const cv::Matx33d ds_to_work_naive(
            naive_scale, 0.0, 0.0,
            0.0, naive_scale, 0.0,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d expected_work_to_fixed =
            center_corrected_local * inverse_matrix(ds_to_work);
        const cv::Matx33d old_scale_bookkeeping_output =
            expected_work_to_fixed * ds_to_work_naive;

        std::ofstream pixels(args.out_dir / "pixel_values.tsv");
        if (!pixels) throw std::runtime_error("cannot write pixel_values.tsv");
        pixels << "sample\tds_x\tds_y\tfixture_b\tfixture_g\tfixture_r\t"
                  "raw_mean_b\traw_mean_g\traw_mean_r\tmax_abs_difference\n";
        pixels << std::setprecision(17);
        std::vector<cv::Point> samples;
        for (int grid_y = 0; grid_y < 9; ++grid_y) {
            for (int grid_x = 0; grid_x < 9; ++grid_x) {
                samples.emplace_back(
                    static_cast<int>(std::lround(grid_x * (roi_ds_size - 1.0) / 8.0)),
                    static_cast<int>(std::lround(grid_y * (roi_ds_size - 1.0) / 8.0))
                );
            }
        }
        samples.emplace_back(1, 1);
        samples.emplace_back(17, 23);
        samples.emplace_back(roi_ds_size - 2, roi_ds_size - 2);
        double raw_fixture_max_difference = 0.0;
        int sample_index = 0;
        for (const cv::Point& point : samples) {
            std::array<double, 3> raw_bgr{};
            for (int dy = 0; dy < args.roi_downsample; ++dy) {
                for (int dx = 0; dx < args.roi_downsample; ++dx) {
                    const auto rgb = microscope.rgb(
                        args.roi_x0 + point.x * args.roi_downsample + dx,
                        args.roi_y0 + point.y * args.roi_downsample + dy
                    );
                    raw_bgr[0] += rgb[2];
                    raw_bgr[1] += rgb[1];
                    raw_bgr[2] += rgb[0];
                }
            }
            const double divisor = args.roi_downsample * args.roi_downsample;
            for (double& value : raw_bgr) value /= divisor;
            const cv::Vec3b fixture = microscope_ds.at<cv::Vec3b>(point.y, point.x);
            double maximum = 0.0;
            for (int channel = 0; channel < 3; ++channel) {
                maximum = std::max(maximum, std::abs(fixture[channel] - raw_bgr[channel]));
            }
            raw_fixture_max_difference = std::max(raw_fixture_max_difference, maximum);
            pixels << sample_index++ << '\t' << point.x << '\t' << point.y << '\t'
                   << static_cast<int>(fixture[0]) << '\t'
                   << static_cast<int>(fixture[1]) << '\t'
                   << static_cast<int>(fixture[2]) << '\t'
                   << raw_bgr[0] << '\t' << raw_bgr[1] << '\t' << raw_bgr[2] << '\t'
                   << maximum << '\n';
        }

        std::ofstream coordinates(args.out_dir / "coordinate_trace.tsv");
        if (!coordinates) throw std::runtime_error("cannot write coordinate_trace.tsv");
        coordinates << std::setprecision(17)
                    << "sample\tds_x\tds_y\traw_center_x\traw_center_y\twork_x\twork_y\t"
                       "roundtrip_ds_x\troundtrip_ds_y\tcorrected_fixed_x\tcorrected_fixed_y\t"
                       "declared_fixed_x\tdeclared_fixed_y\n";
        sample_index = 0;
        for (const cv::Point& point : samples) {
            const cv::Point2d ds(point.x, point.y);
            const cv::Point2d raw = project_point(ds_to_raw_center, ds);
            const cv::Point2d work = project_point(ds_to_work, ds);
            const cv::Point2d roundtrip = project_point(inverse_matrix(ds_to_work), work);
            const cv::Point2d corrected = project_point(center_corrected_local, ds);
            const cv::Point2d declared = project_point(declared_local, ds);
            coordinates << sample_index++ << '\t' << ds.x << '\t' << ds.y << '\t'
                        << raw.x << '\t' << raw.y << '\t' << work.x << '\t' << work.y << '\t'
                        << roundtrip.x << '\t' << roundtrip.y << '\t'
                        << corrected.x << '\t' << corrected.y << '\t'
                        << declared.x << '\t' << declared.y << '\n';
        }

        const GridError fixture_center_error = compare_grid(
            declared_local, center_corrected_local, roi_ds_size, roi_ds_size
        );
        const GridError old_scale_error = compare_grid(
            old_scale_bookkeeping_output,
            center_corrected_local,
            roi_ds_size,
            roi_ds_size
        );
        std::ofstream summary(args.out_dir / "summary.tsv");
        if (!summary) throw std::runtime_error("cannot write summary.tsv");
        summary << std::setprecision(17) << "metric\tvalue\n"
                << "original_microscope_width\t" << microscope.width() << '\n'
                << "original_microscope_height\t" << microscope.height() << '\n'
                << "original_microscope_tile_width\t" << microscope.tile_width() << '\n'
                << "original_microscope_tile_height\t" << microscope.tile_height() << '\n'
                << "cytassist_crop_max_channel_difference\t" << cytassist_max_difference << '\n'
                << "sampled_positions\t" << samples.size() << '\n'
                << "sampled_raw_pixels\t"
                << samples.size() * args.roi_downsample * args.roi_downsample << '\n'
                << "sampled_raw_average_max_channel_difference\t" << raw_fixture_max_difference << '\n'
                << "declared_local_reconstruction_max_matrix_difference\t"
                << matrix_max_abs_difference(declared_local, reconstructed_declared_local) << '\n'
                << "roi_center_offset_raw_pixels\t" << center_offset << '\n'
                << "roi_center_offset_ds4_pixels\t"
                << center_offset / args.roi_downsample << '\n'
                << "internal_resize_offset_work_pixels\t" << ds_to_work(0, 2) << '\n'
                << "fixture_center_error_rmse_fixed_pixels\t" << fixture_center_error.rmse << '\n'
                << "fixture_center_error_mean_x_fixed_pixels\t" << fixture_center_error.mean_x << '\n'
                << "fixture_center_error_mean_y_fixed_pixels\t" << fixture_center_error.mean_y << '\n'
                << "old_scale_bookkeeping_error_rmse_fixed_pixels\t" << old_scale_error.rmse << '\n'
                << "old_scale_bookkeeping_error_mean_x_fixed_pixels\t" << old_scale_error.mean_x << '\n'
                << "old_scale_bookkeeping_error_mean_y_fixed_pixels\t" << old_scale_error.mean_y << '\n';

        std::ofstream matrices(args.out_dir / "matrices.tsv");
        if (!matrices) throw std::runtime_error("cannot write matrices.tsv");
        matrices << std::setprecision(17)
                 << "matrix\tm00\tm01\tm02\tm10\tm11\tm12\tm20\tm21\tm22\n";
        write_matrix(matrices, "sr_full", sr_full);
        write_matrix(matrices, "declared_local", declared_local);
        write_matrix(matrices, "reconstructed_declared_local", reconstructed_declared_local);
        write_matrix(matrices, "center_corrected_local", center_corrected_local);
        write_matrix(matrices, "ds_to_work_pixel_center", ds_to_work);
        write_matrix(matrices, "ds_to_work_naive", ds_to_work_naive);
        write_matrix(matrices, "old_scale_bookkeeping_output", old_scale_bookkeeping_output);

        std::cout << std::setprecision(8)
                  << "cytassist_crop_max=" << cytassist_max_difference
                  << " raw_average_sample_max=" << raw_fixture_max_difference
                  << " fixture_center_rmse=" << fixture_center_error.rmse
                  << " old_scale_rmse=" << old_scale_error.rmse << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visium_hd_registration_pixel_trace: " << error.what() << '\n';
        return 1;
    }
}
