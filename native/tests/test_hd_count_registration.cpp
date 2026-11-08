#include "hd_count_registration.hpp"
#include "hd_json.hpp"
#include "hd_npy.hpp"
#include "hd_projection.hpp"
#include "hd_sha256.hpp"

#include <zstd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace cr = visium_hd::count_registration;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAILED: " << what << "\n";
    }
}

// Reference draws of numpy 1.26: np.random.default_rng(seed).integers(high, size=n), flattened.
void test_numpy_generator() {
    struct Case {
        std::uint64_t seed, high;
        std::vector<std::uint64_t> expected;
    };
    const std::vector<Case> cases = {
        {20261007, 7, {5, 1, 5, 6, 2, 3, 3, 6, 1, 2, 4, 3, 4, 1, 3, 6, 3, 1, 6, 0, 5}},
        {20261007, 1156, {855, 229, 841, 1126, 494, 581, 627, 1050, 213, 381, 772, 495, 662, 311, 629, 1025}},
        {1, 49, {23, 25, 37, 46, 1, 7, 40, 46, 12, 15, 42, 20, 13, 40, 12, 20}},
        {20261007, 2147483648ull, {1588763851, 427182710, 1562524087, 2092349115, 918007280, 1080243662}},
        {20261007, 4294967296ull, {3177527703ull, 854365420ull, 3125048174ull}},
    };
    for (const Case& c : cases) {
        cr::NumpyGenerator generator(c.seed);
        bool same = true;
        for (std::uint64_t value : c.expected) same = same && generator.integers(c.high) == value;
        expect(same, "numpy integers, seed " + std::to_string(c.seed) + ", high " + std::to_string(c.high));
    }
    const std::vector<double> w = cr::block_weights(7, 3, 20261007);
    // Draws of the first resample: 5 1 5 6 2 3 3.
    const std::vector<double> first = {0, 1, 1, 2, 0, 2, 1};
    expect(std::equal(first.begin(), first.end(), w.begin()), "block weights of the first resample");
}

void test_python_float() {
    using visium_hd::json::python_float;
    const std::vector<std::pair<double, std::string>> cases = {
        {0.0001, "0.0001"}, {0.00001, "1e-05"}, {2.0, "2.0"}, {1e16, "1e+16"}, {9999999999999998.0, "9999999999999998.0"},
        {123.456, "123.456"}, {-0.5, "-0.5"}, {0.1 + 0.2, "0.30000000000000004"}, {1.5e-7, "1.5e-07"}, {1e100, "1e+100"},
        {0.75, "0.75"}, {100000.0, "100000.0"}};
    for (const auto& c : cases) expect(python_float(c.first) == c.second, "python float " + c.second + " got " + python_float(c.first));
}

std::vector<std::uint8_t> disks(cr::Grid grid, int count, double radius, std::mt19937& rng) {
    std::vector<std::uint8_t> out(grid.size(), 0);
    std::uniform_real_distribution<double> uy(0, grid.height), ux(0, grid.width);
    for (int k = 0; k < count; ++k) {
        const double cy = uy(rng), cx = ux(rng);
        for (int y = std::max(0, int(cy - radius - 1)); y < std::min(grid.height, int(cy + radius + 2)); ++y) {
            for (int x = std::max(0, int(cx - radius - 1)); x < std::min(grid.width, int(cx + radius + 2)); ++x) {
                if ((y - cy) * (y - cy) + (x - cx) * (x - cx) <= radius * radius) out[size_t(y) * grid.width + x] = 1;
            }
        }
    }
    return out;
}

void test_band_labels() {
    std::mt19937 rng(7);
    const cr::Grid grid{61, 47};
    const std::vector<std::uint8_t> nucleus = disks(grid, 12, 3.5, rng);
    const std::vector<std::int8_t> labels = cr::band_labels(nucleus, grid, 3);
    bool same = true;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const bool in = nucleus[size_t(y) * grid.width + x];
            long best = -1;
            for (int v = 0; v < grid.height; ++v) {
                for (int u = 0; u < grid.width; ++u) {
                    if (bool(nucleus[size_t(v) * grid.width + u]) == in) continue;
                    const long d = long(v - y) * (v - y) + long(u - x) * (u - x);
                    if (best < 0 || d < best) best = d;
                }
            }
            int expected = -1;
            if (in) {
                expected = best < 0 || best > 4 ? 0 : best > 1 ? 1 : 2;
            } else if (best >= 0) {
                for (int k = 1; k <= 6; ++k) {
                    if (best > (k - 1) * (k - 1) && best <= k * k) expected = 2 + k;
                }
            }
            same = same && labels[size_t(y) * grid.width + x] == expected;
        }
    }
    expect(same, "band labels equal brute-force distances");
}

void test_corrected() {
    std::mt19937 rng(3);
    const cr::Grid grid{23, 31};
    std::vector<double> counts(grid.size());
    std::vector<std::uint8_t> support(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        counts[i] = double(rng() % 7);
        support[i] = rng() % 4 != 0;
    }
    const std::vector<double> values = cr::corrected(counts, support, grid, 2, 2);
    double worst = 0.0;
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const size_t i = size_t(y) * grid.width + x;
            double c = 0, f = 0;
            for (int v = y - 2; v <= y + 2; ++v) {
                for (int u = x - 2; u <= x + 2; ++u) {
                    if (v < 0 || u < 0 || v >= grid.height || u >= grid.width) continue;
                    const size_t j = size_t(v) * grid.width + u;
                    if (support[j]) {
                        c += counts[j];
                        f += 1;
                    }
                }
            }
            const double mean = support[i] && f > 0 ? c / f : 0.0;
            const double expected = mean > 0 ? counts[i] / mean : 0.0;
            worst = std::max(worst, std::abs(values[i] - expected));
        }
    }
    expect(worst < 1e-12, "corrected counts equal direct sums");
}

// A planted displacement: counts high in a ring around the nuclei and low under them, the nuclei
// moved by (dy, dx) relative to the counts.
void test_planted_shift() {
    std::mt19937 rng(11);
    const cr::Grid grid{600, 600};
    const int dy = 3, dx = -2;
    const std::vector<std::uint8_t> counts_nuclei = disks(grid, 2500, 3.0, rng);
    const std::vector<std::int8_t> bands = cr::band_labels(counts_nuclei, grid, 4);
    std::poisson_distribution<int> low(2.0), ring(9.0), rest(5.0);
    std::vector<double> counts(grid.size());
    for (size_t i = 0; i < grid.size(); ++i) {
        const int b = bands[i];
        counts[i] = b >= 0 && b <= 2 ? low(rng) : (b == 3 || b == 4) ? ring(rng) : rest(rng);
    }
    const std::vector<std::uint8_t> mask = cr::shift_grid(counts_nuclei, grid, -dy, -dx);
    const std::vector<std::uint8_t> support(grid.size(), 1);
    cr::Options options;
    options.threads = 4;
    options.minimum = 1000;
    const cr::Check check = cr::check(mask, counts, support, grid, options);
    expect(check.whole.usable && check.whole.dy == dy && check.whole.dx == dx,
           "planted shift recovered (" + check.whole.reason + ")");
    int found = 0;
    for (const cr::Region& r : check.regions) found += r.usable() && r.assessment.dy == dy && r.assessment.dx == dx;
    expect(found >= 20, "planted shift recovered in most regions: " + std::to_string(found));
    const cr::AffineModel model = cr::fit_regions(check.regions);
    expect(model.accepted, "affine fit of a constant shift accepted");
    if (model.accepted) {
        expect(std::abs(model.final_fit->coefficients[0][0] - dy) < 1e-9 && std::abs(model.final_fit->coefficients[0][1] - dx) < 1e-9,
               "affine fit of a constant shift is that shift");
    }
    const cr::Check after = cr::check(cr::shift_grid(mask, grid, dy, dx), counts, support, grid, options);
    expect(after.whole.usable && after.whole.dy == 0 && after.whole.dx == 0, "corrected mask lies at zero");
}

void test_affine_moves() {
    const cr::Grid grid{50, 40};
    std::vector<std::uint8_t> labels(grid.size(), 0);
    labels[10 * 40 + 10] = 1;
    labels[49 * 40 + 39] = 1;
    const std::vector<std::uint8_t> moved = cr::apply_affine(labels, grid, {{{1.4, -2.5}, {0, 0}, {0, 0}}});
    // 1.4 rounds to 1; -2.5 rounds half to even, to -2; the corner moves off the grid.
    expect(moved[11 * 40 + 8] == 1 && std::count(moved.begin(), moved.end(), 1) == 1, "affine move rounds half to even");
    const cr::Matrix3 m{{{1.01, 0.002, 3.0}, {-0.001, 0.99, -2.0}, {0.0, 0.0, 1.0}}};
    const cr::Matrix3 product = cr::multiply(m, cr::inverse(m));
    double worst = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) worst = std::max(worst, std::abs(product[i][j] - (i == j ? 1.0 : 0.0)));
    }
    expect(worst < 1e-14, "3 x 3 inverse");
}

void test_npz_round_trip() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "test_hd_count_registration.npz";
    const std::vector<double> values = {1.5, -2.0, 3.25, 0.0, 7.0, 8.0};
    visium_hd::npy::write_npz(path.string(), {{"total", "<f8", {2, 3}, values.data(), values.size() * sizeof(double)}});
    const visium_hd::npy::Array array = visium_hd::npy::read(path.string(), "total");
    expect(array.shape == std::vector<std::size_t>{2, 3} && visium_hd::npy::as_doubles(array) == values, "npz round trip");
    std::filesystem::remove(path);
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

// A zarr v3 uint32 array in 8 x 8 chunks with the zstd codec, as the Cellpose adapter writes.
void write_zarr(const std::filesystem::path& root, const std::vector<std::uint32_t>& values, int height, int width) {
    std::filesystem::create_directories(root);
    write_text(root / "zarr.json",
               "{\"shape\": [" + std::to_string(height) + ", " + std::to_string(width) + "], \"data_type\": \"uint32\", "
               "\"chunk_grid\": {\"name\": \"regular\", \"configuration\": {\"chunk_shape\": [8, 8]}}, "
               "\"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}}, "
               "\"fill_value\": 0, \"codecs\": [{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}, "
               "{\"name\": \"zstd\", \"configuration\": {\"level\": 0, \"checksum\": false}}], \"attributes\": {}, "
               "\"zarr_format\": 3, \"node_type\": \"array\", \"storage_transformers\": []}");
    for (int cy = 0; cy * 8 < height; ++cy) {
        for (int cx = 0; cx * 8 < width; ++cx) {
            std::vector<std::uint32_t> chunk(64, 0);
            bool any = false;
            for (int y = 0; y < 8; ++y) {
                for (int x = 0; x < 8; ++x) {
                    const int gy = cy * 8 + y, gx = cx * 8 + x;
                    if (gy < height && gx < width) chunk[y * 8 + x] = values[size_t(gy) * width + gx];
                    any = any || chunk[y * 8 + x];
                }
            }
            if (!any) continue;   // absent chunks read as the fill value
            std::string packed(ZSTD_compressBound(256), '\0');
            packed.resize(ZSTD_compress(packed.data(), packed.size(), chunk.data(), 256, 3));
            std::filesystem::create_directories(root / "c" / std::to_string(cy));
            write_text(root / "c" / std::to_string(cy) / std::to_string(cx), packed);
        }
    }
}

// A synthetic case: identity registration and a capture grid that puts the
// centre of grid location (row, col) on segmentation pixel (row, col).
void test_projection() {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "test_hd_count_registration_projection";
    fs::remove_all(root);
    const int size = 20;
    std::vector<std::uint32_t> labels(size * size, 0);
    for (int y = 3; y < 9; ++y) for (int x = 5; x < 11; ++x) labels[y * size + x] = 1;
    for (int y = 12; y < 17; ++y) for (int x = 12; x < 18; ++x) labels[y * size + x] = 2;
    write_zarr(root / "segmentation/masks/nucleus_labels.zarr", labels, size, size);
    write_zarr(root / "segmentation/masks/cell_labels.zarr", labels, size, size);
    write_text(root / "segmentation/summary.json",
               "{\"input_contract\": {\"expression_or_vendor_segmentation_loaded\": false, \"source_roi_xyxy\": [0, 0, 40, 40], "
               "\"downsample\": 2, \"segmentation_shape_yx\": [20, 20], \"source_shape_yx\": [40, 40]}, "
               "\"counts\": {\"nucleus_outside_cell_pixels\": 0}, \"outputs\": {\"nucleus_labels.zarr\": \""
               + visium_hd::tree_sha256((root / "segmentation/masks/nucleus_labels.zarr").string()) + "\", \"cell_labels.zarr\": \""
               + visium_hd::tree_sha256((root / "segmentation/masks/cell_labels.zarr").string()) + "\"}}");
    write_text(root / "registration.json",
               "{\"moving_to_fixed\": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], \"moving_shape_yx\": [3, 3], \"fixed_shape_yx\": [3, 3]}");
    write_text(root / "capture.json", "{\"spot_colrow_to_cytassist\": [[0.125, 0, 0.03125], [0, 0.125, 0.03125], [0, 0, 1]]}");
    const visium_hd::projection::Result result = visium_hd::projection::project(
        (root / "segmentation").string(), (root / "registration.json").string(), (root / "capture.json").string(), size, 2);
    bool same = true;
    for (int i = 0; i < size * size; ++i) same = same && result.nucleus[i] == (labels[i] > 0) && result.support[i] == 1;
    expect(same, "projection places each grid centre on its segmentation pixel");
    const auto& h = result.audit.at("grid_to_full_resolution_image");
    expect(h[0][0].as_double() == 2.0 && h[0][2].as_double() == 0.5 && h[1][1].as_double() == 2.0 && h[1][2].as_double() == 0.5,
           "projection's grid-to-image transform");
    write_text(root / "segmentation/masks/cell_labels.zarr/c/0/0", "corrupt");
    bool refused = false;
    try {
        visium_hd::projection::project((root / "segmentation").string(), (root / "registration.json").string(),
                                       (root / "capture.json").string(), size, 2);
    } catch (const std::exception&) {
        refused = true;
    }
    expect(refused, "projection refuses a label tree that differs from its declared hash");
    // Known SHA-256 digests.
    visium_hd::Sha256 empty;
    expect(empty.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 of nothing");
    visium_hd::Sha256 abc;
    abc.update("abc");
    expect(abc.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 of abc");
    fs::remove_all(root);
}

}  // namespace

int main() {
    test_projection();
    test_numpy_generator();
    test_python_float();
    test_band_labels();
    test_corrected();
    test_affine_moves();
    test_npz_round_trip();
    test_planted_shift();
    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "all count-registration tests passed\n";
    return 0;
}
