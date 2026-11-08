#pragma once

// Placing image segmentation on the capture grid: each 2 um location's centre is projected through
// the capture-grid transform (grid to CytAssist image) and the inverse of the native image
// registration (microscope image, decimated 16-fold, to CytAssist image) into the microscope image,
// and then into the Cellpose label images; the nucleus and cell labels at that pixel give the
// location's state. A C++ implementation of select_native_capture_grid_roi and
// load_cellpose_states_from_capture_grid of scripts/publication/score_hd_flex_registration_roi.py, over the
// whole registered extent.

#include "hd_count_registration.hpp"
#include "hd_json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace visium_hd::projection {

using count_registration::Matrix3;

// A zarr v3 array of unsigned 32-bit integers with regular chunks, "bytes" (little-endian) and
// optionally "zstd" codecs, read chunk by chunk.
class ZarrArray {
public:
    explicit ZarrArray(const std::string& path);
    std::int64_t height() const { return shape_[0]; }
    std::int64_t width() const { return shape_[1]; }
    std::int64_t chunk_height() const { return chunk_[0]; }
    std::int64_t chunk_width() const { return chunk_[1]; }
    // One chunk, chunk_height x chunk_width values (the fill value where the chunk is absent).
    std::vector<std::uint32_t> chunk(std::int64_t row, std::int64_t col) const;

private:
    std::string path_;
    std::int64_t shape_[2] = {0, 0};
    std::int64_t chunk_[2] = {0, 0};
    std::uint32_t fill_ = 0;
    bool zstd_ = false;
    std::string separator_ = "/";
};

struct Result {
    std::vector<std::uint8_t> nucleus;   // width x width, nucleus at the location's pixel
    std::vector<std::uint8_t> support;   // width x width, location projects into the segmentation image
    json::Value audit;                   // projection.json
};

// width x width grid; registration.json, capture_grid.json and the segmentation directory (summary.json
// and masks/nucleus_labels.zarr, masks/cell_labels.zarr).
Result project(const std::string& segmentation, const std::string& registration, const std::string& capture,
               int width, int threads);

}  // namespace visium_hd::projection
