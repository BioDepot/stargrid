#include "hd_capture_grid_core.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tiffio.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using visium_hd::capture_grid::DetectionOptions;
using visium_hd::capture_grid::Layout;
using visium_hd::capture_grid::Result;

namespace {

struct Arguments {
    std::string cytassist;
    std::string vlf;
    std::string area;
    std::string out_dir;
    std::string reference_alignment;
    DetectionOptions options;
    double maximum_reference_rmse = 1.0;
};

struct CytassistMetadata {
    std::string slide;
    std::string area;
    uint32_t width = 0;
    uint32_t height = 0;
};

constexpr ttag_t kSlideTag = 65002;
constexpr ttag_t kAreaTag = 65010;
static TIFFExtendProc parent_tag_extender = nullptr;

void register_cytassist_tags(TIFF* handle) {
    static const TIFFFieldInfo fields[] = {
        {kSlideTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, const_cast<char*>("CytAssistSlide")},
        {kAreaTag, -1, -1, TIFF_ASCII, FIELD_CUSTOM, 1, 0, const_cast<char*>("CytAssistArea")},
    };
    TIFFMergeFieldInfo(handle, fields, 2);
    if (parent_tag_extender != nullptr) parent_tag_extender(handle);
}

CytassistMetadata read_cytassist_metadata(const fs::path& path) {
    parent_tag_extender = TIFFSetTagExtender(register_cytassist_tags);
    TIFF* handle = TIFFOpen(path.c_str(), "rm");
    TIFFSetTagExtender(parent_tag_extender);
    if (handle == nullptr) throw std::runtime_error("cannot read CytAssist TIFF metadata");
    CytassistMetadata result;
    char* slide = nullptr;
    char* area = nullptr;
    TIFFGetField(handle, TIFFTAG_IMAGEWIDTH, &result.width);
    TIFFGetField(handle, TIFFTAG_IMAGELENGTH, &result.height);
    if (TIFFGetField(handle, kSlideTag, &slide) != 1 || slide == nullptr ||
        TIFFGetField(handle, kAreaTag, &area) != 1 || area == nullptr) {
        TIFFClose(handle);
        throw std::runtime_error("CytAssist TIFF lacks slide or capture-area metadata");
    }
    result.slide = slide;
    result.area = area;
    TIFFClose(handle);
    return result;
}

int parse_int(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno || end == value.c_str() || *end || parsed < 1 || parsed > 1000000) {
        throw std::runtime_error("invalid integer for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

double parse_double(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno || end == value.c_str() || *end || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid number for " + option + ": " + value);
    }
    return parsed;
}

Arguments parse_args(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::runtime_error("missing value for option: " + std::string(argv[index]));
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--cytassist") args.cytassist = value;
        else if (option == "--vlf") args.vlf = value;
        else if (option == "--area") args.area = value;
        else if (option == "--out-dir") args.out_dir = value;
        else if (option == "--reference-alignment") args.reference_alignment = value;
        else if (option == "--threads") args.options.threads = parse_int(value, option);
        else if (option == "--hough-accumulator-threshold") {
            args.options.hough_accumulator_threshold = parse_double(value, option);
        } else if (option == "--maximum-reference-rmse") {
            args.maximum_reference_rmse = parse_double(value, option);
        } else throw std::runtime_error("unknown option: " + option);
    }
    if (args.cytassist.empty() || args.vlf.empty() || args.area.empty() || args.out_dir.empty()) {
        throw std::runtime_error("--cytassist, --vlf, --area, and --out-dir are required");
    }
    if (args.maximum_reference_rmse <= 0.0) {
        throw std::runtime_error("--maximum-reference-rmse must be positive");
    }
    return args;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference alignment: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

cv::Matx33d read_top_level_transform(const std::string& path) {
    const std::string text = read_text(path);
    const size_t key = text.find("\"transform\"");
    if (key == std::string::npos) throw std::runtime_error("reference alignment lacks transform");
    size_t cursor = text.find('[', key);
    cv::Matx33d result;
    for (double& value : result.val) {
        while (cursor < text.size() &&
               !(text[cursor] == '-' || text[cursor] == '+' || text[cursor] == '.' ||
                 (text[cursor] >= '0' && text[cursor] <= '9'))) ++cursor;
        if (cursor == text.size()) throw std::runtime_error("reference transform has fewer than 9 values");
        char* end = nullptr;
        errno = 0;
        value = std::strtod(text.c_str() + cursor, &end);
        if (errno || end == text.c_str() + cursor || !std::isfinite(value)) {
            throw std::runtime_error("invalid reference transform number");
        }
        cursor = static_cast<size_t>(end - text.c_str());
    }
    return result;
}

void write_matrix(std::ostream& output, const cv::Matx33d& matrix, int indent) {
    const std::string pad(static_cast<size_t>(indent), ' ');
    output << "[\n";
    for (int row = 0; row < 3; ++row) {
        output << pad << "  [";
        for (int column = 0; column < 3; ++column) {
            if (column) output << ", ";
            output << std::setprecision(17) << matrix(row, column);
        }
        output << "]" << (row == 2 ? "\n" : ",\n");
    }
    output << pad << "]";
}

struct ReferenceError {
    int points = 0;
    double rmse = 0.0;
    double maximum = 0.0;
    double mean_delta_x = 0.0;
    double mean_delta_y = 0.0;
    double centered_rmse = 0.0;
};

ReferenceError compare_reference(const cv::Matx33d& estimated, const cv::Matx33d& reference) {
    ReferenceError result;
    double squared = 0.0;
    for (int y = 0; y < 9; ++y) {
        for (int x = 0; x < 9; ++x) {
            const cv::Point2d point(-545.0 + 7790.0 * x / 8.0, -545.0 + 7790.0 * y / 8.0);
            const auto left = visium_hd::capture_grid::project_point(estimated, point);
            const auto right = visium_hd::capture_grid::project_point(reference, point);
            const cv::Point2d delta = left - right;
            const double error = cv::norm(delta);
            squared += error * error;
            result.mean_delta_x += delta.x;
            result.mean_delta_y += delta.y;
            result.maximum = std::max(result.maximum, error);
            ++result.points;
        }
    }
    result.rmse = std::sqrt(squared / result.points);
    result.mean_delta_x /= result.points;
    result.mean_delta_y /= result.points;
    result.centered_rmse = std::sqrt(std::max(
        0.0,
        result.rmse * result.rmse
        - result.mean_delta_x * result.mean_delta_x
        - result.mean_delta_y * result.mean_delta_y
    ));
    return result;
}

void write_overlay(const cv::Mat& source, const Result& result, const fs::path& path) {
    cv::Mat output = source.clone();
    for (const auto& circle : result.circles) {
        cv::circle(output, cv::Point(cvRound(circle[0]), cvRound(circle[1])), cvRound(circle[2]),
                   cv::Scalar(255, 120, 0), 1, cv::LINE_AA);
    }
    for (const auto& match : result.matches) {
        cv::circle(output, cv::Point(cvRound(match.projected_xy.x), cvRound(match.projected_xy.y)),
                   39, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
    if (!cv::imwrite(path.string(), output)) throw std::runtime_error("failed to write overlay");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        const fs::path output(args.out_dir);
        if (fs::exists(output)) throw std::runtime_error("refusing to reuse output directory: " + output.string());
        fs::create_directories(output);
        const auto start = std::chrono::steady_clock::now();
        const cv::Mat image = cv::imread(args.cytassist, cv::IMREAD_UNCHANGED);
        if (image.empty()) throw std::runtime_error("cannot read CytAssist image: " + args.cytassist);
        const Layout layout = visium_hd::capture_grid::read_vlf_layout(args.vlf, args.area);
        const CytassistMetadata metadata = read_cytassist_metadata(args.cytassist);
        if (metadata.slide != layout.slide_uid || metadata.area != layout.area) {
            throw std::runtime_error(
                "CytAssist TIFF slide/area metadata differs from the VLF request"
            );
        }
        if (metadata.width != static_cast<uint32_t>(image.cols) ||
            metadata.height != static_cast<uint32_t>(image.rows)) {
            throw std::runtime_error("CytAssist TIFF metadata shape differs from decoded image");
        }
        const Result result = visium_hd::capture_grid::localize_capture_grid(image, layout, args.options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ReferenceError reference;
        const bool reference_scored = !args.reference_alignment.empty();
        if (reference_scored) {
            reference = compare_reference(
                result.design_to_cytassist, read_top_level_transform(args.reference_alignment)
            );
            if (reference.rmse > args.maximum_reference_rmse) {
                throw std::runtime_error(
                    "reference RMSE gate failed: " + std::to_string(reference.rmse) + " pixels"
                );
            }
        }
        write_overlay(image, result, output / "fiducials_overlay.png");
        std::ofstream transform_tsv(output / "spot_colrow_to_cytassist.tsv");
        transform_tsv << std::setprecision(17);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                if (column) transform_tsv << '\t';
                transform_tsv << result.spot_colrow_to_cytassist(row, column);
            }
            transform_tsv << '\n';
        }
        if (!transform_tsv) throw std::runtime_error("failed to write capture-grid transform");
        std::ofstream json(output / "capture_grid.json");
        json << "{\n"
             << "  \"schema\": \"visium_hd_processing.capture_grid.v1\",\n"
             << "  \"slide\": \"" << layout.slide_uid << "\",\n"
             << "  \"area\": \"" << layout.area << "\",\n"
             << "  \"slide_design\": \"" << layout.slide_design << "\",\n"
             << "  \"grid_shape_row_col\": [3350, 3350],\n"
             << "  \"spot_pitch_microns\": 2.0,\n"
             << "  \"cytassist_coordinate_convention\": \"pixel_corner\",\n"
             << "  \"detector_center_to_exported_corner_offset_xy\": [0.5, 0.5],\n"
             << "  \"cytassist_metadata\": {\"slide\": \"" << metadata.slide
             << "\", \"area\": \"" << metadata.area << "\", \"shape_yx\": ["
             << metadata.height << ", " << metadata.width << "]},\n"
             << "  \"inputs\": {\n"
             << "    \"cytassist_image\": \"" << fs::absolute(args.cytassist).string() << "\",\n"
             << "    \"vlf\": \"" << fs::absolute(args.vlf).string() << "\"\n"
             << "  },\n"
             << "  \"vlf_design_correction\": ";
        write_matrix(json, layout.design_correction, 2);
        json << ",\n  \"design_xy_to_cytassist\": ";
        write_matrix(json, result.design_to_cytassist, 2);
        json << ",\n  \"spot_colrow_to_cytassist\": ";
        write_matrix(json, result.spot_colrow_to_cytassist, 2);
        json << ",\n  \"fit\": {\n"
             << "    \"detected_circles\": " << result.detected_circles << ",\n"
             << "    \"matched_fiducials\": " << result.matched_fiducials << ",\n"
             << "    \"inlier_fiducials\": " << result.inlier_fiducials << ",\n"
             << "    \"median_residual_pixels\": " << std::setprecision(17) << result.median_residual_pixels << ",\n"
             << "    \"root_mean_square_residual_pixels\": " << result.root_mean_square_residual_pixels << ",\n"
             << "    \"maximum_residual_pixels\": " << result.maximum_residual_pixels << "\n"
             << "  },\n"
             << "  \"reference_audit\": {\n"
             << "    \"enabled\": " << (reference_scored ? "true" : "false") << ",\n"
             << "    \"sampled_points\": " << reference.points << ",\n"
             << "    \"root_mean_square_pixels\": " << reference.rmse << ",\n"
             << "    \"maximum_pixels\": " << reference.maximum << ",\n"
             << "    \"mean_delta_x_pixels\": " << reference.mean_delta_x << ",\n"
             << "    \"mean_delta_y_pixels\": " << reference.mean_delta_y << ",\n"
             << "    \"centered_root_mean_square_pixels\": " << reference.centered_rmse << "\n"
             << "  },\n"
             << "  \"invariants\": {\n"
             << "    \"space_ranger_alignment_used_for_fit\": false,\n"
             << "    \"space_ranger_tissue_positions_used_for_fit\": false,\n"
             << "    \"detector_center_to_exported_corner_applied_once\": true,\n"
             << "    \"orientation_preserving_transform\": true\n"
             << "  }\n"
             << "}\n";
        std::cout << "matched=" << result.matched_fiducials
                  << " residual_rmse=" << result.root_mean_square_residual_pixels
                  << " reference_rmse=" << reference.rmse
                  << " seconds=" << seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visium_hd_capture_grid: " << error.what() << '\n';
        return 1;
    }
}
