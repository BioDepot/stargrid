#include "hd_registration_core.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using visium_hd::registration::Options;
using visium_hd::registration::Result;
using visium_hd::registration::AlignmentScore;
using visium_hd::registration::SpatialResidualCell;
using visium_hd::registration::SpatialCorrectionEvaluation;
using visium_hd::registration::GlobalOptimizationEvaluation;
using visium_hd::registration::TransformError;

namespace {

struct Args {
    std::string moving;
    std::string fixed;
    std::string out_dir;
    std::string reference_json;
    std::string reference_key;
    double reference_moving_scale_x = 1.0;
    double reference_moving_scale_y = 1.0;
    double reference_moving_offset_x = 0.0;
    double reference_moving_offset_y = 0.0;
    double maximum_reference_rmse = -1.0;
    int spatial_audit_grid_size = 6;
    double spatial_audit_x0 = -1.0;
    double spatial_audit_y0 = -1.0;
    double spatial_audit_x1 = -1.0;
    double spatial_audit_y1 = -1.0;
    bool apply_spatial_correction = false;
    bool evaluate_global_optimization = false;
    bool select_global_candidate = false;
    int global_optimization_iterations = 60;
    Options options;
};

struct ReferenceMatchSummary {
    int inliers = 0;
    int native_better = 0;
    int reference_better = 0;
    int equal = 0;
    double native_mean_error = 0.0;
    double reference_mean_error = 0.0;
    double native_median_error = 0.0;
    double reference_median_error = 0.0;
};

struct ReferenceImageSummary {
    AlignmentScore native;
    AlignmentScore retained_reference;
    AlignmentScore phase_shifted_native;
};

struct SpatialResidualSummary {
    int cells = 0;
    int quality_pass_cells = 0;
    int neighbor_pairs = 0;
    int landmark_supported_cells = 0;
    double median_phase_magnitude = 0.0;
    double maximum_phase_magnitude = 0.0;
    double neighbor_difference_rmse = 0.0;
    double maximum_landmark_mean_magnitude = 0.0;
};

std::string usage() {
    return R"USAGE(usage: visium_hd_register --moving IMAGE --fixed IMAGE --out-dir DIR [options]

Estimate a microscope-to-CytAssist homography from image pixels. A reference
transform is optional and is used only after estimation for error scoring.

Required:
  --moving FILE                    microscope image or downsampled ROI
  --fixed FILE                     CytAssist image or corresponding ROI
  --out-dir DIR                    new output directory

Registration options:
  --threads INT                    OpenCV/ROI worker threads; >1 enables fast mode
  --max-features INT               SIFT feature limit (default: 12000)
  --work-max-dimension INT         internal maximum dimension (default: 4096)
  --transform-model NAME           similarity (default), affine, or homography
  --ratio-test FLOAT               descriptor ratio threshold (default: 0.70)
  --ransac-threshold FLOAT         RANSAC error in work pixels (default: 1)
  --minimum-good-matches INT       minimum ratio-test matches (default: 20)
  --minimum-inliers INT            minimum transform inliers (default: 14)
  --mutual-matching yes|no         require bidirectional ratio match (default: no)
  --match-scale-policy NAME        smaller (default), larger, or native
  --apply-residual-refinement yes|no apply integer NCC shift (default: no)
  --fixed-frame-offset-x FLOAT     calibrated fixed-frame x offset (default: 0.5)
  --fixed-frame-offset-y FLOAT     calibrated fixed-frame y offset (default: 0)

Spatial residual audit:
  --spatial-audit-grid-size INT    regular grid dimension (default: 6)
  --spatial-audit-x0 FLOAT         optional fixed-image sub-ROI left edge
  --spatial-audit-y0 FLOAT         optional fixed-image sub-ROI top edge
  --spatial-audit-x1 FLOAT         optional fixed-image sub-ROI right edge
  --spatial-audit-y1 FLOAT         optional fixed-image sub-ROI bottom edge
  --apply-spatial-correction yes|no apply experimental smooth ROI field (default: no)
  --global-optimize yes|no        evaluate one global ROI transform (default: no)
  --select-global-candidate yes|no materialize image-only winner (default: no)
  --global-optimize-iterations INT ECC iteration cap (default: 60)

Post-estimation oracle scoring:
  --reference-transform-json FILE  JSON containing a named 3x3 matrix
  --reference-key NAME             matrix field name in that JSON
  --reference-moving-scale FLOAT   apply equal x/y scale before reference
  --reference-moving-scale-x FLOAT apply x scale before reference
  --reference-moving-scale-y FLOAT apply y scale before reference
  --reference-moving-offset-x FLOAT apply x offset before reference
  --reference-moving-offset-y FLOAT apply y offset before reference
  --maximum-reference-rmse FLOAT   fail after writing outputs when exceeded
)USAGE";
}

double parse_double(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value.c_str(), &end);
    if (errno != 0 || end == value.c_str() || *end != '\0' || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid value for " + option + ": " + value);
    }
    return parsed;
}

int parse_int(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno != 0 || end == value.c_str() || *end != '\0' || parsed < 1 || parsed > 100000000) {
        throw std::runtime_error("invalid value for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

bool parse_bool(const std::string& value, const std::string& option) {
    if (value == "yes" || value == "true" || value == "1") return true;
    if (value == "no" || value == "false" || value == "0") return false;
    throw std::runtime_error("invalid value for " + option + ": " + value);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            std::cout << usage();
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::runtime_error("missing value for " + option);
        const std::string value = argv[++index];
        if (option == "--moving") args.moving = value;
        else if (option == "--fixed") args.fixed = value;
        else if (option == "--out-dir") args.out_dir = value;
        else if (option == "--threads") {
            args.options.threads = parse_int(value, option);
            args.options.deterministic = args.options.threads == 1;
        }
        else if (option == "--max-features") args.options.max_features = parse_int(value, option);
        else if (option == "--work-max-dimension") args.options.work_max_dimension = parse_int(value, option);
        else if (option == "--transform-model") args.options.transform_model = value;
        else if (option == "--ratio-test") args.options.ratio_test = parse_double(value, option);
        else if (option == "--ransac-threshold") {
            args.options.ransac_threshold_pixels = parse_double(value, option);
        } else if (option == "--minimum-good-matches") {
            args.options.minimum_good_matches = parse_int(value, option);
        } else if (option == "--minimum-inliers") {
            args.options.minimum_inliers = parse_int(value, option);
        } else if (option == "--mutual-matching") {
            args.options.mutual_matching = parse_bool(value, option);
        } else if (option == "--match-scale-policy") {
            args.options.match_scale_policy = value;
        } else if (option == "--apply-residual-refinement") {
            args.options.apply_residual_refinement = parse_bool(value, option);
        } else if (option == "--fixed-frame-offset-x") {
            args.options.fixed_frame_offset_x_pixels = parse_double(value, option);
        } else if (option == "--fixed-frame-offset-y") {
            args.options.fixed_frame_offset_y_pixels = parse_double(value, option);
        } else if (option == "--spatial-audit-grid-size") {
            args.spatial_audit_grid_size = parse_int(value, option);
        } else if (option == "--spatial-audit-x0") {
            args.spatial_audit_x0 = parse_double(value, option);
        } else if (option == "--spatial-audit-y0") {
            args.spatial_audit_y0 = parse_double(value, option);
        } else if (option == "--spatial-audit-x1") {
            args.spatial_audit_x1 = parse_double(value, option);
        } else if (option == "--spatial-audit-y1") {
            args.spatial_audit_y1 = parse_double(value, option);
        } else if (option == "--apply-spatial-correction") {
            args.apply_spatial_correction = parse_bool(value, option);
        } else if (option == "--global-optimize") {
            args.evaluate_global_optimization = parse_bool(value, option);
        } else if (option == "--select-global-candidate") {
            args.select_global_candidate = parse_bool(value, option);
        } else if (option == "--global-optimize-iterations") {
            args.global_optimization_iterations = parse_int(value, option);
        } else if (option == "--reference-transform-json") args.reference_json = value;
        else if (option == "--reference-key") args.reference_key = value;
        else if (option == "--reference-moving-scale") {
            const double scale = parse_double(value, option);
            args.reference_moving_scale_x = scale;
            args.reference_moving_scale_y = scale;
        } else if (option == "--reference-moving-scale-x") {
            args.reference_moving_scale_x = parse_double(value, option);
        } else if (option == "--reference-moving-scale-y") {
            args.reference_moving_scale_y = parse_double(value, option);
        } else if (option == "--reference-moving-offset-x") {
            args.reference_moving_offset_x = parse_double(value, option);
        } else if (option == "--reference-moving-offset-y") {
            args.reference_moving_offset_y = parse_double(value, option);
        } else if (option == "--maximum-reference-rmse") {
            args.maximum_reference_rmse = parse_double(value, option);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (args.moving.empty() || args.fixed.empty() || args.out_dir.empty()) {
        throw std::runtime_error("--moving, --fixed, and --out-dir are required");
    }
    if (args.reference_json.empty() != args.reference_key.empty()) {
        throw std::runtime_error(
            "--reference-transform-json and --reference-key must be supplied together"
        );
    }
    if (args.options.ratio_test <= 0.0 || args.options.ratio_test >= 1.0) {
        throw std::runtime_error("--ratio-test must be between zero and one");
    }
    if (args.options.ransac_threshold_pixels <= 0.0) {
        throw std::runtime_error("--ransac-threshold must be positive");
    }
    if (args.spatial_audit_grid_size < 2 || args.spatial_audit_grid_size > 32) {
        throw std::runtime_error("--spatial-audit-grid-size must be between 2 and 32");
    }
    const int spatial_roi_values =
        (args.spatial_audit_x0 >= 0.0 ? 1 : 0) +
        (args.spatial_audit_y0 >= 0.0 ? 1 : 0) +
        (args.spatial_audit_x1 >= 0.0 ? 1 : 0) +
        (args.spatial_audit_y1 >= 0.0 ? 1 : 0);
    if (spatial_roi_values != 0 && spatial_roi_values != 4) {
        throw std::runtime_error("all four spatial-audit ROI bounds must be supplied together");
    }
    if (spatial_roi_values == 4 &&
        (args.spatial_audit_x1 <= args.spatial_audit_x0 ||
         args.spatial_audit_y1 <= args.spatial_audit_y0)) {
        throw std::runtime_error("spatial-audit ROI bounds are empty");
    }
    return args;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference JSON: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

cv::Matx33d read_named_matrix(const std::string& path, const std::string& key) {
    const std::string text = read_text(path);
    const std::string quoted_key = "\"" + key + "\"";
    size_t cursor = text.find(quoted_key);
    if (cursor == std::string::npos) throw std::runtime_error("reference key not found: " + key);
    cursor = text.find('[', cursor + quoted_key.size());
    if (cursor == std::string::npos) throw std::runtime_error("reference matrix has no array");
    cv::Matx33d matrix;
    for (int value_index = 0; value_index < 9; ++value_index) {
        while (cursor < text.size() &&
               !(text[cursor] == '-' || text[cursor] == '+' || text[cursor] == '.' ||
                 (text[cursor] >= '0' && text[cursor] <= '9'))) {
            ++cursor;
        }
        if (cursor >= text.size()) throw std::runtime_error("reference matrix has fewer than 9 values");
        const char* begin = text.c_str() + cursor;
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(begin, &end);
        if (errno != 0 || end == begin || !std::isfinite(value)) {
            throw std::runtime_error("invalid number in reference matrix");
        }
        matrix(value_index / 3, value_index % 3) = value;
        cursor = static_cast<size_t>(end - text.c_str());
    }
    return matrix;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (const char character : value) {
        if (character == '\\') output << "\\\\";
        else if (character == '"') output << "\\\"";
        else if (character == '\n') output << "\\n";
        else output << character;
    }
    return output.str();
}

void write_matrix_json(std::ostream& output, const cv::Matx33d& matrix, int indent) {
    const std::string spaces(static_cast<size_t>(indent), ' ');
    output << "[\n";
    for (int row = 0; row < 3; ++row) {
        output << spaces << "  [";
        for (int col = 0; col < 3; ++col) {
            if (col) output << ", ";
            output << std::setprecision(17) << matrix(row, col);
        }
        output << "]" << (row == 2 ? "\n" : ",\n");
    }
    output << spaces << "]";
}

void write_transform_tsv(const fs::path& path, const cv::Matx33d& matrix) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write transform: " + path.string());
    output << std::setprecision(17);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            if (col) output << '\t';
            output << matrix(row, col);
        }
        output << '\n';
    }
}

double normalized_edge_distance(double x, double y, int width, int height) {
    const double distance = std::min({x, y, width - 1.0 - x, height - 1.0 - y});
    return std::max(0.0, distance) / std::max(1.0, static_cast<double>(std::min(width, height)));
}

std::pair<double, double> local_color_summary(const cv::Mat& image, double x, double y) {
    const int center_x = std::clamp(static_cast<int>(std::lround(x)), 0, image.cols - 1);
    const int center_y = std::clamp(static_cast<int>(std::lround(y)), 0, image.rows - 1);
    const int x0 = std::max(0, center_x - 2);
    const int y0 = std::max(0, center_y - 2);
    const int x1 = std::min(image.cols, center_x + 3);
    const int y1 = std::min(image.rows, center_y + 3);
    const cv::Scalar mean = cv::mean(image(cv::Rect(x0, y0, x1 - x0, y1 - y0)));
    const double maximum = std::max({mean[0], mean[1], mean[2]});
    const double minimum = std::min({mean[0], mean[1], mean[2]});
    const double saturation = maximum > 0.0 ? (maximum - minimum) / maximum : 0.0;
    const double brightness = (mean[0] + mean[1] + mean[2]) / (3.0 * 255.0);
    return {saturation, brightness};
}

void write_match_diagnostics(
    const fs::path& path,
    const Result& result,
    const cv::Mat& moving,
    const cv::Mat& fixed
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write match diagnostics: " + path.string());
    output << "match_index\tinlier\tmoving_x\tmoving_y\tfixed_x\tfixed_y\t"
              "projected_x\tprojected_y\tdelta_x\tdelta_y\treprojection_error\t"
              "descriptor_distance\tdescriptor_ratio\tmoving_edge_fraction\t"
              "fixed_edge_fraction\tmoving_saturation\tfixed_saturation\t"
              "moving_brightness\tfixed_brightness\n";
    output << std::setprecision(17);
    for (size_t index = 0; index < result.matches.size(); ++index) {
        const auto& match = result.matches[index];
        const auto moving_color = local_color_summary(moving, match.moving_x, match.moving_y);
        const auto fixed_color = local_color_summary(fixed, match.fixed_x, match.fixed_y);
        output << index << '\t' << (match.inlier ? 1 : 0) << '\t'
               << match.moving_x << '\t' << match.moving_y << '\t'
               << match.fixed_x << '\t' << match.fixed_y << '\t'
               << match.projected_x << '\t' << match.projected_y << '\t'
               << match.projected_x - match.fixed_x << '\t'
               << match.projected_y - match.fixed_y << '\t'
               << match.reprojection_error << '\t'
               << match.descriptor_distance << '\t' << match.descriptor_ratio << '\t'
               << normalized_edge_distance(
                      match.moving_x, match.moving_y, moving.cols, moving.rows
                  ) << '\t'
               << normalized_edge_distance(
                      match.fixed_x, match.fixed_y, fixed.cols, fixed.rows
                  ) << '\t'
               << moving_color.first << '\t' << fixed_color.first << '\t'
               << moving_color.second << '\t' << fixed_color.second << '\n';
    }
}

void write_match_overlay(const fs::path& path, const Result& result, const cv::Mat& fixed) {
    cv::Mat diagnostic = fixed.clone();
    for (const auto& match : result.matches) {
        if (match.inlier) continue;
        const cv::Point observed(
            static_cast<int>(std::lround(match.fixed_x)),
            static_cast<int>(std::lround(match.fixed_y))
        );
        const cv::Point projected(
            static_cast<int>(std::lround(match.projected_x)),
            static_cast<int>(std::lround(match.projected_y))
        );
        cv::line(diagnostic, projected, observed, cv::Scalar(0, 165, 255), 1, cv::LINE_AA);
        cv::circle(diagnostic, observed, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::circle(diagnostic, projected, 2, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
    for (const auto& match : result.matches) {
        if (!match.inlier) continue;
        cv::circle(
            diagnostic,
            cv::Point(
                static_cast<int>(std::lround(match.fixed_x)),
                static_cast<int>(std::lround(match.fixed_y))
            ),
            1,
            cv::Scalar(0, 160, 0),
            cv::FILLED,
            cv::LINE_AA
        );
    }
    if (!cv::imwrite(path.string(), diagnostic)) {
        throw std::runtime_error("failed writing match diagnostic image");
    }
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0
               ? (values[middle - 1] + values[middle]) / 2.0
               : values[middle];
}

ReferenceMatchSummary write_reference_match_diagnostics(
    const fs::path& path,
    const Result& result,
    const cv::Matx33d& reference
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write reference match diagnostics: " + path.string());
    output << "match_index\tinlier\tmoving_x\tmoving_y\tfixed_x\tfixed_y\t"
              "native_projected_x\tnative_projected_y\treference_projected_x\t"
              "reference_projected_y\tnative_error\treference_error\t"
              "native_minus_reference_error\tdescriptor_ratio\n";
    output << std::setprecision(17);
    ReferenceMatchSummary summary;
    std::vector<double> native_errors;
    std::vector<double> reference_errors;
    double native_sum = 0.0;
    double reference_sum = 0.0;
    for (size_t index = 0; index < result.matches.size(); ++index) {
        const auto& match = result.matches[index];
        const cv::Point2d reference_point = visium_hd::registration::project_point(
            reference, {match.moving_x, match.moving_y}
        );
        const double reference_error = cv::norm(
            reference_point - cv::Point2d(match.fixed_x, match.fixed_y)
        );
        const double difference = match.reprojection_error - reference_error;
        output << index << '\t' << (match.inlier ? 1 : 0) << '\t'
               << match.moving_x << '\t' << match.moving_y << '\t'
               << match.fixed_x << '\t' << match.fixed_y << '\t'
               << match.projected_x << '\t' << match.projected_y << '\t'
               << reference_point.x << '\t' << reference_point.y << '\t'
               << match.reprojection_error << '\t' << reference_error << '\t'
               << difference << '\t' << match.descriptor_ratio << '\n';
        if (!match.inlier) continue;
        ++summary.inliers;
        if (difference < -1e-12) ++summary.native_better;
        else if (difference > 1e-12) ++summary.reference_better;
        else ++summary.equal;
        native_sum += match.reprojection_error;
        reference_sum += reference_error;
        native_errors.push_back(match.reprojection_error);
        reference_errors.push_back(reference_error);
    }
    if (summary.inliers) {
        summary.native_mean_error = native_sum / summary.inliers;
        summary.reference_mean_error = reference_sum / summary.inliers;
        summary.native_median_error = median(native_errors);
        summary.reference_median_error = median(reference_errors);
    }
    return summary;
}

void write_reference_grid_diagnostics(
    const fs::path& path,
    const cv::Matx33d& estimated,
    const cv::Matx33d& reference,
    int moving_width,
    int moving_height,
    int grid_size = 9
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write reference grid diagnostics: " + path.string());
    output << "grid_row\tgrid_col\tmoving_x\tmoving_y\tnative_x\tnative_y\t"
              "reference_x\treference_y\tdelta_x\tdelta_y\terror\n";
    output << std::setprecision(17);
    for (int row = 0; row < grid_size; ++row) {
        const double y = (moving_height - 1.0) * row / (grid_size - 1.0);
        for (int col = 0; col < grid_size; ++col) {
            const double x = (moving_width - 1.0) * col / (grid_size - 1.0);
            const cv::Point2d native = visium_hd::registration::project_point(estimated, {x, y});
            const cv::Point2d retained = visium_hd::registration::project_point(reference, {x, y});
            const cv::Point2d delta = native - retained;
            output << row << '\t' << col << '\t' << x << '\t' << y << '\t'
                   << native.x << '\t' << native.y << '\t'
                   << retained.x << '\t' << retained.y << '\t'
                   << delta.x << '\t' << delta.y << '\t' << cv::norm(delta) << '\n';
        }
    }
}

void write_reference_match_overlay(
    const fs::path& path,
    const Result& result,
    const cv::Matx33d& reference,
    const cv::Mat& fixed
) {
    cv::Mat diagnostic = fixed.clone();
    for (const auto& match : result.matches) {
        if (!match.inlier) continue;
        const cv::Point2d reference_point = visium_hd::registration::project_point(
            reference, {match.moving_x, match.moving_y}
        );
        const double reference_error = cv::norm(
            reference_point - cv::Point2d(match.fixed_x, match.fixed_y)
        );
        const bool native_better = match.reprojection_error <= reference_error;
        cv::circle(
            diagnostic,
            cv::Point(
                static_cast<int>(std::lround(match.fixed_x)),
                static_cast<int>(std::lround(match.fixed_y))
            ),
            native_better ? 1 : 3,
            native_better ? cv::Scalar(0, 155, 0) : cv::Scalar(180, 0, 180),
            native_better ? cv::FILLED : 1,
            cv::LINE_AA
        );
    }
    if (!cv::imwrite(path.string(), diagnostic)) {
        throw std::runtime_error("failed writing reference match diagnostic image");
    }
}

void write_residual_translation_sweep(
    const fs::path& path,
    const Result& result,
    const cv::Matx33d& reference,
    const cv::Mat& moving,
    const cv::Mat& fixed,
    int work_max_dimension
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write residual translation sweep: " + path.string());
    output << "phase_fraction\tshift_x\tshift_y\tncc\toverlap_pixels\t"
              "inlier_mean_error\tinlier_median_error\treference_grid_rmse\t"
              "reference_grid_max_error\n";
    output << std::setprecision(17);
    const std::vector<double> fractions{0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5};
    for (const double fraction : fractions) {
        const double shift_x = fraction * result.phase_shift_x;
        const double shift_y = fraction * result.phase_shift_y;
        const cv::Matx33d correction(
            1.0, 0.0, shift_x,
            0.0, 1.0, shift_y,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d candidate = correction * result.moving_to_fixed;
        const AlignmentScore score = visium_hd::registration::score_transform_alignment(
            moving, fixed, candidate, work_max_dimension
        );
        std::vector<double> match_errors;
        double match_sum = 0.0;
        for (const auto& match : result.matches) {
            if (!match.inlier) continue;
            const cv::Point2d projected = visium_hd::registration::project_point(
                candidate, {match.moving_x, match.moving_y}
            );
            const double error = cv::norm(
                projected - cv::Point2d(match.fixed_x, match.fixed_y)
            );
            match_errors.push_back(error);
            match_sum += error;
        }
        const TransformError reference_error = visium_hd::registration::compare_transforms(
            candidate, reference, result.moving_width, result.moving_height
        );
        output << fraction << '\t' << shift_x << '\t' << shift_y << '\t'
               << score.normalized_cross_correlation << '\t' << score.overlap_pixels << '\t'
               << match_sum / match_errors.size() << '\t' << median(match_errors) << '\t'
               << reference_error.root_mean_square_pixels << '\t'
               << reference_error.maximum_pixels << '\n';
    }
}

void write_reference_translation_corrections(
    const fs::path& path,
    const Result& result,
    const cv::Matx33d& reference,
    const cv::Mat& moving,
    const cv::Mat& fixed,
    int work_max_dimension
) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write reference translation corrections: " + path.string());
    }
    const TransformError baseline = visium_hd::registration::compare_transforms(
        result.moving_to_fixed,
        reference,
        result.moving_width,
        result.moving_height
    );
    struct TranslationCandidate {
        std::string name;
        double x;
        double y;
    };
    std::vector<TranslationCandidate> candidates;
    for (const double y : std::array<double, 3>{-0.5, 0.0, 0.5}) {
        for (const double x : std::array<double, 3>{-0.5, 0.0, 0.5}) {
            std::ostringstream name;
            name << "half_pixel_grid_x" << std::showpos << x << "_y" << y;
            candidates.push_back({name.str(), x, y});
        }
    }
    candidates.push_back({
        "reference_mean_translation",
        -baseline.mean_delta_x_pixels,
        -baseline.mean_delta_y_pixels,
    });

    output << std::setprecision(17)
           << "candidate\tshift_x\tshift_y\tncc\toverlap_pixels\t"
              "landmark_mean_error\tlandmark_median_error\treference_rmse\t"
              "reference_maximum_error\treference_mean_delta_x\t"
              "reference_mean_delta_y\treference_centered_rmse\n";
    for (const auto& candidate : candidates) {
        const cv::Matx33d correction(
            1.0, 0.0, candidate.x,
            0.0, 1.0, candidate.y,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d transform = correction * result.moving_to_fixed;
        const AlignmentScore image_score = visium_hd::registration::score_transform_alignment(
            moving, fixed, transform, work_max_dimension
        );
        std::vector<double> landmark_errors;
        double landmark_sum = 0.0;
        for (const auto& match : result.matches) {
            if (!match.inlier) continue;
            const cv::Point2d projected = visium_hd::registration::project_point(
                transform, cv::Point2d(match.moving_x, match.moving_y)
            );
            const double error = cv::norm(
                projected - cv::Point2d(match.fixed_x, match.fixed_y)
            );
            landmark_errors.push_back(error);
            landmark_sum += error;
        }
        const TransformError reference_score = visium_hd::registration::compare_transforms(
            transform, reference, result.moving_width, result.moving_height
        );
        output << candidate.name << '\t' << candidate.x << '\t' << candidate.y << '\t'
               << image_score.normalized_cross_correlation << '\t'
               << image_score.overlap_pixels << '\t'
               << landmark_sum / landmark_errors.size() << '\t'
               << median(landmark_errors) << '\t'
               << reference_score.root_mean_square_pixels << '\t'
               << reference_score.maximum_pixels << '\t'
               << reference_score.mean_delta_x_pixels << '\t'
               << reference_score.mean_delta_y_pixels << '\t'
               << reference_score.residual_after_mean_translation_rmse_pixels << '\n';
    }
}

SpatialResidualSummary write_spatial_residual_grid(
    const fs::path& table_path,
    const fs::path& image_path,
    const std::vector<SpatialResidualCell>& cells,
    const cv::Mat& fixed,
    int grid_size
) {
    if (cells.size() != static_cast<size_t>(grid_size * grid_size)) {
        throw std::runtime_error("spatial residual grid has an unexpected cell count");
    }
    std::ofstream output(table_path);
    if (!output) throw std::runtime_error("cannot write spatial residual grid: " + table_path.string());
    output << "grid_row\tgrid_col\tfixed_x0\tfixed_y0\tfixed_x1\tfixed_y1\t"
              "valid_pixels\tvalid_fraction\tmoving_standard_deviation\t"
              "fixed_standard_deviation\tncc_before\tncc_after\tncc_gain\t"
              "phase_shift_x\tphase_shift_y\tphase_magnitude\tphase_response\t"
              "inlier_matches\tlandmark_mean_delta_x\tlandmark_mean_delta_y\t"
              "landmark_mean_magnitude\tlandmark_median_error\tquality_pass\n";
    output << std::setprecision(17);
    SpatialResidualSummary summary;
    summary.cells = static_cast<int>(cells.size());
    std::vector<double> phase_magnitudes;
    cv::Mat overlay = fixed.clone();
    for (const auto& cell : cells) {
        const double phase_magnitude = std::hypot(cell.phase_shift_x, cell.phase_shift_y);
        const double landmark_magnitude = std::hypot(
            cell.landmark_mean_delta_x, cell.landmark_mean_delta_y
        );
        output << cell.grid_row << '\t' << cell.grid_col << '\t'
               << cell.fixed_x0 << '\t' << cell.fixed_y0 << '\t'
               << cell.fixed_x1 << '\t' << cell.fixed_y1 << '\t'
               << cell.valid_pixels << '\t' << cell.valid_fraction << '\t'
               << cell.moving_standard_deviation << '\t'
               << cell.fixed_standard_deviation << '\t'
               << cell.ncc_before << '\t' << cell.ncc_after << '\t'
               << cell.ncc_after - cell.ncc_before << '\t'
               << cell.phase_shift_x << '\t' << cell.phase_shift_y << '\t'
               << phase_magnitude << '\t' << cell.phase_response << '\t'
               << cell.inlier_matches << '\t'
               << cell.landmark_mean_delta_x << '\t'
               << cell.landmark_mean_delta_y << '\t'
               << landmark_magnitude << '\t' << cell.landmark_median_error << '\t'
               << (cell.quality_pass ? 1 : 0) << '\n';
        const cv::Rect rectangle(
            static_cast<int>(std::lround(cell.fixed_x0)),
            static_cast<int>(std::lround(cell.fixed_y0)),
            std::max(1, static_cast<int>(std::lround(cell.fixed_x1 - cell.fixed_x0))),
            std::max(1, static_cast<int>(std::lround(cell.fixed_y1 - cell.fixed_y0)))
        );
        cv::rectangle(
            overlay,
            rectangle,
            cell.quality_pass ? cv::Scalar(0, 150, 0) : cv::Scalar(100, 100, 100),
            1,
            cv::LINE_AA
        );
        const cv::Point center(
            static_cast<int>(std::lround((cell.fixed_x0 + cell.fixed_x1) / 2.0)),
            static_cast<int>(std::lround((cell.fixed_y0 + cell.fixed_y1) / 2.0))
        );
        if (cell.quality_pass) {
            const cv::Point phase_end(
                static_cast<int>(std::lround(center.x + 12.0 * cell.phase_shift_x)),
                static_cast<int>(std::lround(center.y + 12.0 * cell.phase_shift_y))
            );
            cv::arrowedLine(
                overlay, center, phase_end, cv::Scalar(0, 165, 255), 1, cv::LINE_AA, 0, 0.2
            );
            ++summary.quality_pass_cells;
            phase_magnitudes.push_back(phase_magnitude);
            summary.maximum_phase_magnitude = std::max(
                summary.maximum_phase_magnitude, phase_magnitude
            );
        }
        if (cell.inlier_matches >= 5) {
            const cv::Point landmark_end(
                static_cast<int>(std::lround(center.x + 20.0 * cell.landmark_mean_delta_x)),
                static_cast<int>(std::lround(center.y + 20.0 * cell.landmark_mean_delta_y))
            );
            cv::arrowedLine(
                overlay, center, landmark_end, cv::Scalar(255, 80, 0), 1, cv::LINE_AA, 0, 0.2
            );
            ++summary.landmark_supported_cells;
            summary.maximum_landmark_mean_magnitude = std::max(
                summary.maximum_landmark_mean_magnitude, landmark_magnitude
            );
        }
    }
    summary.median_phase_magnitude = median(phase_magnitudes);
    double neighbor_squared_sum = 0.0;
    for (const auto& cell : cells) {
        if (!cell.quality_pass) continue;
        for (const auto& neighbor : cells) {
            if (!neighbor.quality_pass) continue;
            const bool forward_neighbor =
                (neighbor.grid_row == cell.grid_row && neighbor.grid_col == cell.grid_col + 1) ||
                (neighbor.grid_col == cell.grid_col && neighbor.grid_row == cell.grid_row + 1);
            if (!forward_neighbor) continue;
            const double dx = cell.phase_shift_x - neighbor.phase_shift_x;
            const double dy = cell.phase_shift_y - neighbor.phase_shift_y;
            neighbor_squared_sum += dx * dx + dy * dy;
            ++summary.neighbor_pairs;
        }
    }
    if (summary.neighbor_pairs) {
        summary.neighbor_difference_rmse = std::sqrt(
            neighbor_squared_sum / summary.neighbor_pairs
        );
    }
    if (!cv::imwrite(image_path.string(), overlay)) {
        throw std::runtime_error("failed writing spatial residual grid image");
    }
    return summary;
}

void write_spatial_correction_summary(
    const fs::path& path,
    const SpatialCorrectionEvaluation& correction
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write spatial correction summary");
    output << std::setprecision(17)
           << "support_cells\theldout_cells\theldout_pixels\tncc_before\tncc_after\t"
              "constant_ncc_after\tncc_gain\theldout_ncc_before\theldout_ncc_after\t"
              "heldout_constant_ncc_after\theldout_ncc_gain\t"
              "constant_shift_x\tconstant_shift_y\t"
              "landmark_median_before\tlandmark_median_after\tlandmark_median_change\t"
              "maximum_displacement\n"
           << correction.support_cells << '\t'
           << correction.heldout_cells << '\t'
           << correction.heldout_pixels << '\t'
           << correction.ncc_before << '\t'
           << correction.ncc_after << '\t'
           << correction.constant_ncc_after << '\t'
           << correction.ncc_after - correction.ncc_before << '\t'
           << correction.heldout_ncc_before << '\t'
           << correction.heldout_ncc_after << '\t'
           << correction.heldout_constant_ncc_after << '\t'
           << correction.heldout_ncc_after - correction.heldout_ncc_before << '\t'
           << correction.constant_shift_x << '\t'
           << correction.constant_shift_y << '\t'
           << correction.landmark_median_before << '\t'
           << correction.landmark_median_after << '\t'
           << correction.landmark_median_after - correction.landmark_median_before << '\t'
           << correction.maximum_displacement << '\n';
}

void write_global_optimization_summary(
    const fs::path& path,
    const GlobalOptimizationEvaluation& optimization
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write global optimization summary");
    output << std::setprecision(17)
           << "candidate\tconverged\tecc_objective\tncc\theldout_ncc\t"
              "landmark_median_error\tlandmark_mean_error\ttransform_delta_rmse\t"
              "passes_joint_gate\treference_scored\treference_rmse\t"
              "reference_maximum_error\treference_mean_delta_x\t"
              "reference_mean_delta_y\treference_centered_rmse\t"
              "failure_reason\tm00\tm01\tm02\tm10\tm11\tm12\t"
              "m20\tm21\tm22\n";
    for (const auto& candidate : optimization.candidates) {
        output << candidate.name << '\t'
               << (candidate.converged ? 1 : 0) << '\t'
               << candidate.ecc_objective << '\t'
               << candidate.ncc << '\t'
               << candidate.heldout_ncc << '\t'
               << candidate.landmark_median_error << '\t'
               << candidate.landmark_mean_error << '\t'
               << candidate.transform_delta_rmse << '\t'
               << (candidate.passes_joint_gate ? 1 : 0) << '\t'
               << (candidate.reference_scored ? 1 : 0) << '\t'
               << candidate.reference_rmse << '\t'
               << candidate.reference_maximum_error << '\t'
               << candidate.reference_mean_delta_x << '\t'
               << candidate.reference_mean_delta_y << '\t'
               << candidate.reference_centered_rmse << '\t'
               << candidate.failure_reason;
        for (double value : candidate.moving_to_fixed.val) output << '\t' << value;
        output << '\n';
    }
}

const visium_hd::registration::GlobalOptimizationCandidate* select_global_candidate(
    const GlobalOptimizationEvaluation& optimization
) {
    const visium_hd::registration::GlobalOptimizationCandidate* selected = nullptr;
    for (const auto& candidate : optimization.candidates) {
        if (!candidate.passes_joint_gate) continue;
        if (selected == nullptr ||
            candidate.heldout_ncc > selected->heldout_ncc + 1e-12 ||
            (std::abs(candidate.heldout_ncc - selected->heldout_ncc) <= 1e-12 &&
             candidate.ncc > selected->ncc + 1e-12) ||
            (std::abs(candidate.heldout_ncc - selected->heldout_ncc) <= 1e-12 &&
             std::abs(candidate.ncc - selected->ncc) <= 1e-12 &&
             candidate.name < selected->name)) {
            selected = &candidate;
        }
    }
    return selected;
}

void write_json(
    const fs::path& path,
    const Args& args,
    const Result& result,
    const std::optional<TransformError>& reference_error,
    const std::optional<ReferenceMatchSummary>& reference_match_summary,
    const std::optional<ReferenceImageSummary>& reference_image_summary,
    const SpatialResidualSummary& spatial_summary,
    const std::optional<SpatialCorrectionEvaluation>& spatial_correction,
    const std::optional<GlobalOptimizationEvaluation>& global_optimization
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write result JSON: " + path.string());
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"schema\": \"visium_hd.native_registration.v1\",\n";
    output << "  \"status\": \"accepted\",\n";
    output << "  \"method\": \"hematoxylin_sift_orientation_search_ransac_"
           << json_escape(args.options.transform_model) << "\",\n";
    output << "  \"moving_image\": \"" << json_escape(fs::absolute(args.moving).string()) << "\",\n";
    output << "  \"fixed_image\": \"" << json_escape(fs::absolute(args.fixed).string()) << "\",\n";
    output << "  \"moving_shape_yx\": [" << result.moving_height << ", " << result.moving_width << "],\n";
    output << "  \"fixed_shape_yx\": [" << result.fixed_height << ", " << result.fixed_width << "],\n";
    output << "  \"working_moving_shape_yx\": [" << result.working_moving_height << ", "
           << result.working_moving_width << "],\n";
    output << "  \"working_fixed_shape_yx\": [" << result.working_fixed_height << ", "
           << result.working_fixed_width << "],\n";
    output << "  \"parameters\": {\"threads\": " << args.options.threads
           << ", \"deterministic\": "
           << (args.options.deterministic ? "true" : "false")
           << ", \"max_features\": " << args.options.max_features
           << ", \"work_max_dimension\": " << args.options.work_max_dimension
           << ", \"transform_model\": \"" << json_escape(args.options.transform_model) << "\""
           << ", \"ratio_test\": " << args.options.ratio_test
           << ", \"ransac_threshold_pixels\": "
           << args.options.ransac_threshold_pixels
           << ", \"minimum_good_matches\": "
           << args.options.minimum_good_matches
           << ", \"minimum_inliers\": " << args.options.minimum_inliers
           << ", \"mutual_matching\": "
           << (args.options.mutual_matching ? "true" : "false")
           << ", \"match_scale_policy\": \""
           << json_escape(args.options.match_scale_policy) << "\""
           << ", \"apply_residual_refinement\": "
           << (args.options.apply_residual_refinement ? "true" : "false")
           << ", \"fixed_frame_offset_x_pixels\": "
           << args.options.fixed_frame_offset_x_pixels
           << ", \"fixed_frame_offset_y_pixels\": "
           << args.options.fixed_frame_offset_y_pixels
           << "},\n";
    output << "  \"selected_orientation\": \"" << result.orientation << "\",\n";
    output << "  \"moving_to_fixed\": ";
    write_matrix_json(output, result.moving_to_fixed, 2);
    output << ",\n";
    output << "  \"fit\": {\"ratio_matches\": " << result.good_matches
           << ", \"inliers\": " << result.inliers
           << ", \"inlier_fraction\": " << result.inlier_fraction
           << ", \"median_inlier_error_work_pixels\": " << result.median_inlier_error << "},\n";
    output << "  \"spatial_residual_audit\": {\"grid_size\": "
           << args.spatial_audit_grid_size
           << ", \"fixed_roi_xyxy\": ["
           << (args.spatial_audit_x0 >= 0.0 ? args.spatial_audit_x0 : 0.0) << ", "
           << (args.spatial_audit_y0 >= 0.0 ? args.spatial_audit_y0 : 0.0) << ", "
           << (args.spatial_audit_x1 >= 0.0 ? args.spatial_audit_x1 : result.fixed_width) << ", "
           << (args.spatial_audit_y1 >= 0.0 ? args.spatial_audit_y1 : result.fixed_height) << "]"
           << ", \"cells\": " << spatial_summary.cells
           << ", \"quality_pass_cells\": " << spatial_summary.quality_pass_cells
           << ", \"median_phase_magnitude_pixels\": "
           << spatial_summary.median_phase_magnitude
           << ", \"maximum_phase_magnitude_pixels\": "
           << spatial_summary.maximum_phase_magnitude
           << ", \"neighbor_difference_rmse_pixels\": "
           << spatial_summary.neighbor_difference_rmse
           << ", \"landmark_supported_cells\": "
           << spatial_summary.landmark_supported_cells
           << ", \"maximum_landmark_mean_magnitude_pixels\": "
           << spatial_summary.maximum_landmark_mean_magnitude
           << ", \"quality_rule\": \"valid_fraction>=0.60; both_stddev>=5; "
              "phase_response>=0.20; ncc_gain>=0.001\"},\n";
    output << "  \"spatial_correction\": {\"enabled\": "
           << (spatial_correction ? "true" : "false");
    if (spatial_correction) {
        output << ", \"method\": \"gaussian_smoothed_virtual_tile_translation\""
               << ", \"validation\": \"checkerboard_two_fold_heldout_tiles\""
               << ", \"support_cells\": " << spatial_correction->support_cells
               << ", \"heldout_cells\": " << spatial_correction->heldout_cells
               << ", \"heldout_pixels\": " << spatial_correction->heldout_pixels
               << ", \"ncc_before\": " << spatial_correction->ncc_before
               << ", \"ncc_after\": " << spatial_correction->ncc_after
               << ", \"constant_ncc_after\": " << spatial_correction->constant_ncc_after
               << ", \"heldout_ncc_before\": " << spatial_correction->heldout_ncc_before
               << ", \"heldout_ncc_after\": " << spatial_correction->heldout_ncc_after
               << ", \"heldout_constant_ncc_after\": "
               << spatial_correction->heldout_constant_ncc_after
               << ", \"constant_shift_x\": " << spatial_correction->constant_shift_x
               << ", \"constant_shift_y\": " << spatial_correction->constant_shift_y
               << ", \"landmark_median_before\": "
               << spatial_correction->landmark_median_before
               << ", \"landmark_median_after\": "
               << spatial_correction->landmark_median_after
               << ", \"maximum_displacement\": "
               << spatial_correction->maximum_displacement;
    }
    output << "},\n";
    output << "  \"global_optimization\": {\"enabled\": "
           << (global_optimization ? "true" : "false");
    if (global_optimization) {
        const auto* selected = args.select_global_candidate
                                   ? select_global_candidate(*global_optimization)
                                   : nullptr;
        output << ", \"scope\": \"one_global_transform_over_entire_fixed_roi\""
               << ", \"heldout_tiles_are_scoring_masks_only\" : true"
               << ", \"selection_enabled\": "
               << (args.select_global_candidate ? "true" : "false")
               << ", \"iterations\": " << args.global_optimization_iterations
               << ", \"selected_candidate\": ";
        if (selected == nullptr) output << "null";
        else output << "\"" << json_escape(selected->name) << "\"";
        output
               << ", \"candidates\": [";
        for (size_t index = 0; index < global_optimization->candidates.size(); ++index) {
            const auto& candidate = global_optimization->candidates[index];
            if (index) output << ", ";
            output << "{\"name\": \"" << candidate.name
                   << "\", \"converged\": " << (candidate.converged ? "true" : "false")
                   << ", \"ecc_objective\": " << candidate.ecc_objective
                   << ", \"ncc\": " << candidate.ncc
                   << ", \"heldout_ncc\": " << candidate.heldout_ncc
                   << ", \"landmark_median_error\": "
                   << candidate.landmark_median_error
                   << ", \"landmark_mean_error\": " << candidate.landmark_mean_error
                   << ", \"transform_delta_rmse\": " << candidate.transform_delta_rmse
                   << ", \"passes_joint_gate\": "
                   << (candidate.passes_joint_gate ? "true" : "false")
                   << ", \"reference_scored\": "
                   << (candidate.reference_scored ? "true" : "false")
                   << ", \"reference_rmse\": " << candidate.reference_rmse
                   << ", \"reference_maximum_error\": "
                   << candidate.reference_maximum_error
                   << ", \"reference_mean_delta_x\": "
                   << candidate.reference_mean_delta_x
                   << ", \"reference_mean_delta_y\": "
                   << candidate.reference_mean_delta_y
                   << ", \"reference_centered_rmse\": "
                   << candidate.reference_centered_rmse
                   << ", \"failure_reason\": \"" << json_escape(candidate.failure_reason)
                   << "\"}";
        }
        output << "]";
    }
    output << "},\n";
    output << "  \"overlap_refinement\": {\"method\": \"hann_phase_proposal_local_ncc\""
           << ", \"overlap_pixels\": " << result.overlap_pixels
           << ", \"phase_shift_x\": " << result.phase_shift_x
           << ", \"phase_shift_y\": " << result.phase_shift_y
           << ", \"phase_response\": " << result.phase_response
           << ", \"residual_shift_x\": " << result.residual_shift_x
           << ", \"residual_shift_y\": " << result.residual_shift_y
           << ", \"ncc_before\": " << result.ncc_before_refinement
           << ", \"ncc_after\": " << result.ncc_after_refinement
           << ", \"applied\": " << (result.residual_refinement_applied ? "true" : "false")
           << "},\n";
    output << "  \"orientation_candidates\": [\n";
    for (size_t index = 0; index < result.candidates.size(); ++index) {
        const auto& candidate = result.candidates[index];
        output << "    {\"name\": \"" << candidate.name
               << "\", \"moving_keypoints\": " << candidate.moving_keypoints
               << ", \"fixed_keypoints\": " << candidate.fixed_keypoints
               << ", \"ratio_matches\": " << candidate.ratio_matches
               << ", \"inliers\": " << candidate.inliers
               << ", \"inlier_fraction\": " << candidate.inlier_fraction
               << ", \"median_inlier_error_work_pixels\": " << candidate.median_inlier_error
               << ", \"accepted\": " << (candidate.accepted ? "true" : "false")
               << ", \"rejection_reason\": \"" << candidate.rejection_reason << "\"}"
               << (index + 1 == result.candidates.size() ? "\n" : ",\n");
    }
    output << "  ]";
    if (reference_error) {
        output << ",\n  \"post_estimation_reference_score\": {\n";
        output << "    \"reference_json\": \"" << json_escape(fs::absolute(args.reference_json).string()) << "\",\n";
        output << "    \"reference_key\": \"" << json_escape(args.reference_key) << "\",\n";
        output << "    \"sampled_points\": " << reference_error->sampled_points << ",\n";
        output << "    \"mean_pixels\": " << reference_error->mean_pixels << ",\n";
        output << "    \"root_mean_square_pixels\": " << reference_error->root_mean_square_pixels << ",\n";
        output << "    \"median_pixels\": " << reference_error->median_pixels << ",\n";
        output << "    \"maximum_pixels\": " << reference_error->maximum_pixels << ",\n";
        output << "    \"mean_delta_x_pixels\": " << reference_error->mean_delta_x_pixels << ",\n";
        output << "    \"mean_delta_y_pixels\": " << reference_error->mean_delta_y_pixels << ",\n";
        output << "    \"residual_after_mean_translation_rmse_pixels\": "
               << reference_error->residual_after_mean_translation_rmse_pixels << ",\n";
        output << "    \"inlier_match_comparison\": {\"inliers\": "
               << reference_match_summary->inliers
               << ", \"native_better\": " << reference_match_summary->native_better
               << ", \"reference_better\": " << reference_match_summary->reference_better
               << ", \"equal\": " << reference_match_summary->equal
               << ", \"native_mean_error\": " << reference_match_summary->native_mean_error
               << ", \"reference_mean_error\": " << reference_match_summary->reference_mean_error
               << ", \"native_median_error\": " << reference_match_summary->native_median_error
               << ", \"reference_median_error\": " << reference_match_summary->reference_median_error
               << "},\n";
        output << "    \"image_alignment\": {\"native_ncc\": "
               << reference_image_summary->native.normalized_cross_correlation
               << ", \"retained_reference_ncc\": "
               << reference_image_summary->retained_reference.normalized_cross_correlation
               << ", \"phase_shifted_native_ncc\": "
               << reference_image_summary->phase_shifted_native.normalized_cross_correlation
               << ", \"phase_shift_x\": " << result.phase_shift_x
               << ", \"phase_shift_y\": " << result.phase_shift_y
               << "}\n";
        output << "  }\n";
    } else {
        output << "\n";
    }
    output << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path output_dir = fs::absolute(args.out_dir);
        if (fs::exists(output_dir)) {
            throw std::runtime_error("output directory already exists: " + output_dir.string());
        }
        fs::create_directories(output_dir);

        const cv::Mat moving = cv::imread(args.moving, cv::IMREAD_COLOR);
        const cv::Mat fixed = cv::imread(args.fixed, cv::IMREAD_COLOR);
        if (moving.empty()) throw std::runtime_error("cannot read moving image: " + args.moving);
        if (fixed.empty()) throw std::runtime_error("cannot read fixed image: " + args.fixed);

        const Result result = visium_hd::registration::register_images(
            moving, fixed, args.options
        );
        std::optional<TransformError> reference_error;
        std::optional<cv::Matx33d> reference_transform;
        if (!args.reference_json.empty()) {
            const cv::Matx33d reference_base = read_named_matrix(
                args.reference_json, args.reference_key
            );
            const cv::Matx33d moving_to_reference(
                args.reference_moving_scale_x,
                0.0,
                args.reference_moving_offset_x,
                0.0,
                args.reference_moving_scale_y,
                args.reference_moving_offset_y,
                0.0,
                0.0,
                1.0
            );
            const cv::Matx33d reference = reference_base * moving_to_reference;
            reference_transform = reference;
            reference_error = visium_hd::registration::compare_transforms(
                result.moving_to_fixed,
                reference,
                moving.cols,
                moving.rows
            );
        }

        cv::Mat registered;
        cv::warpPerspective(
            moving,
            registered,
            cv::Mat(result.moving_to_fixed),
            fixed.size(),
            cv::INTER_AREA,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );
        cv::Mat overlay;
        cv::addWeighted(fixed, 0.5, registered, 0.5, 0.0, overlay);
        if (!cv::imwrite((output_dir / "registered_moving.png").string(), registered)) {
            throw std::runtime_error("failed writing registered image");
        }
        if (!cv::imwrite((output_dir / "overlay.png").string(), overlay)) {
            throw std::runtime_error("failed writing overlay image");
        }
        write_match_diagnostics(
            output_dir / "match_diagnostics.tsv", result, moving, fixed
        );
        write_match_overlay(output_dir / "match_diagnostics.png", result, fixed);
        cv::Rect2d spatial_roi;
        if (args.spatial_audit_x0 >= 0.0) {
            if (args.spatial_audit_x1 > fixed.cols || args.spatial_audit_y1 > fixed.rows) {
                throw std::runtime_error("spatial-audit ROI lies outside the fixed image");
            }
            spatial_roi = cv::Rect2d(
                args.spatial_audit_x0,
                args.spatial_audit_y0,
                args.spatial_audit_x1 - args.spatial_audit_x0,
                args.spatial_audit_y1 - args.spatial_audit_y0
            );
        }
        const std::vector<SpatialResidualCell> spatial_cells =
            visium_hd::registration::audit_spatial_residual_grid(
                moving,
                fixed,
                result,
                args.spatial_audit_grid_size,
                args.options.work_max_dimension,
                spatial_roi
            );
        const SpatialResidualSummary spatial_summary = write_spatial_residual_grid(
            output_dir / "spatial_residual_grid.tsv",
            output_dir / "spatial_residual_grid.png",
            spatial_cells,
            fixed,
            args.spatial_audit_grid_size
        );
        std::optional<SpatialCorrectionEvaluation> spatial_correction;
        if (args.apply_spatial_correction) {
            spatial_correction = visium_hd::registration::evaluate_spatial_correction(
                moving, fixed, result, spatial_cells
            );
            if (!cv::imwrite(
                    (output_dir / "spatially_corrected_registered_moving.png").string(),
                    spatial_correction->corrected_registered_bgr
                )) {
                throw std::runtime_error("failed writing spatially corrected registered image");
            }
            cv::Mat corrected_overlay;
            cv::addWeighted(
                fixed,
                0.5,
                spatial_correction->corrected_registered_bgr,
                0.5,
                0.0,
                corrected_overlay
            );
            if (!cv::imwrite(
                    (output_dir / "spatially_corrected_overlay.png").string(),
                    corrected_overlay
                )) {
                throw std::runtime_error("failed writing spatially corrected overlay");
            }
            write_spatial_correction_summary(
                output_dir / "spatial_correction_summary.tsv", *spatial_correction
            );
        }
        std::optional<GlobalOptimizationEvaluation> global_optimization;
        if (args.evaluate_global_optimization) {
            global_optimization = visium_hd::registration::evaluate_global_optimization(
                moving,
                fixed,
                result,
                spatial_cells,
                args.global_optimization_iterations
            );
            if (reference_transform) {
                for (auto& candidate : global_optimization->candidates) {
                    if (!candidate.converged) continue;
                    const TransformError score = visium_hd::registration::compare_transforms(
                        candidate.moving_to_fixed,
                        *reference_transform,
                        moving.cols,
                        moving.rows
                    );
                    candidate.reference_scored = true;
                    candidate.reference_rmse = score.root_mean_square_pixels;
                    candidate.reference_maximum_error = score.maximum_pixels;
                    candidate.reference_mean_delta_x = score.mean_delta_x_pixels;
                    candidate.reference_mean_delta_y = score.mean_delta_y_pixels;
                    candidate.reference_centered_rmse =
                        score.residual_after_mean_translation_rmse_pixels;
                }
            }
            write_global_optimization_summary(
                output_dir / "global_optimization_candidates.tsv", *global_optimization
            );
            for (const auto& candidate : global_optimization->candidates) {
                if (!candidate.converged) continue;
                cv::Mat candidate_registered;
                cv::warpPerspective(
                    moving,
                    candidate_registered,
                    cv::Mat(candidate.moving_to_fixed),
                    fixed.size(),
                    cv::INTER_AREA,
                    cv::BORDER_CONSTANT,
                    cv::Scalar(255, 255, 255)
                );
                cv::Mat candidate_overlay;
                cv::addWeighted(fixed, 0.5, candidate_registered, 0.5, 0.0, candidate_overlay);
                if (!cv::imwrite(
                        (output_dir / ("global_" + candidate.name + "_overlay.png")).string(),
                        candidate_overlay
                    )) {
                    throw std::runtime_error("failed writing global optimization overlay");
                }
            }
            const auto* selected = args.select_global_candidate
                                       ? select_global_candidate(*global_optimization)
                                       : nullptr;
            if (selected != nullptr) {
                write_transform_tsv(
                    output_dir / "selected_global_moving_to_fixed.tsv",
                    selected->moving_to_fixed
                );
                cv::Mat selected_registered;
                cv::warpPerspective(
                    moving,
                    selected_registered,
                    cv::Mat(selected->moving_to_fixed),
                    fixed.size(),
                    cv::INTER_AREA,
                    cv::BORDER_CONSTANT,
                    cv::Scalar(255, 255, 255)
                );
                if (!cv::imwrite(
                        (output_dir / "selected_global_registered_moving.png").string(),
                        selected_registered
                    )) {
                    throw std::runtime_error("failed writing selected global registration");
                }
                cv::Mat selected_overlay;
                cv::addWeighted(fixed, 0.5, selected_registered, 0.5, 0.0, selected_overlay);
                if (!cv::imwrite(
                        (output_dir / "selected_global_overlay.png").string(),
                        selected_overlay
                    )) {
                    throw std::runtime_error("failed writing selected global overlay");
                }
            }
        }
        std::optional<ReferenceMatchSummary> reference_match_summary;
        std::optional<ReferenceImageSummary> reference_image_summary;
        if (reference_transform) {
            reference_match_summary = write_reference_match_diagnostics(
                output_dir / "reference_match_diagnostics.tsv", result, *reference_transform
            );
            write_reference_grid_diagnostics(
                output_dir / "reference_grid_diagnostics.tsv",
                result.moving_to_fixed,
                *reference_transform,
                moving.cols,
                moving.rows
            );
            write_reference_match_overlay(
                output_dir / "reference_match_diagnostics.png",
                result,
                *reference_transform,
                fixed
            );
            write_residual_translation_sweep(
                output_dir / "residual_translation_sweep.tsv",
                result,
                *reference_transform,
                moving,
                fixed,
                args.options.work_max_dimension
            );
            write_reference_translation_corrections(
                output_dir / "reference_translation_corrections.tsv",
                result,
                *reference_transform,
                moving,
                fixed,
                args.options.work_max_dimension
            );
            const cv::Matx33d phase_correction(
                1.0, 0.0, result.phase_shift_x,
                0.0, 1.0, result.phase_shift_y,
                0.0, 0.0, 1.0
            );
            reference_image_summary = ReferenceImageSummary{
                visium_hd::registration::score_transform_alignment(
                    moving, fixed, result.moving_to_fixed, args.options.work_max_dimension
                ),
                visium_hd::registration::score_transform_alignment(
                    moving, fixed, *reference_transform, args.options.work_max_dimension
                ),
                visium_hd::registration::score_transform_alignment(
                    moving,
                    fixed,
                    phase_correction * result.moving_to_fixed,
                    args.options.work_max_dimension
                ),
            };
        }
        write_transform_tsv(output_dir / "moving_to_fixed.tsv", result.moving_to_fixed);
        write_json(
            output_dir / "registration.json",
            args,
            result,
            reference_error,
            reference_match_summary,
            reference_image_summary,
            spatial_summary,
            spatial_correction,
            global_optimization
        );

        std::cout << std::setprecision(8)
                  << "orientation=" << result.orientation
                  << " matches=" << result.good_matches
                  << " inliers=" << result.inliers
                  << " inlier_fraction=" << result.inlier_fraction
                  << " median_inlier_error=" << result.median_inlier_error
                  << " residual_shift=" << result.residual_shift_x << ','
                  << result.residual_shift_y
                  << " ncc=" << result.ncc_before_refinement << "->"
                  << result.ncc_after_refinement;
        if (spatial_correction) {
            std::cout << " spatial_ncc=" << spatial_correction->ncc_before
                      << "->" << spatial_correction->ncc_after
                      << " heldout_ncc=" << spatial_correction->heldout_ncc_before
                      << "->" << spatial_correction->heldout_ncc_after
                      << " heldout_constant="
                      << spatial_correction->heldout_constant_ncc_after
                      << " landmark_median=" << spatial_correction->landmark_median_before
                      << "->" << spatial_correction->landmark_median_after;
        }
        if (global_optimization) {
            for (const auto& candidate : global_optimization->candidates) {
                if (!candidate.passes_joint_gate) continue;
                std::cout << " global_pass=" << candidate.name
                          << ':' << candidate.ncc
                          << ':' << candidate.heldout_ncc
                          << ':' << candidate.landmark_median_error;
            }
        }
        if (reference_error) {
            std::cout << " reference_rmse=" << reference_error->root_mean_square_pixels
                      << " reference_max=" << reference_error->maximum_pixels;
        }
        std::cout << '\n';
        if (reference_error && args.maximum_reference_rmse >= 0.0 &&
            reference_error->root_mean_square_pixels > args.maximum_reference_rmse) {
            std::cerr << "reference RMSE exceeded threshold: "
                      << reference_error->root_mean_square_pixels << " > "
                      << args.maximum_reference_rmse << '\n';
            return 3;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visium_hd_register: " << error.what() << '\n';
        return 2;
    }
}
