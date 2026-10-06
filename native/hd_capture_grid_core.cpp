#include "hd_capture_grid_core.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace visium_hd::capture_grid {
namespace {

constexpr int kGridRows = 3350;
constexpr int kGridColumns = 3350;
constexpr double kSpotPitchMicrons = 2.0;
constexpr double kFiducialPitchMicrons = 410.0;
constexpr double kDetectorCenterToExportedCorner = 0.5;

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open JSON input: " + path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

size_t find_key(const std::string& text, const std::string& key, size_t begin = 0) {
    const std::string quoted = "\"" + key + "\"";
    const size_t location = text.find(quoted, begin);
    if (location == std::string::npos) throw std::runtime_error("missing JSON key: " + key);
    return location + quoted.size();
}

std::string parse_string_after_key(const std::string& text, const std::string& key) {
    size_t cursor = text.find(':', find_key(text, key));
    if (cursor == std::string::npos) throw std::runtime_error("missing JSON value: " + key);
    cursor = text.find('"', cursor + 1);
    if (cursor == std::string::npos) throw std::runtime_error("JSON value is not a string: " + key);
    const size_t end = text.find('"', cursor + 1);
    if (end == std::string::npos) throw std::runtime_error("unterminated JSON string: " + key);
    return text.substr(cursor + 1, end - cursor - 1);
}

std::array<double, 9> parse_matrix_after(const std::string& text, size_t cursor) {
    cursor = text.find('[', cursor);
    if (cursor == std::string::npos) throw std::runtime_error("JSON matrix has no opening bracket");
    std::array<double, 9> values{};
    for (double& value : values) {
        while (cursor < text.size() &&
               !(text[cursor] == '-' || text[cursor] == '+' || text[cursor] == '.' ||
                 (text[cursor] >= '0' && text[cursor] <= '9'))) {
            ++cursor;
        }
        if (cursor == text.size()) throw std::runtime_error("JSON matrix has fewer than 9 numbers");
        const char* begin = text.c_str() + cursor;
        char* end = nullptr;
        errno = 0;
        value = std::strtod(begin, &end);
        if (errno != 0 || end == begin || !std::isfinite(value)) {
            throw std::runtime_error("invalid JSON matrix number");
        }
        cursor = static_cast<size_t>(end - text.c_str());
    }
    return values;
}

cv::Matx33d to_matrix(const std::array<double, 9>& values) {
    return cv::Matx33d(
        values[0], values[1], values[2],
        values[3], values[4], values[5],
        values[6], values[7], values[8]
    );
}

cv::Matx33d mat_to_matx(const cv::Mat& matrix) {
    cv::Mat converted;
    matrix.convertTo(converted, CV_64F);
    if (converted.rows != 3 || converted.cols != 3) {
        throw std::runtime_error("capture-grid transform is not 3x3");
    }
    cv::Matx33d result;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result(row, column) = converted.at<double>(row, column);
        }
    }
    return result;
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(middle), values.end());
    const double upper = values[middle];
    if (values.size() % 2) return upper;
    std::nth_element(values.begin(), values.begin() + static_cast<long>(middle - 1), values.end());
    return 0.5 * (values[middle - 1] + upper);
}

struct InitialFit {
    cv::Matx33d transform = cv::Matx33d::eye();
    int matches = 0;
    double median_error = std::numeric_limits<double>::infinity();
};

bool is_canonical_slide_orientation(const cv::Matx33d& transform) {
    const cv::Point2d upper_left = project_point(transform, {-545.0, -545.0});
    const cv::Point2d upper_right = project_point(transform, {7245.0, -545.0});
    const cv::Point2d lower_left = project_point(transform, {-545.0, 7245.0});
    const cv::Point2d right = upper_right - upper_left;
    const cv::Point2d down = lower_left - upper_left;
    return right.x > 0.0 && std::abs(right.x) > std::abs(right.y) &&
           down.y > 0.0 && std::abs(down.y) > std::abs(down.x);
}

double bilinear_u8(const cv::Mat& image, double x, double y) {
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    if (x0 < 0 || y0 < 0 || x0 + 1 >= image.cols || y0 + 1 >= image.rows) {
        return 0.0;
    }
    const double fx = x - x0;
    const double fy = y - y0;
    const auto* upper = image.ptr<unsigned char>(y0);
    const auto* lower = image.ptr<unsigned char>(y0 + 1);
    return
        (1.0 - fx) * (1.0 - fy) * upper[x0] +
        fx * (1.0 - fy) * upper[x0 + 1] +
        (1.0 - fx) * fy * lower[x0] +
        fx * fy * lower[x0 + 1];
}

cv::Point2d refine_concentric_center(const cv::Mat& gray, cv::Point2d center) {
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::vector<cv::Point2f> edge_points;
        edge_points.reserve(180);
        for (int angle_index = 0; angle_index < 180; ++angle_index) {
            const double angle = angle_index * CV_PI / 90.0;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            double best_gradient = 0.0;
            double best_radius = 0.0;
            for (double radius = 29.0; radius <= 43.0; radius += 0.5) {
                const double outside = bilinear_u8(
                    gray, center.x + (radius + 1.0) * cosine,
                    center.y + (radius + 1.0) * sine
                );
                const double inside = bilinear_u8(
                    gray, center.x + (radius - 1.0) * cosine,
                    center.y + (radius - 1.0) * sine
                );
                const double gradient = outside - inside;
                if (gradient > best_gradient) {
                    best_gradient = gradient;
                    best_radius = radius;
                }
            }
            if (best_gradient >= 4.0) {
                edge_points.emplace_back(
                    center.x + best_radius * cosine,
                    center.y + best_radius * sine
                );
            }
        }
        if (edge_points.size() < 100) break;
        const cv::RotatedRect ellipse = cv::fitEllipseAMS(edge_points);
        const cv::Point2d updated(ellipse.center.x, ellipse.center.y);
        if (cv::norm(updated - center) > 5.0) break;
        const double shift = cv::norm(updated - center);
        center = updated;
        if (shift < 0.005) break;
    }
    return center;
}

cv::Mat grayscale_u8(const cv::Mat& source) {
    cv::Mat gray;
    if (source.channels() == 3) cv::cvtColor(source, gray, cv::COLOR_BGR2GRAY);
    else if (source.channels() == 4) cv::cvtColor(source, gray, cv::COLOR_BGRA2GRAY);
    else if (source.channels() == 1) gray = source;
    else throw std::runtime_error("unsupported CytAssist image channel count");
    if (gray.depth() != CV_8U) {
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(gray, &minimum, &maximum);
        if (!(maximum > minimum)) throw std::runtime_error("constant CytAssist image");
        gray.convertTo(gray, CV_8U, 255.0 / (maximum - minimum), -255.0 * minimum / (maximum - minimum));
    }
    return gray;
}

std::pair<int, double> score_transform(
    const cv::Matx33d& transform,
    const std::vector<cv::Point2d>& design,
    const std::vector<cv::Vec3f>& detected,
    double tolerance
) {
    int count = 0;
    std::vector<double> errors;
    errors.reserve(design.size());
    for (const cv::Point2d& point : design) {
        const cv::Point2d projected = project_point(transform, point);
        double best = std::numeric_limits<double>::infinity();
        for (const cv::Vec3f& circle : detected) {
            best = std::min(best, cv::norm(projected - cv::Point2d(circle[0], circle[1])));
        }
        if (best <= tolerance) {
            ++count;
            errors.push_back(best);
        }
    }
    return {count, median(std::move(errors))};
}

InitialFit initial_similarity_fit(
    const std::vector<cv::Point2d>& design,
    const std::vector<cv::Vec3f>& detected,
    double tolerance
) {
    std::vector<std::pair<int, int>> design_edges;
    for (size_t left = 0; left < design.size(); ++left) {
        for (size_t right = left + 1; right < design.size(); ++right) {
            if (std::abs(cv::norm(design[right] - design[left]) - kFiducialPitchMicrons) < 1e-6) {
                design_edges.emplace_back(static_cast<int>(left), static_cast<int>(right));
                design_edges.emplace_back(static_cast<int>(right), static_cast<int>(left));
            }
        }
    }
    std::vector<std::pair<int, int>> detected_edges;
    for (size_t left = 0; left < detected.size(); ++left) {
        for (size_t right = left + 1; right < detected.size(); ++right) {
            const cv::Point2d delta(
                detected[right][0] - detected[left][0], detected[right][1] - detected[left][1]
            );
            const double distance = cv::norm(delta);
            if (distance >= 65.0 && distance <= 115.0) {
                detected_edges.emplace_back(static_cast<int>(left), static_cast<int>(right));
                detected_edges.emplace_back(static_cast<int>(right), static_cast<int>(left));
            }
        }
    }
    if (design_edges.empty() || detected_edges.empty()) {
        throw std::runtime_error("insufficient adjacent fiducial pairs for capture-grid fit");
    }

    InitialFit best;
    for (const auto& [design_a_index, design_b_index] : design_edges) {
        const cv::Point2d design_a = design[design_a_index];
        const cv::Point2d design_delta = design[design_b_index] - design_a;
        const double denominator = design_delta.dot(design_delta);
        for (const auto& [detected_a_index, detected_b_index] : detected_edges) {
            const cv::Point2d detected_a(detected[detected_a_index][0], detected[detected_a_index][1]);
            const cv::Point2d detected_delta(
                detected[detected_b_index][0] - detected_a.x,
                detected[detected_b_index][1] - detected_a.y
            );
            // Complex multiplication is an orientation-preserving similarity.
            const double real = design_delta.dot(detected_delta) / denominator;
            const double imaginary =
                (design_delta.x * detected_delta.y - design_delta.y * detected_delta.x) /
                denominator;
            const cv::Matx33d transform(
                real, -imaginary, 0.0,
                imaginary, real, 0.0,
                0.0, 0.0, 1.0
            );
            // CytAssist exports the slide in its canonical scanner frame:
            // increasing design x points right and increasing design y points
            // down. The circular perimeter is nearly quarter-turn symmetric,
            // so inlier support alone cannot establish barcode row/column
            // orientation.
            if (!is_canonical_slide_orientation(transform)) continue;
            const cv::Point2d mapped_a = project_point(transform, design_a);
            cv::Matx33d placed = transform;
            placed(0, 2) = detected_a.x - mapped_a.x;
            placed(1, 2) = detected_a.y - mapped_a.y;
            const auto [count, med] = score_transform(placed, design, detected, tolerance);
            if (count > best.matches || (count == best.matches && med < best.median_error)) {
                best = {placed, count, med};
            }
        }
    }
    return best;
}

std::vector<FiducialMatch> match_points(
    const cv::Matx33d& transform,
    const std::vector<cv::Point2d>& design,
    const std::vector<cv::Vec3f>& detected,
    double tolerance
) {
    std::vector<FiducialMatch> matches;
    std::vector<bool> used(detected.size(), false);
    for (size_t index = 0; index < design.size(); ++index) {
        const cv::Point2d projected = project_point(transform, design[index]);
        int best_index = -1;
        double best = tolerance;
        for (size_t candidate = 0; candidate < detected.size(); ++candidate) {
            if (used[candidate]) continue;
            const double error = cv::norm(
                projected - cv::Point2d(detected[candidate][0], detected[candidate][1])
            );
            if (error <= best) {
                best = error;
                best_index = static_cast<int>(candidate);
            }
        }
        if (best_index >= 0) {
            used[best_index] = true;
            matches.push_back({
                static_cast<int>(index), design[index],
                cv::Point2d(detected[best_index][0], detected[best_index][1]),
                projected, best,
            });
        }
    }
    return matches;
}

cv::Matx33d refine_homography(
    const std::vector<FiducialMatch>& matches,
    double ransac_threshold,
    std::vector<unsigned char>* inliers
) {
    std::vector<cv::Point2f> source;
    std::vector<cv::Point2f> target;
    source.reserve(matches.size());
    target.reserve(matches.size());
    for (const auto& match : matches) {
        source.emplace_back(match.design_xy);
        target.emplace_back(match.detected_xy);
    }
    cv::Mat mask;
    const cv::Mat homography = cv::findHomography(
        source, target, cv::RANSAC, ransac_threshold, mask, 10000, 0.999
    );
    if (homography.empty()) throw std::runtime_error("fiducial homography fit failed");
    inliers->assign(matches.size(), 0);
    for (int row = 0; row < mask.rows; ++row) (*inliers)[row] = mask.at<unsigned char>(row);
    return mat_to_matx(homography);
}

}  // namespace

std::vector<cv::Point2d> visium_hd_v1_fiducial_centers() {
    std::vector<cv::Point2d> result;
    result.reserve(73);
    for (int index = 0; index < 18; ++index) result.emplace_back(-135.0 + 410.0 * index, -545.0);
    for (int index = 0; index < 18; ++index) result.emplace_back(7245.0, -135.0 + 410.0 * index);
    result.emplace_back(7245.0, 7245.0);
    for (int index = 0; index < 18; ++index) result.emplace_back(6835.0 - 410.0 * index, 7245.0);
    for (int index = 0; index < 18; ++index) result.emplace_back(-545.0, 6835.0 - 410.0 * index);
    return result;
}

Layout read_vlf_layout(const std::string& path, const std::string& area) {
    const std::string text = read_text(path);
    Layout result;
    result.slide_uid = parse_string_after_key(text, "slide_uid");
    result.file_format = parse_string_after_key(text, "file_format");
    result.aligner_version = parse_string_after_key(text, "aligner_version");
    result.input_hash = parse_string_after_key(text, "input_hash");
    result.slide_design = parse_string_after_key(text, "slide_design");
    result.area = area;
    if (result.slide_design != "visium_hd_rc1") {
        throw std::runtime_error("unsupported slide design: " + result.slide_design);
    }
    const size_t capture_areas = find_key(text, "capture_areas");
    const size_t area_location = find_key(text, area, capture_areas);
    result.design_correction = to_matrix(parse_matrix_after(text, area_location));
    if (std::abs(cv::determinant(cv::Mat(result.design_correction))) < 1e-12) {
        throw std::runtime_error("VLF capture-area transform is singular");
    }
    return result;
}

std::vector<cv::Vec3f> detect_circular_fiducials(
    const cv::Mat& cytassist_bgr,
    const DetectionOptions& options
) {
    if (cytassist_bgr.empty()) throw std::runtime_error("empty CytAssist image");
    cv::setNumThreads(options.threads);
    cv::Mat gray = grayscale_u8(cytassist_bgr);
    cv::medianBlur(gray, gray, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(
        gray, circles, cv::HOUGH_GRADIENT,
        options.hough_dp, options.hough_min_distance,
        options.hough_edge_threshold, options.hough_accumulator_threshold,
        options.minimum_radius, options.maximum_radius
    );
    std::sort(circles.begin(), circles.end(), [](const cv::Vec3f& left, const cv::Vec3f& right) {
        if (left[1] != right[1]) return left[1] < right[1];
        if (left[0] != right[0]) return left[0] < right[0];
        return left[2] < right[2];
    });
    return circles;
}

Result fit_capture_grid(
    const std::vector<cv::Vec3f>& detected_circles,
    const Layout& layout,
    const DetectionOptions& options
) {
    if (detected_circles.size() < static_cast<size_t>(options.minimum_matched_fiducials)) {
        throw std::runtime_error("too few circular fiducial candidates");
    }
    const std::vector<cv::Point2d> design = visium_hd_v1_fiducial_centers();
    const InitialFit initial = initial_similarity_fit(
        design, detected_circles, options.initial_match_tolerance
    );
    if (initial.matches < options.minimum_matched_fiducials) {
        throw std::runtime_error(
            "fiducial point-cloud fit matched only " + std::to_string(initial.matches) +
            " of " + std::to_string(design.size()) + " expected fiducials from " +
            std::to_string(detected_circles.size()) + " detected circle candidates"
        );
    }
    cv::Matx33d transform = initial.transform;
    std::vector<FiducialMatch> matches;
    std::vector<unsigned char> inliers;
    for (int iteration = 0; iteration < 3; ++iteration) {
        matches = match_points(transform, design, detected_circles, options.initial_match_tolerance);
        transform = refine_homography(matches, options.refined_match_tolerance, &inliers);
    }
    matches = match_points(transform, design, detected_circles, options.refined_match_tolerance);
    if (matches.size() < static_cast<size_t>(options.minimum_matched_fiducials)) {
        throw std::runtime_error("refined fiducial fit failed its minimum-match gate");
    }
    transform = refine_homography(matches, options.refined_match_tolerance, &inliers);
    std::vector<double> residuals;
    residuals.reserve(matches.size());
    double squared_sum = 0.0;
    double maximum = 0.0;
    for (auto& match : matches) {
        match.projected_xy = project_point(transform, match.design_xy);
        match.residual_pixels = cv::norm(match.projected_xy - match.detected_xy);
        residuals.push_back(match.residual_pixels);
        squared_sum += match.residual_pixels * match.residual_pixels;
        maximum = std::max(maximum, match.residual_pixels);
    }
    const cv::Matx33d spot_to_design(
        kSpotPitchMicrons, 0.0, kSpotPitchMicrons / 2.0,
        0.0, kSpotPitchMicrons, kSpotPitchMicrons / 2.0,
        0.0, 0.0, 1.0
    );
    Result result;
    // OpenCV feature coordinates put the center of the first image pixel at
    // (0, 0). Exported Visium image transforms use pixel-corner coordinates,
    // where that center is (0.5, 0.5). Keep residual fitting in the detector
    // frame, then perform this fixed convention conversion exactly once.
    const cv::Matx33d center_to_corner(
        1.0, 0.0, kDetectorCenterToExportedCorner,
        0.0, 1.0, kDetectorCenterToExportedCorner,
        0.0, 0.0, 1.0
    );
    result.design_to_cytassist = center_to_corner * transform;
    result.spot_colrow_to_cytassist =
        center_to_corner * transform * layout.design_correction * spot_to_design;
    result.detected_circles = static_cast<int>(detected_circles.size());
    result.matched_fiducials = static_cast<int>(matches.size());
    result.inlier_fiducials = static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
    if (result.inlier_fiducials < options.minimum_matched_fiducials) {
        throw std::runtime_error("refined fiducial fit failed its minimum-inlier gate");
    }
    result.median_residual_pixels = median(residuals);
    result.root_mean_square_residual_pixels = std::sqrt(squared_sum / residuals.size());
    result.maximum_residual_pixels = maximum;
    result.circles = detected_circles;
    result.matches = std::move(matches);
    if (cv::determinant(cv::Mat(transform)) <= 0.0 ||
        !is_canonical_slide_orientation(transform)) {
        throw std::runtime_error("capture-grid transform violates canonical slide orientation");
    }
    const auto upper_left = project_point(result.spot_colrow_to_cytassist, {0.0, 0.0});
    const auto lower_right = project_point(
        result.spot_colrow_to_cytassist,
        {static_cast<double>(kGridColumns - 1), static_cast<double>(kGridRows - 1)}
    );
    if (!std::isfinite(upper_left.x) || !std::isfinite(lower_right.x)) {
        throw std::runtime_error("capture-grid transform projects grid to infinity");
    }
    return result;
}

Result localize_capture_grid(
    const cv::Mat& cytassist_bgr,
    const Layout& layout,
    const DetectionOptions& options
) {
    const std::vector<cv::Vec3f> detected = detect_circular_fiducials(cytassist_bgr, options);
    const Result preliminary = fit_capture_grid(detected, layout, options);
    const cv::Mat gray = grayscale_u8(cytassist_bgr);
    std::vector<cv::Vec3f> refined;
    refined.reserve(preliminary.matches.size());
    // Refine only circles that the initial detector actually matched. Adding
    // predicted centers for absent design fiducials would turn the fitted model
    // back into its own evidence and could conceal a damaged or missing mark.
    for (const FiducialMatch& match : preliminary.matches) {
        const cv::Point2d center = refine_concentric_center(gray, match.detected_xy);
        refined.emplace_back(
            static_cast<float>(center.x), static_cast<float>(center.y), 36.0F
        );
    }
    Result result = fit_capture_grid(refined, layout, options);
    result.detected_circles = static_cast<int>(detected.size());
    result.circles = std::move(refined);
    return result;
}

cv::Point2d project_point(const cv::Matx33d& transform, const cv::Point2d& point) {
    const cv::Vec3d value = transform * cv::Vec3d(point.x, point.y, 1.0);
    if (std::abs(value[2]) < 1e-12) throw std::runtime_error("point projects to infinity");
    return {value[0] / value[2], value[1] / value[2]};
}

}  // namespace visium_hd::capture_grid
