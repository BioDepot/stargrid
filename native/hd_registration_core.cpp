#include "hd_registration_core.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace visium_hd::registration {
namespace {

struct ScaledImage {
    cv::Mat image;
    cv::Matx33d source_to_scaled = cv::Matx33d::eye();
};

struct OrientedImage {
    std::string name;
    cv::Mat image;
    cv::Matx33d source_to_oriented = cv::Matx33d::eye();
};

struct Candidate {
    OrientationMetrics metrics;
    cv::Matx33d source_working_to_fixed_working = cv::Matx33d::eye();
    std::vector<MatchDiagnostic> matches;
};

struct GoodMatch {
    cv::DMatch match;
    double ratio = 0.0;
};

struct NccEvaluation {
    double score = -std::numeric_limits<double>::infinity();
    int pixels = 0;
};

struct ResidualRefinement {
    cv::Matx33d refined = cv::Matx33d::eye();
    int overlap_pixels = 0;
    double phase_shift_x = 0.0;
    double phase_shift_y = 0.0;
    double phase_response = 0.0;
    int shift_x = 0;
    int shift_y = 0;
    double ncc_before = 0.0;
    double ncc_after = 0.0;
    bool applied = false;
};

struct DisplacementField {
    cv::Mat x;
    cv::Mat y;
    double maximum_magnitude = 0.0;
};

cv::Matx33d mat_to_matx(const cv::Mat& matrix) {
    cv::Mat converted;
    matrix.convertTo(converted, CV_64F);
    if (converted.rows != 3 || converted.cols != 3) {
        throw std::runtime_error("registration transform is not 3x3");
    }
    cv::Matx33d result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) result(row, col) = converted.at<double>(row, col);
    }
    return result;
}

cv::Matx33d affine_to_matx(const cv::Mat& matrix) {
    cv::Mat converted;
    matrix.convertTo(converted, CV_64F);
    if (converted.rows != 2 || converted.cols != 3) {
        throw std::runtime_error("registration affine transform is not 2x3");
    }
    return cv::Matx33d(
        converted.at<double>(0, 0),
        converted.at<double>(0, 1),
        converted.at<double>(0, 2),
        converted.at<double>(1, 0),
        converted.at<double>(1, 1),
        converted.at<double>(1, 2),
        0.0,
        0.0,
        1.0
    );
}

cv::Matx33d inverse(const cv::Matx33d& matrix) {
    cv::Mat source(3, 3, CV_64F, const_cast<double*>(matrix.val));
    cv::Mat inverted;
    if (cv::invert(source, inverted, cv::DECOMP_LU) == 0.0) {
        throw std::runtime_error("singular registration transform");
    }
    return mat_to_matx(inverted);
}

ScaledImage scale_for_work(const cv::Mat& source, int max_dimension, bool allow_upscale = false) {
    const int source_max_dimension = std::max(source.cols, source.rows);
    if (max_dimension <= 0 || source_max_dimension == max_dimension ||
        (!allow_upscale && source_max_dimension < max_dimension)) {
        return {source, cv::Matx33d::eye()};
    }
    const double nominal = static_cast<double>(max_dimension) /
                           static_cast<double>(source_max_dimension);
    const int width = std::max(1, static_cast<int>(std::lround(source.cols * nominal)));
    const int height = std::max(1, static_cast<int>(std::lround(source.rows * nominal)));
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_AREA);
    return {resized, pixel_center_resize_transform(
        source.cols, source.rows, width, height
    )};
}

cv::Mat contrast_normalize(const cv::Mat& source) {
    CV_Assert(source.type() == CV_8U);
    int histogram[256] = {};
    for (int row = 0; row < source.rows; ++row) {
        const auto* values = source.ptr<unsigned char>(row);
        for (int col = 0; col < source.cols; ++col) ++histogram[values[col]];
    }
    const long long count = static_cast<long long>(source.rows) * source.cols;
    const long long low_target = count / 100;
    const long long high_target = count - low_target;
    long long cumulative = 0;
    int low = 0;
    int high = 255;
    for (int value = 0; value < 256; ++value) {
        cumulative += histogram[value];
        if (cumulative >= low_target) {
            low = value;
            break;
        }
    }
    cumulative = 0;
    for (int value = 0; value < 256; ++value) {
        cumulative += histogram[value];
        if (cumulative >= high_target) {
            high = value;
            break;
        }
    }
    if (high <= low + 1) return source.clone();
    cv::Mat normalized;
    source.convertTo(
        normalized,
        CV_8U,
        255.0 / static_cast<double>(high - low),
        -255.0 * low / static_cast<double>(high - low)
    );
    return normalized;
}

cv::Mat histology_features(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("registration requires a non-empty 8-bit three-channel image");
    }
    cv::Mat floating;
    bgr.convertTo(floating, CV_32FC3, 1.0 / 256.0, 1.0 / 256.0);
    std::vector<cv::Mat> channels;
    cv::split(floating, channels);  // B, G, R
    for (auto& channel : channels) {
        cv::log(channel, channel);
        channel *= -1.0F;
    }
    // Ruifrok-style hematoxylin direction. Exact stain separation is not
    // required; this provides a stable morphology image across the two scans.
    cv::Mat hematoxylin =
        0.290F * channels[0] + 0.704F * channels[1] + 0.650F * channels[2];
    cv::Mat hematoxylin_u8;
    cv::normalize(hematoxylin, hematoxylin_u8, 0, 255, cv::NORM_MINMAX, CV_8U);
    hematoxylin_u8 = contrast_normalize(hematoxylin_u8);
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(hematoxylin_u8, enhanced);
    cv::GaussianBlur(enhanced, enhanced, cv::Size(3, 3), 0.6);
    return enhanced;
}

std::vector<OrientedImage> orientations(const cv::Mat& source) {
    const double w = source.cols - 1.0;
    const double h = source.rows - 1.0;
    std::vector<OrientedImage> result;
    result.push_back({"identity", source, cv::Matx33d::eye()});

    cv::Mat transformed;
    cv::flip(source, transformed, 1);
    result.push_back({
        "flip_x",
        transformed,
        cv::Matx33d(-1.0, 0.0, w, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    });
    cv::flip(source, transformed, 0);
    result.push_back({
        "flip_y",
        transformed,
        cv::Matx33d(1.0, 0.0, 0.0, 0.0, -1.0, h, 0.0, 0.0, 1.0),
    });
    cv::rotate(source, transformed, cv::ROTATE_180);
    result.push_back({
        "rotate_180",
        transformed,
        cv::Matx33d(-1.0, 0.0, w, 0.0, -1.0, h, 0.0, 0.0, 1.0),
    });
    cv::transpose(source, transformed);
    result.push_back({
        "transpose",
        transformed,
        cv::Matx33d(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    });
    cv::rotate(source, transformed, cv::ROTATE_90_CLOCKWISE);
    result.push_back({
        "rotate_90_cw",
        transformed,
        cv::Matx33d(0.0, -1.0, h, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    });
    cv::rotate(source, transformed, cv::ROTATE_90_COUNTERCLOCKWISE);
    result.push_back({
        "rotate_90_ccw",
        transformed,
        cv::Matx33d(0.0, 1.0, 0.0, -1.0, 0.0, w, 0.0, 0.0, 1.0),
    });
    cv::transpose(source, transformed);
    cv::flip(transformed, transformed, -1);
    result.push_back({
        "anti_transpose",
        transformed,
        cv::Matx33d(0.0, -1.0, h, -1.0, 0.0, w, 0.0, 0.0, 1.0),
    });
    return result;
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 == 1) return upper;
    std::nth_element(values.begin(), values.begin() + middle - 1, values.end());
    return 0.5 * (values[middle - 1] + upper);
}

bool finite_transform(const cv::Matx33d& transform) {
    for (double value : transform.val) {
        if (!std::isfinite(value)) return false;
    }
    return std::abs(cv::determinant(cv::Mat(transform))) > 1e-12;
}

bool plausible_projection(
    const cv::Matx33d& transform,
    int moving_width,
    int moving_height,
    int fixed_width,
    int fixed_height
) {
    std::vector<cv::Point2f> projected;
    for (const auto& point : std::vector<cv::Point2d>{
             {0.0, 0.0},
             {moving_width - 1.0, 0.0},
             {moving_width - 1.0, moving_height - 1.0},
             {0.0, moving_height - 1.0},
         }) {
        const cv::Point2d value = project_point(transform, point);
        projected.emplace_back(static_cast<float>(value.x), static_cast<float>(value.y));
    }
    for (const auto& point : projected) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) return false;
    }
    const double area = std::abs(cv::contourArea(projected));
    const double fixed_area = static_cast<double>(fixed_width) * fixed_height;
    if (area < fixed_area * 0.005 || area > fixed_area * 200.0) return false;
    const cv::Rect2d projected_bounds = cv::boundingRect(projected);
    const cv::Rect2d fixed_bounds(0.0, 0.0, fixed_width, fixed_height);
    return (projected_bounds & fixed_bounds).area() > fixed_area * 0.005;
}

Candidate fit_candidate(
    const OrientedImage& moving,
    const std::vector<cv::KeyPoint>& fixed_keypoints,
    const cv::Mat& fixed_descriptors,
    const Options& options
) {
    Candidate candidate;
    candidate.metrics.name = moving.name;
    candidate.metrics.fixed_keypoints = static_cast<int>(fixed_keypoints.size());

    auto sift = cv::SIFT::create(options.max_features);
    std::vector<cv::KeyPoint> moving_keypoints;
    cv::Mat moving_descriptors;
    sift->detectAndCompute(moving.image, cv::noArray(), moving_keypoints, moving_descriptors);
    candidate.metrics.moving_keypoints = static_cast<int>(moving_keypoints.size());
    if (moving_descriptors.empty() || fixed_descriptors.empty()) {
        candidate.metrics.rejection_reason = "no_descriptors";
        return candidate;
    }

    cv::BFMatcher matcher(cv::NORM_L2, false);
    std::vector<std::vector<cv::DMatch>> neighbors;
    matcher.knnMatch(moving_descriptors, fixed_descriptors, neighbors, 2);
    std::vector<GoodMatch> good;
    good.reserve(neighbors.size());
    for (const auto& row : neighbors) {
        if (row.size() == 2 && row[0].distance < options.ratio_test * row[1].distance) {
            good.push_back({row[0], row[0].distance / row[1].distance});
        }
    }
    if (options.mutual_matching && !good.empty()) {
        std::vector<std::vector<cv::DMatch>> reverse_neighbors;
        matcher.knnMatch(fixed_descriptors, moving_descriptors, reverse_neighbors, 2);
        std::vector<int> reverse_best(static_cast<size_t>(fixed_descriptors.rows), -1);
        for (size_t fixed_index = 0; fixed_index < reverse_neighbors.size(); ++fixed_index) {
            const auto& row = reverse_neighbors[fixed_index];
            if (row.size() == 2 && row[0].distance < options.ratio_test * row[1].distance) {
                reverse_best[fixed_index] = row[0].trainIdx;
            }
        }
        good.erase(
            std::remove_if(
                good.begin(),
                good.end(),
                [&](const GoodMatch& row) {
                    return reverse_best[static_cast<size_t>(row.match.trainIdx)] !=
                           row.match.queryIdx;
                }
            ),
            good.end()
        );
    }
    candidate.metrics.ratio_matches = static_cast<int>(good.size());
    if (static_cast<int>(good.size()) < options.minimum_good_matches) {
        candidate.metrics.rejection_reason = "too_few_ratio_matches";
        return candidate;
    }

    std::vector<cv::Point2f> moving_points;
    std::vector<cv::Point2f> fixed_points;
    moving_points.reserve(good.size());
    fixed_points.reserve(good.size());
    for (const auto& row : good) {
        moving_points.push_back(moving_keypoints[row.match.queryIdx].pt);
        fixed_points.push_back(fixed_keypoints[row.match.trainIdx].pt);
    }
    cv::Mat inlier_mask;
    cv::Matx33d oriented_to_fixed_matrix;
    cv::Mat fitted;
    if (options.transform_model == "homography") {
        fitted = cv::findHomography(
            moving_points,
            fixed_points,
            cv::RANSAC,
            options.ransac_threshold_pixels,
            inlier_mask,
            10000,
            0.999
        );
        if (!fitted.empty()) oriented_to_fixed_matrix = mat_to_matx(fitted);
    } else if (options.transform_model == "affine") {
        fitted = cv::estimateAffine2D(
            moving_points,
            fixed_points,
            inlier_mask,
            cv::RANSAC,
            options.ransac_threshold_pixels,
            10000,
            0.999,
            10
        );
        if (!fitted.empty()) oriented_to_fixed_matrix = affine_to_matx(fitted);
    } else if (options.transform_model == "similarity") {
        fitted = cv::estimateAffinePartial2D(
            moving_points,
            fixed_points,
            inlier_mask,
            cv::RANSAC,
            options.ransac_threshold_pixels,
            10000,
            0.999,
            10
        );
        if (!fitted.empty()) oriented_to_fixed_matrix = affine_to_matx(fitted);
    } else {
        throw std::runtime_error("unknown registration transform model: " + options.transform_model);
    }
    if (fitted.empty()) {
        candidate.metrics.rejection_reason = "transform_fit_failed";
        return candidate;
    }
    const cv::Matx33d transform = oriented_to_fixed_matrix * moving.source_to_oriented;
    if (!finite_transform(transform)) {
        candidate.metrics.rejection_reason = "non_finite_or_singular_transform";
        return candidate;
    }

    std::vector<double> errors;
    for (size_t index = 0; index < good.size(); ++index) {
        if (!inlier_mask.at<unsigned char>(static_cast<int>(index))) continue;
        const cv::Point2d projected = project_point(
            oriented_to_fixed_matrix, moving_points[index]
        );
        errors.push_back(cv::norm(projected - cv::Point2d(fixed_points[index])));
    }
    candidate.metrics.inliers = static_cast<int>(errors.size());
    candidate.metrics.inlier_fraction =
        static_cast<double>(errors.size()) / static_cast<double>(good.size());
    candidate.metrics.median_inlier_error = median(errors);
    if (candidate.metrics.inliers < options.minimum_inliers) {
        candidate.metrics.rejection_reason = "too_few_inliers";
        return candidate;
    }
    candidate.metrics.accepted = true;
    candidate.source_working_to_fixed_working = transform;
    const cv::Matx33d oriented_to_source = inverse(moving.source_to_oriented);
    candidate.matches.reserve(good.size());
    for (size_t index = 0; index < good.size(); ++index) {
        const cv::Point2d source_working = project_point(
            oriented_to_source, moving_points[index]
        );
        const cv::Point2d fixed_working = fixed_points[index];
        const cv::Point2d projected = project_point(transform, source_working);
        MatchDiagnostic diagnostic;
        diagnostic.moving_x = source_working.x;
        diagnostic.moving_y = source_working.y;
        diagnostic.fixed_x = fixed_working.x;
        diagnostic.fixed_y = fixed_working.y;
        diagnostic.projected_x = projected.x;
        diagnostic.projected_y = projected.y;
        diagnostic.descriptor_distance = good[index].match.distance;
        diagnostic.descriptor_ratio = good[index].ratio;
        diagnostic.reprojection_error = cv::norm(projected - fixed_working);
        diagnostic.inlier = inlier_mask.at<unsigned char>(static_cast<int>(index)) != 0;
        candidate.matches.push_back(diagnostic);
    }
    return candidate;
}

bool better(const Candidate& left, const Candidate& right) {
    if (left.metrics.accepted != right.metrics.accepted) return left.metrics.accepted;
    if (left.metrics.inliers != right.metrics.inliers) return left.metrics.inliers > right.metrics.inliers;
    if (left.metrics.inlier_fraction != right.metrics.inlier_fraction) {
        return left.metrics.inlier_fraction > right.metrics.inlier_fraction;
    }
    if (left.metrics.median_inlier_error != right.metrics.median_inlier_error) {
        return left.metrics.median_inlier_error < right.metrics.median_inlier_error;
    }
    return left.metrics.name < right.metrics.name;
}

NccEvaluation masked_ncc(
    const cv::Mat& moving,
    const cv::Mat& fixed,
    const cv::Mat& moving_valid,
    const cv::Rect& region,
    int shift_x,
    int shift_y
) {
    double sum_moving = 0.0;
    double sum_fixed = 0.0;
    int count = 0;
    for (int y = region.y; y < region.y + region.height; ++y) {
        const int moving_y = y - shift_y;
        if (moving_y < 0 || moving_y >= moving.rows) continue;
        for (int x = region.x; x < region.x + region.width; ++x) {
            const int moving_x = x - shift_x;
            if (moving_x < 0 || moving_x >= moving.cols ||
                moving_valid.at<unsigned char>(moving_y, moving_x) == 0) {
                continue;
            }
            sum_moving += moving.at<unsigned char>(moving_y, moving_x);
            sum_fixed += fixed.at<unsigned char>(y, x);
            ++count;
        }
    }
    NccEvaluation result;
    result.pixels = count;
    if (count < 4096) return result;
    const double mean_moving = sum_moving / count;
    const double mean_fixed = sum_fixed / count;
    double covariance = 0.0;
    double moving_variance = 0.0;
    double fixed_variance = 0.0;
    for (int y = region.y; y < region.y + region.height; ++y) {
        const int moving_y = y - shift_y;
        if (moving_y < 0 || moving_y >= moving.rows) continue;
        for (int x = region.x; x < region.x + region.width; ++x) {
            const int moving_x = x - shift_x;
            if (moving_x < 0 || moving_x >= moving.cols ||
                moving_valid.at<unsigned char>(moving_y, moving_x) == 0) {
                continue;
            }
            const double moving_delta =
                moving.at<unsigned char>(moving_y, moving_x) - mean_moving;
            const double fixed_delta = fixed.at<unsigned char>(y, x) - mean_fixed;
            covariance += moving_delta * fixed_delta;
            moving_variance += moving_delta * moving_delta;
            fixed_variance += fixed_delta * fixed_delta;
        }
    }
    if (moving_variance <= 1e-9 || fixed_variance <= 1e-9) return result;
    result.score = covariance / std::sqrt(moving_variance * fixed_variance);
    return result;
}

NccEvaluation evaluate_transform_ncc(
    const cv::Mat& moving_features,
    const cv::Mat& fixed_features,
    const cv::Matx33d& moving_to_fixed
) {
    cv::Mat warped;
    cv::warpPerspective(
        moving_features,
        warped,
        cv::Mat(moving_to_fixed),
        fixed_features.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::Mat source_valid(moving_features.size(), CV_8U, cv::Scalar(255));
    cv::Mat valid;
    cv::warpPerspective(
        source_valid,
        valid,
        cv::Mat(moving_to_fixed),
        fixed_features.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::erode(valid, valid, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    std::vector<cv::Point> valid_points;
    cv::findNonZero(valid, valid_points);
    if (valid_points.empty()) return {};
    const cv::Rect region = cv::boundingRect(valid_points);
    if (region.width < 64 || region.height < 64) return {};
    return masked_ncc(warped, fixed_features, valid, region, 0, 0);
}

NccEvaluation evaluate_fractional_shift_ncc(
    const cv::Mat& warped,
    const cv::Mat& fixed,
    const cv::Mat& valid,
    const cv::Rect& region,
    double shift_x,
    double shift_y
) {
    const cv::Matx23d translation(
        1.0, 0.0, shift_x - region.x,
        0.0, 1.0, shift_y - region.y
    );
    cv::Mat shifted;
    cv::Mat shifted_valid;
    cv::warpAffine(
        warped,
        shifted,
        cv::Mat(translation),
        region.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::warpAffine(
        valid,
        shifted_valid,
        cv::Mat(translation),
        region.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    return masked_ncc(
        shifted,
        fixed(region),
        shifted_valid,
        cv::Rect(0, 0, region.width, region.height),
        0,
        0
    );
}

ResidualRefinement refine_residual_translation(
    const cv::Mat& moving_features,
    const cv::Mat& fixed_features,
    const cv::Matx33d& moving_to_fixed
) {
    ResidualRefinement result;
    result.refined = moving_to_fixed;
    cv::Mat warped;
    cv::warpPerspective(
        moving_features,
        warped,
        cv::Mat(moving_to_fixed),
        fixed_features.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::Mat source_valid(moving_features.size(), CV_8U, cv::Scalar(255));
    cv::Mat valid;
    cv::warpPerspective(
        source_valid,
        valid,
        cv::Mat(moving_to_fixed),
        fixed_features.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::erode(valid, valid, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));
    std::vector<cv::Point> valid_points;
    cv::findNonZero(valid, valid_points);
    if (valid_points.empty()) return result;
    const cv::Rect region = cv::boundingRect(valid_points);
    if (region.width < 64 || region.height < 64) return result;

    const NccEvaluation before = masked_ncc(warped, fixed_features, valid, region, 0, 0);
    result.overlap_pixels = before.pixels;
    result.ncc_before = std::isfinite(before.score) ? before.score : 0.0;
    if (!std::isfinite(before.score)) return result;

    cv::Mat phase_moving(region.size(), CV_64F, cv::Scalar(0));
    cv::Mat phase_fixed(region.size(), CV_64F, cv::Scalar(0));
    cv::Mat region_valid = valid(region);
    const cv::Scalar moving_mean = cv::mean(warped(region), region_valid);
    const cv::Scalar fixed_mean = cv::mean(fixed_features(region), region_valid);
    for (int y = 0; y < region.height; ++y) {
        const auto* mask_row = region_valid.ptr<unsigned char>(y);
        const auto* moving_row = warped.ptr<unsigned char>(region.y + y);
        const auto* fixed_row = fixed_features.ptr<unsigned char>(region.y + y);
        auto* phase_moving_row = phase_moving.ptr<double>(y);
        auto* phase_fixed_row = phase_fixed.ptr<double>(y);
        for (int x = 0; x < region.width; ++x) {
            if (mask_row[x] == 0) continue;
            phase_moving_row[x] = moving_row[region.x + x] - moving_mean[0];
            phase_fixed_row[x] = fixed_row[region.x + x] - fixed_mean[0];
        }
    }
    cv::Mat window;
    cv::createHanningWindow(window, region.size(), CV_64F);
    const cv::Point2d phase = cv::phaseCorrelate(
        phase_moving, phase_fixed, window, &result.phase_response
    );
    result.phase_shift_x = phase.x;
    result.phase_shift_y = phase.y;

    std::vector<cv::Point> centers{{0, 0}};
    if (std::isfinite(phase.x) && std::isfinite(phase.y)) {
        const int phase_x = static_cast<int>(std::lround(phase.x));
        const int phase_y = static_cast<int>(std::lround(phase.y));
        centers.emplace_back(phase_x, phase_y);
        centers.emplace_back(-phase_x, -phase_y);
    }
    NccEvaluation best = before;
    int best_x = 0;
    int best_y = 0;
    for (const auto& center : centers) {
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                const int shift_x = std::clamp(center.x + dx, -8, 8);
                const int shift_y = std::clamp(center.y + dy, -8, 8);
                const NccEvaluation candidate = masked_ncc(
                    warped, fixed_features, valid, region, shift_x, shift_y
                );
                const int candidate_distance = std::abs(shift_x) + std::abs(shift_y);
                const int best_distance = std::abs(best_x) + std::abs(best_y);
                if (candidate.score > best.score + 1e-12 ||
                    (std::abs(candidate.score - best.score) <= 1e-12 &&
                     candidate_distance < best_distance)) {
                    best = candidate;
                    best_x = shift_x;
                    best_y = shift_y;
                }
            }
        }
    }
    result.shift_x = best_x;
    result.shift_y = best_y;
    result.ncc_after = std::isfinite(best.score) ? best.score : result.ncc_before;
    if ((best_x != 0 || best_y != 0) && best.score >= before.score + 0.001) {
        const cv::Matx33d correction(
            1.0, 0.0, best_x,
            0.0, 1.0, best_y,
            0.0, 0.0, 1.0
        );
        result.refined = correction * moving_to_fixed;
        result.applied = true;
    } else {
        result.shift_x = 0;
        result.shift_y = 0;
        result.ncc_after = result.ncc_before;
    }
    return result;
}

}  // namespace

cv::Matx33d pixel_center_resize_transform(
    int source_width,
    int source_height,
    int target_width,
    int target_height
) {
    if (source_width <= 0 || source_height <= 0 ||
        target_width <= 0 || target_height <= 0) {
        throw std::runtime_error("resize dimensions must be positive");
    }
    const double scale_x = static_cast<double>(target_width) / source_width;
    const double scale_y = static_cast<double>(target_height) / source_height;
    // OpenCV samples resized pixels by their centers:
    // target = scale * (source + 0.5) - 0.5.
    return cv::Matx33d(
        scale_x, 0.0, 0.5 * (scale_x - 1.0),
        0.0, scale_y, 0.5 * (scale_y - 1.0),
        0.0, 0.0, 1.0
    );
}

cv::Point2d project_point(const cv::Matx33d& transform, const cv::Point2d& point) {
    const double denominator = transform(2, 0) * point.x +
                               transform(2, 1) * point.y + transform(2, 2);
    if (std::abs(denominator) < 1e-15) {
        return {
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
        };
    }
    return {
        (transform(0, 0) * point.x + transform(0, 1) * point.y + transform(0, 2)) /
            denominator,
        (transform(1, 0) * point.x + transform(1, 1) * point.y + transform(1, 2)) /
            denominator,
    };
}

Result register_images(const cv::Mat& moving_bgr, const cv::Mat& fixed_bgr, const Options& options) {
    if (moving_bgr.empty() || fixed_bgr.empty()) {
        throw std::runtime_error("registration input image is empty");
    }
    if (options.deterministic) {
        cv::setNumThreads(1);
        cv::setRNGSeed(0x5A17);
    } else if (options.threads > 0) {
        cv::setNumThreads(options.threads);
    }
    const int moving_max_dimension = std::max(moving_bgr.cols, moving_bgr.rows);
    const int fixed_max_dimension = std::max(fixed_bgr.cols, fixed_bgr.rows);
    ScaledImage moving_scaled;
    ScaledImage fixed_scaled;
    if (options.match_scale_policy == "smaller" ||
        options.match_scale_policy == "larger") {
        const int paired_dimension = options.match_scale_policy == "smaller"
                                         ? std::min(moving_max_dimension, fixed_max_dimension)
                                         : std::max(moving_max_dimension, fixed_max_dimension);
        const int work_dimension = std::min(options.work_max_dimension, paired_dimension);
        const bool allow_upscale = options.match_scale_policy == "larger";
        moving_scaled = scale_for_work(moving_bgr, work_dimension, allow_upscale);
        fixed_scaled = scale_for_work(fixed_bgr, work_dimension, allow_upscale);
    } else if (options.match_scale_policy == "native") {
        moving_scaled = scale_for_work(moving_bgr, options.work_max_dimension);
        fixed_scaled = scale_for_work(fixed_bgr, options.work_max_dimension);
    } else {
        throw std::runtime_error(
            "unknown registration match-scale policy: " + options.match_scale_policy
        );
    }
    const cv::Mat moving_features = histology_features(moving_scaled.image);
    const cv::Mat fixed_features = histology_features(fixed_scaled.image);

    auto sift = cv::SIFT::create(options.max_features);
    std::vector<cv::KeyPoint> fixed_keypoints;
    cv::Mat fixed_descriptors;
    sift->detectAndCompute(fixed_features, cv::noArray(), fixed_keypoints, fixed_descriptors);
    if (fixed_descriptors.empty()) throw std::runtime_error("no fixed-image features detected");

    const std::vector<OrientedImage> oriented_images = orientations(moving_features);
    std::vector<Candidate> candidates(oriented_images.size());
    cv::parallel_for_(
        cv::Range(0, static_cast<int>(oriented_images.size())),
        [&](const cv::Range& range) {
            for (int index = range.start; index < range.end; ++index) {
                Candidate candidate = fit_candidate(
                    oriented_images[static_cast<size_t>(index)],
                    fixed_keypoints,
                    fixed_descriptors,
                    options
                );
                if (candidate.metrics.accepted &&
                    !plausible_projection(
                        candidate.source_working_to_fixed_working,
                        moving_scaled.image.cols,
                        moving_scaled.image.rows,
                        fixed_scaled.image.cols,
                        fixed_scaled.image.rows
                    )) {
                    candidate.metrics.accepted = false;
                    candidate.metrics.rejection_reason = "implausible_projected_bounds";
                }
                candidates[static_cast<size_t>(index)] = std::move(candidate);
            }
        }
    );
    const auto best_it = std::max_element(
        candidates.begin(),
        candidates.end(),
        [](const Candidate& left, const Candidate& right) { return better(right, left); }
    );
    if (best_it == candidates.end() || !best_it->metrics.accepted) {
        std::string detail = "no orientation produced an accepted registration transform";
        for (const auto& candidate : candidates) {
            detail += " [" + candidate.metrics.name +
                      ":matches=" + std::to_string(candidate.metrics.ratio_matches) +
                      ",inliers=" + std::to_string(candidate.metrics.inliers) +
                      ",reason=" + candidate.metrics.rejection_reason + "]";
        }
        throw std::runtime_error(detail);
    }
    ResidualRefinement refinement = refine_residual_translation(
        moving_features,
        fixed_features,
        best_it->source_working_to_fixed_working
    );
    if (!options.apply_residual_refinement && refinement.applied) {
        refinement.refined = best_it->source_working_to_fixed_working;
        refinement.shift_x = 0;
        refinement.shift_y = 0;
        refinement.ncc_after = refinement.ncc_before;
        refinement.applied = false;
    }
    Result result;
    const cv::Matx33d fixed_frame_offset(
        1.0, 0.0, options.fixed_frame_offset_x_pixels,
        0.0, 1.0, options.fixed_frame_offset_y_pixels,
        0.0, 0.0, 1.0
    );
    result.moving_to_fixed = fixed_frame_offset *
                             inverse(fixed_scaled.source_to_scaled) *
                             refinement.refined * moving_scaled.source_to_scaled;
    result.orientation = best_it->metrics.name;
    result.moving_width = moving_bgr.cols;
    result.moving_height = moving_bgr.rows;
    result.fixed_width = fixed_bgr.cols;
    result.fixed_height = fixed_bgr.rows;
    result.working_moving_width = moving_scaled.image.cols;
    result.working_moving_height = moving_scaled.image.rows;
    result.working_fixed_width = fixed_scaled.image.cols;
    result.working_fixed_height = fixed_scaled.image.rows;
    result.good_matches = best_it->metrics.ratio_matches;
    result.inliers = best_it->metrics.inliers;
    result.inlier_fraction = best_it->metrics.inlier_fraction;
    result.median_inlier_error = best_it->metrics.median_inlier_error;
    result.overlap_pixels = refinement.overlap_pixels;
    result.phase_shift_x = refinement.phase_shift_x;
    result.phase_shift_y = refinement.phase_shift_y;
    result.phase_response = refinement.phase_response;
    result.residual_shift_x = refinement.shift_x;
    result.residual_shift_y = refinement.shift_y;
    result.ncc_before_refinement = refinement.ncc_before;
    result.ncc_after_refinement = refinement.ncc_after;
    result.residual_refinement_applied = refinement.applied;
    for (const auto& candidate : candidates) result.candidates.push_back(candidate.metrics);
    const cv::Matx33d moving_working_to_source = inverse(moving_scaled.source_to_scaled);
    const cv::Matx33d fixed_working_to_source = inverse(fixed_scaled.source_to_scaled);
    result.matches.reserve(best_it->matches.size());
    for (const auto& working : best_it->matches) {
        const cv::Point2d moving_source = project_point(
            moving_working_to_source, {working.moving_x, working.moving_y}
        );
        const cv::Point2d fixed_source = project_point(
            fixed_working_to_source, {working.fixed_x, working.fixed_y}
        );
        const cv::Point2d projected_source = project_point(result.moving_to_fixed, moving_source);
        MatchDiagnostic diagnostic = working;
        diagnostic.moving_x = moving_source.x;
        diagnostic.moving_y = moving_source.y;
        diagnostic.fixed_x = fixed_source.x;
        diagnostic.fixed_y = fixed_source.y;
        diagnostic.projected_x = projected_source.x;
        diagnostic.projected_y = projected_source.y;
        diagnostic.reprojection_error = cv::norm(projected_source - fixed_source);
        result.matches.push_back(diagnostic);
    }
    return result;
}

TransformError compare_transforms(
    const cv::Matx33d& estimated,
    const cv::Matx33d& reference,
    int moving_width,
    int moving_height,
    int grid_size
) {
    if (grid_size < 2) throw std::runtime_error("transform comparison grid must be at least 2x2");
    std::vector<double> errors;
    errors.reserve(static_cast<size_t>(grid_size * grid_size));
    std::vector<cv::Point2d> deltas;
    deltas.reserve(static_cast<size_t>(grid_size * grid_size));
    double squared_sum = 0.0;
    double sum = 0.0;
    double maximum = 0.0;
    cv::Point2d delta_sum(0.0, 0.0);
    for (int row = 0; row < grid_size; ++row) {
        const double y = (moving_height - 1.0) * row / (grid_size - 1.0);
        for (int col = 0; col < grid_size; ++col) {
            const double x = (moving_width - 1.0) * col / (grid_size - 1.0);
            const cv::Point2d point(x, y);
            const cv::Point2d estimated_point = project_point(estimated, point);
            const cv::Point2d reference_point = project_point(reference, point);
            const cv::Point2d delta = estimated_point - reference_point;
            const double error = cv::norm(delta);
            if (!std::isfinite(error)) throw std::runtime_error("non-finite transform comparison");
            errors.push_back(error);
            deltas.push_back(delta);
            sum += error;
            squared_sum += error * error;
            maximum = std::max(maximum, error);
            delta_sum += delta;
        }
    }
    TransformError result;
    result.sampled_points = static_cast<int>(errors.size());
    result.mean_pixels = sum / errors.size();
    result.root_mean_square_pixels = std::sqrt(squared_sum / errors.size());
    result.median_pixels = median(errors);
    result.maximum_pixels = maximum;
    const cv::Point2d mean_delta = delta_sum * (1.0 / errors.size());
    result.mean_delta_x_pixels = mean_delta.x;
    result.mean_delta_y_pixels = mean_delta.y;
    double centered_squared_sum = 0.0;
    for (const auto& delta : deltas) {
        const cv::Point2d centered = delta - mean_delta;
        centered_squared_sum += centered.dot(centered);
    }
    result.residual_after_mean_translation_rmse_pixels =
        std::sqrt(centered_squared_sum / deltas.size());
    return result;
}

AlignmentScore score_transform_alignment(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const cv::Matx33d& moving_to_fixed,
    int work_max_dimension
) {
    if (moving_bgr.empty() || fixed_bgr.empty()) {
        throw std::runtime_error("alignment scoring input image is empty");
    }
    const int paired_work_dimension = std::min(
        work_max_dimension,
        std::min(
            std::max(moving_bgr.cols, moving_bgr.rows),
            std::max(fixed_bgr.cols, fixed_bgr.rows)
        )
    );
    const ScaledImage moving_scaled = scale_for_work(moving_bgr, paired_work_dimension);
    const ScaledImage fixed_scaled = scale_for_work(fixed_bgr, paired_work_dimension);
    const cv::Mat moving_features = histology_features(moving_scaled.image);
    const cv::Mat fixed_features = histology_features(fixed_scaled.image);
    const cv::Matx33d working_transform =
        fixed_scaled.source_to_scaled * moving_to_fixed * inverse(moving_scaled.source_to_scaled);
    const NccEvaluation score = evaluate_transform_ncc(
        moving_features, fixed_features, working_transform
    );
    AlignmentScore result;
    result.overlap_pixels = score.pixels;
    result.normalized_cross_correlation =
        std::isfinite(score.score) ? score.score : 0.0;
    return result;
}

std::vector<SpatialResidualCell> audit_spatial_residual_grid(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    int grid_size,
    int work_max_dimension,
    const cv::Rect2d& fixed_roi
) {
    if (moving_bgr.empty() || fixed_bgr.empty()) {
        throw std::runtime_error("spatial residual input image is empty");
    }
    if (grid_size < 2 || grid_size > 32) {
        throw std::runtime_error("spatial residual grid size must be between 2 and 32");
    }
    const int paired_work_dimension = std::min(
        work_max_dimension,
        std::min(
            std::max(moving_bgr.cols, moving_bgr.rows),
            std::max(fixed_bgr.cols, fixed_bgr.rows)
        )
    );
    const ScaledImage moving_scaled = scale_for_work(moving_bgr, paired_work_dimension);
    const ScaledImage fixed_scaled = scale_for_work(fixed_bgr, paired_work_dimension);
    const cv::Mat moving_features = histology_features(moving_scaled.image);
    const cv::Mat fixed_features = histology_features(fixed_scaled.image);
    const cv::Matx33d working_transform =
        fixed_scaled.source_to_scaled * result.moving_to_fixed *
        inverse(moving_scaled.source_to_scaled);
    cv::Mat warped;
    cv::warpPerspective(
        moving_features,
        warped,
        cv::Mat(working_transform),
        fixed_features.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::Mat source_valid(moving_features.size(), CV_8U, cv::Scalar(255));
    cv::Mat valid;
    cv::warpPerspective(
        source_valid,
        valid,
        cv::Mat(working_transform),
        fixed_features.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::erode(valid, valid, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));

    cv::Rect audit_region(0, 0, fixed_features.cols, fixed_features.rows);
    if (fixed_roi.width > 0.0 && fixed_roi.height > 0.0) {
        const int roi_x0 = static_cast<int>(std::floor(
            fixed_roi.x * fixed_scaled.source_to_scaled(0, 0)
        ));
        const int roi_y0 = static_cast<int>(std::floor(
            fixed_roi.y * fixed_scaled.source_to_scaled(1, 1)
        ));
        const int roi_x1 = static_cast<int>(std::ceil(
            (fixed_roi.x + fixed_roi.width) * fixed_scaled.source_to_scaled(0, 0)
        ));
        const int roi_y1 = static_cast<int>(std::ceil(
            (fixed_roi.y + fixed_roi.height) * fixed_scaled.source_to_scaled(1, 1)
        ));
        audit_region = cv::Rect(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0) &
                       cv::Rect(0, 0, fixed_features.cols, fixed_features.rows);
        if (audit_region.width < grid_size * 32 || audit_region.height < grid_size * 32) {
            throw std::runtime_error("spatial residual ROI is too small for the grid");
        }
    }

    std::vector<SpatialResidualCell> cells(static_cast<size_t>(grid_size * grid_size));
    cv::parallel_for_(cv::Range(0, grid_size * grid_size), [&](const cv::Range& range) {
        for (int cell_index = range.start; cell_index < range.end; ++cell_index) {
            const int grid_row = cell_index / grid_size;
            const int grid_col = cell_index % grid_size;
            const int y0 = audit_region.y + audit_region.height * grid_row / grid_size;
            const int y1 = audit_region.y + audit_region.height * (grid_row + 1) / grid_size;
            const int x0 = audit_region.x + audit_region.width * grid_col / grid_size;
            const int x1 = audit_region.x + audit_region.width * (grid_col + 1) / grid_size;
            const cv::Rect region(x0, y0, x1 - x0, y1 - y0);
            const cv::Mat region_valid = valid(region);
            SpatialResidualCell cell;
            cell.grid_row = grid_row;
            cell.grid_col = grid_col;
            cell.fixed_x0 = x0 / fixed_scaled.source_to_scaled(0, 0);
            cell.fixed_y0 = y0 / fixed_scaled.source_to_scaled(1, 1);
            cell.fixed_x1 = x1 / fixed_scaled.source_to_scaled(0, 0);
            cell.fixed_y1 = y1 / fixed_scaled.source_to_scaled(1, 1);
            cell.valid_pixels = cv::countNonZero(region_valid);
            cell.valid_fraction = static_cast<double>(cell.valid_pixels) / region.area();
            cv::Scalar moving_mean;
            cv::Scalar moving_deviation;
            cv::Scalar fixed_mean;
            cv::Scalar fixed_deviation;
            cv::meanStdDev(warped(region), moving_mean, moving_deviation, region_valid);
            cv::meanStdDev(fixed_features(region), fixed_mean, fixed_deviation, region_valid);
            cell.moving_standard_deviation = moving_deviation[0];
            cell.fixed_standard_deviation = fixed_deviation[0];
            const NccEvaluation before = masked_ncc(
                warped, fixed_features, valid, region, 0, 0
            );
            cell.ncc_before = std::isfinite(before.score) ? before.score : 0.0;
            cell.ncc_after = cell.ncc_before;

            if (cell.valid_pixels >= 4096) {
                cv::Mat phase_moving(region.size(), CV_64F, cv::Scalar(0));
                cv::Mat phase_fixed(region.size(), CV_64F, cv::Scalar(0));
                for (int y = 0; y < region.height; ++y) {
                    const auto* mask_row = region_valid.ptr<unsigned char>(y);
                    const auto* moving_row = warped.ptr<unsigned char>(region.y + y);
                    const auto* fixed_row = fixed_features.ptr<unsigned char>(region.y + y);
                    auto* phase_moving_row = phase_moving.ptr<double>(y);
                    auto* phase_fixed_row = phase_fixed.ptr<double>(y);
                    for (int x = 0; x < region.width; ++x) {
                        if (mask_row[x] == 0) continue;
                        phase_moving_row[x] = moving_row[region.x + x] - moving_mean[0];
                        phase_fixed_row[x] = fixed_row[region.x + x] - fixed_mean[0];
                    }
                }
                cv::Mat window;
                cv::createHanningWindow(window, region.size(), CV_64F);
                const cv::Point2d phase = cv::phaseCorrelate(
                    phase_moving, phase_fixed, window, &cell.phase_response
                );
                const NccEvaluation positive = evaluate_fractional_shift_ncc(
                    warped, fixed_features, valid, region, phase.x, phase.y
                );
                const NccEvaluation negative = evaluate_fractional_shift_ncc(
                    warped, fixed_features, valid, region, -phase.x, -phase.y
                );
                const bool use_positive = positive.score >= negative.score;
                const cv::Point2d selected = use_positive ? phase : -phase;
                const NccEvaluation selected_score = use_positive ? positive : negative;
                cell.phase_shift_x =
                    selected.x / fixed_scaled.source_to_scaled(0, 0);
                cell.phase_shift_y =
                    selected.y / fixed_scaled.source_to_scaled(1, 1);
                cell.ncc_after = std::isfinite(selected_score.score)
                                     ? selected_score.score
                                     : cell.ncc_before;
            }

            std::vector<double> landmark_errors;
            cv::Point2d landmark_sum(0.0, 0.0);
            for (const auto& match : result.matches) {
                if (!match.inlier ||
                    match.fixed_x < cell.fixed_x0 || match.fixed_x >= cell.fixed_x1 ||
                    match.fixed_y < cell.fixed_y0 || match.fixed_y >= cell.fixed_y1) {
                    continue;
                }
                const cv::Point2d delta(
                    match.projected_x - match.fixed_x,
                    match.projected_y - match.fixed_y
                );
                landmark_sum += delta;
                landmark_errors.push_back(match.reprojection_error);
            }
            cell.inlier_matches = static_cast<int>(landmark_errors.size());
            if (cell.inlier_matches) {
                cell.landmark_mean_delta_x = landmark_sum.x / cell.inlier_matches;
                cell.landmark_mean_delta_y = landmark_sum.y / cell.inlier_matches;
                cell.landmark_median_error = median(landmark_errors);
            }
            cell.quality_pass =
                cell.valid_fraction >= 0.60 &&
                cell.moving_standard_deviation >= 5.0 &&
                cell.fixed_standard_deviation >= 5.0 &&
                cell.phase_response >= 0.20 &&
                cell.ncc_after >= cell.ncc_before + 0.001;
            cells[static_cast<size_t>(cell_index)] = cell;
        }
    });
    return cells;
}

SpatialCorrectionEvaluation evaluate_spatial_correction(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    const std::vector<SpatialResidualCell>& cells
) {
    if (moving_bgr.empty() || fixed_bgr.empty()) {
        throw std::runtime_error("spatial correction input image is empty");
    }
    std::vector<double> cell_spans;
    int support_cells = 0;
    cv::Rect audit_region;
    bool have_region = false;
    for (const auto& cell : cells) {
        if (!cell.quality_pass) continue;
        ++support_cells;
        cell_spans.push_back(
            0.5 * ((cell.fixed_x1 - cell.fixed_x0) + (cell.fixed_y1 - cell.fixed_y0))
        );
        const cv::Rect cell_region(
            static_cast<int>(std::floor(cell.fixed_x0)),
            static_cast<int>(std::floor(cell.fixed_y0)),
            std::max(1, static_cast<int>(std::ceil(cell.fixed_x1 - cell.fixed_x0))),
            std::max(1, static_cast<int>(std::ceil(cell.fixed_y1 - cell.fixed_y0)))
        );
        audit_region = have_region ? (audit_region | cell_region) : cell_region;
        have_region = true;
    }
    if (support_cells < 4 || !have_region) {
        throw std::runtime_error("spatial correction requires at least four supported cells");
    }
    audit_region &= cv::Rect(0, 0, fixed_bgr.cols, fixed_bgr.rows);
    const double spacing = median(cell_spans);
    const double sigma = std::max(8.0, 1.25 * spacing);
    const double sigma_squared = sigma * sigma;

    auto make_field = [&](int parity) {
        DisplacementField field;
        field.x = cv::Mat::zeros(fixed_bgr.size(), CV_32F);
        field.y = cv::Mat::zeros(fixed_bgr.size(), CV_32F);
        cv::parallel_for_(cv::Range(0, fixed_bgr.rows),
            [&](const cv::Range& range) {
                for (int y = range.start; y < range.end; ++y) {
                    float* out_x = field.x.ptr<float>(y);
                    float* out_y = field.y.ptr<float>(y);
                    for (int x = 0; x < fixed_bgr.cols; ++x) {
                        double weighted_x = 0.0;
                        double weighted_y = 0.0;
                        double weight_sum = 0.0;
                        for (const auto& cell : cells) {
                            if (!cell.quality_pass) continue;
                            if (parity >= 0 && ((cell.grid_row + cell.grid_col) & 1) != parity) {
                                continue;
                            }
                            const double center_x = 0.5 * (cell.fixed_x0 + cell.fixed_x1);
                            const double center_y = 0.5 * (cell.fixed_y0 + cell.fixed_y1);
                            const double delta_x = x - center_x;
                            const double delta_y = y - center_y;
                            const double distance_squared = delta_x * delta_x + delta_y * delta_y;
                            const double confidence = std::clamp(cell.phase_response, 0.20, 1.0);
                            const double weight = confidence *
                                std::exp(-0.5 * distance_squared / sigma_squared);
                            weighted_x += weight * cell.phase_shift_x;
                            weighted_y += weight * cell.phase_shift_y;
                            weight_sum += weight;
                        }
                        if (weight_sum > 1e-12) {
                            out_x[x] = static_cast<float>(weighted_x / weight_sum);
                            out_y[x] = static_cast<float>(weighted_y / weight_sum);
                        }
                    }
                }
            }
        );
        cv::Mat magnitude;
        cv::magnitude(field.x, field.y, magnitude);
        cv::minMaxLoc(magnitude, nullptr, &field.maximum_magnitude);
        return field;
    };

    auto make_constant_field = [&](int parity) {
        DisplacementField field;
        field.x = cv::Mat::zeros(fixed_bgr.size(), CV_32F);
        field.y = cv::Mat::zeros(fixed_bgr.size(), CV_32F);
        double weighted_x = 0.0;
        double weighted_y = 0.0;
        double weight_sum = 0.0;
        for (const auto& cell : cells) {
            if (!cell.quality_pass) continue;
            if (parity >= 0 && ((cell.grid_row + cell.grid_col) & 1) != parity) continue;
            const double weight = std::clamp(cell.phase_response, 0.20, 1.0);
            weighted_x += weight * cell.phase_shift_x;
            weighted_y += weight * cell.phase_shift_y;
            weight_sum += weight;
        }
        if (weight_sum <= 0.0) return field;
        const float shift_x = static_cast<float>(weighted_x / weight_sum);
        const float shift_y = static_cast<float>(weighted_y / weight_sum);
        field.x.setTo(shift_x);
        field.y.setTo(shift_y);
        field.maximum_magnitude = std::hypot(shift_x, shift_y);
        return field;
    };

    auto remap_with_field = [&](const cv::Mat& source, const DisplacementField& field,
                                int interpolation, const cv::Scalar& border) {
        cv::Mat map_x(source.size(), CV_32F);
        cv::Mat map_y(source.size(), CV_32F);
        cv::parallel_for_(cv::Range(0, source.rows), [&](const cv::Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                float* mx = map_x.ptr<float>(y);
                float* my = map_y.ptr<float>(y);
                const float* dx = field.x.ptr<float>(y);
                const float* dy = field.y.ptr<float>(y);
                for (int x = 0; x < source.cols; ++x) {
                    mx[x] = static_cast<float>(x) - dx[x];
                    my[x] = static_cast<float>(y) - dy[x];
                }
            }
        });
        cv::Mat corrected;
        cv::remap(source, corrected, map_x, map_y, interpolation, cv::BORDER_CONSTANT, border);
        return corrected;
    };

    // Match the established registration scorer: downsample the moving image
    // before feature extraction and warping, while keeping the fixed ROI at
    // its native dimensions. Computing stain features at full moving-image
    // resolution and shrinking them only during the warp creates a large,
    // artificial interpolation benefit for any second remap.
    const int fixed_work_dimension = std::max(fixed_bgr.cols, fixed_bgr.rows);
    const ScaledImage moving_scaled = scale_for_work(moving_bgr, fixed_work_dimension);
    const ScaledImage fixed_scaled = scale_for_work(fixed_bgr, fixed_work_dimension);
    const cv::Mat moving_features = histology_features(moving_scaled.image);
    const cv::Mat fixed_features = histology_features(fixed_scaled.image);
    const cv::Matx33d working_transform =
        fixed_scaled.source_to_scaled * result.moving_to_fixed *
        inverse(moving_scaled.source_to_scaled);
    cv::Mat warped_features;
    cv::warpPerspective(
        moving_features,
        warped_features,
        cv::Mat(working_transform),
        fixed_features.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::Mat source_valid(moving_features.size(), CV_8U, cv::Scalar(255));
    cv::Mat valid;
    cv::warpPerspective(
        source_valid,
        valid,
        cv::Mat(working_transform),
        fixed_features.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::erode(valid, valid, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9)));

    SpatialCorrectionEvaluation evaluation;
    evaluation.support_cells = support_cells;
    const DisplacementField final_field = make_field(-1);
    evaluation.displacement_x = final_field.x;
    evaluation.displacement_y = final_field.y;
    evaluation.maximum_displacement = final_field.maximum_magnitude;
    const cv::Mat corrected_features = remap_with_field(
        warped_features, final_field, cv::INTER_LINEAR, cv::Scalar(0)
    );
    const cv::Mat corrected_valid = remap_with_field(
        valid, final_field, cv::INTER_NEAREST, cv::Scalar(0)
    );
    cv::Mat common_valid;
    cv::bitwise_and(valid, corrected_valid, common_valid);
    const NccEvaluation before = masked_ncc(
        warped_features, fixed_features, common_valid, audit_region, 0, 0
    );
    const NccEvaluation after = masked_ncc(
        corrected_features, fixed_features, common_valid, audit_region, 0, 0
    );
    evaluation.ncc_before = std::isfinite(before.score) ? before.score : 0.0;
    evaluation.ncc_after = std::isfinite(after.score) ? after.score : 0.0;
    const DisplacementField constant_field = make_constant_field(-1);
    evaluation.constant_shift_x = constant_field.x.at<float>(
        audit_region.y + audit_region.height / 2,
        audit_region.x + audit_region.width / 2
    );
    evaluation.constant_shift_y = constant_field.y.at<float>(
        audit_region.y + audit_region.height / 2,
        audit_region.x + audit_region.width / 2
    );
    const cv::Mat constant_features = remap_with_field(
        warped_features, constant_field, cv::INTER_LINEAR, cv::Scalar(0)
    );
    const cv::Mat constant_valid = remap_with_field(
        valid, constant_field, cv::INTER_NEAREST, cv::Scalar(0)
    );
    cv::Mat constant_common;
    cv::bitwise_and(valid, constant_valid, constant_common);
    const NccEvaluation constant_after = masked_ncc(
        constant_features, fixed_features, constant_common, audit_region, 0, 0
    );
    evaluation.constant_ncc_after =
        std::isfinite(constant_after.score) ? constant_after.score : 0.0;

    double heldout_before_sum = 0.0;
    double heldout_after_sum = 0.0;
    double heldout_constant_after_sum = 0.0;
    int heldout_weight = 0;
    for (int training_parity = 0; training_parity < 2; ++training_parity) {
        const DisplacementField fold_field = make_field(training_parity);
        const DisplacementField fold_constant_field = make_constant_field(training_parity);
        const cv::Mat fold_features = remap_with_field(
            warped_features, fold_field, cv::INTER_LINEAR, cv::Scalar(0)
        );
        const cv::Mat fold_valid = remap_with_field(
            valid, fold_field, cv::INTER_NEAREST, cv::Scalar(0)
        );
        const cv::Mat fold_constant_features = remap_with_field(
            warped_features, fold_constant_field, cv::INTER_LINEAR, cv::Scalar(0)
        );
        const cv::Mat fold_constant_valid = remap_with_field(
            valid, fold_constant_field, cv::INTER_NEAREST, cv::Scalar(0)
        );
        cv::Mat heldout_mask = cv::Mat::zeros(fixed_bgr.size(), CV_8U);
        int fold_cells = 0;
        for (const auto& cell : cells) {
            if (!cell.quality_pass || ((cell.grid_row + cell.grid_col) & 1) == training_parity) {
                continue;
            }
            const cv::Rect region(
                static_cast<int>(std::floor(cell.fixed_x0)),
                static_cast<int>(std::floor(cell.fixed_y0)),
                std::max(1, static_cast<int>(std::ceil(cell.fixed_x1 - cell.fixed_x0))),
                std::max(1, static_cast<int>(std::ceil(cell.fixed_y1 - cell.fixed_y0)))
            );
            heldout_mask(region & cv::Rect(0, 0, fixed_bgr.cols, fixed_bgr.rows)).setTo(255);
            ++fold_cells;
        }
        cv::Mat fold_common;
        cv::bitwise_and(valid, fold_valid, fold_common);
        cv::bitwise_and(fold_common, heldout_mask, fold_common);
        const NccEvaluation fold_before = masked_ncc(
            warped_features, fixed_features, fold_common, audit_region, 0, 0
        );
        const NccEvaluation fold_after = masked_ncc(
            fold_features, fixed_features, fold_common, audit_region, 0, 0
        );
        cv::Mat fold_constant_common;
        cv::bitwise_and(valid, fold_constant_valid, fold_constant_common);
        cv::bitwise_and(fold_constant_common, heldout_mask, fold_constant_common);
        const NccEvaluation fold_constant_after = masked_ncc(
            fold_constant_features,
            fixed_features,
            fold_constant_common,
            audit_region,
            0,
            0
        );
        const int fold_weight = std::min(fold_before.pixels, fold_after.pixels);
        if (fold_weight > 0 && std::isfinite(fold_before.score) &&
            std::isfinite(fold_after.score) && std::isfinite(fold_constant_after.score)) {
            heldout_before_sum += fold_weight * fold_before.score;
            heldout_after_sum += fold_weight * fold_after.score;
            heldout_constant_after_sum += fold_weight * fold_constant_after.score;
            heldout_weight += fold_weight;
            evaluation.heldout_cells += fold_cells;
        }
    }
    evaluation.heldout_pixels = heldout_weight;
    if (heldout_weight > 0) {
        evaluation.heldout_ncc_before = heldout_before_sum / heldout_weight;
        evaluation.heldout_ncc_after = heldout_after_sum / heldout_weight;
        evaluation.heldout_constant_ncc_after =
            heldout_constant_after_sum / heldout_weight;
    }

    std::vector<double> landmark_before;
    std::vector<double> landmark_after;
    for (const auto& match : result.matches) {
        if (!match.inlier) continue;
        const int x = std::clamp(
            static_cast<int>(std::lround(match.projected_x)), 0, fixed_bgr.cols - 1
        );
        const int y = std::clamp(
            static_cast<int>(std::lround(match.projected_y)), 0, fixed_bgr.rows - 1
        );
        landmark_before.push_back(match.reprojection_error);
        const cv::Point2d corrected(
            match.projected_x + final_field.x.at<float>(y, x),
            match.projected_y + final_field.y.at<float>(y, x)
        );
        landmark_after.push_back(cv::norm(corrected - cv::Point2d(match.fixed_x, match.fixed_y)));
    }
    evaluation.landmark_median_before = median(landmark_before);
    evaluation.landmark_median_after = median(landmark_after);

    // Compose the global inverse homography and local displacement in one
    // sampling operation. A warp followed by a second remap visibly changes
    // stain intensity inside the projected crop and is not a fair image output.
    const cv::Matx33d fixed_to_moving = inverse(result.moving_to_fixed);
    cv::Mat direct_map_x(fixed_bgr.size(), CV_32F);
    cv::Mat direct_map_y(fixed_bgr.size(), CV_32F);
    cv::parallel_for_(cv::Range(0, fixed_bgr.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float* map_x = direct_map_x.ptr<float>(y);
            float* map_y = direct_map_y.ptr<float>(y);
            const float* shift_x = final_field.x.ptr<float>(y);
            const float* shift_y = final_field.y.ptr<float>(y);
            for (int x = 0; x < fixed_bgr.cols; ++x) {
                const cv::Point2d source = project_point(
                    fixed_to_moving,
                    cv::Point2d(x - shift_x[x], y - shift_y[x])
                );
                map_x[x] = static_cast<float>(source.x);
                map_y[x] = static_cast<float>(source.y);
            }
        }
    });
    cv::remap(
        moving_bgr,
        evaluation.corrected_registered_bgr,
        direct_map_x,
        direct_map_y,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(255, 255, 255)
    );
    return evaluation;
}

GlobalOptimizationEvaluation evaluate_global_optimization(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    const std::vector<SpatialResidualCell>& cells,
    int maximum_iterations
) {
    if (moving_bgr.empty() || fixed_bgr.empty()) {
        throw std::runtime_error("global optimization input image is empty");
    }
    if (maximum_iterations < 1) {
        throw std::runtime_error("global optimization iterations must be positive");
    }
    const int fixed_work_dimension = std::max(fixed_bgr.cols, fixed_bgr.rows);
    const ScaledImage moving_scaled = scale_for_work(moving_bgr, fixed_work_dimension);
    const ScaledImage fixed_scaled = scale_for_work(fixed_bgr, fixed_work_dimension);
    if (moving_scaled.image.size() != fixed_scaled.image.size()) {
        throw std::runtime_error("global ECC optimization requires equal working image sizes");
    }
    const cv::Mat moving_features = histology_features(moving_scaled.image);
    const cv::Mat fixed_features = histology_features(fixed_scaled.image);
    const cv::Matx33d base_working =
        fixed_scaled.source_to_scaled * result.moving_to_fixed *
        inverse(moving_scaled.source_to_scaled);
    cv::Mat base_warped;
    cv::warpPerspective(
        moving_features,
        base_warped,
        cv::Mat(base_working),
        fixed_features.size(),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::Mat moving_valid(moving_features.size(), CV_8U, cv::Scalar(255));
    cv::Mat base_valid;
    cv::warpPerspective(
        moving_valid,
        base_valid,
        cv::Mat(base_working),
        fixed_features.size(),
        cv::INTER_NEAREST,
        cv::BORDER_CONSTANT,
        cv::Scalar(0)
    );
    cv::erode(
        base_valid,
        base_valid,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9))
    );
    const cv::Rect full_region(0, 0, fixed_features.cols, fixed_features.rows);

    std::array<cv::Mat, 2> parity_masks{
        cv::Mat::zeros(fixed_features.size(), CV_8U),
        cv::Mat::zeros(fixed_features.size(), CV_8U),
    };
    for (const auto& cell : cells) {
        if (!cell.quality_pass) continue;
        const cv::Rect region(
            static_cast<int>(std::floor(cell.fixed_x0)),
            static_cast<int>(std::floor(cell.fixed_y0)),
            std::max(1, static_cast<int>(std::ceil(cell.fixed_x1 - cell.fixed_x0))),
            std::max(1, static_cast<int>(std::ceil(cell.fixed_y1 - cell.fixed_y0)))
        );
        parity_masks[static_cast<size_t>((cell.grid_row + cell.grid_col) & 1)](
            region & full_region
        ).setTo(255);
    }
    for (auto& mask : parity_masks) cv::bitwise_and(mask, base_valid, mask);

    auto source_from_working = [&](const cv::Matx33d& working) {
        return inverse(fixed_scaled.source_to_scaled) * working *
               moving_scaled.source_to_scaled;
    };
    auto working_from_source = [&](const cv::Matx33d& source) {
        return fixed_scaled.source_to_scaled * source *
               inverse(moving_scaled.source_to_scaled);
    };
    auto score_working = [&](const cv::Matx33d& working, const cv::Mat& requested_mask) {
        cv::Mat warped;
        cv::warpPerspective(
            moving_features,
            warped,
            cv::Mat(working),
            fixed_features.size(),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0)
        );
        cv::Mat candidate_valid;
        cv::warpPerspective(
            moving_valid,
            candidate_valid,
            cv::Mat(working),
            fixed_features.size(),
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT,
            cv::Scalar(0)
        );
        cv::erode(
            candidate_valid,
            candidate_valid,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9))
        );
        cv::Mat common;
        cv::bitwise_and(base_valid, candidate_valid, common);
        if (!requested_mask.empty()) cv::bitwise_and(common, requested_mask, common);
        return masked_ncc(warped, fixed_features, common, full_region, 0, 0);
    };
    auto landmark_scores = [&](GlobalOptimizationCandidate& candidate) {
        std::vector<double> errors;
        double sum = 0.0;
        for (const auto& match : result.matches) {
            if (!match.inlier) continue;
            const cv::Point2d projected = project_point(
                candidate.moving_to_fixed,
                cv::Point2d(match.moving_x, match.moving_y)
            );
            const double error = cv::norm(projected - cv::Point2d(match.fixed_x, match.fixed_y));
            errors.push_back(error);
            sum += error;
        }
        candidate.landmark_median_error = median(errors);
        candidate.landmark_mean_error = errors.empty() ? 0.0 : sum / errors.size();
        candidate.transform_delta_rmse = compare_transforms(
            candidate.moving_to_fixed,
            result.moving_to_fixed,
            moving_bgr.cols,
            moving_bgr.rows
        ).root_mean_square_pixels;
    };
    auto evaluate_source_candidate = [&](GlobalOptimizationCandidate& candidate) {
        const cv::Matx33d working = working_from_source(candidate.moving_to_fixed);
        const NccEvaluation score = score_working(working, cv::Mat());
        candidate.ncc = std::isfinite(score.score) ? score.score : 0.0;
        double heldout_sum = 0.0;
        int heldout_weight = 0;
        for (const auto& mask : parity_masks) {
            const NccEvaluation heldout = score_working(working, mask);
            if (!std::isfinite(heldout.score) || heldout.pixels <= 0) continue;
            heldout_sum += heldout.score * heldout.pixels;
            heldout_weight += heldout.pixels;
        }
        candidate.heldout_ncc = heldout_weight ? heldout_sum / heldout_weight : 0.0;
        landmark_scores(candidate);
    };

    auto residual_from_ecc = [&](int motion_type, const cv::Mat& training_mask,
                                 double& objective) {
        cv::Mat template_to_input;
        if (motion_type == cv::MOTION_HOMOGRAPHY) {
            template_to_input = cv::Mat::eye(3, 3, CV_32F);
        } else {
            template_to_input = cv::Mat::eye(2, 3, CV_32F);
        }
        objective = cv::findTransformECC(
            fixed_features,
            base_warped,
            template_to_input,
            motion_type,
            cv::TermCriteria(
                cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                maximum_iterations,
                1e-6
            ),
            training_mask,
            5
        );
        cv::Mat template_to_input_3 = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat converted;
        template_to_input.convertTo(converted, CV_64F);
        converted.copyTo(
            template_to_input_3(cv::Rect(0, 0, converted.cols, converted.rows))
        );
        return inverse(mat_to_matx(template_to_input_3));
    };

    GlobalOptimizationEvaluation evaluation;
    GlobalOptimizationCandidate baseline;
    baseline.name = "baseline_ransac_homography";
    baseline.moving_to_fixed = result.moving_to_fixed;
    baseline.converged = true;
    evaluate_source_candidate(baseline);
    evaluation.candidates.push_back(baseline);

    GlobalOptimizationCandidate phase;
    phase.name = "global_phase_translation";
    phase.moving_to_fixed = cv::Matx33d(
        1.0, 0.0, result.phase_shift_x,
        0.0, 1.0, result.phase_shift_y,
        0.0, 0.0, 1.0
    ) * result.moving_to_fixed;
    phase.converged = true;
    evaluate_source_candidate(phase);
    evaluation.candidates.push_back(phase);

    const std::array<std::pair<const char*, int>, 4> modes{{
        {"ecc_translation", cv::MOTION_TRANSLATION},
        {"ecc_euclidean", cv::MOTION_EUCLIDEAN},
        {"ecc_affine", cv::MOTION_AFFINE},
        {"ecc_homography", cv::MOTION_HOMOGRAPHY},
    }};
    std::array<GlobalOptimizationCandidate, 4> optimized;
    cv::parallel_for_(cv::Range(0, static_cast<int>(modes.size())), [&](const cv::Range& range) {
        for (int mode_index = range.start; mode_index < range.end; ++mode_index) {
            GlobalOptimizationCandidate candidate;
            candidate.name = modes[static_cast<size_t>(mode_index)].first;
            try {
                const int motion_type = modes[static_cast<size_t>(mode_index)].second;
                double objective = 0.0;
                const cv::Matx33d residual = residual_from_ecc(
                    motion_type, base_valid, objective
                );
                candidate.ecc_objective = objective;
                candidate.moving_to_fixed = source_from_working(residual * base_working);
                candidate.converged = finite_transform(candidate.moving_to_fixed);
                if (!candidate.converged) {
                    candidate.failure_reason = "non_finite_or_singular_transform";
                } else {
                    const NccEvaluation score = score_working(
                        residual * base_working, cv::Mat()
                    );
                    candidate.ncc = std::isfinite(score.score) ? score.score : 0.0;
                    double heldout_sum = 0.0;
                    int heldout_weight = 0;
                    for (int training_parity = 0; training_parity < 2; ++training_parity) {
                        double fold_objective = 0.0;
                        const cv::Matx33d fold_residual = residual_from_ecc(
                            motion_type,
                            parity_masks[static_cast<size_t>(training_parity)],
                            fold_objective
                        );
                        const cv::Mat& heldout_mask =
                            parity_masks[static_cast<size_t>(1 - training_parity)];
                        const NccEvaluation heldout = score_working(
                            fold_residual * base_working, heldout_mask
                        );
                        if (!std::isfinite(heldout.score) || heldout.pixels <= 0) continue;
                        heldout_sum += heldout.score * heldout.pixels;
                        heldout_weight += heldout.pixels;
                    }
                    candidate.heldout_ncc =
                        heldout_weight ? heldout_sum / heldout_weight : 0.0;
                    landmark_scores(candidate);
                }
            } catch (const cv::Exception& error) {
                candidate.failure_reason = error.what();
            }
            optimized[static_cast<size_t>(mode_index)] = std::move(candidate);
        }
    });
    for (auto& candidate : optimized) evaluation.candidates.push_back(std::move(candidate));

    const double baseline_ncc = evaluation.candidates.front().ncc;
    const double baseline_heldout = evaluation.candidates.front().heldout_ncc;
    const double baseline_landmark_mean =
        evaluation.candidates.front().landmark_mean_error;
    const double baseline_landmark_median =
        evaluation.candidates.front().landmark_median_error;
    for (auto& candidate : evaluation.candidates) {
        candidate.passes_joint_gate = candidate.name != "baseline_ransac_homography" &&
            candidate.converged &&
            candidate.ncc > baseline_ncc + 0.001 &&
            candidate.heldout_ncc > baseline_heldout + 0.001 &&
            candidate.landmark_mean_error <= baseline_landmark_mean + 1e-12 &&
            candidate.landmark_median_error <= baseline_landmark_median + 1e-12 &&
            candidate.transform_delta_rmse <= 1.0;
    }
    return evaluation;
}

}  // namespace visium_hd::registration
