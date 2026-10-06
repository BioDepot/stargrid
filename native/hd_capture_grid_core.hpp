#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visium_hd::capture_grid {

struct Layout {
    std::string slide_uid;
    std::string file_format;
    std::string aligner_version;
    std::string input_hash;
    std::string slide_design;
    std::string area;
    cv::Matx33d design_correction = cv::Matx33d::eye();
};

struct DetectionOptions {
    double hough_dp = 1.0;
    double hough_min_distance = 55.0;
    double hough_edge_threshold = 110.0;
    double hough_accumulator_threshold = 32.0;
    int minimum_radius = 22;
    int maximum_radius = 48;
    // Hough centers are quantized at the seed stage. A one-pixel error over a
    // single approximately 89-pixel adjacent pair can accumulate across the
    // complete fiducial frame, so allow ten pixels only for initialization and
    // retain the tight three-pixel gate for the refined homography.
    double initial_match_tolerance = 10.0;
    double refined_match_tolerance = 3.0;
    int minimum_matched_fiducials = 60;
    int threads = 1;
};

struct FiducialMatch {
    int design_index = -1;
    cv::Point2d design_xy;
    cv::Point2d detected_xy;
    cv::Point2d projected_xy;
    double residual_pixels = 0.0;
};

struct Result {
    cv::Matx33d design_to_cytassist = cv::Matx33d::eye();
    cv::Matx33d spot_colrow_to_cytassist = cv::Matx33d::eye();
    int detected_circles = 0;
    int matched_fiducials = 0;
    int inlier_fiducials = 0;
    double median_residual_pixels = 0.0;
    double root_mean_square_residual_pixels = 0.0;
    double maximum_residual_pixels = 0.0;
    std::vector<cv::Vec3f> circles;
    std::vector<FiducialMatch> matches;
};

std::vector<cv::Point2d> visium_hd_v1_fiducial_centers();

Layout read_vlf_layout(const std::string& path, const std::string& area);

std::vector<cv::Vec3f> detect_circular_fiducials(
    const cv::Mat& cytassist_bgr,
    const DetectionOptions& options = DetectionOptions{}
);

Result fit_capture_grid(
    const std::vector<cv::Vec3f>& detected_circles,
    const Layout& layout,
    const DetectionOptions& options = DetectionOptions{}
);

Result localize_capture_grid(
    const cv::Mat& cytassist_bgr,
    const Layout& layout,
    const DetectionOptions& options = DetectionOptions{}
);

cv::Point2d project_point(const cv::Matx33d& transform, const cv::Point2d& point);

}  // namespace visium_hd::capture_grid
