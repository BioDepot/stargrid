#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace visium_hd::registration {

struct Options {
    int max_features = 12000;
    int work_max_dimension = 4096;
    std::string transform_model = "similarity";
    double ratio_test = 0.70;
    double ransac_threshold_pixels = 1.0;
    int minimum_good_matches = 20;
    int minimum_inliers = 14;
    bool mutual_matching = false;
    std::string match_scale_policy = "smaller";
    bool apply_residual_refinement = false;
    double fixed_frame_offset_x_pixels = 0.5;
    double fixed_frame_offset_y_pixels = 0.0;
    bool deterministic = true;
    int threads = 1;
};

struct OrientationMetrics {
    std::string name;
    int moving_keypoints = 0;
    int fixed_keypoints = 0;
    int ratio_matches = 0;
    int inliers = 0;
    double inlier_fraction = 0.0;
    double median_inlier_error = 0.0;
    bool accepted = false;
    std::string rejection_reason;
};

struct MatchDiagnostic {
    double moving_x = 0.0;
    double moving_y = 0.0;
    double fixed_x = 0.0;
    double fixed_y = 0.0;
    double projected_x = 0.0;
    double projected_y = 0.0;
    double descriptor_distance = 0.0;
    double descriptor_ratio = 0.0;
    double reprojection_error = 0.0;
    bool inlier = false;
};

struct Result {
    cv::Matx33d moving_to_fixed = cv::Matx33d::eye();
    std::string orientation;
    int moving_width = 0;
    int moving_height = 0;
    int fixed_width = 0;
    int fixed_height = 0;
    int working_moving_width = 0;
    int working_moving_height = 0;
    int working_fixed_width = 0;
    int working_fixed_height = 0;
    int good_matches = 0;
    int inliers = 0;
    double inlier_fraction = 0.0;
    double median_inlier_error = 0.0;
    int overlap_pixels = 0;
    double phase_shift_x = 0.0;
    double phase_shift_y = 0.0;
    double phase_response = 0.0;
    int residual_shift_x = 0;
    int residual_shift_y = 0;
    double ncc_before_refinement = 0.0;
    double ncc_after_refinement = 0.0;
    bool residual_refinement_applied = false;
    std::vector<OrientationMetrics> candidates;
    std::vector<MatchDiagnostic> matches;
};

struct TransformError {
    int sampled_points = 0;
    double mean_pixels = 0.0;
    double root_mean_square_pixels = 0.0;
    double median_pixels = 0.0;
    double maximum_pixels = 0.0;
    double mean_delta_x_pixels = 0.0;
    double mean_delta_y_pixels = 0.0;
    double residual_after_mean_translation_rmse_pixels = 0.0;
};

struct AlignmentScore {
    int overlap_pixels = 0;
    double normalized_cross_correlation = 0.0;
};

struct SpatialResidualCell {
    int grid_row = 0;
    int grid_col = 0;
    double fixed_x0 = 0.0;
    double fixed_y0 = 0.0;
    double fixed_x1 = 0.0;
    double fixed_y1 = 0.0;
    int valid_pixels = 0;
    double valid_fraction = 0.0;
    double moving_standard_deviation = 0.0;
    double fixed_standard_deviation = 0.0;
    double ncc_before = 0.0;
    double ncc_after = 0.0;
    double phase_shift_x = 0.0;
    double phase_shift_y = 0.0;
    double phase_response = 0.0;
    int inlier_matches = 0;
    double landmark_mean_delta_x = 0.0;
    double landmark_mean_delta_y = 0.0;
    double landmark_median_error = 0.0;
    bool quality_pass = false;
};

struct SpatialCorrectionEvaluation {
    cv::Mat corrected_registered_bgr;
    cv::Mat displacement_x;
    cv::Mat displacement_y;
    int support_cells = 0;
    int heldout_cells = 0;
    int heldout_pixels = 0;
    double ncc_before = 0.0;
    double ncc_after = 0.0;
    double constant_ncc_after = 0.0;
    double heldout_ncc_before = 0.0;
    double heldout_ncc_after = 0.0;
    double heldout_constant_ncc_after = 0.0;
    double constant_shift_x = 0.0;
    double constant_shift_y = 0.0;
    double landmark_median_before = 0.0;
    double landmark_median_after = 0.0;
    double maximum_displacement = 0.0;
};

struct GlobalOptimizationCandidate {
    std::string name;
    cv::Matx33d moving_to_fixed = cv::Matx33d::eye();
    bool converged = false;
    std::string failure_reason;
    double ecc_objective = 0.0;
    double ncc = 0.0;
    double heldout_ncc = 0.0;
    double landmark_median_error = 0.0;
    double landmark_mean_error = 0.0;
    double transform_delta_rmse = 0.0;
    bool passes_joint_gate = false;
    bool reference_scored = false;
    double reference_rmse = 0.0;
    double reference_maximum_error = 0.0;
    double reference_mean_delta_x = 0.0;
    double reference_mean_delta_y = 0.0;
    double reference_centered_rmse = 0.0;
};

struct GlobalOptimizationEvaluation {
    std::vector<GlobalOptimizationCandidate> candidates;
};

cv::Matx33d pixel_center_resize_transform(
    int source_width,
    int source_height,
    int target_width,
    int target_height
);

Result register_images(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Options& options = Options{}
);

TransformError compare_transforms(
    const cv::Matx33d& estimated,
    const cv::Matx33d& reference,
    int moving_width,
    int moving_height,
    int grid_size = 9
);

AlignmentScore score_transform_alignment(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const cv::Matx33d& moving_to_fixed,
    int work_max_dimension = 4096
);

std::vector<SpatialResidualCell> audit_spatial_residual_grid(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    int grid_size = 6,
    int work_max_dimension = 4096,
    const cv::Rect2d& fixed_roi = cv::Rect2d()
);

SpatialCorrectionEvaluation evaluate_spatial_correction(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    const std::vector<SpatialResidualCell>& cells
);

GlobalOptimizationEvaluation evaluate_global_optimization(
    const cv::Mat& moving_bgr,
    const cv::Mat& fixed_bgr,
    const Result& result,
    const std::vector<SpatialResidualCell>& cells,
    int maximum_iterations = 60
);

cv::Point2d project_point(const cv::Matx33d& transform, const cv::Point2d& point);

}  // namespace visium_hd::registration
