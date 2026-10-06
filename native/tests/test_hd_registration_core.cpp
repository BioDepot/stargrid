#include "hd_registration_core.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

cv::Mat synthetic_histology() {
    cv::Mat image(700, 900, CV_8UC3, cv::Scalar(245, 242, 247));
    cv::RNG random(0x51A7);
    for (int index = 0; index < 180; ++index) {
        const cv::Point center(random.uniform(30, 870), random.uniform(30, 670));
        const cv::Size axes(random.uniform(5, 28), random.uniform(4, 22));
        const double angle = random.uniform(0.0, 180.0);
        const cv::Scalar fill(
            random.uniform(115, 205),
            random.uniform(45, 150),
            random.uniform(105, 225)
        );
        cv::ellipse(image, center, axes, angle, 0.0, 360.0, fill, cv::FILLED, cv::LINE_AA);
        if (index % 3 == 0) {
            cv::ellipse(
                image,
                center,
                cv::Size(std::max(2, axes.width / 3), std::max(2, axes.height / 3)),
                angle,
                0.0,
                360.0,
                cv::Scalar(235, 226, 238),
                cv::FILLED,
                cv::LINE_AA
            );
        }
    }
    cv::putText(
        image,
        "CRC",
        cv::Point(70, 120),
        cv::FONT_HERSHEY_SIMPLEX,
        2.1,
        cv::Scalar(90, 35, 130),
        5,
        cv::LINE_AA
    );
    cv::line(image, cv::Point(80, 610), cv::Point(760, 520), cv::Scalar(180, 80, 160), 9);
    return image;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const visium_hd::registration::Options defaults;
        require(
            defaults.transform_model == "similarity" &&
                defaults.ransac_threshold_pixels == 1.0 &&
                defaults.match_scale_policy == "smaller" &&
                defaults.fixed_frame_offset_x_pixels == 0.5 &&
                defaults.fixed_frame_offset_y_pixels == 0.0 &&
                !defaults.apply_residual_refinement,
            "Visium registration defaults lost the oracle-validated bounded model"
        );
        const cv::Matx33d roi_resize =
            visium_hd::registration::pixel_center_resize_transform(
                2176, 2176, 607, 607
            );
        const double roi_scale = 607.0 / 2176.0;
        require(
            std::abs(roi_resize(0, 0) - roi_scale) < 1e-15 &&
                std::abs(roi_resize(1, 1) - roi_scale) < 1e-15 &&
                std::abs(roi_resize(0, 2) - 0.5 * (roi_scale - 1.0)) < 1e-15 &&
                std::abs(roi_resize(1, 2) - 0.5 * (roi_scale - 1.0)) < 1e-15,
            "working-image resize lost the OpenCV pixel-center offset"
        );
        const cv::Point2d first_source_center =
            visium_hd::registration::project_point(roi_resize, cv::Point2d(0.0, 0.0));
        require(
            std::abs(first_source_center.x + 0.36052389705882354) < 1e-15 &&
                std::abs(first_source_center.y + 0.36052389705882354) < 1e-15,
            "ROI resize pixel-center regression has the wrong offset"
        );
        const cv::Mat fixed = synthetic_histology();
        const cv::Matx33d unequal_expected(
            0.0, -0.30, 250.0,
            -0.30, 0.0, 285.0,
            0.0, 0.0, 1.0
        );
        cv::Mat unequal_fixed;
        cv::warpPerspective(
            fixed,
            unequal_fixed,
            cv::Mat(unequal_expected),
            cv::Size(300, 300),
            cv::INTER_AREA,
            cv::BORDER_CONSTANT,
            cv::Scalar(250, 250, 250)
        );
        visium_hd::registration::Options unequal_options;
        unequal_options.max_features = 8000;
        unequal_options.work_max_dimension = 300;
        unequal_options.minimum_good_matches = 20;
        unequal_options.minimum_inliers = 14;
        unequal_options.transform_model = "similarity";
        unequal_options.ransac_threshold_pixels = 1.5;
        unequal_options.fixed_frame_offset_x_pixels = 0.0;
        const auto unequal_result = visium_hd::registration::register_images(
            fixed, unequal_fixed, unequal_options
        );
        const auto unequal_error = visium_hd::registration::compare_transforms(
            unequal_result.moving_to_fixed,
            unequal_expected,
            fixed.cols,
            fixed.rows
        );
        require(
            unequal_error.root_mean_square_pixels < 0.20,
            "unequal-resolution pixel-center registration RMSE exceeded 0.20 pixels"
        );
        const cv::Matx33d expected(
            -0.980, -0.030, 880.0,
            -0.025, 0.985, 29.0,
            0.0, 0.0, 1.0
        );
        cv::Mat moving;
        cv::warpPerspective(
            fixed,
            moving,
            cv::Mat(expected),
            fixed.size(),
            cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
            cv::BORDER_CONSTANT,
            cv::Scalar(250, 250, 250)
        );

        visium_hd::registration::Options options;
        options.max_features = 8000;
        options.work_max_dimension = 1600;
        options.minimum_good_matches = 30;
        options.minimum_inliers = 24;
        options.transform_model = "homography";
        options.ransac_threshold_pixels = 3.0;
        options.fixed_frame_offset_x_pixels = 0.0;
        const auto first = visium_hd::registration::register_images(moving, fixed, options);
        const auto error = visium_hd::registration::compare_transforms(
            first.moving_to_fixed,
            expected,
            moving.cols,
            moving.rows
        );
        const double affine_determinant =
            first.moving_to_fixed(0, 0) * first.moving_to_fixed(1, 1) -
            first.moving_to_fixed(0, 1) * first.moving_to_fixed(1, 0);
        require(affine_determinant < 0.0, "synthetic reflected transform was not recovered");
        require(first.inliers >= 100, "too few synthetic inliers");
        require(error.root_mean_square_pixels < 1.5, "synthetic transform RMSE exceeded 1.5 pixels");
        const cv::Matx33d translated = cv::Matx33d(
            1.0, 0.0, 2.5,
            0.0, 1.0, -1.25,
            0.0, 0.0, 1.0
        ) * expected;
        const auto translation_error = visium_hd::registration::compare_transforms(
            translated, expected, moving.cols, moving.rows
        );
        require(
            std::abs(translation_error.mean_delta_x_pixels - 2.5) < 1e-10 &&
                std::abs(translation_error.mean_delta_y_pixels + 1.25) < 1e-10,
            "transform comparison did not preserve a coherent translation"
        );
        require(
            translation_error.residual_after_mean_translation_rmse_pixels < 1e-10,
            "pure translation left a nonzero centered residual"
        );
        const auto expected_score = visium_hd::registration::score_transform_alignment(
            moving, fixed, expected, 1600
        );
        const auto shifted_score = visium_hd::registration::score_transform_alignment(
            moving, fixed, translated, 1600
        );
        require(expected_score.overlap_pixels > 100000, "too little synthetic scoring overlap");
        require(
            expected_score.normalized_cross_correlation > shifted_score.normalized_cross_correlation,
            "alignment score did not prefer the generating transform"
        );
        const cv::Rect2d audit_roi(120.0, 120.0, 600.0, 420.0);
        const auto spatial_first = visium_hd::registration::audit_spatial_residual_grid(
            moving, fixed, first, 4, 1600, audit_roi
        );
        const auto spatial_second = visium_hd::registration::audit_spatial_residual_grid(
            moving, fixed, first, 4, 1600, audit_roi
        );
        require(spatial_first.size() == 16, "synthetic spatial audit did not emit a 4x4 grid");
        require(
            spatial_first.size() == spatial_second.size(),
            "synthetic spatial audit changed cell count"
        );
        int spatial_quality_cells = 0;
        for (size_t index = 0; index < spatial_first.size(); ++index) {
            const auto& left = spatial_first[index];
            const auto& right = spatial_second[index];
            require(
                left.grid_row == right.grid_row && left.grid_col == right.grid_col &&
                    left.phase_shift_x == right.phase_shift_x &&
                    left.phase_shift_y == right.phase_shift_y &&
                    left.phase_response == right.phase_response,
                "synthetic spatial audit was not deterministic"
            );
            require(
                std::isfinite(left.phase_shift_x) && std::isfinite(left.phase_shift_y) &&
                    std::isfinite(left.ncc_before) && std::isfinite(left.ncc_after),
                "synthetic spatial audit emitted a non-finite value"
            );
            if (left.quality_pass) ++spatial_quality_cells;
        }
        require(spatial_quality_cells >= 8, "too few supported synthetic spatial audit cells");
        const auto global_optimization =
            visium_hd::registration::evaluate_global_optimization(
                moving, fixed, first, spatial_first, 10
            );
        require(
            global_optimization.candidates.size() == 6,
            "global ROI optimization did not emit all declared candidates"
        );
        require(
            global_optimization.candidates.front().name == "baseline_ransac_homography" &&
                global_optimization.candidates.front().converged,
            "global ROI optimization lost its baseline candidate"
        );
        require(
            std::isfinite(global_optimization.candidates.front().ncc) &&
                std::isfinite(global_optimization.candidates.front().heldout_ncc),
            "global ROI optimization emitted non-finite baseline scores"
        );
        require(
            !global_optimization.candidates.front().reference_scored,
            "core global optimization unexpectedly opened a reference oracle"
        );
        for (const auto& candidate : global_optimization.candidates) {
            if (!candidate.passes_joint_gate) continue;
            require(
                    candidate.ncc > global_optimization.candidates.front().ncc + 0.001 &&
                    candidate.heldout_ncc >
                        global_optimization.candidates.front().heldout_ncc + 0.001 &&
                    candidate.landmark_mean_error <=
                        global_optimization.candidates.front().landmark_mean_error + 1e-12 &&
                    candidate.landmark_median_error <=
                        global_optimization.candidates.front().landmark_median_error + 1e-12 &&
                    candidate.transform_delta_rmse <= 1.0,
                "global ROI candidate bypassed the declared joint gate"
            );
        }
        auto deliberately_shifted = first;
        deliberately_shifted.moving_to_fixed = cv::Matx33d(
            1.0, 0.0, 3.0,
            0.0, 1.0, -2.0,
            0.0, 0.0, 1.0
        ) * first.moving_to_fixed;
        const auto shifted_spatial = visium_hd::registration::audit_spatial_residual_grid(
            moving, fixed, deliberately_shifted, 4, 1600, audit_roi
        );
        cv::Point2d mean_recovery(0.0, 0.0);
        int recovery_cells = 0;
        for (const auto& cell : shifted_spatial) {
            if (!cell.quality_pass) continue;
            mean_recovery.x += cell.phase_shift_x;
            mean_recovery.y += cell.phase_shift_y;
            ++recovery_cells;
        }
        require(recovery_cells >= 8, "too few supported shifted spatial audit cells");
        mean_recovery *= 1.0 / recovery_cells;
        require(
            mean_recovery.x < -1.5 && mean_recovery.y > 0.75,
            "spatial audit did not recover the direction of a controlled displacement"
        );
        require(
            first.matches.size() == static_cast<size_t>(first.good_matches),
            "match diagnostic count does not equal ratio-match count"
        );
        int diagnostic_inliers = 0;
        for (const auto& match : first.matches) {
            if (match.inlier) ++diagnostic_inliers;
        }
        require(
            diagnostic_inliers == first.inliers,
            "match diagnostic inlier count does not equal fit inlier count"
        );

        auto offset_options = options;
        offset_options.fixed_frame_offset_x_pixels = 0.5;
        offset_options.fixed_frame_offset_y_pixels = -0.25;
        const auto offset_result =
            visium_hd::registration::register_images(moving, fixed, offset_options);
        const cv::Matx33d expected_offset(
            1.0, 0.0, 0.5,
            0.0, 1.0, -0.25,
            0.0, 0.0, 1.0
        );
        const cv::Matx33d expected_offset_transform =
            expected_offset * first.moving_to_fixed;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                require(
                    offset_result.moving_to_fixed(row, col) ==
                        expected_offset_transform(row, col),
                    "fixed-frame offset was not composed after source scaling"
                );
            }
        }

        const auto second = visium_hd::registration::register_images(moving, fixed, options);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                require(
                    first.moving_to_fixed(row, col) == second.moving_to_fixed(row, col),
                    "registration was not bitwise deterministic"
                );
            }
        }
        std::cout << "orientation=" << first.orientation
                  << " inliers=" << first.inliers
                  << " rmse=" << error.root_mean_square_pixels
                  << " unequal_rmse=" << unequal_error.root_mean_square_pixels << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_hd_registration_core: " << error.what() << '\n';
        return 1;
    }
}
