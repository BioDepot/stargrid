#include "hd_capture_grid_core.hpp"

#include <opencv2/imgproc.hpp>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace visium_hd::capture_grid;

int main() {
    const auto design = visium_hd_v1_fiducial_centers();
    assert(design.size() == 73);
    assert(design.front() == cv::Point2d(-135.0, -545.0));
    assert(design[36] == cv::Point2d(7245.0, 7245.0));

    const cv::Matx33d expected(
        0.2171, -0.0007, 338.5,
        0.00065, 0.2172, 840.3,
        3.0e-9, 2.0e-8, 1.0
    );
    cv::Mat synthetic(3000, 3200, CV_8UC3, cv::Scalar(242, 239, 238));
    for (const cv::Point2d& point : design) {
        const cv::Point2d center = project_point(expected, point);
        cv::circle(synthetic, center, 35, cv::Scalar(25, 25, 25), 3, cv::LINE_AA);
        cv::circle(synthetic, center, 24, cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
        cv::circle(synthetic, center, 12, cv::Scalar(35, 35, 35), 3, cv::LINE_AA);
    }
    DetectionOptions options;
    options.hough_accumulator_threshold = 28.0;
    const auto circles = detect_circular_fiducials(synthetic, options);
    assert(circles.size() >= 70);
    Layout layout;
    layout.slide_design = "visium_hd_rc1";
    const Result fit = fit_capture_grid(circles, layout, options);
    assert(fit.matched_fiducials >= 70);
    double squared = 0.0;
    for (const cv::Point2d& point : design) {
        const double error = cv::norm(
            project_point(fit.design_to_cytassist, point)
            - project_point(expected, point) - cv::Point2d(0.5, 0.5)
        );
        squared += error * error;
    }
    const double rmse = std::sqrt(squared / design.size());
    assert(rmse < 0.75);
    const cv::Point2d spot = project_point(fit.spot_colrow_to_cytassist, {0.0, 0.0});
    const cv::Point2d expected_spot =
        project_point(expected, {1.0, 1.0}) + cv::Point2d(0.5, 0.5);
    assert(cv::norm(spot - expected_spot) < 0.75);

    cv::Mat incomplete(3000, 3200, CV_8UC3, cv::Scalar(242, 239, 238));
    for (size_t index = 0; index < 65; ++index) {
        const cv::Point2d center = project_point(expected, design[index]);
        cv::circle(incomplete, center, 35, cv::Scalar(25, 25, 25), 3, cv::LINE_AA);
        cv::circle(incomplete, center, 24, cv::Scalar(30, 30, 30), 3, cv::LINE_AA);
        cv::circle(incomplete, center, 12, cv::Scalar(35, 35, 35), 3, cv::LINE_AA);
    }
    const Result incomplete_fit = localize_capture_grid(incomplete, layout, options);
    assert(incomplete_fit.matched_fiducials >= options.minimum_matched_fiducials);
    assert(incomplete_fit.circles.size() < design.size());

    // A real CytAssist frame can combine half-pixel Hough-center quantization
    // with mild perspective. Seeding from one adjacent pair then needs a
    // broader correspondence radius than the final homography fit. Four
    // corner-like distractors ensure the broad seed does not enter the result.
    const cv::Matx33d quantized_expected(
        0.21623664220455119, -0.0013941794474318166, 312.62737816115373,
        0.0014392775277962712, 0.21635252395437296, 767.1149018722399,
        -2.7100973184045807e-8, 4.5748637710594702e-8, 1.0
    );
    std::vector<cv::Vec3f> quantized;
    quantized.reserve(design.size() + 4);
    for (const cv::Point2d& point : design) {
        const cv::Point2d center = project_point(quantized_expected, point);
        quantized.emplace_back(
            static_cast<float>(std::floor(center.x) + 0.5),
            static_cast<float>(std::floor(center.y) + 0.5), 35.0F
        );
    }
    quantized.emplace_back(248.5F, 702.5F, 23.0F);
    quantized.emplace_back(1826.5F, 712.5F, 23.0F);
    quantized.emplace_back(238.5F, 2280.5F, 23.0F);
    quantized.emplace_back(1816.5F, 2291.5F, 23.0F);
    std::sort(quantized.begin(), quantized.end(), [](const cv::Vec3f& left, const cv::Vec3f& right) {
        if (left[1] != right[1]) return left[1] < right[1];
        return left[0] < right[0];
    });
    const Result quantized_fit = fit_capture_grid(quantized, layout, DetectionOptions{});
    assert(quantized_fit.matched_fiducials >= 70);
    double quantized_squared = 0.0;
    for (const cv::Point2d& point : design) {
        const double error = cv::norm(
            project_point(quantized_fit.design_to_cytassist, point)
            - project_point(quantized_expected, point) - cv::Point2d(0.5, 0.5)
        );
        quantized_squared += error * error;
    }
    assert(std::sqrt(quantized_squared / design.size()) < 0.75);

    const std::filesystem::path vlf =
        std::filesystem::temp_directory_path() / "visium_hd_capture_grid_test.vlf";
    {
        std::ofstream output(vlf);
        output << R"({"slide_uid":"TEST-SLIDE","file_format":"1.0","aligner_version":"1.0","input_hash":"abc","slide_design":"visium_hd_rc1","capture_areas":{"A1":[1,0,3,0,1,-4,0,0,1]}})";
    }
    const Layout parsed = read_vlf_layout(vlf.string(), "A1");
    std::filesystem::remove(vlf);
    assert(parsed.slide_uid == "TEST-SLIDE");
    assert(parsed.area == "A1");
    assert(parsed.design_correction(0, 2) == 3.0);
    assert(parsed.design_correction(1, 2) == -4.0);
    std::cout << "detected=" << circles.size() << " matched=" << fit.matched_fiducials
              << " transform_rmse=" << rmse << '\n';
    return 0;
}
