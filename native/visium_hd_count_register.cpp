// visium_hd_count_register: the registration check and correction of STARgrid from the counts.
//
//   check    the correlation-ratio check of a nucleus map against 2 um counts (the whole slide and
//            5 x 5 regions), as `stargrid-register check`
//   project  place the Cellpose nuclei on the capture grid through the native image registration and
//            the capture-grid transform (the step before the correction, on the segmentation branch)
//   correct  the pipeline step after STAR: check, fit and apply the correction to the nucleus map,
//            check again, and write the registered positions of the 2, 8 and 16 um bins

#include "hd_count_registration.hpp"
#include "hd_json.hpp"
#include "hd_mex.hpp"
#include "hd_npy.hpp"
#include "hd_parallel.hpp"
#include "hd_parquet.hpp"
#include "hd_projection.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace cr = visium_hd::count_registration;
using visium_hd::json::Value;

namespace {

constexpr const char* kTool = "visium_hd_count_register";

const char* const kUsage = R"(usage:
  visium_hd_count_register check --counts PATH --tissue PATH --nuclei PATH --out FILE
      [--nuclei-key NAME] [--width 3350] [--height WIDTH] [--search 12] [--draws 200] [--seed 20261007]
      [--divisions 5] [--baseline 2] [--surface FILE.npy] [--threads N]
      [--areas FILE.json]   (named areas [{"name", "bounds": [y0, y1, x0, x1]}] instead of the
                             whole slide and its regions; --surface then writes one .npz)
  visium_hd_count_register project --segmentation DIR --registration FILE --capture FILE --out DIR
      [--width 3350] [--threads N]
  visium_hd_count_register correct --mex DIR --projection DIR --segmentation DIR --out DIR
      [--width 3350] [--threads N]

check:   --counts is a Matrix Market directory (barcodes.tsv, matrix.mtx), or .npz (key "total") or
         .npy; --tissue is .npz (key "tissue") or .npy; --nuclei is .npz or .npy (nonzero = nucleus).
project: --segmentation holds summary.json and masks/{nucleus,cell}_labels.zarr (zarr v3, zstd);
         --registration is registration.json, --capture capture_grid.json. --out must not exist.
correct: --mex is STAR's SpatialGex.out or SpatialFlex.out (hard/square_002um, _008um, _016um); --projection holds
         nuclei.npz, support.npz and projection.json; --segmentation holds summary.json. --out must
         not exist.
)";

std::string utc_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
    const long micros = static_cast<long>(
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1000000);
    std::tm parts{};
    gmtime_r(&seconds, &parts);
    char text[64];
    std::snprintf(text, sizeof(text), "%04d-%02d-%02dT%02d:%02d:%02d.%06ld+00:00", parts.tm_year + 1900,
                  parts.tm_mon + 1, parts.tm_mday, parts.tm_hour, parts.tm_min, parts.tm_sec, micros);
    return text;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct Arguments {
    std::string step;
    std::map<std::string, std::string> values;
    std::vector<std::string> argv;

    bool has(const std::string& name) const { return values.count(name) > 0; }
    std::string get(const std::string& name) const {
        const auto found = values.find(name);
        if (found == values.end()) throw std::runtime_error("missing --" + name);
        return found->second;
    }
    std::string get(const std::string& name, const std::string& fallback) const { return has(name) ? get(name) : fallback; }
    long long number(const std::string& name, long long fallback) const {
        if (!has(name)) return fallback;
        std::size_t used = 0;
        const long long value = std::stoll(get(name), &used);
        if (used != get(name).size()) throw std::runtime_error("invalid --" + name + ": " + get(name));
        return value;
    }
};

Arguments parse_args(int argc, char** argv) {
    Arguments args;
    for (int i = 0; i < argc; ++i) args.argv.emplace_back(argv[i]);
    if (argc < 2) throw std::runtime_error("missing step");
    args.step = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option.rfind("--", 0) != 0 || i + 1 >= argc) throw std::runtime_error("bad argument: " + option);
        args.values[option.substr(2)] = argv[++i];
    }
    return args;
}

int threads_of(const Arguments& args) {
    const long long fallback = std::max(1u, std::thread::hardware_concurrency());
    const long long n = args.number("threads", fallback);
    if (n < 1 || n > 4096) throw std::runtime_error("--threads must be between 1 and 4096");
    return static_cast<int>(n);
}

// ---------------------------------------------------------------------------------------------
// Inputs on the 2 um grid.

std::vector<double> load_counts(const std::string& path, cr::Grid grid, int threads) {
    if (fs::is_directory(path)) {
        const fs::path dir(path);
        if (!fs::exists(dir / "barcodes.tsv") || !fs::exists(dir / "matrix.mtx")) {
            throw std::runtime_error("a count directory needs uncompressed barcodes.tsv and matrix.mtx: " + path);
        }
        const visium_hd::mex::Barcodes barcodes =
            visium_hd::mex::read_barcodes((dir / "barcodes.tsv").string(), 2, grid.width, true, threads);
        const std::vector<double> totals = visium_hd::mex::column_totals((dir / "matrix.mtx").string(), threads);
        if (totals.size() != barcodes.size()) throw std::runtime_error("matrix columns and barcodes differ in number");
        std::vector<double> counts(grid.size(), 0.0);
        for (std::size_t i = 0; i < barcodes.size(); ++i) {
            counts[static_cast<std::size_t>(barcodes.row[i]) * grid.width + barcodes.col[i]] = totals[i];
        }
        return counts;
    }
    const visium_hd::npy::Array array =
        visium_hd::npy::read(path, fs::path(path).extension() == ".npz" ? "total" : "");
    if (array.count() != grid.size()) throw std::runtime_error("counts do not match the grid: " + path);
    return visium_hd::npy::as_doubles(array);
}

std::vector<std::uint8_t> load_flags(const std::string& path, const std::string& key, cr::Grid grid) {
    visium_hd::npy::Array array;
    if (fs::path(path).extension() == ".npz" && key.empty()) {
        try {
            array = visium_hd::npy::read(path, "labels");
        } catch (const std::exception&) {
            array = visium_hd::npy::read(path, "");
        }
    } else {
        array = visium_hd::npy::read(path, key);
    }
    if (array.count() != grid.size()) throw std::runtime_error("grid array does not match the grid: " + path);
    return visium_hd::npy::as_flags(array);
}

// ---------------------------------------------------------------------------------------------
// JSON in the form stargrid-register writes.

Value integer_or_null(const std::optional<int>& value, int scale = 1) {
    return value ? Value(*value * scale) : Value();
}

void set_assessment(Value& out, const cr::Assessment& a) {
    out.set("status", a.status);
    out.set("usable", a.usable);
    out.set("method", "correlation ratio");
    out.set("dy", integer_or_null(a.dy));
    out.set("dx", integer_or_null(a.dx));
    out.set("dy_um", integer_or_null(a.dy, 2));
    out.set("dx_um", integer_or_null(a.dx, 2));
    out.set("eta2", Value(a.eta2));
    out.set("eta2_at_zero", Value(a.eta2_at_zero));
    out.set("interior", Value(a.interior));
    out.set("share_any_peak", Value(a.share_any_peak));
    out.set("share_within_one", Value(a.share_within_one));
    out.set("share_exact", Value(a.share_exact));
    out.set("candidate_dy", Value(a.candidate_dy));
    out.set("candidate_dx", Value(a.candidate_dx));
    if (a.has_profile) {
        Value profile = Value::object();
        for (int j = 0; j <= cr::kBands; ++j) profile.set(cr::kBandNames[j], Value(a.profile[j]));
        out.set("profile", profile);
    } else {
        out.set("profile", Value());
    }
    out.set("baseline", a.baseline);
    out.set("blocks", a.blocks);
    out.set("draws", a.draws);
    out.set("seed", static_cast<unsigned long long>(a.seed));
    out.set("search", a.search);
    out.set("reason", a.reason);
}

Value assessment_json(const cr::Assessment& a) {
    Value out = Value::object();
    set_assessment(out, a);
    return out;
}

Value offset_um(const cr::Assessment& a) {
    return a.usable ? Value(2.0 * std::hypot(static_cast<double>(*a.dy), static_cast<double>(*a.dx))) : Value();
}

Value regions_json(const std::vector<cr::Region>& regions) {
    Value out = Value::array();
    for (const cr::Region& r : regions) {
        Value row = Value::object();
        row.set("region", r.name);
        row.set("row", r.row);
        row.set("col", r.col);
        row.set("nucleus_squares_in_tissue", static_cast<long long>(r.nucleus_locations));
        row.set("usable", false);
        row.set("centroid_y", Value(r.centroid_y));
        row.set("centroid_x", Value(r.centroid_x));
        if (r.tested) {
            set_assessment(row, r.assessment);
        } else {
            row.set("status", "not tested");
            row.set("reason", r.reason);
        }
        out.push(row);
    }
    return out;
}

// The check of the whole slide and its regions, as the pipeline records it.
Value check_json(const cr::Check& check) {
    Value whole = assessment_json(check.whole);
    whole.set("offset_um", offset_um(check.whole));
    Value out = Value::object();
    out.set("whole", whole);
    out.set("regions", regions_json(check.regions));
    out.set("usable_regions", check.usable_regions);
    out.set("regions_at_zero", check.regions_at_zero);
    out.set("nucleus_locations_in_support", static_cast<long long>(check.nucleus_locations_in_support));
    return out;
}

Value coefficients_json(const std::array<std::array<double, 2>, 3>& c) {
    Value out = Value::array();
    for (const auto& row : c) {
        Value r = Value::array();
        r.push(row[0]);
        r.push(row[1]);
        out.push(r);
    }
    return out;
}

Value pair_json(const std::array<double, 2>& p) {
    Value out = Value::array();
    out.push(p[0]);
    out.push(p[1]);
    return out;
}

Value linear_fit_json(const cr::LinearFit& fit) {
    Value out = Value::object();
    out.set("coefficients", coefficients_json(fit.coefficients));
    out.set("first_coefficients", coefficients_json(fit.first_coefficients));
    Value residuals = Value::array();
    for (const auto& r : fit.first_residuals) residuals.push(pair_json(r));
    out.set("first_residuals", residuals);
    out.set("retained", Value::array_of(fit.retained));
    out.set("rejected", Value::array_of(fit.rejected));
    return out;
}

Value model_json(const cr::AffineModel& m) {
    Value out = Value::object();
    out.set("usable_regions", m.usable_regions);
    out.set("training_regions", m.training_regions);
    out.set("held_out_total", m.held_out_total);
    out.set("held_out_pass", m.held_out_pass);
    out.set("accepted", m.accepted);
    out.set("model", "[dy,dx] = [1,source_row,source_col] @ coefficients");
    Value held = Value::array();
    for (const cr::HeldOut& h : m.held_out) {
        Value row = Value::object();
        row.set("region", h.region);
        row.set("prediction", pair_json(h.prediction));
        row.set("residual", pair_json(h.residual));
        row.set("within_one_square", h.within_one_square);
        held.push(row);
    }
    out.set("held_out", held);
    out.set("final_fit", m.final_fit ? linear_fit_json(*m.final_fit) : Value());
    if (m.training_fit) out.set("training_fit", linear_fit_json(*m.training_fit));
    if (m.has_held_out_fraction) out.set("held_out_fraction", Value(m.held_out_fraction));
    if (m.stop_reason) out.set("stop_reason", *m.stop_reason);
    return out;
}

Value matrix_json(const cr::Matrix3& m) {
    Value out = Value::array();
    for (const auto& row : m) {
        Value r = Value::array();
        for (double v : row) r.push(v);
        out.push(r);
    }
    return out;
}

cr::Matrix3 matrix_from(const Value& value) {
    cr::Matrix3 m{};
    if (value.size() != 3) throw std::runtime_error("expected a 3 x 3 matrix");
    for (int i = 0; i < 3; ++i) {
        if (value[i].size() != 3) throw std::runtime_error("expected a 3 x 3 matrix");
        for (int j = 0; j < 3; ++j) m[i][j] = value[i][j].as_double();
    }
    return m;
}

cr::Options options_of(const Arguments& args) {
    cr::Options o;
    o.search = static_cast<int>(args.number("search", o.search));
    o.draws = static_cast<int>(args.number("draws", o.draws));
    o.seed = static_cast<std::uint64_t>(args.number("seed", static_cast<long long>(o.seed)));
    o.divisions = static_cast<int>(args.number("divisions", o.divisions));
    o.baseline = static_cast<int>(args.number("baseline", o.baseline));
    o.threads = threads_of(args);
    return o;
}

std::string method_name() {
    return std::string(kTool) + " (C++), stargrid-register " + cr::kMethodVersion;
}

// ---------------------------------------------------------------------------------------------

int run_check(const Arguments& args) {
    const int width = static_cast<int>(args.number("width", 3350));
    const cr::Grid grid{static_cast<int>(args.number("height", width)), width};
    const cr::Options options = options_of(args);
    const std::vector<double> counts = load_counts(args.get("counts"), grid, options.threads);
    const std::vector<std::uint8_t> tissue = load_flags(
        args.get("tissue"), fs::path(args.get("tissue")).extension() == ".npz" ? "tissue" : "", grid);
    const std::vector<std::uint8_t> nuclei = load_flags(args.get("nuclei"), args.get("nuclei-key", ""), grid);
    if (args.has("areas")) {
        // Named areas instead of the whole slide and its regions: [{"name": ..., "bounds": [y0, y1, x0, x1]}].
        const Value listed = visium_hd::json::read_file(args.get("areas"));
        std::vector<cr::Bounds> areas;
        for (std::size_t i = 0; i < listed.size(); ++i) {
            const Value& b = listed[i].at("bounds");
            areas.push_back(cr::Bounds{static_cast<int>(b[0].as_int()), static_cast<int>(b[1].as_int()),
                                       static_cast<int>(b[2].as_int()), static_cast<int>(b[3].as_int())});
        }
        const std::vector<cr::Assessment> assessed = cr::assess_areas(nuclei, counts, tissue, grid, areas, options);
        Value rows = Value::array();
        std::vector<visium_hd::npy::Member> surfaces;
        const std::size_t size = static_cast<std::size_t>(2 * options.search + 3);
        for (std::size_t i = 0; i < assessed.size(); ++i) {
            Value row = Value::object();
            row.set("name", listed[i].at("name").as_string());
            row.set("bounds", listed[i].at("bounds"));
            set_assessment(row, assessed[i]);
            rows.push(row);
            surfaces.push_back({listed[i].at("name").as_string(), "<f8", {size, size}, assessed[i].surface.data(),
                                assessed[i].surface.size() * sizeof(double)});
        }
        Value out = Value::object();
        out.set("tool", method_name());
        out.set("step", "check");
        out.set("baseline", options.baseline);
        out.set("areas", rows);
        visium_hd::json::write_file(args.get("out"), out);
        if (args.has("surface")) visium_hd::npy::write_npz(args.get("surface"), surfaces);
        return 0;
    }
    const cr::Check check = cr::check(nuclei, counts, tissue, grid, options);
    Value whole = assessment_json(check.whole);
    whole.set("nucleus_locations_in_tissue", static_cast<long long>(check.nucleus_locations_in_support));
    whole.set("offset_um", offset_um(check.whole));
    Value out = Value::object();
    out.set("tool", method_name());
    out.set("step", "check");
    out.set("baseline", options.baseline);
    out.set("whole", whole);
    if (options.divisions) {
        out.set("regions", regions_json(check.regions));
        out.set("usable_regions", check.usable_regions);
        out.set("regions_at_zero", check.regions_at_zero);
    }
    visium_hd::json::write_file(args.get("out"), out);
    if (args.has("surface")) {
        const int size = 2 * options.search + 3;
        visium_hd::npy::write_npy(args.get("surface"), {"surface", "<f8", {static_cast<std::size_t>(size), static_cast<std::size_t>(size)},
                                                         check.whole.surface.data(), check.whole.surface.size() * sizeof(double)});
    }
    std::cout << "{\"status\": \"" << check.whole.status << "\", \"reason\": \"" << check.whole.reason << "\"}\n";
    return 0;
}

// The registered position of every bin of one size, in the emitted MEX column order.
Value write_positions(const fs::path& barcodes_path, const fs::path& output, int bin_um, const cr::Matrix3& h,
                      std::array<double, 2> image_shape, int width, std::uint64_t expected_columns, int threads) {
    const double factor = bin_um / 2.0;
    const int side = static_cast<int>(std::ceil(width / factor));
    if (expected_columns > static_cast<std::uint64_t>(side) * static_cast<std::uint64_t>(side)) {
        throw std::runtime_error("more matrix columns than bins");
    }
    const visium_hd::mex::Barcodes barcodes = visium_hd::mex::read_barcodes(barcodes_path.string(), bin_um, side, false, threads);
    if (barcodes.size() != expected_columns) {
        throw std::runtime_error("barcodes and matrix columns differ in number: " + barcodes_path.string());
    }
    const std::size_t rows = barcodes.size();
    std::vector<double> pixel_row(rows), pixel_col(rows);
    std::vector<std::uint8_t> inside(rows);
    const std::size_t chunk = 1 << 16;
    visium_hd::parallel_for((rows + chunk - 1) / chunk, threads, [&](std::size_t k) {
        for (std::size_t i = k * chunk; i < std::min(rows, (k + 1) * chunk); ++i) {
            const double gx = factor * barcodes.col[i] + (factor - 1) / 2;
            const double gy = factor * barcodes.row[i] + (factor - 1) / 2;
            const double m0 = gx * h[0][0] + gy * h[0][1] + h[0][2];
            const double m1 = gx * h[1][0] + gy * h[1][1] + h[1][2];
            const double m2 = gx * h[2][0] + gy * h[2][1] + h[2][2];
            if (!std::isfinite(m0) || !std::isfinite(m1) || !std::isfinite(m2) || !(std::abs(m2) > 1e-12)) {
                throw std::runtime_error("a bin centre maps to infinity");
            }
            const double xx = m0 / m2;
            const double yy = m1 / m2;
            pixel_col[i] = xx;
            pixel_row[i] = yy;
            inside[i] = xx >= 0 && yy >= 0 && xx < image_shape[1] && yy < image_shape[0];
        }
    });
    using visium_hd::parquet::Column;
    using visium_hd::parquet::Type;
    std::vector<Column> columns(6);
    columns[0].name = "barcode";
    columns[0].type = Type::String;
    columns[0].text = barcodes.text.data();
    columns[0].begin = barcodes.begin.data();
    columns[1].name = "array_row";
    columns[1].type = Type::Int32;
    columns[1].int32s = barcodes.row.data();
    columns[2].name = "array_col";
    columns[2].type = Type::Int32;
    columns[2].int32s = barcodes.col.data();
    columns[3].name = "pxl_row_in_fullres";
    columns[3].doubles = pixel_row.data();
    columns[4].name = "pxl_col_in_fullres";
    columns[4].doubles = pixel_col.data();
    columns[5].name = "in_image";
    columns[5].type = Type::Boolean;
    columns[5].booleans = inside.data();
    visium_hd::parquet::Options options;
    options.threads = threads;
    options.created_by = method_name();
    visium_hd::parquet::write(output.string(), columns, rows, options);
    Value out = Value::object();
    out.set("bin_um", bin_um);
    out.set("rows", static_cast<unsigned long long>(rows));
    out.set("path", output.string());
    out.set("matrix_columns", static_cast<unsigned long long>(expected_columns));
    out.set("barcode_order", "exact emitted MEX column order");
    out.set("full_grid_coverage_required", false);
    out.set("support_column", "in_image; no tissue classification or count filtering");
    return out;
}

Value run_project(const Arguments& args, const fs::path& out) {
    const int width = static_cast<int>(args.number("width", 3350));
    const int threads = threads_of(args);
    const auto started = std::chrono::steady_clock::now();
    visium_hd::projection::Result result = visium_hd::projection::project(
        args.get("segmentation"), args.get("registration"), args.get("capture"), width, threads);
    const std::vector<std::size_t> shape{static_cast<std::size_t>(width), static_cast<std::size_t>(width)};
    visium_hd::npy::write_npz((out / "nuclei.npz").string(), {{"labels", "|b1", shape, result.nucleus.data(), result.nucleus.size()}});
    visium_hd::npy::write_npz((out / "support.npz").string(), {{"tissue", "|b1", shape, result.support.data(), result.support.size()}});
    visium_hd::json::write_file((out / "projection.json").string(), result.audit);
    Value complete = result.audit;
    complete.set("implementation", method_name());
    complete.set("seconds", seconds_since(started));
    return complete;
}

Value run_correct(const Arguments& args, const fs::path& out) {
    const int width = static_cast<int>(args.number("width", 3350));
    const cr::Grid grid{width, width};
    cr::Options options;
    options.threads = threads_of(args);
    const int threads = options.threads;
    const fs::path mex = args.get("mex");
    const fs::path projection = args.get("projection");
    const fs::path segmentation = args.get("segmentation");
    Value times = Value::object();
    const auto started = std::chrono::steady_clock::now();

    const std::vector<double> counts = load_counts((mex / "hard/square_002um").string(), grid, threads);
    times.set("load_counts_seconds", seconds_since(started));
    visium_hd::npy::write_npz((out / "square_totals.npz").string(),
                              {{"total", "<f8", {grid.size()}, counts.data(), counts.size() * sizeof(double)}});
    const std::vector<std::uint8_t> nucleus = load_flags((projection / "nuclei.npz").string(), "", grid);
    const std::vector<std::uint8_t> support = load_flags((projection / "support.npz").string(), "tissue", grid);

    auto before = std::chrono::steady_clock::now();
    const cr::Check initial = cr::check(nucleus, counts, support, grid, options);
    const cr::AffineModel model = cr::fit_regions(initial.regions);
    times.set("check_and_fit_seconds", seconds_since(before));
    visium_hd::json::write_file((out / "check_before.json").string(), check_json(initial));
    {
        Value fit = Value::object();
        fit.set("fit", model_json(model));
        visium_hd::json::write_file((out / "fit.json").string(), fit);
    }

    std::vector<std::uint8_t> moved;
    cr::Matrix3 forward{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    Value correction = Value::object();
    if (model.accepted) {
        const auto& coefficients = model.final_fit->coefficients;
        moved = cr::apply_affine(nucleus, grid, coefficients);
        forward = cr::displacement_matrix(coefficients);
        correction.set("kind", "accepted_regional_affine");
        correction.set("coefficients", coefficients_json(coefficients));
        correction.set("mask_resampling", "forward rounded grid positions; later row-order location wins");
    } else if (initial.whole.usable) {
        const int dy = *initial.whole.dy;
        const int dx = *initial.whole.dx;
        moved = cr::shift_grid(nucleus, grid, dy, dx);
        forward = cr::displacement_matrix({{{static_cast<double>(dy), static_cast<double>(dx)}, {0, 0}, {0, 0}}});
        correction.set("kind", "whole_slide_shift");
        correction.set("dy", dy);
        correction.set("dx", dx);
        correction.set("affine_rejection", Value(model.stop_reason));
    } else {
        // Not registered: the first registration stands.
        moved = nucleus;
        correction.set("kind", "not_registered");
        correction.set("reason", initial.whole.reason);
        correction.set("affine_rejection", Value(model.stop_reason));
    }
    visium_hd::npy::write_npz((out / "corrected_nuclei.npz").string(),
                              {{"labels", "|b1", {static_cast<std::size_t>(width), static_cast<std::size_t>(width)},
                                moved.data(), moved.size()}});
    before = std::chrono::steady_clock::now();
    const cr::Check final_check = cr::check(moved, counts, support, grid, options);
    times.set("check_after_seconds", seconds_since(before));
    visium_hd::json::write_file((out / "check_after.json").string(), check_json(final_check));

    const Value projected = visium_hd::json::read_file((projection / "projection.json").string());
    const cr::Matrix3 original = matrix_from(projected.at("grid_to_full_resolution_image"));
    cr::Matrix3 registered = cr::multiply(original, cr::inverse(forward));
    const double scale = registered[2][2];
    for (auto& row : registered) {
        for (double& v : row) v /= scale;
    }
    const Value contract = visium_hd::json::read_file((segmentation / "summary.json").string()).at("input_contract");
    const Value& shape = contract.at("source_shape_yx");
    const std::array<double, 2> image_shape{shape[0].as_double(), shape[1].as_double()};
    before = std::chrono::steady_clock::now();
    Value tables = Value::array();
    for (int size : {2, 8, 16}) {
        char name[32];
        std::snprintf(name, sizeof(name), "square_%03dum", size);
        char file[48];
        std::snprintf(file, sizeof(file), "positions_%03dum.parquet", size);
        const fs::path dir = mex / "hard" / name;
        tables.push(write_positions(dir / "barcodes.tsv", out / file, size, registered, image_shape, width,
                                    visium_hd::mex::matrix_columns((dir / "matrix.mtx").string()), threads));
    }
    times.set("write_registered_positions_seconds", seconds_since(before));

    Value result = Value::object();
    result.set("matrix_root", args.get("mex"));
    result.set("count_matrices_modified", false);
    result.set("support_kind", "native_registered_image_coverage");
    result.set("uncorrected_grid_to_image", matrix_json(original));
    result.set("correction", correction);
    result.set("corrected_grid_to_image", matrix_json(registered));
    result.set("positions", tables);
    result.set("component_seconds", times);
    result.set("total_seconds", seconds_since(started));
    result.set("implementation", method_name());
    result.set("threads", threads);
    visium_hd::json::write_file((out / "registered_geometry.json").string(), result);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    Arguments args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n" << kUsage;
        return 2;
    }
    try {
        if (args.step == "check") return run_check(args);
        if (args.step != "correct" && args.step != "project") {
            std::cerr << "unknown step: " << args.step << "\n" << kUsage;
            return 2;
        }
        const fs::path out = args.get("out");
        if (fs::exists(out)) throw std::runtime_error("output directory exists: " + out.string());
        fs::create_directories(out);
        Value invocation = Value::object();
        invocation.set("argv", Value::array_of(args.argv));
        invocation.set("started_utc", utc_now());
        visium_hd::json::write_file((out / "invocation.json").string(), invocation);
        try {
            const Value result = args.step == "correct" ? run_correct(args, out) : run_project(args, out);
            Value complete = Value::object();
            complete.set("utc", utc_now());
            complete.set("step", args.step);
            complete.set("result", result);
            visium_hd::json::write_file((out / "COMPLETE.json").string(), complete);
        } catch (const std::exception& error) {
            Value failed = Value::object();
            failed.set("utc", utc_now());
            failed.set("reason", error.what());
            visium_hd::json::write_file((out / "FAILED.json").string(), failed);
            throw;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << kTool << ": " << error.what() << "\n";
        return 1;
    }
}
