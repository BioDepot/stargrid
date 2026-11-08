#pragma once

// The registration check and correction of STARgrid from the counts: a C++ implementation of
// stargrid-register 0.6.0 (register/ in this repository), with the same defaults, the same
// spatial-block resamples (numpy's default_rng draws, reproduced exactly) and the same decisions.
//
// The capture locations of a nucleus map are labelled by their distance from the nucleus edge, in
// bands one location (2 um) wide: three inside the nuclei, six outside them, and the rest. Each count
// is divided by the mean count of the supported locations in the square of 2 * baseline + 1
// locations around it. At each shift of the nucleus map, the correlation ratio eta^2 (Roche et al.,
// 1998) measures how much of the variance of these corrected counts the bands explain. The shift is
// the highest local maximum of eta^2 within the search, accepted if it is interior to the search and
// stable when the slide is resampled in spatial blocks. A displacement (dy, dx) is in capture
// locations, rows downward, from mask to counts: a nucleus at (row, col) lies over the counts at
// (row + dy, col + dx).
//
// The sums over bands are taken directly, not by FFT, so they are exact up to the rounding of
// double-precision addition; the Python package's FFT sums agree with them to about 1e-10.

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace visium_hd::count_registration {

constexpr const char* kMethodVersion = "0.6.0";
constexpr int kBands = 9;
extern const std::array<const char*, kBands + 1> kBandNames;   // I3, I2, I1, O1 ... O6, far

struct Options {
    int search = 12;                 // largest shift searched on each axis, in locations
    int draws = 200;                 // spatial-block resamples of the stability test
    std::uint64_t seed = 20261007;
    int block = 100;                 // block side, in locations
    double stable = 0.95;            // share of resamples within one location of the shift
    int baseline = 2;                // half-width of the baseline window, in locations
    int divisions = 5;               // regions per axis for the regional check (0: whole slide only)
    std::int64_t minimum = 30000;    // nucleus locations in the support for a region to be tested
    int threads = 1;
};

// A grid of capture locations, stored row by row.
struct Grid {
    int height = 0;
    int width = 0;
    std::size_t size() const { return static_cast<std::size_t>(height) * static_cast<std::size_t>(width); }
};

struct Bounds {
    int y0 = 0, y1 = 0, x0 = 0, x1 = 0;
};

// The result of assessing one area (the whole slide or a region), with the fields of
// stargrid-register's ratio.assess.
struct Assessment {
    std::string status = "not registered";
    bool usable = false;
    std::optional<int> dy, dx;
    std::optional<double> eta2, eta2_at_zero;
    std::optional<bool> interior;
    std::optional<double> share_any_peak, share_within_one, share_exact;
    std::optional<int> candidate_dy, candidate_dx;
    bool has_profile = false;
    std::array<std::optional<double>, kBands + 1> profile{};   // mean corrected count by band, at the shift
    int baseline = 0;
    int blocks = 0;
    int draws = 0;
    std::uint64_t seed = 0;
    int search = 0;
    std::string reason;
    std::vector<double> surface;     // eta^2 of the full data at every shift, (2 search + 3)^2, row-major
};

struct Region {
    std::string name;                // r<row>c<col>
    int row = 0, col = 0;
    std::int64_t nucleus_locations = 0;   // nucleus locations in the support
    std::optional<double> centroid_y, centroid_x;
    bool tested = false;
    Assessment assessment;           // when tested
    std::string reason;              // when not tested
    bool usable() const { return tested && assessment.usable; }
};

struct Check {
    Assessment whole;
    std::vector<Region> regions;
    int usable_regions = 0;
    int regions_at_zero = 0;
    std::int64_t nucleus_locations_in_support = 0;
};

// The check of a nucleus map against the counts on the supported locations: the whole slide and,
// with divisions > 0, each region of a divisions x divisions split.
Check check(const std::vector<std::uint8_t>& nucleus, const std::vector<double>& counts,
            const std::vector<std::uint8_t>& support, Grid grid, const Options& options);

// ratio.assess of each area (y0, y1, x0, x1), sharing one pass over the locations.
std::vector<Assessment> assess_areas(const std::vector<std::uint8_t>& nucleus, const std::vector<double>& counts,
                                     const std::vector<std::uint8_t>& support, Grid grid,
                                     const std::vector<Bounds>& areas, const Options& options);

// A displacement that varies linearly across the slide, [dy, dx] = [1, row, col] @ coefficients,
// fitted to the usable regions as in stargrid-register's affine.fit_regions: fitted to alternate
// regions, tested on the others (75% within one location), and refitted to all usable regions.
struct LinearFit {
    std::array<std::array<double, 2>, 3> coefficients{};
    std::array<std::array<double, 2>, 3> first_coefficients{};
    std::vector<std::array<double, 2>> first_residuals;
    std::vector<std::string> retained, rejected;
};

struct HeldOut {
    std::string region;
    std::array<double, 2> prediction{}, residual{};
    bool within_one_square = false;
};

struct AffineModel {
    int usable_regions = 0, training_regions = 0, held_out_total = 0, held_out_pass = 0;
    bool accepted = false;
    std::vector<HeldOut> held_out;
    std::optional<LinearFit> training_fit, final_fit;
    bool has_held_out_fraction = false;
    std::optional<double> held_out_fraction;
    std::optional<std::string> stop_reason;
};

AffineModel fit_regions(const std::vector<Region>& regions, int minimum_final = 12, double held_out_tolerance = 1.0,
                        double minimum_held_out_fraction = 0.75);

// Moving a nucleus map: by whole locations, or by the modelled displacement at each location,
// rounded (where two locations land on one, the later in row order wins; off-grid ones are dropped).
std::vector<std::uint8_t> shift_grid(const std::vector<std::uint8_t>& labels, Grid grid, int dy, int dx);
std::vector<std::uint8_t> apply_affine(const std::vector<std::uint8_t>& labels, Grid grid,
                                       const std::array<std::array<double, 2>, 3>& coefficients);

using Matrix3 = std::array<std::array<double, 3>, 3>;

// The forward map from the uncorrected mask grid to the count grid, in (column, row) coordinates.
Matrix3 displacement_matrix(const std::array<std::array<double, 2>, 3>& coefficients);
Matrix3 multiply(const Matrix3& a, const Matrix3& b);
Matrix3 inverse(const Matrix3& m);

// numpy.random.default_rng(seed) for an integer seed: SeedSequence, PCG64 and the bounded integers
// of Generator.integers, reproduced exactly.
class NumpyGenerator {
public:
    explicit NumpyGenerator(std::uint64_t seed);
    std::uint32_t next_uint32();
    std::uint64_t next_uint64();
    std::uint64_t integers(std::uint64_t high);   // one draw of Generator.integers(high), high <= 2^32

private:
    unsigned __int128 state_ = 0, increment_ = 0;
    bool has_uint32_ = false;
    std::uint32_t uinteger_ = 0;
};

// Multinomial weights of `draws` resamples of `blocks` spatial blocks, (draws, blocks) row-major,
// as stargrid-register's dip.block_weights.
std::vector<double> block_weights(int blocks, int draws, std::uint64_t seed);

// The band of every location (0-8, or -1 beyond the bands), as stargrid-register's ratio.band_labels.
std::vector<std::int8_t> band_labels(const std::vector<std::uint8_t>& nucleus, Grid grid, int threads);

// Each supported count over the mean count of the supported locations within `baseline` locations
// on each axis; zero off the support and where that mean is zero (ratio.corrected).
std::vector<double> corrected(const std::vector<double>& counts, const std::vector<std::uint8_t>& support, Grid grid,
                              int baseline, int threads);

}  // namespace visium_hd::count_registration
