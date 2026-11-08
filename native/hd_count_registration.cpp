#include "hd_count_registration.hpp"

#include "hd_json.hpp"
#include "hd_parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace visium_hd::count_registration {

const std::array<const char*, kBands + 1> kBandNames = {"I3", "I2", "I1", "O1", "O2", "O3", "O4", "O5", "O6", "far"};

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::int64_t kNone = std::numeric_limits<std::int64_t>::max() / 4;

// ---------------------------------------------------------------------------------------------
// numpy.random.default_rng: SeedSequence (numpy/random/bit_generator.pyx), PCG64 (pcg64.h) and the
// bounded integers of Generator.integers (Lemire's method, distributions.c).

constexpr std::uint32_t kInitA = 0x43b0d7e5, kMultA = 0x931e8875, kInitB = 0x8b51f9dd, kMultB = 0x58f38ded;
constexpr std::uint32_t kMixL = 0xca01f9dd, kMixR = 0x4973f715;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash) {
    value ^= hash;
    hash *= kMultA;
    value *= hash;
    value ^= value >> 16;
    return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
    std::uint32_t result = kMixL * x - kMixR * y;
    result ^= result >> 16;
    return result;
}

constexpr unsigned __int128 kPcgMultiplier =
    (static_cast<unsigned __int128>(0x2360ED051FC65DA4ull) << 64) | 0x4385DF649FCCF645ull;

}  // namespace

NumpyGenerator::NumpyGenerator(std::uint64_t seed) {
    std::vector<std::uint32_t> entropy;
    if (seed == 0) entropy.push_back(0);
    for (; seed > 0; seed >>= 32) entropy.push_back(static_cast<std::uint32_t>(seed & 0xFFFFFFFFu));
    std::uint32_t pool[4];
    std::uint32_t hash = kInitA;
    for (std::size_t i = 0; i < 4; ++i) pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0u, hash);
    for (int source = 0; source < 4; ++source) {
        for (int target = 0; target < 4; ++target) {
            if (source != target) pool[target] = mix(pool[target], hashmix(pool[source], hash));
        }
    }
    for (std::size_t source = 4; source < entropy.size(); ++source) {
        for (int target = 0; target < 4; ++target) pool[target] = mix(pool[target], hashmix(entropy[source], hash));
    }
    std::uint32_t words[8];
    std::uint32_t constant = kInitB;
    for (int i = 0; i < 8; ++i) {
        std::uint32_t value = pool[i % 4];
        value ^= constant;
        constant *= kMultB;
        value *= constant;
        value ^= value >> 16;
        words[i] = value;
    }
    std::uint64_t state[4];
    for (int i = 0; i < 4; ++i) state[i] = words[2 * i] | (static_cast<std::uint64_t>(words[2 * i + 1]) << 32);
    const unsigned __int128 initial = (static_cast<unsigned __int128>(state[0]) << 64) | state[1];
    const unsigned __int128 sequence = (static_cast<unsigned __int128>(state[2]) << 64) | state[3];
    state_ = 0;
    increment_ = (sequence << 1) | 1u;
    state_ = state_ * kPcgMultiplier + increment_;
    state_ += initial;
    state_ = state_ * kPcgMultiplier + increment_;
}

std::uint64_t NumpyGenerator::next_uint64() {
    state_ = state_ * kPcgMultiplier + increment_;
    const std::uint64_t value = static_cast<std::uint64_t>(state_ >> 64) ^ static_cast<std::uint64_t>(state_);
    const unsigned rotation = static_cast<unsigned>(state_ >> 122);
    return (value >> rotation) | (value << ((64u - rotation) & 63u));
}

std::uint32_t NumpyGenerator::next_uint32() {
    if (has_uint32_) {
        has_uint32_ = false;
        return uinteger_;
    }
    const std::uint64_t next = next_uint64();
    has_uint32_ = true;
    uinteger_ = static_cast<std::uint32_t>(next >> 32);
    return static_cast<std::uint32_t>(next & 0xFFFFFFFFu);
}

std::uint64_t NumpyGenerator::integers(std::uint64_t high) {
    if (high == 0) throw std::invalid_argument("integers: high must be positive");
    const std::uint64_t range = high - 1;
    if (range == 0) return 0;
    if (range > 0xFFFFFFFFull) throw std::invalid_argument("integers: ranges above 2^32 are not implemented");
    if (range == 0xFFFFFFFFull) return next_uint32();
    const std::uint32_t exclusive = static_cast<std::uint32_t>(range) + 1u;
    std::uint64_t m = static_cast<std::uint64_t>(next_uint32()) * exclusive;
    std::uint32_t leftover = static_cast<std::uint32_t>(m & 0xFFFFFFFFu);
    if (leftover < exclusive) {
        const std::uint32_t threshold = (0xFFFFFFFFu - static_cast<std::uint32_t>(range)) % exclusive;
        while (leftover < threshold) {
            m = static_cast<std::uint64_t>(next_uint32()) * exclusive;
            leftover = static_cast<std::uint32_t>(m & 0xFFFFFFFFu);
        }
    }
    return m >> 32;
}

std::vector<double> block_weights(int blocks, int draws, std::uint64_t seed) {
    std::vector<double> weights(static_cast<std::size_t>(blocks) * static_cast<std::size_t>(draws), 0.0);
    if (blocks <= 0) return weights;
    NumpyGenerator generator(seed);
    for (int draw = 0; draw < draws; ++draw) {
        double* row = weights.data() + static_cast<std::size_t>(draw) * blocks;
        for (int i = 0; i < blocks; ++i) row[generator.integers(static_cast<std::uint64_t>(blocks))] += 1.0;
    }
    return weights;
}

// ---------------------------------------------------------------------------------------------
// Band labels: exact squared Euclidean distances (Felzenszwalb and Huttenlocher), compared with
// squared thresholds, so the labels equal those from scipy's distance_transform_edt.

namespace {

std::vector<std::int64_t> squared_distance(const std::vector<std::uint8_t>& feature, Grid grid, int threads) {
    const int h = grid.height;
    const int w = grid.width;
    std::vector<std::int64_t> out(grid.size(), kNone);
    // Along columns, in strips of columns so that each pass reads rows contiguously.
    const int strip = 256;
    const int strips = (w + strip - 1) / strip;
    parallel_for(static_cast<std::size_t>(strips), threads, [&](std::size_t s) {
        const int x0 = static_cast<int>(s) * strip;
        const int x1 = std::min(w, x0 + strip);
        std::vector<std::int64_t> last(static_cast<std::size_t>(x1 - x0), -1);
        for (int y = 0; y < h; ++y) {
            const std::size_t row = static_cast<std::size_t>(y) * w;
            for (int x = x0; x < x1; ++x) {
                if (feature[row + x]) last[x - x0] = y;
                const std::int64_t at = last[x - x0];
                out[row + x] = at < 0 ? kNone : (y - at) * (y - at);
            }
        }
        std::fill(last.begin(), last.end(), -1);
        for (int y = h - 1; y >= 0; --y) {
            const std::size_t row = static_cast<std::size_t>(y) * w;
            for (int x = x0; x < x1; ++x) {
                if (feature[row + x]) last[x - x0] = y;
                const std::int64_t at = last[x - x0];
                if (at >= 0) out[row + x] = std::min(out[row + x], (at - y) * (at - y));
            }
        }
    });
    // Along rows: the lower envelope of the parabolas of the column distances.
    parallel_for(static_cast<std::size_t>(h), threads, [&](std::size_t y) {
        std::int64_t* row = out.data() + y * static_cast<std::size_t>(w);
        std::vector<std::int64_t> f(row, row + w);
        std::vector<int> v(static_cast<std::size_t>(w));
        std::vector<double> z(static_cast<std::size_t>(w) + 1);
        int k = -1;
        for (int q = 0; q < w; ++q) {
            if (f[q] >= kNone) continue;
            while (k >= 0) {
                const double s = (static_cast<double>(f[q] + static_cast<std::int64_t>(q) * q)
                                  - static_cast<double>(f[v[k]] + static_cast<std::int64_t>(v[k]) * v[k]))
                    / (2.0 * (q - v[k]));
                if (s <= z[k]) {
                    --k;
                    continue;
                }
                ++k;
                v[k] = q;
                z[k] = s;
                z[k + 1] = std::numeric_limits<double>::infinity();
                break;
            }
            if (k < 0) {
                k = 0;
                v[0] = q;
                z[0] = -std::numeric_limits<double>::infinity();
                z[1] = std::numeric_limits<double>::infinity();
            }
        }
        if (k < 0) return;
        int j = 0;
        for (int q = 0; q < w; ++q) {
            while (z[j + 1] < q) ++j;
            const std::int64_t d = static_cast<std::int64_t>(q - v[j]);
            row[q] = d * d + f[v[j]];
        }
    });
    return out;
}

}  // namespace

std::vector<std::int8_t> band_labels(const std::vector<std::uint8_t>& nucleus, Grid grid, int threads) {
    if (nucleus.size() != grid.size()) throw std::invalid_argument("nucleus map does not match the grid");
    std::vector<std::uint8_t> background(nucleus.size());
    for (std::size_t i = 0; i < nucleus.size(); ++i) background[i] = nucleus[i] ? 0 : 1;
    const std::vector<std::int64_t> inside = squared_distance(background, grid, threads);
    const std::vector<std::int64_t> outside = squared_distance(nucleus, grid, threads);
    std::vector<std::int8_t> labels(grid.size(), -1);
    parallel_for(static_cast<std::size_t>(grid.height), threads, [&](std::size_t y) {
        const std::size_t begin = y * static_cast<std::size_t>(grid.width);
        for (std::size_t i = begin; i < begin + static_cast<std::size_t>(grid.width); ++i) {
            if (nucleus[i]) {
                const std::int64_t d = inside[i];
                labels[i] = d > 4 ? 0 : d > 1 ? 1 : 2;
            } else {
                const std::int64_t d = outside[i];
                for (int k = 1; k <= 6; ++k) {
                    if (d > static_cast<std::int64_t>(k - 1) * (k - 1) && d <= static_cast<std::int64_t>(k) * k) {
                        labels[i] = static_cast<std::int8_t>(2 + k);
                        break;
                    }
                }
            }
        }
    });
    return labels;
}

std::vector<double> corrected(const std::vector<double>& counts, const std::vector<std::uint8_t>& support, Grid grid,
                              int baseline, int threads) {
    if (counts.size() != grid.size() || support.size() != grid.size()) {
        throw std::invalid_argument("counts or support do not match the grid");
    }
    const int h = grid.height;
    const int w = grid.width;
    // Sums over the square of 2 * baseline + 1 locations, zero beyond the grid: along rows, then columns.
    std::vector<double> row_counts(grid.size());
    std::vector<std::int32_t> row_flags(grid.size());
    parallel_for(static_cast<std::size_t>(h), threads, [&](std::size_t y) {
        const std::size_t row = y * static_cast<std::size_t>(w);
        double c = 0.0;
        std::int32_t f = 0;
        for (int x = 0; x < std::min(baseline, w); ++x) {
            if (support[row + x]) {
                c += counts[row + x];
                ++f;
            }
        }
        for (int x = 0; x < w; ++x) {
            const int enter = x + baseline;
            const int leave = x - baseline - 1;
            if (enter < w && support[row + enter]) {
                c += counts[row + enter];
                ++f;
            }
            if (leave >= 0 && support[row + leave]) {
                c -= counts[row + leave];
                --f;
            }
            row_counts[row + x] = c;
            row_flags[row + x] = f;
        }
    });
    std::vector<double> values(grid.size(), 0.0);
    const int strip = 256;
    const int strips = (w + strip - 1) / strip;
    parallel_for(static_cast<std::size_t>(strips), threads, [&](std::size_t s) {
        const int x0 = static_cast<int>(s) * strip;
        const int x1 = std::min(w, x0 + strip);
        for (int x = x0; x < x1; ++x) {
            double c = 0.0;
            std::int64_t f = 0;
            for (int y = 0; y < std::min(baseline, h); ++y) {
                c += row_counts[static_cast<std::size_t>(y) * w + x];
                f += row_flags[static_cast<std::size_t>(y) * w + x];
            }
            for (int y = 0; y < h; ++y) {
                const int enter = y + baseline;
                const int leave = y - baseline - 1;
                if (enter < h) {
                    c += row_counts[static_cast<std::size_t>(enter) * w + x];
                    f += row_flags[static_cast<std::size_t>(enter) * w + x];
                }
                if (leave >= 0) {
                    c -= row_counts[static_cast<std::size_t>(leave) * w + x];
                    f -= row_flags[static_cast<std::size_t>(leave) * w + x];
                }
                const std::size_t i = static_cast<std::size_t>(y) * w + x;
                if (!support[i] || f <= 0) continue;
                const double mean = c / static_cast<double>(f);
                values[i] = mean > 0.0 ? counts[i] / mean : 0.0;
            }
        }
    });
    return values;
}

namespace {

// ---------------------------------------------------------------------------------------------
// The check.

constexpr int kStatsPerShift = kBands * 2;   // (locations, sum of corrected counts) per band

struct Geometry {
    int radius = 0;   // search + 1
    int size = 0;     // 2 * radius + 1
    int shifts = 0;   // size * size
    std::size_t stride = 0;   // per block: shifts * bands * 2, then locations, sum, sum of squares
};

Geometry geometry_for(const Options& options) {
    Geometry g;
    g.radius = options.search + 1;
    g.size = 2 * g.radius + 1;
    g.shifts = g.size * g.size;
    g.stride = static_cast<std::size_t>(g.shifts) * kStatsPerShift + 3;
    return g;
}

// Counts of supported locations in rectangles, from an integral image.
class SupportSums {
public:
    SupportSums(const std::vector<std::uint8_t>& support, Grid grid) : grid_(grid), sums_((grid.size() + grid.height + grid.width + 1), 0) {
        const std::size_t stride = static_cast<std::size_t>(grid.width) + 1;
        for (int y = 0; y < grid.height; ++y) {
            std::int64_t running = 0;
            for (int x = 0; x < grid.width; ++x) {
                running += support[static_cast<std::size_t>(y) * grid.width + x] ? 1 : 0;
                sums_[(y + 1) * stride + x + 1] = sums_[y * stride + x + 1] + running;
            }
        }
    }
    std::int64_t count(int y0, int y1, int x0, int x1) const {
        y0 = std::max(y0, 0);
        x0 = std::max(x0, 0);
        y1 = std::min(y1, grid_.height);
        x1 = std::min(x1, grid_.width);
        if (y1 <= y0 || x1 <= x0) return 0;
        const std::size_t stride = static_cast<std::size_t>(grid_.width) + 1;
        return sums_[y1 * stride + x1] - sums_[y0 * stride + x1] - sums_[y1 * stride + x0] + sums_[y0 * stride + x0];
    }

private:
    Grid grid_;
    std::vector<std::int64_t> sums_;
};

// The blocks of `block` x `block` locations over `bounds` that lie within `margin` of the support,
// in row-major order and clipped to the bounds (dip.blocks_for).
struct Partition {
    Bounds bounds;
    int ny = 0, nx = 0;
    std::vector<int> index;        // block grid position -> eligible block, or -1
    std::vector<Bounds> blocks;
    std::size_t first_target = 0;
};

Partition partition_for(const SupportSums& sums, Bounds bounds, int block, int margin) {
    Partition p;
    p.bounds = bounds;
    p.ny = (bounds.y1 - bounds.y0 + block - 1) / block;
    p.nx = (bounds.x1 - bounds.x0 + block - 1) / block;
    p.index.assign(static_cast<std::size_t>(p.ny) * p.nx, -1);
    for (int i = 0; i < p.ny * p.nx; ++i) {
        Bounds b;
        b.y0 = bounds.y0 + (i / p.nx) * block;
        b.y1 = std::min(b.y0 + block, bounds.y1);
        b.x0 = bounds.x0 + (i % p.nx) * block;
        b.x1 = std::min(b.x0 + block, bounds.x1);
        if (sums.count(b.y0 - margin, b.y1 + margin, b.x0 - margin, b.x1 + margin) > 0) {
            p.index[i] = static_cast<int>(p.blocks.size());
            p.blocks.push_back(b);
        }
    }
    return p;
}

int block_of(const Partition& p, int y, int x, int block) {
    const int by = (y - p.bounds.y0) / block;
    const int bx = (x - p.bounds.x0) / block;
    return p.index[static_cast<std::size_t>(by) * p.nx + bx];
}

// numpy.linspace(0, length, divisions + 1).astype(int).
std::vector<int> edges_for(int length, int divisions) {
    std::vector<int> edges(static_cast<std::size_t>(divisions) + 1);
    const double step = static_cast<double>(length) / divisions;
    for (int i = 0; i <= divisions; ++i) edges[i] = static_cast<int>(i * step);
    edges[divisions] = length;
    return edges;
}

// The band sums of one rectangle of target locations (an atom: the intersection of a whole-slide
// block and a regional block) at every shift, added to the blocks that contain it.
void atom_sums(const std::vector<std::int8_t>& padded, int padded_width, const std::vector<double>& values,
               const std::vector<std::uint8_t>& support, Grid grid, Bounds atom, const Geometry& g,
               std::vector<double>& out) {
    std::vector<int> xs;
    std::vector<double> vs;
    std::vector<int> row_begin(static_cast<std::size_t>(atom.y1 - atom.y0) + 1, 0);
    double tissue_n = 0.0, tissue_s = 0.0, tissue_q = 0.0;
    for (int y = atom.y0; y < atom.y1; ++y) {
        const std::size_t row = static_cast<std::size_t>(y) * grid.width;
        for (int x = atom.x0; x < atom.x1; ++x) {
            if (!support[row + x]) continue;
            const double v = values[row + x];
            xs.push_back(x);
            vs.push_back(v);
            tissue_n += 1.0;
            tissue_s += v;
            tissue_q += v * v;
        }
        row_begin[y - atom.y0 + 1] = static_cast<int>(xs.size());
    }
    out.assign(g.stride, 0.0);
    out[g.stride - 3] = tissue_n;
    out[g.stride - 2] = tissue_s;
    out[g.stride - 1] = tissue_q;
    if (xs.empty()) return;
    // Two interleaved sets of accumulators per shift; index 0 collects locations beyond the bands.
    std::vector<std::int32_t> n(static_cast<std::size_t>(g.shifts) * 20, 0);
    std::vector<double> s(static_cast<std::size_t>(g.shifts) * 20, 0.0);
    const int r = g.radius;
    for (int y = atom.y0; y < atom.y1; ++y) {
        const int b = row_begin[y - atom.y0];
        const int e = row_begin[y - atom.y0 + 1];
        if (b == e) continue;
        for (int dy = -r; dy <= r; ++dy) {
            const std::int8_t* labels_row = padded.data() + static_cast<std::size_t>(y - dy + r) * padded_width + r;
            for (int dx = -r; dx <= r; ++dx) {
                const std::size_t p = static_cast<std::size_t>(dy + r) * g.size + (dx + r);
                const std::int8_t* l = labels_row - dx;
                std::int32_t* n0 = n.data() + p * 20;
                std::int32_t* n1 = n0 + 10;
                double* s0 = s.data() + p * 20;
                double* s1 = s0 + 10;
                int i = b;
                for (; i + 1 < e; i += 2) {
                    const int j0 = l[xs[i]] + 1;
                    const int j1 = l[xs[i + 1]] + 1;
                    n0[j0] += 1;
                    s0[j0] += vs[i];
                    n1[j1] += 1;
                    s1[j1] += vs[i + 1];
                }
                if (i < e) {
                    const int j0 = l[xs[i]] + 1;
                    n0[j0] += 1;
                    s0[j0] += vs[i];
                }
            }
        }
    }
    for (int p = 0; p < g.shifts; ++p) {
        for (int j = 0; j < kBands; ++j) {
            const std::size_t at = static_cast<std::size_t>(p) * 20 + j + 1;
            out[(static_cast<std::size_t>(p) * kBands + j) * 2] = static_cast<double>(n[at] + n[at + 10]);
            out[(static_cast<std::size_t>(p) * kBands + j) * 2 + 1] = s[at] + s[at + 10];
        }
    }
}

// eta^2 at every shift from summed block statistics (ratio.eta2).
void eta2_surface(const double* sums, const Geometry& g, double* out) {
    const double n = sums[g.stride - 3];
    const double s = sums[g.stride - 2];
    const double q = sums[g.stride - 1];
    for (int p = 0; p < g.shifts; ++p) {
        const double* band = sums + static_cast<std::size_t>(p) * kStatsPerShift;
        double explained = 0.0, n_bands = 0.0, s_bands = 0.0;
        for (int j = 0; j < kBands; ++j) {
            const double nj = band[2 * j];
            const double sj = band[2 * j + 1];
            n_bands += nj;
            s_bands += sj;
            explained += nj > 0 ? sj * sj / nj : 0.0;
        }
        const double n_far = n - n_bands;
        const double s_far = s - s_bands;
        const double between = explained + (n_far > 0 ? s_far * s_far / n_far : 0.0) - s * s / n;
        out[p] = between / (q - s * s / n);
    }
}

struct Peak {
    bool found = false;
    int dy = 0, dx = 0;
};

// The highest local maximum (above all eight neighbours) within `search`; ties go to the first
// shift in row-major order (ratio.highest).
Peak highest(const double* surface, const Geometry& g, int search) {
    const int lo = g.radius - search;
    const int side = 2 * search + 1;
    double best = -std::numeric_limits<double>::infinity();
    int best_k = 0;
    for (int a = 0; a < side; ++a) {
        for (int b = 0; b < side; ++b) {
            const int y = lo + a;
            const int x = lo + b;
            const double c = surface[y * g.size + x];
            if (!std::isfinite(c)) continue;
            bool hit = true;
            for (int jy = -1; jy <= 1 && hit; ++jy) {
                for (int jx = -1; jx <= 1; ++jx) {
                    if (jy == 0 && jx == 0) continue;
                    const double v = surface[(y + jy) * g.size + x + jx];
                    if (!std::isfinite(v) || !(c > v)) {
                        hit = false;
                        break;
                    }
                }
            }
            if (hit && c > best) {
                best = c;
                best_k = a * side + b;
            }
        }
    }
    Peak peak;
    peak.found = std::isfinite(best);
    peak.dy = best_k / side - search;
    peak.dx = best_k % side - search;
    return peak;
}

std::string percent(double share) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.0f%%", share * 100.0);
    return text;
}

// ratio.assess on blocks whose statistics are already summed.
Assessment assess_blocks(const std::vector<const double*>& blocks, const Geometry& g, const Options& options) {
    Assessment row;
    row.baseline = options.baseline;
    row.blocks = static_cast<int>(blocks.size());
    row.draws = options.draws;
    row.seed = options.seed;
    row.search = options.search;
    if (blocks.empty()) {
        row.reason = "no tissue";
        row.surface.assign(static_cast<std::size_t>(g.shifts), kNaN);
        return row;
    }
    const int threads = options.threads;
    std::vector<double> total(g.stride, 0.0);
    {
        const std::size_t tile = 1024;
        const std::size_t tiles = (g.stride + tile - 1) / tile;
        parallel_for(tiles, threads, [&](std::size_t t) {
            const std::size_t c0 = t * tile;
            const std::size_t c1 = std::min(g.stride, c0 + tile);
            for (const double* block : blocks) {
                for (std::size_t c = c0; c < c1; ++c) total[c] += block[c];
            }
        });
    }
    row.surface.assign(static_cast<std::size_t>(g.shifts), 0.0);
    eta2_surface(total.data(), g, row.surface.data());
    const int centre = g.radius * g.size + g.radius;
    row.eta2_at_zero = row.surface[centre];
    const Peak peak = highest(row.surface.data(), g, options.search);
    if (!peak.found) {
        row.reason = "no local maximum";
        return row;
    }
    const int dy = peak.dy;
    const int dx = peak.dx;

    // The resamples: weighted sums of the block statistics, in tiles of columns and groups of draws.
    const int count = static_cast<int>(blocks.size());
    const std::vector<double> weights = block_weights(count, options.draws, options.seed);
    std::vector<double> resampled(static_cast<std::size_t>(options.draws) * g.stride, 0.0);
    {
        const std::size_t tile = 512;
        const int group = 8;
        const std::size_t tiles = (g.stride + tile - 1) / tile;
        const std::size_t groups = static_cast<std::size_t>((options.draws + group - 1) / group);
        parallel_for(tiles * groups, threads, [&](std::size_t task) {
            const std::size_t c0 = (task % tiles) * tile;
            const std::size_t c1 = std::min(g.stride, c0 + tile);
            const int d0 = static_cast<int>(task / tiles) * group;
            const int d1 = std::min(options.draws, d0 + group);
            for (int b = 0; b < count; ++b) {
                const double* source = blocks[b];
                for (int d = d0; d < d1; ++d) {
                    const double w = weights[static_cast<std::size_t>(d) * count + b];
                    if (w == 0.0) continue;
                    double* target = resampled.data() + static_cast<std::size_t>(d) * g.stride;
                    for (std::size_t c = c0; c < c1; ++c) target[c] += w * source[c];
                }
            }
        });
    }
    std::vector<Peak> peaks(static_cast<std::size_t>(options.draws));
    parallel_for(static_cast<std::size_t>(options.draws), threads, [&](std::size_t d) {
        std::vector<double> surface(static_cast<std::size_t>(g.shifts));
        eta2_surface(resampled.data() + d * g.stride, g, surface.data());
        peaks[d] = highest(surface.data(), g, options.search);
    });
    int any = 0, within_one = 0, exact = 0;
    for (const Peak& p : peaks) {
        if (!p.found) continue;
        ++any;
        const int far = std::max(std::abs(p.dy - dy), std::abs(p.dx - dx));
        if (far <= 1) ++within_one;
        if (far == 0) ++exact;
    }
    const int k = (g.radius + dy) * g.size + g.radius + dx;
    const double* band = total.data() + static_cast<std::size_t>(k) * kStatsPerShift;
    double n_bands = 0.0, s_bands = 0.0;
    for (int j = 0; j < kBands; ++j) {
        const double mean = band[2 * j + 1] / band[2 * j];
        row.profile[j] = std::isfinite(mean) ? std::optional<double>(mean) : std::nullopt;
        n_bands += band[2 * j];
        s_bands += band[2 * j + 1];
    }
    const double far_mean = (total[g.stride - 2] - s_bands) / (total[g.stride - 3] - n_bands);
    row.profile[kBands] = std::isfinite(far_mean) ? std::optional<double>(far_mean) : std::nullopt;
    row.has_profile = true;
    row.eta2 = row.surface[k];
    row.interior = std::max(std::abs(dy), std::abs(dx)) < options.search;
    row.share_any_peak = static_cast<double>(any) / options.draws;
    row.share_within_one = static_cast<double>(within_one) / options.draws;
    row.share_exact = static_cast<double>(exact) / options.draws;
    row.candidate_dy = dy;
    row.candidate_dx = dx;
    if (*row.interior && *row.share_within_one >= options.stable) {
        row.status = "usable";
        row.usable = true;
        row.dy = dy;
        row.dx = dx;
        row.reason = "usable";
    } else {
        row.reason = *row.interior
            ? "highest maximum within one location in fewer than " + percent(options.stable) + " of resamples"
            : "highest maximum on the edge of the search";
    }
    return row;
}

}  // namespace

namespace {

void validate(const std::vector<std::uint8_t>& nucleus, const std::vector<double>& counts,
              const std::vector<std::uint8_t>& support, Grid grid, const Options& options) {
    if (nucleus.size() != grid.size() || counts.size() != grid.size() || support.size() != grid.size()) {
        throw std::invalid_argument("nucleus map, counts and support must match the grid");
    }
    if (options.search < 1 || options.draws < 1 || options.block < 1 || options.baseline < 0 || options.divisions < 0) {
        throw std::invalid_argument("invalid check options");
    }
    for (double c : counts) {
        if (!std::isfinite(c) || c < 0) throw std::invalid_argument("counts must be finite and nonnegative");
    }
}

// The assessments of several areas of one nucleus map in one pass over the target locations: the
// grid is cut along the block edges of every area, and the band sums of each piece are added to the
// block of every area that contains it.
std::vector<Assessment> assess_prepared(const std::vector<std::int8_t>& labels, const std::vector<double>& values,
                                        const std::vector<std::uint8_t>& support, Grid grid,
                                        const std::vector<Bounds>& areas, const Options& options) {
    const int threads = options.threads;
    const Geometry g = geometry_for(options);
    // Band labels with a margin of the search radius plus one, beyond the bands outside the grid.
    const int r = g.radius;
    const int padded_width = grid.width + 2 * r;
    std::vector<std::int8_t> padded(static_cast<std::size_t>(grid.height + 2 * r) * padded_width, -1);
    for (int y = 0; y < grid.height; ++y) {
        std::copy(labels.begin() + static_cast<std::ptrdiff_t>(y) * grid.width,
                  labels.begin() + static_cast<std::ptrdiff_t>(y + 1) * grid.width,
                  padded.begin() + static_cast<std::ptrdiff_t>(y + r) * padded_width + r);
    }
    const SupportSums sums(support, grid);
    std::vector<Partition> partitions;
    std::size_t targets = 0;
    std::vector<int> ycuts{0, grid.height}, xcuts{0, grid.width};
    for (const Bounds& area : areas) {
        if (area.y0 < 0 || area.x0 < 0 || area.y1 > grid.height || area.x1 > grid.width || area.y1 <= area.y0
            || area.x1 <= area.x0) {
            throw std::invalid_argument("area outside the grid");
        }
        partitions.push_back(partition_for(sums, area, options.block, r));
        partitions.back().first_target = targets;
        targets += partitions.back().blocks.size();
        for (int at = area.y0; at < area.y1; at += options.block) ycuts.push_back(at);
        for (int at = area.x0; at < area.x1; at += options.block) xcuts.push_back(at);
        ycuts.push_back(area.y1);
        xcuts.push_back(area.x1);
    }
    for (std::vector<int>* c : {&ycuts, &xcuts}) {
        std::sort(c->begin(), c->end());
        c->erase(std::unique(c->begin(), c->end()), c->end());
    }
    struct Atom {
        Bounds bounds;
        std::vector<int> targets;
    };
    std::vector<Atom> atoms;
    for (std::size_t a = 0; a + 1 < ycuts.size(); ++a) {
        for (std::size_t b = 0; b + 1 < xcuts.size(); ++b) {
            Atom atom;
            atom.bounds = Bounds{ycuts[a], ycuts[a + 1], xcuts[b], xcuts[b + 1]};
            if (sums.count(atom.bounds.y0, atom.bounds.y1, atom.bounds.x0, atom.bounds.x1) == 0) continue;
            for (const Partition& p : partitions) {
                if (atom.bounds.y0 < p.bounds.y0 || atom.bounds.y0 >= p.bounds.y1 || atom.bounds.x0 < p.bounds.x0
                    || atom.bounds.x0 >= p.bounds.x1) {
                    continue;
                }
                const int k = block_of(p, atom.bounds.y0, atom.bounds.x0, options.block);
                if (k >= 0) atom.targets.push_back(static_cast<int>(p.first_target) + k);
            }
            if (!atom.targets.empty()) atoms.push_back(std::move(atom));
        }
    }
    std::vector<double> stats(targets * g.stride, 0.0);
    std::vector<std::mutex> locks(targets);
    parallel_for(atoms.size(), threads, [&](std::size_t i) {
        std::vector<double> local;
        atom_sums(padded, padded_width, values, support, grid, atoms[i].bounds, g, local);
        for (int target : atoms[i].targets) {
            std::lock_guard<std::mutex> hold(locks[target]);
            double* out = stats.data() + static_cast<std::size_t>(target) * g.stride;
            for (std::size_t c = 0; c < g.stride; ++c) out[c] += local[c];
        }
    });
    std::vector<Assessment> out;
    for (const Partition& p : partitions) {
        std::vector<const double*> blocks;
        for (std::size_t k = 0; k < p.blocks.size(); ++k) blocks.push_back(stats.data() + (p.first_target + k) * g.stride);
        out.push_back(assess_blocks(blocks, g, options));
    }
    return out;
}

}  // namespace

std::vector<Assessment> assess_areas(const std::vector<std::uint8_t>& nucleus, const std::vector<double>& counts,
                                     const std::vector<std::uint8_t>& support, Grid grid,
                                     const std::vector<Bounds>& areas, const Options& options) {
    validate(nucleus, counts, support, grid, options);
    const std::vector<std::int8_t> labels = band_labels(nucleus, grid, options.threads);
    const std::vector<double> values = corrected(counts, support, grid, options.baseline, options.threads);
    return assess_prepared(labels, values, support, grid, areas, options);
}

Check check(const std::vector<std::uint8_t>& nucleus, const std::vector<double>& counts,
            const std::vector<std::uint8_t>& support, Grid grid, const Options& options) {
    validate(nucleus, counts, support, grid, options);
    const int threads = options.threads;
    const std::vector<std::int8_t> labels = band_labels(nucleus, grid, threads);
    const std::vector<double> values = corrected(counts, support, grid, options.baseline, threads);

    Check result;
    std::vector<Bounds> areas{Bounds{0, grid.height, 0, grid.width}};
    std::vector<int> area_of_region;
    if (options.divisions > 0) {
        const std::vector<int> ey = edges_for(grid.height, options.divisions);
        const std::vector<int> ex = edges_for(grid.width, options.divisions);
        const int regions = options.divisions * options.divisions;
        result.regions.resize(static_cast<std::size_t>(regions));
        parallel_for(static_cast<std::size_t>(regions), threads, [&](std::size_t i) {
            Region& region = result.regions[i];
            region.row = static_cast<int>(i) / options.divisions;
            region.col = static_cast<int>(i) % options.divisions;
            region.name = "r" + std::to_string(region.row) + "c" + std::to_string(region.col);
            std::int64_t n = 0, sy = 0, sx = 0;
            for (int y = ey[region.row]; y < ey[region.row + 1]; ++y) {
                for (int x = ex[region.col]; x < ex[region.col + 1]; ++x) {
                    const std::size_t at = static_cast<std::size_t>(y) * grid.width + x;
                    if (nucleus[at] && support[at]) {
                        ++n;
                        sy += y - ey[region.row];
                        sx += x - ex[region.col];
                    }
                }
            }
            region.nucleus_locations = n;
            if (n > 0) {
                region.centroid_y = static_cast<double>(sy) / static_cast<double>(n) + ey[region.row];
                region.centroid_x = static_cast<double>(sx) / static_cast<double>(n) + ex[region.col];
            }
            region.tested = n >= options.minimum;
            if (!region.tested) region.reason = "fewer than " + std::to_string(options.minimum) + " nucleus squares in tissue";
        });
        area_of_region.assign(static_cast<std::size_t>(regions), -1);
        for (int i = 0; i < regions; ++i) {
            const Region& region = result.regions[i];
            if (!region.tested) continue;
            area_of_region[i] = static_cast<int>(areas.size());
            areas.push_back(Bounds{ey[region.row], ey[region.row + 1], ex[region.col], ex[region.col + 1]});
        }
    }
    std::vector<Assessment> assessed = assess_prepared(labels, values, support, grid, areas, options);
    result.whole = std::move(assessed[0]);
    for (std::size_t i = 0; i < result.regions.size(); ++i) {
        Region& region = result.regions[i];
        if (!region.tested) continue;
        region.assessment = std::move(assessed[area_of_region[i]]);
        if (region.usable()) {
            ++result.usable_regions;
            if (*region.assessment.dy == 0 && *region.assessment.dx == 0) ++result.regions_at_zero;
        }
    }
    for (std::size_t i = 0; i < nucleus.size(); ++i) {
        if (nucleus[i] && support[i]) ++result.nucleus_locations_in_support;
    }
    return result;
}

// ---------------------------------------------------------------------------------------------
// The regional fit (affine.fit_regions), with least squares by a one-sided Jacobi SVD.

namespace {

struct Svd {
    std::vector<std::array<double, 3>> u;   // m x 3, columns of unit length (or zero)
    std::array<double, 3> sigma{};
    std::array<std::array<double, 3>, 3> v{};
};

Svd svd(const std::vector<std::array<double, 3>>& a) {
    Svd out;
    out.u = a;
    for (int i = 0; i < 3; ++i) out.v[i] = {0.0, 0.0, 0.0}, out.v[i][i] = 1.0;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int sweep = 0; sweep < 60; ++sweep) {
        bool rotated = false;
        for (int i = 0; i < 2; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (const auto& row : out.u) {
                    alpha += row[i] * row[i];
                    beta += row[j] * row[j];
                    gamma += row[i] * row[j];
                }
                if (gamma == 0.0 || std::abs(gamma) <= eps * std::sqrt(alpha * beta)) continue;
                rotated = true;
                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::abs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = c * t;
                for (auto& row : out.u) {
                    const double ui = row[i];
                    const double uj = row[j];
                    row[i] = c * ui - s * uj;
                    row[j] = s * ui + c * uj;
                }
                for (auto& row : out.v) {
                    const double vi = row[i];
                    const double vj = row[j];
                    row[i] = c * vi - s * vj;
                    row[j] = s * vi + c * vj;
                }
            }
        }
        if (!rotated) break;
    }
    for (int k = 0; k < 3; ++k) {
        double norm = 0.0;
        for (const auto& row : out.u) norm += row[k] * row[k];
        out.sigma[k] = std::sqrt(norm);
        if (out.sigma[k] > 0) {
            for (auto& row : out.u) row[k] /= out.sigma[k];
        }
    }
    return out;
}

int matrix_rank(const std::vector<std::array<double, 3>>& a) {
    if (a.empty()) return 0;
    const Svd d = svd(a);
    const double top = *std::max_element(d.sigma.begin(), d.sigma.end());
    const double tolerance = top * static_cast<double>(std::max<std::size_t>(a.size(), 3)) * std::numeric_limits<double>::epsilon();
    return static_cast<int>(std::count_if(d.sigma.begin(), d.sigma.end(), [&](double s) { return s > tolerance; }));
}

std::array<std::array<double, 2>, 3> lstsq(const std::vector<std::array<double, 3>>& a,
                                           const std::vector<std::array<double, 2>>& b) {
    const Svd d = svd(a);
    const double top = *std::max_element(d.sigma.begin(), d.sigma.end());
    const double cut = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max<std::size_t>(a.size(), 3)) * top;
    std::array<std::array<double, 2>, 3> x{};
    for (int k = 0; k < 3; ++k) {
        if (!(d.sigma[k] > cut)) continue;
        for (int c = 0; c < 2; ++c) {
            double projection = 0.0;
            for (std::size_t m = 0; m < a.size(); ++m) projection += d.u[m][k] * b[m][c];
            projection /= d.sigma[k];
            for (int i = 0; i < 3; ++i) x[i][c] += d.v[i][k] * projection;
        }
    }
    return x;
}

std::array<double, 2> predict(const std::array<double, 3>& design, const std::array<std::array<double, 2>, 3>& c) {
    return {design[0] * c[0][0] + design[1] * c[1][0] + design[2] * c[2][0],
            design[0] * c[0][1] + design[1] * c[1][1] + design[2] * c[2][1]};
}

struct FitError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

LinearFit fit_once_with_rejection(const std::vector<const Region*>& rows) {
    std::vector<std::array<double, 3>> x;
    std::vector<std::array<double, 2>> y;
    for (const Region* r : rows) {
        x.push_back({1.0, *r->centroid_y, *r->centroid_x});
        y.push_back({static_cast<double>(*r->assessment.dy), static_cast<double>(*r->assessment.dx)});
    }
    if (rows.size() < 3 || matrix_rank(x) < 3) throw FitError("affine design has fewer than three independent positions");
    LinearFit fit;
    fit.first_coefficients = lstsq(x, y);
    std::vector<std::array<double, 3>> kept_x;
    std::vector<std::array<double, 2>> kept_y;
    std::vector<bool> keep;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::array<double, 2> p = predict(x[i], fit.first_coefficients);
        const std::array<double, 2> residual{y[i][0] - p[0], y[i][1] - p[1]};
        fit.first_residuals.push_back(residual);
        keep.push_back(std::max(std::abs(residual[0]), std::abs(residual[1])) <= 2);
        if (keep.back()) {
            kept_x.push_back(x[i]);
            kept_y.push_back(y[i]);
        }
    }
    if (kept_x.size() < 3 || matrix_rank(kept_x) < 3) {
        throw FitError("outlier rejection leaves fewer than three independent positions");
    }
    fit.coefficients = lstsq(kept_x, kept_y);
    for (std::size_t i = 0; i < rows.size(); ++i) (keep[i] ? fit.retained : fit.rejected).push_back(rows[i]->name);
    return fit;
}

}  // namespace

AffineModel fit_regions(const std::vector<Region>& regions, int minimum_final, double held_out_tolerance,
                        double minimum_held_out_fraction) {
    std::vector<const Region*> usable, train, held;
    for (const Region& r : regions) {
        if (!r.usable()) continue;
        usable.push_back(&r);
        ((r.row + r.col) % 2 == 0 ? train : held).push_back(&r);
    }
    AffineModel out;
    out.usable_regions = static_cast<int>(usable.size());
    out.training_regions = static_cast<int>(train.size());
    out.held_out_total = static_cast<int>(held.size());
    try {
        out.training_fit = fit_once_with_rejection(train);
    } catch (const FitError& error) {
        out.stop_reason = error.what();
        return out;
    }
    for (const Region* r : held) {
        HeldOut h;
        h.region = r->name;
        h.prediction = predict({1.0, *r->centroid_y, *r->centroid_x}, out.training_fit->coefficients);
        h.residual = {*r->assessment.dy - h.prediction[0], *r->assessment.dx - h.prediction[1]};
        h.within_one_square = std::max(std::abs(h.residual[0]), std::abs(h.residual[1])) <= held_out_tolerance;
        out.held_out.push_back(h);
        if (h.within_one_square) ++out.held_out_pass;
    }
    out.has_held_out_fraction = true;
    if (!held.empty()) out.held_out_fraction = static_cast<double>(out.held_out_pass) / static_cast<double>(held.size());
    if (held.empty() || *out.held_out_fraction < minimum_held_out_fraction) {
        out.stop_reason = "held-out prediction fraction below " + json::python_float(minimum_held_out_fraction);
        return out;
    }
    try {
        out.final_fit = fit_once_with_rejection(usable);
    } catch (const FitError& error) {
        out.stop_reason = error.what();
        return out;
    }
    if (static_cast<int>(out.final_fit->retained.size()) < minimum_final) {
        out.stop_reason = "fewer than " + std::to_string(minimum_final) + " regions remain";
        return out;
    }
    out.accepted = true;
    return out;
}

std::vector<std::uint8_t> shift_grid(const std::vector<std::uint8_t>& labels, Grid grid, int dy, int dx) {
    std::vector<std::uint8_t> out(labels.size(), 0);
    for (int y = std::max(0, -dy); y < std::min(grid.height, grid.height - dy); ++y) {
        for (int x = std::max(0, -dx); x < std::min(grid.width, grid.width - dx); ++x) {
            out[static_cast<std::size_t>(y + dy) * grid.width + x + dx] = labels[static_cast<std::size_t>(y) * grid.width + x];
        }
    }
    return out;
}

std::vector<std::uint8_t> apply_affine(const std::vector<std::uint8_t>& labels, Grid grid,
                                       const std::array<std::array<double, 2>, 3>& c) {
    std::vector<std::uint8_t> out(labels.size(), 0);
    for (int y = 0; y < grid.height; ++y) {
        for (int x = 0; x < grid.width; ++x) {
            const std::uint8_t label = labels[static_cast<std::size_t>(y) * grid.width + x];
            if (!label) continue;
            const double d0 = 1.0 * c[0][0] + y * c[1][0] + x * c[2][0];
            const double d1 = 1.0 * c[0][1] + y * c[1][1] + x * c[2][1];
            const double ny = std::nearbyint(y + d0);
            const double nx = std::nearbyint(x + d1);
            if (ny < 0 || ny >= grid.height || nx < 0 || nx >= grid.width) continue;
            out[static_cast<std::size_t>(ny) * grid.width + static_cast<std::size_t>(nx)] = label;
        }
    }
    return out;
}

Matrix3 displacement_matrix(const std::array<std::array<double, 2>, 3>& c) {
    for (const auto& row : c) {
        for (double v : row) {
            if (!std::isfinite(v)) throw std::invalid_argument("non-finite displacement coefficient");
        }
    }
    return Matrix3{{{1 + c[2][1], c[1][1], c[0][1]}, {c[2][0], 1 + c[1][0], c[0][0]}, {0.0, 0.0, 1.0}}};
}

Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
    Matrix3 out{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
    return out;
}

Matrix3 inverse(const Matrix3& m) {
    // Gauss-Jordan elimination with partial pivoting.
    Matrix3 a = m;
    Matrix3 out{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 3; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) pivot = r;
        }
        if (a[pivot][col] == 0.0) throw std::runtime_error("singular 3 x 3 matrix");
        std::swap(a[col], a[pivot]);
        std::swap(out[col], out[pivot]);
        const double scale = a[col][col];
        for (int j = 0; j < 3; ++j) {
            a[col][j] /= scale;
            out[col][j] /= scale;
        }
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = a[r][col];
            for (int j = 0; j < 3; ++j) {
                a[r][j] -= f * a[col][j];
                out[r][j] -= f * out[col][j];
            }
        }
    }
    return out;
}

}  // namespace visium_hd::count_registration
