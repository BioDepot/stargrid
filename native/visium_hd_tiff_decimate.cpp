#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <tiffio.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Arguments {
    fs::path input;
    fs::path output;
    fs::path summary;
    int factor = 16;
    int threads = 1;
};

int parse_positive(const std::string& value, const std::string& option) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value.c_str(), &end, 10);
    if (errno || end == value.c_str() || *end || parsed < 1 || parsed > 4096) {
        throw std::runtime_error("invalid positive integer for " + option + ": " + value);
    }
    return static_cast<int>(parsed);
}

Arguments parse_args(int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::runtime_error("missing option value");
        const std::string option = argv[index];
        const std::string value = argv[index + 1];
        if (option == "--input") args.input = value;
        else if (option == "--output") args.output = value;
        else if (option == "--summary") args.summary = value;
        else if (option == "--factor") args.factor = parse_positive(value, option);
        else if (option == "--threads") args.threads = parse_positive(value, option);
        else throw std::runtime_error("unknown option: " + option);
    }
    if (args.input.empty() || args.output.empty() || args.summary.empty()) {
        throw std::runtime_error("--input, --output, and --summary are required");
    }
    return args;
}

struct TiffGeometry {
    uint32_t width = 0;
    uint32_t height = 0;
    bool tiled = false;
    uint32_t tile_width = 0;
    uint32_t tile_height = 0;
    tmsize_t tile_bytes = 0;
    tmsize_t scanline_bytes = 0;
};

TiffGeometry inspect(TIFF* handle) {
    TiffGeometry geometry;
    uint16_t bits = 0;
    uint16_t samples = 0;
    uint16_t planar = PLANARCONFIG_CONTIG;
    uint16_t orientation = ORIENTATION_TOPLEFT;
    TIFFGetField(handle, TIFFTAG_IMAGEWIDTH, &geometry.width);
    TIFFGetField(handle, TIFFTAG_IMAGELENGTH, &geometry.height);
    TIFFGetField(handle, TIFFTAG_BITSPERSAMPLE, &bits);
    TIFFGetField(handle, TIFFTAG_SAMPLESPERPIXEL, &samples);
    TIFFGetFieldDefaulted(handle, TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(handle, TIFFTAG_ORIENTATION, &orientation);
    if (bits != 8 || samples != 3 || planar != PLANARCONFIG_CONTIG ||
        orientation != ORIENTATION_TOPLEFT) {
        throw std::runtime_error("input must be RGB8 contiguous, top-left TIFF");
    }
    geometry.tiled = TIFFIsTiled(handle);
    if (geometry.tiled) {
        TIFFGetField(handle, TIFFTAG_TILEWIDTH, &geometry.tile_width);
        TIFFGetField(handle, TIFFTAG_TILELENGTH, &geometry.tile_height);
        geometry.tile_bytes = TIFFTileSize(handle);
    } else {
        geometry.scanline_bytes = TIFFScanlineSize(handle);
        if (geometry.scanline_bytes < static_cast<tmsize_t>(geometry.width) * 3) {
            throw std::runtime_error("stripped TIFF scanline is shorter than RGB width");
        }
    }
    return geometry;
}

void decimate_tiled_worker(
    const fs::path& path,
    const TiffGeometry& geometry,
    int factor,
    cv::Mat* output,
    std::atomic<uint32_t>* next_tile_row
) {
    // 'm' disables libtiff's whole-file mmap. The source is an uncompressed
    // multi-gigabyte BigTIFF; bounded tile reads must not make the complete
    // file resident merely to retain a 16-fold sample.
    TIFF* handle = TIFFOpen(path.c_str(), "rm");
    if (handle == nullptr) throw std::runtime_error("cannot open input TIFF in worker");
    try {
        const uint32_t tile_rows =
            (geometry.height + geometry.tile_height - 1) / geometry.tile_height;
        const uint32_t tile_columns =
            (geometry.width + geometry.tile_width - 1) / geometry.tile_width;
        std::vector<unsigned char> tile(static_cast<size_t>(geometry.tile_bytes));
        while (true) {
            const uint32_t tile_row = next_tile_row->fetch_add(1);
            if (tile_row >= tile_rows) break;
            const uint32_t source_y0 = tile_row * geometry.tile_height;
            const uint32_t first_output_y = (source_y0 + factor - 1) / factor;
            const uint32_t source_y1 = std::min(source_y0 + geometry.tile_height, geometry.height);
            const uint32_t last_output_y = (source_y1 - 1) / factor;
            for (uint32_t tile_column = 0; tile_column < tile_columns; ++tile_column) {
                const uint32_t source_x0 = tile_column * geometry.tile_width;
                const ttile_t tile_id = TIFFComputeTile(handle, source_x0, source_y0, 0, 0);
                if (TIFFReadEncodedTile(handle, tile_id, tile.data(), geometry.tile_bytes) < 0) {
                    throw std::runtime_error("failed reading encoded TIFF tile");
                }
                const uint32_t source_x1 = std::min(source_x0 + geometry.tile_width, geometry.width);
                const uint32_t first_output_x = (source_x0 + factor - 1) / factor;
                const uint32_t last_output_x = (source_x1 - 1) / factor;
                for (uint32_t output_y = first_output_y; output_y <= last_output_y; ++output_y) {
                    const uint32_t source_y = output_y * factor;
                    if (source_y < source_y0 || source_y >= source_y1) continue;
                    cv::Vec3b* row = output->ptr<cv::Vec3b>(output_y);
                    const uint32_t local_y = source_y - source_y0;
                    for (uint32_t output_x = first_output_x; output_x <= last_output_x; ++output_x) {
                        const uint32_t source_x = output_x * factor;
                        if (source_x < source_x0 || source_x >= source_x1) continue;
                        const uint32_t local_x = source_x - source_x0;
                        const size_t offset =
                            (static_cast<size_t>(local_y) * geometry.tile_width + local_x) * 3;
                        // libtiff returns RGB; OpenCV encodes BGR.
                        row[output_x] = cv::Vec3b(
                            tile[offset + 2], tile[offset + 1], tile[offset]
                        );
                    }
                }
            }
        }
        TIFFClose(handle);
    } catch (...) {
        TIFFClose(handle);
        throw;
    }
}

void decimate_stripped_worker(
    const fs::path& path,
    const TiffGeometry& geometry,
    int factor,
    cv::Mat* output,
    std::atomic<uint32_t>* next_output_row
) {
    // Each worker owns a libtiff handle. RowsPerStrip=1 is common for
    // public scanner exports and makes direct sampled scanline reads bounded.
    TIFF* handle = TIFFOpen(path.c_str(), "rm");
    if (handle == nullptr) throw std::runtime_error("cannot open input TIFF in worker");
    try {
        std::vector<unsigned char> scanline(
            static_cast<size_t>(geometry.scanline_bytes)
        );
        while (true) {
            const uint32_t output_y = next_output_row->fetch_add(1);
            if (output_y >= static_cast<uint32_t>(output->rows)) break;
            const uint32_t source_y = output_y * factor;
            if (TIFFReadScanline(handle, scanline.data(), source_y, 0) < 0) {
                throw std::runtime_error("failed reading TIFF scanline");
            }
            cv::Vec3b* row = output->ptr<cv::Vec3b>(output_y);
            for (int output_x = 0; output_x < output->cols; ++output_x) {
                const uint32_t source_x = static_cast<uint32_t>(output_x) * factor;
                const size_t offset = static_cast<size_t>(source_x) * 3;
                // libtiff returns RGB; OpenCV encodes BGR.
                row[output_x] = cv::Vec3b(
                    scanline[offset + 2], scanline[offset + 1], scanline[offset]
                );
            }
        }
        TIFFClose(handle);
    } catch (...) {
        TIFFClose(handle);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        if (fs::exists(args.output) || fs::exists(args.summary)) {
            throw std::runtime_error("refusing to overwrite output or summary");
        }
        TIFF* primary = TIFFOpen(args.input.c_str(), "rm");
        if (primary == nullptr) throw std::runtime_error("cannot open input TIFF");
        const TiffGeometry geometry = inspect(primary);
        TIFFClose(primary);
        const int output_width = static_cast<int>((geometry.width + args.factor - 1) / args.factor);
        const int output_height = static_cast<int>((geometry.height + args.factor - 1) / args.factor);
        cv::Mat output(output_height, output_width, CV_8UC3, cv::Scalar(0, 0, 0));
        std::atomic<uint32_t> next_work_row{0};
        std::vector<std::thread> workers;
        std::vector<std::exception_ptr> failures(static_cast<size_t>(args.threads));
        const auto start = std::chrono::steady_clock::now();
        for (int thread_index = 0; thread_index < args.threads; ++thread_index) {
            workers.emplace_back([&, thread_index] {
                try {
                    if (geometry.tiled) {
                        decimate_tiled_worker(
                            args.input, geometry, args.factor, &output,
                            &next_work_row
                        );
                    } else {
                        decimate_stripped_worker(
                            args.input, geometry, args.factor, &output,
                            &next_work_row
                        );
                    }
                } catch (...) {
                    failures[thread_index] = std::current_exception();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
        fs::create_directories(args.output.parent_path());
        if (!cv::imwrite(args.output.string(), output)) {
            throw std::runtime_error("failed writing decimated TIFF");
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();
        std::ofstream summary(args.summary);
        summary << "{\n"
                << "  \"schema\": \"visium_hd_processing.tiff_decimation.v1\",\n"
                << "  \"input\": \"" << fs::absolute(args.input).string() << "\",\n"
                << "  \"output\": \"" << fs::absolute(args.output).string() << "\",\n"
                << "  \"source_shape_yx\": [" << geometry.height << ", " << geometry.width << "],\n"
                << "  \"output_shape_yx\": [" << output_height << ", " << output_width << "],\n"
                << "  \"factor\": " << args.factor << ",\n"
                << "  \"sampling\": \"direct_source_pixel_decimation\",\n"
                << "  \"source_storage\": \""
                << (geometry.tiled ? "tiled" : "stripped") << "\",\n"
                << "  \"source_pixel_for_output_xy\": \"source_xy = factor * output_xy\",\n"
                << "  \"threads\": " << args.threads << ",\n"
                << "  \"runtime_seconds\": " << seconds << "\n"
                << "}\n";
        std::cout << "source=" << geometry.width << "x" << geometry.height
                  << " output=" << output_width << "x" << output_height
                  << " seconds=" << seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "visium_hd_tiff_decimate: " << error.what() << '\n';
        return 1;
    }
}
