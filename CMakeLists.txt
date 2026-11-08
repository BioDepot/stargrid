cmake_minimum_required(VERSION 3.20)
project(visium_hd_suite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VISIUM_HD_BUILD_REGISTRATION "Build the image registration and capture-grid tools (needs OpenCV and libtiff)" ON)
option(VISIUM_HD_BUILD_READ_TOOLS "Build the read-level comparison tools used by the manuscript (needs htslib and zlib)" OFF)
option(VISIUM_HD_BUILD_COUNT_REGISTRATION "Build the projection of nuclei onto the grid and the registration check and correction from the counts (needs zlib and zstd)" ON)

find_package(Threads REQUIRED)

function(visium_hd_release_compile_options target)
    target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
            "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=.")
    endif()
endfunction()

if(VISIUM_HD_BUILD_COUNT_REGISTRATION)
    find_package(ZLIB REQUIRED)
    # The check and correction need only the C++ standard library and threads, so that they can be
    # built into STAR; the file formats (Matrix Market, .npy/.npz, Parquet) add zlib.
    add_library(hd_count_registration_core native/hd_count_registration.cpp native/hd_json.cpp)
    target_include_directories(hd_count_registration_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native)
    target_link_libraries(hd_count_registration_core PUBLIC Threads::Threads)
    visium_hd_release_compile_options(hd_count_registration_core)

    add_library(hd_count_io native/hd_mex.cpp native/hd_npy.cpp native/hd_parquet.cpp)
    target_include_directories(hd_count_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native)
    target_link_libraries(hd_count_io PUBLIC Threads::Threads ZLIB::ZLIB)
    visium_hd_release_compile_options(hd_count_io)

    # Reading Cellpose's zarr v3 label images needs zstd.
    find_path(ZSTD_INCLUDE_DIR zstd.h REQUIRED)
    find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
    add_library(hd_projection native/hd_projection.cpp native/hd_sha256.cpp)
    target_include_directories(hd_projection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native PRIVATE ${ZSTD_INCLUDE_DIR})
    target_link_libraries(hd_projection PUBLIC hd_count_registration_core PRIVATE ${ZSTD_LIBRARY})
    visium_hd_release_compile_options(hd_projection)

    add_executable(visium_hd_count_register native/visium_hd_count_register.cpp)
    target_link_libraries(visium_hd_count_register PRIVATE hd_count_registration_core hd_count_io hd_projection)
    visium_hd_release_compile_options(visium_hd_count_register)
endif()

if(VISIUM_HD_BUILD_REGISTRATION)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d video)
    find_package(TIFF REQUIRED)
    add_library(hd_registration_core native/hd_registration_core.cpp)
    target_include_directories(hd_registration_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native)
    target_link_libraries(hd_registration_core PUBLIC ${OpenCV_LIBS})
    visium_hd_release_compile_options(hd_registration_core)

    add_executable(visium_hd_register native/visium_hd_register.cpp)
    target_link_libraries(visium_hd_register PRIVATE hd_registration_core)
    visium_hd_release_compile_options(visium_hd_register)

    add_executable(visium_hd_registration_pixel_trace native/visium_hd_registration_pixel_trace.cpp)
    target_link_libraries(visium_hd_registration_pixel_trace PRIVATE hd_registration_core TIFF::TIFF)
    visium_hd_release_compile_options(visium_hd_registration_pixel_trace)

    add_library(hd_capture_grid_core native/hd_capture_grid_core.cpp)
    target_include_directories(hd_capture_grid_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/native)
    target_link_libraries(hd_capture_grid_core PUBLIC ${OpenCV_LIBS})
    visium_hd_release_compile_options(hd_capture_grid_core)

    add_executable(visium_hd_capture_grid native/visium_hd_capture_grid.cpp)
    target_link_libraries(visium_hd_capture_grid PRIVATE hd_capture_grid_core TIFF::TIFF)
    visium_hd_release_compile_options(visium_hd_capture_grid)

    add_executable(visium_hd_tiff_decimate native/visium_hd_tiff_decimate.cpp)
    target_link_libraries(visium_hd_tiff_decimate PRIVATE ${OpenCV_LIBS} TIFF::TIFF Threads::Threads)
    visium_hd_release_compile_options(visium_hd_tiff_decimate)
endif()

if(VISIUM_HD_BUILD_READ_TOOLS)
    find_package(PkgConfig REQUIRED)
    find_package(ZLIB REQUIRED)
    pkg_check_modules(HTSLIB REQUIRED IMPORTED_TARGET htslib)

    # native/hd_candidate_preserving_reference.cpp is not built here: it includes the reference
    # read decoder, whose source is not in this repository yet.

    add_executable(hd_sr_disagreement_extract native/hd_sr_disagreement_extract.cpp)
    target_link_libraries(hd_sr_disagreement_extract PRIVATE ZLIB::ZLIB)
    target_compile_options(hd_sr_disagreement_extract PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)

    add_executable(hd_sr_disagreement_raw_edit_audit native/hd_sr_disagreement_raw_edit_audit.cpp)
    target_link_libraries(hd_sr_disagreement_raw_edit_audit PRIVATE PkgConfig::HTSLIB Threads::Threads ZLIB::ZLIB)
    target_compile_options(hd_sr_disagreement_raw_edit_audit PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
endif()

include(CTest)
if(BUILD_TESTING AND VISIUM_HD_BUILD_COUNT_REGISTRATION)
    add_executable(test_hd_count_registration native/tests/test_hd_count_registration.cpp)
    target_link_libraries(test_hd_count_registration PRIVATE hd_count_registration_core hd_count_io hd_projection ${ZSTD_LIBRARY})
    target_include_directories(test_hd_count_registration PRIVATE ${ZSTD_INCLUDE_DIR})
    add_test(NAME hd_count_registration_synthetic COMMAND test_hd_count_registration)
endif()
if(BUILD_TESTING AND VISIUM_HD_BUILD_REGISTRATION)
    add_executable(test_hd_registration_core native/tests/test_hd_registration_core.cpp)
    target_link_libraries(test_hd_registration_core PRIVATE hd_registration_core)
    add_test(NAME hd_registration_core_synthetic COMMAND test_hd_registration_core)

    add_executable(test_hd_capture_grid_core native/tests/test_hd_capture_grid_core.cpp)
    target_link_libraries(test_hd_capture_grid_core PRIVATE hd_capture_grid_core)
    add_test(NAME hd_capture_grid_core_synthetic COMMAND test_hd_capture_grid_core)
endif()
