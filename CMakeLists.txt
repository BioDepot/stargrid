cmake_minimum_required(VERSION 3.20)
project(visium_hd_suite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VISIUM_HD_BUILD_REGISTRATION "Build the image registration and capture-grid tools (needs OpenCV and libtiff)" ON)
option(VISIUM_HD_BUILD_READ_TOOLS "Build the read-level comparison tools used by the manuscript (needs htslib and zlib)" OFF)

find_package(Threads REQUIRED)

if(VISIUM_HD_BUILD_REGISTRATION)
    find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d video)
    find_package(TIFF REQUIRED)
    function(visium_hd_release_compile_options target)
        target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
                "-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=.")
        endif()
    endfunction()
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
if(BUILD_TESTING AND VISIUM_HD_BUILD_REGISTRATION)
    add_executable(test_hd_registration_core native/tests/test_hd_registration_core.cpp)
    target_link_libraries(test_hd_registration_core PRIVATE hd_registration_core)
    add_test(NAME hd_registration_core_synthetic COMMAND test_hd_registration_core)

    add_executable(test_hd_capture_grid_core native/tests/test_hd_capture_grid_core.cpp)
    target_link_libraries(test_hd_capture_grid_core PRIVATE hd_capture_grid_core)
    add_test(NAME hd_capture_grid_core_synthetic COMMAND test_hd_capture_grid_core)
endif()
