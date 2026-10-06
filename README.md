# Visium HD Suite

Tools for Visium HD spatial transcriptomics that sit around
[STAR Suite](https://github.com/morphic-bio/STAR-suite).

STAR Suite takes sequencing reads to count matrices at 2, 8 and 16 µm, including the decoding of
the spatial barcode. This repository holds the Visium HD code that is not part of STAR Suite:
placing the tissue image on the capture grid, detecting nuclei, checking that placement against
the counts, building cell matrices, measuring the quality of a slide, and the evaluation code that
compares the counts with Space Ranger's and with other assays.

**Status: early.** The layout mirrors our development repository. Most tools are command-line
scripts, each with `--help`; there is no stable Python API yet.

## What is here

| Task | Where | Notes |
|---|---|---|
| Place the microscope image on the capture grid | `native/` (`visium_hd_capture_grid`, `visium_hd_register`, `visium_hd_tiff_decimate`) | C++, built with CMake. Uses the images only, no counts. See the warning below. |
| Detect nuclei in the H&E image | `scripts/publication/segment_hd_he_cellpose.py`, `publication/methods/cellpose_profiles.yaml`, `containers/cellpose-paper/`; `scripts/segment_hd_he_stardist.py` | Cellpose, and StarDist at the settings of one published tool. Which detector and settings to prefer is still being evaluated. |
| Check where the nuclei lie relative to the counts | `scripts/measure_hd_qc_reference.py` | Works from output files: a count for each 2 µm square, a tissue mask and a nucleus mask. |
| Move nuclei into register | `scripts/publication/hd_mask_shift.py`, `fit_hd_mask_affine.py`, `check_hd_mask_affine.py` | A whole-square shift, or a displacement that varies linearly across the slide, fitted on some regions and checked on the others. |
| Quality measures for a slide | `scripts/measure_hd_qc_reference.py`, `scripts/summarize_hd_qc_reference.py` | Counts beyond the tissue and their decay with distance; the contrast of counts under nuclei. |
| Cell matrices from square counts | `scripts/build_hd_cell_matrix.py`, `validate_hd_cell_matrix_truth.py`, `stage_hd_output_matrix.py`, `compare_hd_cell_matrices.py` | Grows each nucleus by a fixed distance and sums the squares. Works on hard counts and on expected counts. |
| Evaluation code for our manuscript | `scripts/publication/`, and the other scripts in `scripts/` | Agreement with Space Ranger, H&E scoring, adjacent-section Xenium and CODEX. |
| Parts of the reference model | `src/star_spatial/` | Only the modules the scripts above import. |

### Check the placement of the nuclei

**The image registration here can be wrong without saying so.** On two of the three slides we
examined, it placed the nuclei several micrometres from where the counts place them, and on one of
those the error drifted across the slide. Its own fit statistics gave no warning. Run the check
before using image-derived nuclei with counts.

The check rests on one observation: on the slides we have measured, the count on squares under a
nucleus is lower than on the squares around it. So the mean count under the nucleus mask, as the
mask is moved over the grid, is lowest where mask and counts are in register, with a ring of
higher values around that position. `measure_hd_qc_reference.py` reports the shift at which the
mean is lowest relative to its ring, for the whole slide and for regions of it.

Two cautions:
- **A cross-correlation peak does not locate the nuclei.** It lies on the ring.
  `scripts/measure_hd_global_offset_fft.py` computes that peak. It is superseded and is here only
  because other tools import its readers.
- **The lower count under nuclei has been seen on a few slides only**, covering two chemistries.
  Do not assume it for a tissue that has not been measured.

## What is not here

- **The pipeline.** Reads to count matrices, including barcode decoding, is STAR Suite.
- **Data.** No reads, images, count matrices or vendor outputs. The barcode layout of a slide is
  read from reference files distributed with Space Ranger; obtain them under its licence.
- **Result tables and figures** of the manuscript. They are released with it.
- **A benchmark of cell-assignment methods** against known cell identities. It will be added.
- **`hd_candidate_preserving_reference`** cannot be built here yet: it includes the reference read
  decoder, whose source is not in this repository.

Scripts that reproduce particular runs name their inputs with placeholders such as `<runs>/…` and
`<datasets>/…`. Replace them with your own paths.

## Install and test

Python 3.10 or later.

```sh
python3 -m pip install numpy scipy pandas h5py pyarrow anndata matplotlib scikit-learn \
    threadpoolctl opencv-python-headless tifffile zarr pytest
python3 -m pytest
```

The image tools need a C++17 compiler, CMake 3.20 or later, OpenCV 4.5 or later and libtiff:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Add `-DVISIUM_HD_BUILD_READ_TOOLS=ON` for the read-level comparison tools, which also need htslib
and zlib. The segmentation scripts need Cellpose or StarDist; the container definition in
`containers/cellpose-paper/` records the Cellpose environment we used.

Run the Python tests from a git checkout: some tools record the source revision in their output.

## How this was written

No 10x Genomics source code was consulted in designing or implementing anything here. What was
used: the applicable licences, 10x Genomics' published documentation, reference files distributed
with the software, and output files, either published with datasets or from runs we performed.
Space Ranger's outputs are used as comparators, never as ground truth. The comparisons describe
observed output and make no statement about how Space Ranger produces it.

## Licence and citation

MIT; see `LICENSE`.

A manuscript describing the processing and its evaluation is in preparation. Until it is
available, please cite this repository and STAR Suite.
