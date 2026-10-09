# STARgrid: results and Visium HD tools

**STARgrid** is the Visium HD pipeline of
[STAR Suite](https://github.com/morphic-bio/STAR-suite). It takes sequencing reads to count
matrices at 2, 8 and 16 µm, with Flex or RNA-seq quantitation.

**The pipeline itself is not in this repository. It is part of STAR Suite**, and is installed and
run from there. This repository is its companion. It holds:

- the result tables, numbers and figure generators of our manuscript on STARgrid (L.-H. Hung and
  K. Y. Yeung; in preparation);
- the evaluation code that compares STARgrid's counts with Space Ranger's and with other assays;
- the Visium HD tools that are not part of STAR Suite: placing the tissue image on the capture
  grid, detecting nuclei, checking that placement against the counts, building cell matrices and
  measuring the quality of a slide.

This repository was called `visium-hd-suite` until October 2026. The old address redirects here.

**Status: early.** The layout mirrors our development repository. Most tools are command-line
scripts, each with `--help`; there is no stable Python API yet.

## What STARgrid does

Each capture location of a Visium HD slide has two barcodes, one for its column and one for its
row. They are read one after the other in Read 1, and their lengths vary, so nothing in the read
marks where the first ends and the second begins. STARgrid:

1. **Decodes the barcodes.** It finds every capture location whose two barcodes match the read
   with the fewest edits (substituted, inserted or deleted bases), allowing up to two edits in
   each barcode. When several locations match equally well it keeps them all.
2. **Assigns the gene**, by alignment to the genome for RNA-seq quantitation or by probe matching
   for Flex quantitation, with the methods of STAR Suite.
3. **Groups the reads of each molecule** (same gene, same UMI, a location in common) and chooses
   one location for the group from all its reads.
4. **Counts molecules** after correcting UMIs at each location, and writes matrices at 2, 8 and
   16 µm.

What is counted when a molecule still has more than one possible location is set by a
*resolution policy*. The manuscript reports the HARD policy, which assigns each molecule to its
most probable location, and uses the STRICT policy, which drops such molecules, as a conservative
control.

## What is here

| Task | Where | Notes |
|---|---|---|
| Place the microscope image on the capture grid | `native/` (`visium_hd_capture_grid`, `visium_hd_register`, `visium_hd_tiff_decimate`) | C++, built with CMake. Uses the images only, no counts. See the warning below. |
| Detect nuclei in the H&E image | `scripts/publication/segment_hd_he_cellpose.py`, `publication/methods/cellpose_profiles.yaml`, `containers/cellpose-paper/`; `scripts/segment_hd_he_stardist.py` | Cellpose, and StarDist at the settings of one published tool. Which detector and settings to prefer is still being evaluated. |
| Check where the nuclei lie relative to the counts | `scripts/measure_hd_qc_reference.py` | Works from output files: a count for each 2 µm capture location, a tissue mask and a nucleus mask. |
| Move nuclei into register | `scripts/publication/hd_mask_shift.py`, `fit_hd_mask_affine.py`, `check_hd_mask_affine.py` | A shift by whole capture locations, or a displacement that varies linearly across the slide, fitted on some regions and checked on the others. |
| Quality measures for a slide | `scripts/measure_hd_qc_reference.py`, `scripts/summarize_hd_qc_reference.py` | Counts beyond the tissue and their decay with distance; the contrast of counts under nuclei. |
| Cell matrices from 2 µm counts | `scripts/build_hd_cell_matrix.py`, `validate_hd_cell_matrix_truth.py`, `stage_hd_output_matrix.py`, `compare_hd_cell_matrices.py` | Grows each nucleus by a fixed distance and sums the capture locations. Works on the counts under the HARD policy and under the expected-count policy. |
| Evaluation code for our manuscript | `scripts/publication/`, and the other scripts in `scripts/` | Agreement with Space Ranger, H&E scoring, adjacent-section Xenium and CODEX. |
| Results of our manuscript | `paper_results/`, `paper/`, `docs/DATASETS.md` | Result tables, numbers, and the figure and table generators. See below. |
| Parts of the reference model | `src/star_spatial/` | Only the modules the scripts above import. |

### Check the placement of the nuclei

**The image registration here can be wrong without saying so.** On two of the three slides we
examined, it placed the nuclei several micrometres from where the counts place them, and on one of
those the error drifted across the slide. Its own fit statistics gave no warning. Run the check
before using image-derived nuclei with counts.

The check rests on one observation: on the slides we have measured, the count on capture
locations under a nucleus is lower than on those around it. So the mean count under the nucleus
mask, as the mask is moved over the grid, is lowest where mask and counts are in register, with a
ring of higher values around that position. `measure_hd_qc_reference.py` reports the shift at
which the mean is lowest relative to its ring, for the whole slide and for regions of it.

Two cautions:
- **A cross-correlation peak does not locate the nuclei.** It lies on the ring.
  `scripts/measure_hd_global_offset_fft.py` computes that peak. It is superseded and is here only
  because other tools import its readers.
- **The lower count under nuclei has been seen on a few slides only**, covering both kinds of
  quantitation. Do not assume it for a tissue that has not been measured.

## Results of the manuscript

```
paper_results/spatial_v1_9_5_20260915/   accepted result tables for the three slides; its README lists them
paper_results/frame_edit_distance_sr_taxonomy_20260720/   examination of Space Ranger's recorded locations
paper_results/xenium_ovarian_adjacent_20251007/           Xenium dataset record (files, sizes, MD5)
paper/numbers.tex and *_numbers.tex      numbers used in the manuscript, one macro each
paper/figures/                           figures, their generators (scripts/) and source data (data/)
paper/supplementary/                     Additional file 1 and its generator
docs/DATASETS.md                         datasets, sources and licences
```

The analyses were run with STAR Suite v1.9.5 (commit `c95c57d`). The cited release, v1.9.5.a,
writes byte-identical matrices on the four official downsampled spatial checks.

Run a generator from the repository root, for example
`python3 paper/figures/scripts/fig2_agreement.py /tmp/out`; `paper/figures/scripts/README.md` lists
them. They need numpy, pandas and matplotlib. `paper/numbers.tex` is written by
`scripts/publication/export_spatial_paper_numbers.py` from the accepted tables, and each macro
records the result it came from.

The H&E tables include the placement corrections described above: a shift of the masks by whole
capture locations on one slide and a fitted displacement on another.

The manuscript is still being revised, and the figures and supplementary tables here can lag
behind it.

## What is not here

- **The pipeline.** STARgrid, from reads to count matrices including barcode decoding, is in
  STAR Suite.
- **Data.** No reads, images, count matrices or vendor outputs; the result tables are summaries
  computed from them. `docs/DATASETS.md` gives the source and licence of each dataset. The barcode
  layout of a slide is read from reference files distributed with Space Ranger; obtain them under
  its licence.
- **A benchmark of cell-assignment methods** against known cell identities. It will be added.
- **`hd_candidate_preserving_reference`** cannot be built here yet: it includes the reference read
  decoder, whose source is not in this repository.

Scripts that reproduce particular runs, and paths inside result records, use placeholders such as
`<runs>/…`, `<datasets>/…`, `<storage>/…` and `<local>/…` in place of our own paths.

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

The Python package and the CMake project still carry the repository's earlier name
(`visium-hd-suite`, `visium_hd_suite`).

## How this was written

We wrote STARgrid and the tools here without reading any 10x Genomics source code. We used only
the licences, 10x Genomics' published documentation, reference files distributed with Space
Ranger, and Space Ranger's output, either published with datasets or from our own runs. We used
that output to check our results.

## Licence and citation

MIT; see `LICENSE`.

The manuscript is in preparation. Until it is available, please cite this repository and STAR
Suite.
