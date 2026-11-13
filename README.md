# STARgrid: image registration, results and Visium HD tools

**STARgrid** is an open pipeline for Visium HD, from sequencing reads to count matrices registered
to the tissue image. It has four parts:

1. **Decoding the barcodes.** Each read is given every capture location whose two barcodes match it
   with the fewest edits.
2. **Assigning the gene**, by alignment to the genome (RNA-seq quantitation) or by probe matching
   (Flex), with the methods of STAR Suite.
3. **Assigning the counts.** Reads of one molecule are grouped before a location is chosen, and the
   locations still tied are resolved under a *resolution policy*; matrices are written at 2, 8 and
   16 µm.
4. **Registering the image**: the microscope image is placed on the capture grid, and the placement
   is then checked and corrected against the counts themselves.

The first three run inside STAR, in [STAR Suite](https://github.com/morphic-bio/STAR-suite). The
registration is in this repository, as C++ programs that run after STAR. This repository also holds
the evaluation code and results of our manuscript (L.-H. Hung and K. Y. Yeung, bioRxiv,
[doi:10.64898/2026.10.09.757882](https://doi.org/10.64898/2026.10.09.757882)) and other Visium HD
tools.

This repository was called `visium-hd-suite` until October 2026. The old address redirects here.

## Reproduce a slide

These steps run STARgrid on a slide of the manuscript from its public data, with the settings of the
manuscript, and check the result. The ovarian 3′ slide (RNA-seq quantitation) is shown; the
colorectal Flex slide differs only where noted.

**What you need**

- Linux on x86-64. For the ovarian slide: about 150 GB of disk for the data and the genome index,
  and about 170 GB more of temporary space for STAR (less for the colorectal slide). STAR needs at
  least 9 GB of memory besides the genome index (about 35 GB), and runs faster with more. Building
  the genome index needs about 40 GB.
- [STAR Suite](https://github.com/morphic-bio/STAR-suite) release 1.12.2 or later (the release
  binary), and a clone of its repository for the scripts.
- `visium_hd_v1.slide` from a Space Ranger installation (`lib/python/cellranger/barcodes/`). The
  barcode layout of the slides is 10x Genomics reference data and is not redistributed.
- This repository, with its C++ tools built (see [Install and test](#install-and-test)).
- For the nuclei: Cellpose 4.1.1 with the Cellpose-SAM weights (`cpsam`), preferably on a GPU. A
  container definition is in `containers/cellpose-paper/`; or `pip install cellpose==4.1.1 tifffile
  zarr imagecodecs opencv-python-headless`. Cellpose saves the weights in `~/.cellpose/models/` the
  first time it loads them:
  `python -c "from cellpose import models; models.CellposeModel(pretrained_model='cpsam')"`.

**1. Download the data.** `scripts/download_dataset.sh` downloads the reads, the H&E and CytAssist
images and the 10x Genomics reference GRCh38 2024-A, and checks each file against the checksums of
the copies we used:

```sh
scripts/download_dataset.sh ovarian data/        # or: colorectal
```

The registration also needs the layout file of the slide, which 10x Genomics provides by serial
number. Download it from their slide layout file page for slide `H1-YQJBZ7X` (colorectal:
`H1-GMHFWPH`), save it as `data/H1-YQJBZ7X.vlf`, and check it with the SHA-256 the script prints.

**2. Count.** For RNA-seq quantitation, build the genome index once, with the parameters of the
manuscript's index:

```sh
gunzip -c data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz > data/genes.gtf
STAR --runMode genomeGenerate --runThreadN 16 --genomeDir data/star_index \
    --genomeFastaFiles data/refdata-gex-GRCh38-2024-A/fasta/genome.fa \
    --sjdbGTFfile data/genes.gtf --genomeSAindexNbases 14 --genomeChrBinNbits 18 \
    --genomeSAsparseD 3 --limitGenomeGenerateRAM 110000000000
```

Then run STARgrid with the script from STAR Suite:

```sh
SLIDE=/path/to/spaceranger/lib/python/cellranger/barcodes/visium_hd_v1.slide
P=Visium_HD_3prime_Human_Ovarian_Cancer_FF_Min_Depth
STAR-suite/scripts/run_visium_hd.sh --mode gex --fastq-dir data/${P}_fastqs \
    --slide "$SLIDE" --genome-dir data/star_index --out-dir run --threads 16
```

For the colorectal Flex slide no genome index is needed; the probe cache is built from the probe
set published with the dataset and the GTF:

```sh
P=Visium_HD_6p5mm_Human_Colon_Cancer
STAR-suite/scripts/run_visium_hd.sh --mode flex --fastq-dir data/${P}_fastqs \
    --slide "$SLIDE" --flex-reference data/flex_reference --probe-set data/${P}_probe_set.csv \
    --gtf data/refdata-gex-GRCh38-2024-A/genes/genes.gtf.gz --out-dir run --threads 16
```

The matrices are in `run/SpatialGex.out` (Flex: `run/SpatialFlex.out`), one folder for each
resolution policy, each with `square_002um`, `square_008um` and `square_016um` in Matrix Market
format. The manuscript reports the counts under the `hard` policy. Check that yours are the same as
ours, file for file:

```sh
(cd run/SpatialGex.out && sha256sum -c ../../stargrid/docs/expected/ovarian_hard.sha256)
# colorectal: (cd run/SpatialFlex.out && sha256sum -c .../docs/expected/colorectal_hard.sha256)
```

**3. Register the image.**

```sh
scripts/run_registration.sh --he data/${P}_tissue_image.btf --cytassist data/${P}_image.tif \
    --layout data/H1-YQJBZ7X.vlf --area D1 --counts run/SpatialGex.out \
    --out-dir run/registration --threads 16
```

Add `--container IMAGE` to run Cellpose in a Docker image, or `--python PATH` for a Python
environment that has it. The script runs, in order:

| Step | Program | Output |
|---|---|---|
| H&E image reduced 16-fold for matching | `visium_hd_tiff_decimate` | `inputs/` |
| CytAssist image to the capture grid, by the fiducial marks and the slide's layout file | `visium_hd_capture_grid` | `capture_grid/` |
| H&E image to the CytAssist image, by image features (SIFT) and RANSAC | `visium_hd_register` | `registration/` |
| Nuclei segmented from the H&E image with Cellpose-SAM | `scripts/publication/segment_hd_he_cellpose.py` | `segmentation/` |
| Nuclei placed on the capture grid by the first two registrations | `visium_hd_count_register project` | `projection/` |
| The shift of the nuclei that best matches the counts, for the whole slide and its regions; the corrected positions | `visium_hd_count_register correct` | `registered/` |

The first stage uses the images only. The second uses the nuclei and the counts under the `hard`
policy over the part of the grid that the image covers; no Space Ranger output enters either. At
the end the script prints the shift it found and the correction it applied. `registered/` holds
`positions_002um.parquet`, `_008um` and `_016um` (the position of each bin of the matrices in the
H&E image, after the correction), `corrected_nuclei.npz` (the nuclei on the grid),
`check_before.json` and `check_after.json` (the shift for the whole slide and each region, before
and after the correction) and `fit.json` (the displacement fitted across the slide). Each step is
skipped if its output is complete, so a stopped run can be resumed. `--dry-run` prints the
commands; `--help` lists the options.

On the ovarian slide the script reports that the counts place the nuclei −4 rows and −3 columns of
capture locations (−8 and −6 µm) from where the first stage put them, and that the displacement
fitted across the slide predicts all ten held-out regions within one location. Given the nuclei of
the manuscript's timed runs, every output equals theirs. With Cellpose run again on a different GPU, the
nuclei differ slightly, but the whole-slide shift, the shift of each of the 25 regions and the
fitted displacement are the same. The run takes about ten minutes, most of it Cellpose on an 8 GB
GPU.

**Checks we ran on these steps.** The genome index built with the command above is byte-identical
to the manuscript's. On 100,000-read subsets of the ovarian and colorectal slides,
`run_visium_hd.sh` writes the same matrices as the commands recorded for the manuscript, and the
barcode inputs and probe cache it builds are byte-identical to ours. On the three slides of the
manuscript's timing, the first stage of `run_registration.sh` gives the same transforms as the
manuscript's runs.

## What else is here

| Task | Where | Notes |
|---|---|---|
| Registration, both stages | `native/` (`visium_hd_tiff_decimate`, `visium_hd_capture_grid`, `visium_hd_register`, `visium_hd_count_register`); `scripts/run_registration.sh` | C++. The count step is parallel and needs only the C++ standard library at its core, so that it can be built into STAR; reading and writing the files needs zlib and zstd. |
| Nucleus segmentation | `scripts/publication/segment_hd_he_cellpose.py`, `publication/methods/cellpose_profiles.yaml`, `containers/cellpose-paper/`; `scripts/segment_hd_he_stardist.py` | Cellpose-SAM is the default and was used in every analysis. StarDist, the control, gives comparable shifts. |
| The registration check as a Python package | `register/` (`stargrid-register`: `check`, `refine`, `fit`, `apply`, `place`) | The reference implementation, used for the manuscript's analyses; `visium_hd_count_register` gives the same results. It also works on Space Ranger's output. See `register/README.md`. |
| Quality measures for a slide | `scripts/measure_hd_qc_reference.py`, `scripts/summarize_hd_qc_reference.py` | Counts beyond the tissue and their decay with distance; the contrast of counts under nuclei. |
| Cell matrices from 2 µm counts | `scripts/build_hd_cell_matrix.py`, `validate_hd_cell_matrix_truth.py`, `stage_hd_output_matrix.py`, `compare_hd_cell_matrices.py` | Grows each nucleus by a fixed distance and sums the capture locations. |
| Evaluation code of the manuscript | `scripts/publication/`, and the other scripts in `scripts/` | Agreement with Space Ranger, H&E scoring, adjacent-section Xenium and CODEX. |
| Results of the manuscript | `paper_results/`, `paper/` | See below. |
| Datasets | `docs/DATASETS.md`, `scripts/download_dataset.sh` | Sources and licences of every dataset. |
| Parts of the reference model | `src/star_spatial/` | Only the modules the scripts above import. |

### How the second stage of the registration works

The first stage can place the image several micrometres from the counts without any sign of it in
its own fit. The second stage measures this from the counts. On most slides we examined, the counts
change with the distance from the edge of a nucleus: usually fewer under the nucleus and more in a
ring just around it. The capture locations are grouped by their distance from the edge of the
nearest nucleus, in bands one location (2 µm) wide, and each count is divided by the mean count
around it, so that only the local contrast remains. The nuclei are then moved over the grid, and
the shift kept is the one at which the groups explain the most variation in these counts (the
correlation ratio, from multimodal image registration; Roche et al., MICCAI 1998), provided it is
stable when the slide is resampled in spatial blocks. Where no shift is stable, the first stage
stands. The shift is found for the whole slide and for the regions of a 5 × 5 division, and a
displacement that varies linearly across the slide is fitted to alternate regions, tested on the
others and applied if they agree. `register/README.md` gives the details and how to read the
output.

`scripts/measure_hd_global_offset_fft.py` computes a cross-correlation peak, which lies on the
ring around the nuclei rather than on them. It is superseded and is kept only because other tools
import its readers.

## Results of the manuscript

```
paper/*_numbers.tex      every number in the manuscript, one macro each, with its source in comments
paper/figures/           the figures, their scripts (scripts/) and source data (data/)
paper/supplementary/     the generated supplementary tables and their script
paper_results/           the result records these are written from
```

These are the manuscript as submitted to Genome Research. `paper/regenerate.sh` writes every
number file, figure and generated supplementary table again from `paper_results/` and compares
each with the copy in `paper/`; `paper/README.md` lists which script writes which file and the few
inputs that are not in this repository. It needs Python 3.10 or later with numpy, pandas, scipy
and matplotlib. The results of version 1 of the preprint are kept in the tag
`preprint-v1-results`.

## What is not here

- **The decoding, gene assignment and count assignment**, which are part of STAR in STAR Suite.
- **Data.** No reads, images or count matrices, and of Space Ranger's output only the version,
  command line and summary metrics of one of our runs (`paper_results/number_inputs/`); the result
  tables are summaries computed from the data. `docs/DATASETS.md` gives the source and licence of
  each dataset.
- **A benchmark of cell-assignment methods** against known cell identities. It will be added.
- **`hd_candidate_preserving_reference`** cannot be built here yet: it includes the reference read
  decoder, whose source is not in this repository.

Scripts that reproduce particular runs, and paths inside result records, use placeholders such as
`<runs>/…`, `<datasets>/…`, `<storage>/…` and `<local>/…` in place of our own paths.

## Install and test

The C++ tools need a C++17 compiler, CMake 3.20 or later, OpenCV 4.5 or later, libtiff, zlib and
zstd:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

`scripts/run_registration.sh` finds them in `build/`. `-DVISIUM_HD_BUILD_REGISTRATION=OFF` builds
only `visium_hd_count_register`, without OpenCV and libtiff. Add `-DVISIUM_HD_BUILD_READ_TOOLS=ON`
for the read-level comparison tools, which also need htslib.

The Python scripts need Python 3.10 or later:

```sh
python3 -m pip install numpy scipy pandas h5py pyarrow anndata matplotlib scikit-learn \
    threadpoolctl opencv-python-headless tifffile zarr pytest
python3 -m pytest
```

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

Please cite: L.-H. Hung and K. Y. Yeung, STARgrid: an open-source pipeline from Visium HD
sequencing reads to count matrices, bioRxiv (2026),
[doi:10.64898/2026.10.09.757882](https://doi.org/10.64898/2026.10.09.757882), and STAR Suite.
