# stargrid-register

Check, and correct, how a microscope image is registered to the Visium HD capture grid, using the
counts themselves.

On the Visium HD tumour slides we have examined, the total count on 2 µm capture locations under
cell nuclei is, on average, lower than on the same locations moved a few micrometres. The
comparison is local: it does not require nuclei to hold fewer counts than the rest of the tissue.
Moving a nucleus mask, segmented from the slide's H&E or DAPI image, over the capture grid and
recording the mean count under it therefore traces a dip, lower than the ring of shifts around it,
where the mask lies on the nuclei. The shift of the dip from zero is how far the registration
places the image from the counts, for the whole slide or for regions of it, and it can be used to
correct the registration.

The check needs only a 2 µm count matrix, the tissue flags and a nucleus mask on the grid. It works
on the output of [STARgrid](https://github.com/morphic-bio/STAR-suite), the open Visium HD pipeline
of STAR Suite, and on Space Ranger's output.

**Status: early (version 0.2.0).** The methods are those of the STARgrid manuscript (see Citation),
and the package reproduces its registration results exactly (`docs/REGRESSION.md`). The
command-line interface and output fields may still change.

This directory is the reference copy of the package. New methods are developed separately and
merged here when they are ready.

## What is here

| Part | What it does |
|---|---|
| `stargrid-register check` | The dip for the whole slide (shift, contrast against the ring 3 to 5 locations away, profile) and for the regions of a 5 × 5 division |
| `stargrid-register refine` | The same displacement by band-pass normalised cross-correlation or mutual information, with a sub-location estimate |
| `stargrid-register fit` | A displacement that varies linearly across the slide, fitted to alternate regions and accepted only if it predicts the regions left out |
| `stargrid-register apply` | Moves a nucleus label grid by a shift or by the fitted displacement |
| `stargrid-register place` | Places nucleus labels segmented from the image on the grid with Space Ranger's registration, read from its 2 µm (or 8 µm) `tissue_positions` table |
| `../native/` (C++) | Our image registration, from the two images of a slide and no counts: the microscope image to the CytAssist image by image features, and the CytAssist image to the grid by the fiducial marks of the slide design (`visium_hd_register`, `visium_hd_capture_grid`, `visium_hd_tiff_decimate`); built with the repository's top-level `CMakeLists.txt` |

Nucleus segmentation itself (Cellpose, StarDist) is not part of this package; the repository's
`scripts/` hold the segmentation used in the manuscript.

The module `other_methods` keeps three further registration measures (phase correlation, ECC
and SimpleITK mutual information, with the settings used in the manuscript's tests) for later
work. They did not meet the manuscript's criteria and the command-line tool does not use them;
they need `pip install './register[other]'`.

## Install

Python 3.10 or later:

```sh
python3 -m pip install "git+https://github.com/BioDepot/stargrid#subdirectory=register"
# or, from a checkout of this repository:
python3 -m pip install -e './register[test]' && (cd register && pytest)
```

The image-registration tools need a C++17 compiler, CMake 3.20 or later, OpenCV 4.5 or later and
libtiff; build them from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Use

```sh
# Where do the nuclei lie relative to the counts?
stargrid-register check \
    --counts outs/binned_outputs/square_002um/filtered_feature_bc_matrix.h5 \
    --tissue outs/binned_outputs/square_002um/spatial/tissue_positions.parquet \
    --nuclei nuclei_on_grid.npz --out check.json

# Confirm with a second measure, and fit an offset that varies across the slide
stargrid-register refine --method ncc --counts ... --tissue ... --nuclei ... --out refine.json
stargrid-register fit --counts ... --tissue ... --nuclei ... --out fit.json

# Correct the mask
stargrid-register apply --nuclei nuclei_on_grid.npz --shift=-3,-3 --out-grid corrected.npz --out apply.json
stargrid-register apply --nuclei nuclei_on_grid.npz --affine fit.json --out-grid corrected.npz --out apply.json

# Place image-derived nucleus labels with Space Ranger's registration
stargrid-register place --tissue-positions tissue_positions.parquet --labels nucleus_labels.npy \
    --image-scale 0.5 --out-grid nuclei_on_grid.npz --out place.json
```

`place` samples the label image at the nearest pixel by default. `--sampling archived-floor`
rounds positions down to whole pixels, as the manuscript's runs did; with `--image-to-labels
frame.json` (a 3 × 3 matrix from full-resolution image pixels to label-image pixels) it reproduces
their placements exactly. The two conventions gave the same shifts on every slide tested.

Inputs:
- **Counts**: a feature-barcode matrix in HDF5, a Matrix Market directory (STARgrid's or Space
  Ranger's `square_002um` output), or `.npz` with an array `total` of length 3,350 × 3,350.
- **Tissue**: the 2 µm `tissue_positions.parquet`, or `.npz` with a boolean array `tissue`.
- **Nuclei**: a label grid (`.npz`/`.npy`, 0 for background) of 3,350 × 3,350, or Space Ranger's
  `barcode_mappings.parquet` (its own nucleus map).

Displacements are (dy, dx) in capture locations of 2 µm, rows downward, from mask to counts: a
nucleus at (row, col) lies over the counts at (row + dy, col + dx).

## Reading the result

- `whole.dy`, `whole.dx` and `offset_um`: where the dip lies. Zero means the mask is in register
  with the counts, to the nearest location.
- `whole.contrast`: how far the count under the mask is below its ring, as a fraction; positive for
  a dip. On the tumour slides examined so far it was 0.047 to 0.156 with the masks in register.
- `whole.border`: the strongest feature lies on the edge of the search; widen `--search` or treat
  the result as unresolved.
- `regions`, `usable_regions`, `regions_at_zero`: the same in each region with enough nuclei.
- `local_dip`: a weaker criterion, the deepest shift whose count is below all eight neighbouring
  shifts, for slides where the count under nuclei is lower than at the neighbouring shifts but
  not than further away (as on the lung cores of the manuscript).

Cautions:
- The dip has been seen on a small number of slides, from colorectal and ovarian tumours, with
  both probe-based and 3′ quantitation; on lung cores it was weaker and only local. Its cause is
  not known. Check that a slide shows a clear dip before relying on the offset.
- Space Ranger 4.0 and later refine their registration against the counts by default, so agreement
  between this check and Space Ranger's registration is not independent of the counts.
- A cross-correlation peak between nuclei and counts lies on the ring around the dip, not at the
  nuclei. Use the dip.

## Citation

Hung L-H, Yeung KY. STARgrid: an open-source pipeline from Visium HD sequencing reads to count
matrices. bioRxiv 2026. doi:10.64898/2026.10.09.757882

## How this was written

Without reading any 10x Genomics source code. The package reads files that Space Ranger writes,
whose formats 10x Genomics documents, and uses no Space Ranger output in its own registration.
`docs/PROVENANCE.md` lists where each part comes from.

## Licence

MIT; see `LICENSE`.
