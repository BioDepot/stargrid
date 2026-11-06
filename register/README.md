# stargrid-register

Check, and correct, how a microscope image is registered to the Visium HD capture grid, using the
counts themselves.

On the Visium HD tumour slides we have examined, the total count on 2 µm capture locations under
cell nuclei is, on average, lower than on the same locations moved a few micrometres. The
comparison is local: it does not require nuclei to hold fewer counts than the rest of the tissue.
Moving a nucleus mask, segmented from the slide's H&E or DAPI image, over the capture grid and
recording the mean count under it therefore traces a dip, lower than the shifts next to it, where
the mask lies on the nuclei. The shift of the deepest dip from zero is how far the registration
places the image from the counts, for the whole slide or for regions of it, and it can be used to
correct the registration. The shift is reported only if the dip is stable when the slide is
resampled in spatial blocks; where there is no dip, the check gives no shift.

The check needs only a 2 µm count matrix, the tissue flags and a nucleus mask on the grid. It works
on the output of [STARgrid](https://github.com/morphic-bio/STAR-suite), the open Visium HD pipeline
of STAR Suite, and on Space Ranger's output.

**Status: early (version 0.3.0).** The methods are those of the STARgrid manuscript (see Citation),
and the package reproduces its registration results exactly (`docs/REGRESSION.md`). The
command-line interface and output fields may still change.

This directory is the reference copy of the package. New methods are developed separately and
merged here when they are ready.

## What is here

| Part | What it does |
|---|---|
| `stargrid-register check` | The deepest dip for the whole slide (shift, depth, stability under block resampling) and for the regions of a 5 × 5 division, or no shift where there is no stable dip |
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
# Where do the nuclei lie relative to the counts? (--experimental-fallback: see below)
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

A dip is a shift at which the mean count under the mask is lower than at all eight neighbouring
shifts. Its depth is one minus that count over the median count at the eight neighbours. The
check takes the deepest dip within 12 locations (24 µm) of zero in each direction.

- `whole.status`: `usable`, `no dip` (no shift is lower than all its neighbours) or `unstable`.
  A shift is given only when the status is `usable` (or, with an experimental fallback below,
  `fringe peak (experimental)` or `peak (experimental)`). `whole.method` says which.
- `whole.dy`, `whole.dx` and `offset_um`: where the dip lies; empty unless usable. Zero means the
  mask is in register with the counts, to the nearest location.
- `whole.usable`: true if the deepest dip lies inside the search (|dy| and |dx| below 12, not on its
  edge) and is stable. A region must also hold at least 30,000 nucleus locations in tissue.
- `whole.depth`: the depth of the deepest dip, as a fraction. `depth_lo`, `depth_hi`: the 95% range
  of the depth at that shift over the resamples.
- Stability: the slide is resampled 200 times in blocks of 100 × 100 locations (seed 20261007), the
  deepest dip is found in each resample, and the dip is stable if it lies within one location of
  the full-data dip in at least 95% of them. `share_within_one` is that share, `share_exact` the
  share at exactly the same shift, and `share_any_dip` the share of resamples with any dip.
- `candidate_dy`, `candidate_dx`: the deepest dip even when it is not usable, for inspection only;
  do not use it as a shift.
- `ring_contrast_at_dip`: descriptive only. How far the count at the shift is below the median
  count 3 to 5 locations away. No decision uses it.
- `regions`, `usable_regions`, `regions_at_zero`: the same in each region with enough nuclei;
  `fit` uses only the usable regions.

Cautions:
- The dip has been seen on colorectal, ovarian, lymph-node, prostate and lung tissue. Its cause is
  not known. On the public kidney and mouse-brain slides we examined there was no dip; the check
  then gives no shift. Do not use the shift of a slide whose status is not `usable`.
- On lung cores measured on the same section by Xenium and Visium HD, the local dip lay at the true
  shift, where a comparison against a ring of shifts gave a peak one or two locations away.
- Space Ranger 4.0 and later refine their registration against the counts by default, so agreement
  between this check and Space Ranger's registration is not independent of the counts.
- A cross-correlation peak between nuclei and counts lies on the ring around the dip, not at the
  nuclei. Use the dip.

## Experimental fallbacks, where there is no usable dip

For research only, and not part of the method of the STARgrid manuscript. `--experimental-fallback`
(on `check` and `fit`; off by default) gives a shift where the dip does not, accepted by the same
stability test as a dip. `status` and `method` then say `fringe peak (experimental)` or
`peak (experimental)`, and `peak_dy`, `peak_dx`, `peak_height` and `peak_share_within_one` are
recorded whether or not the fallback passes. A usable dip is never replaced.

- `fringe`: where there is no usable dip, the mean count under the band around the nuclei (the
  locations within `--fringe-width` locations of a nucleus, default 3, that is 0 to 6 µm outside its
  edge) is followed as the band is moved, and its highest stable peak is taken. Where counts are
  higher just around nuclei than under them, the band is highest when it lies on that band, even
  where the dip under the nuclei is too weak to pass.
- `peak`: where the surface of the nuclei has no dip at all, its highest stable peak. An unstable
  dip never falls back to it, because the highest peak then lies on the rim of the dip.

What has been tested:
- On lung tissue imaged by Xenium and Visium HD on the same section (four cores and one whole
  section), with nuclei from the H&E image or imaged by Xenium, the fringe peak (width 3) was never
  two or more locations from the true shift where it passed, on whole sections or regions. Where the dip did not pass, it
  gave a shift in 7 of 44 cases, 6 of them exactly right, including one whole section whose dip
  lay at the true shift but was not stable. A width of 1 location, chosen first, did not meet this
  standard; the width of 3 was chosen after seeing these results.
- On the tumour slides of the manuscript, the fringe peak equals the dip on every whole slide but
  one, where it is one location away, and is within one location of it in every region where
  both pass.
- On the public kidney and mouse-brain slides, which have no dip, the fringe peak moves with its
  width, and the peak of the nuclei is within one location of Space Ranger 4's registration. No
  independent reference has tested either there, and Space Ranger 4's registration also uses
  the counts.

Treat a shift from either fallback as a hypothesis to check against an independent reference,
such as nuclei imaged on the same section.

## Citation

Hung L-H, Yeung KY. STARgrid: an open-source pipeline from Visium HD sequencing reads to count
matrices. bioRxiv 2026. doi:10.64898/2026.10.09.757882

## How this was written

Without reading any 10x Genomics source code. The package reads files that Space Ranger writes,
whose formats 10x Genomics documents, and uses no Space Ranger output in its own registration.
`docs/PROVENANCE.md` lists where each part comes from.

## Licence

MIT; see `LICENSE`.
