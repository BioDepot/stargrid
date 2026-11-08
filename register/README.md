# stargrid-register

Check, and correct, how a microscope image is registered to the Visium HD capture grid, using the
counts themselves.

On most Visium HD slides we have examined, the counts around cell nuclei follow a pattern: more
molecules in a ring of 2 µm capture locations just around each nucleus and, usually, fewer under
the nucleus itself. Moving a nucleus mask, segmented from the slide's H&E or DAPI image, over the
capture grid and recording the mean count under it therefore traces a dip, lower than the shifts
next to it, where the mask lies on the nuclei; moving a mask of the ring traces a peak at the same
shift. The comparison is local: it does not require nuclei to hold fewer counts than the rest of
the tissue. The check takes the dip where it is stable when the slide is resampled in spatial
blocks and, where a dip is present but weak, the peak of the ring if that is stable. In some
tissues the nuclei hold more counts than the tissue around them, so there is no dip; there the
check can, if asked to (`--disk`), move the nuclei and their ring together, the disk, and take its
peak. The dip, the ring and the disk are three forms of one pattern. The shift from
zero is how far the registration places the image from the counts, for the whole slide or for
regions of it, and it can be used to correct the registration. Where no shift is given, the first
registration stands.

The check needs only a 2 µm count matrix, the tissue flags and a nucleus mask on the grid. It works
on the output of [STARgrid](https://github.com/morphic-bio/STAR-suite), the open Visium HD pipeline
of STAR Suite, and on Space Ranger's output.

**Status: early (version 0.5.0).** The methods are those of the STARgrid manuscript (see Citation),
and the package reproduces its registration results exactly (`docs/REGRESSION.md`). The
command-line interface and output fields may still change.

This directory is the reference copy of the package. New methods are developed separately and
merged here when they are ready.

## What is here

| Part | What it does |
|---|---|
| `stargrid-register check` | The shift for the whole slide and for the regions of a 5 × 5 division: the deepest stable dip (shift, depth, stability under block resampling) or, where the dip is weak, the highest stable peak of the ring; with `--disk`, where there is no dip, the highest stable peak of the disk (the nuclei and their ring); no shift otherwise |
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
# Where do the nuclei lie relative to the counts? (the dip, or the ring where the dip is weak;
# add --disk to use the disk, the nuclei and their ring, where there is no dip)
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
check takes the deepest dip within 12 locations (24 µm) of zero in each direction. The dip is
usable if it lies inside the search (|dy| and |dx| below 12, not on its edge) and is stable.

Where a dip is present but not usable, the ring is used: the capture locations within
`--ring-width` locations (default 3, that is 6 µm, by Euclidean distance) of a capture location
under a nucleus that are not themselves under a nucleus. Its mask is moved in the same way, and its
highest peak, a shift at which the mean count is higher than at all eight neighbours, is taken if
it is inside the search and stable by the same rule. `--fallback none` turns the ring off.

Where there is no dip at all, the ring is not used and, by default, no shift is given. With
`--disk`, the disk is used: the nuclei and their ring together, the whole area of the dip and the
ring. Its sums are those of the nuclei plus those of the ring, so it is the same measurement
weighted differently. It is moved in the same way, and its highest peak is taken if it is inside
the search and stable by the same rule. Where the nuclei and the tissue just around them hold more
counts than the tissue further away, the mean count under the disk is highest where it lies on
them. Where a dip is present, even an unstable one, the disk is not used.

- `whole.status`: `usable` (a shift is given) or `not registered`. `whole.method`: `dip`, `ring`
  or `disk`, the surface that gave the shift. `fallback` and `disk` record the options used.
- `whole.dy`, `whole.dx` and `offset_um`: the shift; empty unless usable. Zero means the mask is
  in register with the counts, to the nearest location.
- `whole.dip_status`: `usable`, `no dip` (no shift is lower than all its neighbours) or `unstable`.
  `whole.ring_status` and `whole.disk_status`: `usable`, `no peak` or `unstable`; present only
  where that surface was measured.
  `reason` says why a surface was not used.
- `whole.depth`: the depth of the deepest dip, as a fraction. `depth_lo`, `depth_hi`: the 95% range
  of the depth at that shift over the resamples.
- `ring_dy`, `ring_dx`, `ring_height`, `ring_share_within_one`, `ring_width`: the highest peak of
  the ring, its height (the mean count there over the median at the eight neighbours, minus one)
  and its stability, recorded whenever the ring was measured, whether or not it passed;
  `disk_dy`, `disk_dx`, `disk_height` and `disk_share_within_one` the same for the disk, whose
  ring has the same `ring_width`.
- Stability: the slide is resampled 200 times in blocks of 100 × 100 locations (seed 20261007), the
  deepest dip (or highest peak) is found in each resample, and it is stable if it lies within one
  location of the full-data one in at least 95% of them. `share_within_one` is that share for the
  dip, `share_exact` the share at exactly the same shift, and `share_any_dip` the share of
  resamples with any dip.
- `candidate_dy`, `candidate_dx`: the deepest dip even when it is not usable, for inspection only;
  do not use it as a shift.
- `contrast_at_dip`: descriptive only. How far the count at the dip is below the median count at
  the shifts 3 to 5 locations away. No decision uses it.
- `regions`, `usable_regions`, `regions_at_zero`: the same in each region with at least 30,000
  nucleus locations in tissue (`status` `not tested` otherwise); `fit` uses only the usable
  regions, whichever surface gave their shift.

What has been tested:
- On lung tissue imaged by Xenium and Visium HD on the same section (four cores and two whole
  sections), the Xenium nuclei show where the nuclei lie. The dip of the nuclei segmented from the
  H&E image lay there in all four cores. On both whole sections it lay there but was not stable,
  and the ring gave that shift.
- Region by region, with nuclei from the H&E image or imaged by Xenium, the ring was never two or
  more locations from the true shift where it passed. On the cores and the first whole section,
  where the dip did not pass, the ring gave a shift in 7 of 44 cases, 6 of them exactly right. Its
  width of 3 locations was chosen after seeing these results (a width of 1 location, chosen first,
  did not meet this standard); on the second whole section, tested afterwards, it gave a shift in
  5 of 29 such cases, all exactly right.
- On the tumour slides of the manuscript, the peak of the ring equals the dip on every whole slide
  but one, where it is one location away, and is within one location of it in every region where
  both pass.
- On the lung cores and sections, the disk passed on six of the eight surfaces with no dip, four
  of them exactly at the true shift and two one location away. Wherever it passed, with or without
  a dip, it lay within one location of the true shift (27 surfaces, 21 of them exactly).
- On the public kidney and mouse-brain slides, which have no dip, the disk lies at Space Ranger 4's
  registration on the kidney slide and one location from it on the mouse-brain slide, and passes
  in all four regions with no dip, each within one location of it. On the public prostate slide,
  whose dip lies at zero shift, it passes in five of the six regions with no dip, each within one
  location of zero.
- On the tumour slides, wherever both the dip and the disk pass, they lie within one location of
  each other.

Cautions:
- The dip has been seen on colorectal, ovarian, lymph-node, prostate and lung tissue. On the public
  kidney and mouse-brain slides we examined there was no dip. No reference imaged on the same
  section has tested the disk on kidney or mouse-brain tissue. Do not use the shift of a slide
  whose status is not `usable`.
- On lung cores measured on the same section by Xenium and Visium HD, the local dip lay at the true
  shift, where the comparison against the shifts 3 to 5 locations away (the rule of versions 0.1
  and 0.2) gave a shift one or two locations away.
- Space Ranger 4.0 and later refine their registration against the counts by default, so agreement
  between this check and Space Ranger's registration is not independent of the counts.
- A cross-correlation peak between nuclei and counts lies on the rim of the dip, not at the
  nuclei. Use the dip, the ring or, where there is no dip, the disk.

## Changes in 0.5.0

- The ring is used only where a dip is present but not usable (`dip_status` `unstable`). Where
  there is no dip at all, 0.4.0 used the ring; 0.5.0 gives no shift there, and the first
  registration stands, unless `--disk` is given. On the manuscript's inputs this changes three
  surfaces, all on slides without a dip (the kidney slide as a whole and one of its regions, and
  one mouse-brain region), which are now not registered by default.
- `--disk` (on `check` and `fit`; `disk=True` in `dip.assess` and `dip.regions`) replaces
  `--fallback peak`: where there is no dip, the highest stable peak of the disk, the nuclei and
  their ring of `--ring-width` together (the experimental peak of 0.4.0 used the nuclei alone).
  `method` is `disk` (was `peak (experimental)`), its fields are `disk_*` (were `peak_*`) and
  `disk_status`. `--fallback` takes `ring` or `none`.

## Changes in 0.4.0

- The ring is used by default where the dip is not usable (`--fallback ring`, `--ring-width 3`).
  In 0.3.0 it was off by default (`--experimental-fallback fringe`); `--fallback none` gives the
  0.3.0 default. Where the dip is usable, results are unchanged.
- The highest peak is chosen by its height (the mean count over the median at its eight
  neighbours, minus one), as in the manuscript's analysis; 0.3.0 chose it on the reciprocal of the
  surface, which can pick a different peak in a few resamples.
- `status` is `usable` or `not registered`; the dip's own status is now `dip_status` and the ring's
  `ring_status`. `method` is `dip`, `ring` or `peak (experimental)`.
- Renamed: the ring's `peak_*` fields to `ring_*`, `fringe_width` to `ring_width`,
  `ring_contrast_at_dip` to `contrast_at_dip`; in the Python interface `dip.fringe_mask` to
  `dip.ring_mask`, `dip.FRINGE` to `dip.RING_WIDTH`, `dip.RING` to `dip.CONTRAST_RING`, and the
  `fringe` and `fringe_width` arguments of `dip.assess` and `dip.regions` to `ring` and
  `ring_width`.

## Citation

Hung L-H, Yeung KY. STARgrid: an open-source pipeline from Visium HD sequencing reads to count
matrices. bioRxiv 2026. doi:10.64898/2026.10.09.757882

## How this was written

Without reading any 10x Genomics source code. The package reads files that Space Ranger writes,
whose formats 10x Genomics documents, and uses no Space Ranger output in its own registration.
`docs/PROVENANCE.md` lists where each part comes from.

## Licence

MIT; see `LICENSE`.
