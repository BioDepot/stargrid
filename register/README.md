# stargrid-register

Check, and correct, how a microscope image is registered to the Visium HD capture grid, using the
counts themselves.

On most Visium HD slides we have examined, the counts around cell nuclei follow a pattern that
changes with the distance from the nucleus edge: usually more molecules in a ring of 2 µm capture
locations just around each nucleus and fewer under the nucleus itself (a dip), and in some tissues
more under the nuclei and their ring than farther away. The check groups the capture locations by
their distance from the edge of the nearest nucleus, segmented from the slide's H&E or DAPI image,
divides each count by the mean count around it so that only the local contrast remains, and moves
the nuclei to the shift at which the groups explain the most variation in these counts: the
correlation ratio, a similarity measure from multimodal image registration (Roche et al., MICCAI
1998). It finds the shift where the dip is clear, where it is weak and where there is none. The
shift is accepted where it is stable when the slide is resampled in spatial blocks. The shift from
zero is how far the registration places the image from the counts, for the whole slide or for
regions of it, and it can be used to correct the registration. Where no shift is given, the first
registration stands.

The check needs only a 2 µm count matrix, the tissue flags and a nucleus mask on the grid. It works
on the output of [STARgrid](https://github.com/morphic-bio/STAR-suite), the open Visium HD pipeline
of STAR Suite, and on Space Ranger's output.

**Status: early (version 0.6.0).** The methods are those of the STARgrid manuscript (see Citation),
and the package reproduces its registration results exactly (`docs/REGRESSION.md`). The
command-line interface and output fields may still change.

This directory is the reference copy of the package. New methods are developed separately and
merged here when they are ready.

## What is here

| Part | What it does |
|---|---|
| `stargrid-register check` | The shift for the whole slide and for the regions of a 5 × 5 division: the highest stable maximum of the correlation ratio between the counts and the distance from the nucleus edge (shift, stability under block resampling, and the profile of counts across the edge at the shift); no shift otherwise |
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

Each capture location of the nucleus map is put in a band by its Euclidean distance from the nucleus
edge, one location (2 µm) wide: three inside the nuclei (at most 1, 1 to 2 and more than 2 locations
from the edge) and six outside (0 to 1, ..., 5 to 6 locations from a nucleus); the other locations in
tissue form one more group. Each count is divided by the mean count of the tissue locations in the
square of 2 × `--baseline` + 1 locations around it (default 5 × 5, ±4 µm). At each shift of the
nucleus map the correlation ratio η² is the share of the variance of these divided counts, over the
tissue, that the group means explain. The check takes the highest local maximum of η² (higher than at
all eight neighbouring shifts) within 12 locations (24 µm) of zero in each direction. It is usable if
it lies inside the search (|dy| and |dx| below 12, not on its edge) and is stable.

- `whole.status`: `usable` (a shift is given) or `not registered`; `whole.method`: `correlation ratio`.
- `whole.dy`, `whole.dx` and `offset_um`: the shift; empty unless usable. Zero means the mask is in
  register with the counts, to the nearest location.
- `whole.eta2`, `whole.eta2_at_zero`: η² at the shift and at zero shift.
- `whole.profile`: the mean divided count in each band (`I3`, `I2`, `I1` inside, deepest first; `O1` to
  `O6` outside) and in the rest of the tissue (`far`), at the shift. Below 1 inside and above 1 just
  outside is the dip and the ring; above 1 inside is the pattern of tissues without a dip.
- Stability: the slide is resampled 200 times in blocks of 100 × 100 locations (seed 20261007), the
  highest local maximum is found in each resample, and the shift is stable if it lies within one
  location of the full-data one in at least 95% of them. `share_within_one` is that share,
  `share_exact` the share at exactly the same shift, and `share_any_peak` the share of resamples with
  any local maximum.
- `candidate_dy`, `candidate_dx`: the highest local maximum even when it is not usable, for inspection
  only; do not use it as a shift. `reason` says why a shift was not given.
- `baseline`: the half-width of the window of the local mean, in locations.
- `regions`, `usable_regions`, `regions_at_zero`: the same in each region with at least 30,000
  nucleus locations in tissue (`status` `not tested` otherwise); a location counts in the region it
  lies in, under whichever band the shift brings onto it. `fit` uses only the usable regions.

The dip itself (the mean count under the nuclei at each shift) is still measured by the functions
of `dip` (`shift_surface`, `local_dip`, `feature`), for description.

What has been tested (the STARgrid manuscript's inputs; `docs/REGRESSION.md`):
- On lung tissue imaged by Xenium and Visium HD on the same section (four cores and two whole
  sections, with their regions; nuclei from the H&E image and imaged by Xenium: 112 surfaces), the
  Xenium nuclei show where the nuclei lie. The check gave a shift on 100 of them, 81 at that position,
  18 one location away and 1 two locations away; on every core and whole section, the H&E nuclei were
  moved onto the Xenium nuclei. The dip alone gave a shift on 39 of the 112.
- On the tumour slides of the manuscript, under each registration tested, it gives the shift of the
  dip on every whole slide, and wherever both are usable it lies within one location of the dip (the
  same shift on 237 of 256 slides and regions).
- On four public Space Ranger 4.0.1 slides (lymph node, prostate cancer, kidney, mouse brain), it gave
  a shift on 64 of the 69 slides and regions, each within one location of Space Ranger 4's
  registration, including the kidney and mouse-brain slides, which have no dip.
- The window of the local mean was chosen from half-widths of 1 to 30 locations: 2 locations gave the
  most shifts at the Xenium position on the lung tissue; from 10 locations up, a growing number of
  tumour regions lay two or more locations from the dip, because a wide window passes the density of
  the tissue on to the correlation ratio.

Cautions:
- Nuclei imaged on the same section are the only independent reference, and the public datasets that
  have them are of lung. Kidney and mouse-brain tissue have been compared only with Space Ranger 4's
  registration. Do not use the shift of a slide whose status is not `usable`.
- Space Ranger 4.0 and later refine their registration against the counts by default, so agreement
  between this check and Space Ranger's registration is not independent of the counts.
- A plain cross-correlation peak between nuclei and counts lies on the rim of the dip, not at the
  nuclei.

## Changes in 0.6.0

- The shift comes from the correlation ratio of the counts, divided by their local mean, on bands of
  distance from the nucleus edge (`ratio.assess`, `ratio.regions`; `--baseline`, default 2). The dip,
  the ring and the disk of 0.3.0 to 0.5.0 are each part of this profile; it finds the shift where the
  dip is clear, weak or absent, so `--fallback`, `--disk` and `--ring-width` are gone, and `dip.assess`,
  `dip.regions`, `dip.peak` and `dip.ring_mask` with them. `check` and `fit` output `baseline` instead of
  `fallback`, `disk` and `ring_width`; `whole.method` is `correlation ratio`; the dip's fields (`depth`,
  `dip_status`, `contrast_at_dip`, ...) and the ring's and disk's are replaced by `eta2`,
  `eta2_at_zero`, `share_any_peak` and `profile`.
- `dip.block_weights` gives the weights of the block resampling that `dip.resample` and `ratio` share.

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
