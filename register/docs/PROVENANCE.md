# Provenance

The STARgrid manuscript's registration results were computed with analysis scripts in the group's
development repository (private) and in this repository's `scripts/`. This package takes the
methods from them; the runs of those scripts remain the record of the published numbers. Each run
saved a copy of the scripts it ran, with their SHA-256. `docs/REGRESSION.md` shows the package
reproducing the results.

| Here | Taken from | Version used by the manuscript's runs | How |
|---|---|---|---|
| `dip.shift_surface`, `dip.contrast`, `dip.feature` (the contrast against the shifts 3 to 5 locations away, now descriptive only) | `scripts/measure_hd_qc_reference.py` | commit e1ff8d5, SHA-256 8225df02…26ad0143 | Verbatim, with a guard on the surface radius; the constant `RING` renamed `CONTRAST_RING` in 0.4.0 |
| `dip.regions` (version 0.2.0 and earlier) | `scripts/publication/fit_hd_mask_affine.py`, `measure_regions` | commit e1ff8d5, SHA-256 0f64edf0…3811b0ab | Verbatim; thresholds made arguments. From 0.3.0 the region split is kept but each region is assessed with `dip.assess` |
| `affine.design`, `affine.fit_once_with_rejection`, `affine.fit_regions` | the same file | the same | Verbatim; thresholds made arguments |
| `methods.prepare`, `slices`, `plant`, `ncc_surface`, `histogram_codes`, `mutual_information`, `mi_surface`, `optimum` | `hd_registration_methods.py` (development repository) | SHA-256 1576808f…dbdd10e8 | Verbatim |
| `other_methods.phase`, `other_methods.ecc`, `other_methods.sitk_mi`; `other_methods.FROZEN_2026_10_07` | the same file; settings from `registration_methods_frozen_20261007.json` (SHA-256 01d0a695…fb9ab6) | as above | Verbatim; kept for later work, not used by the command-line tool |
| `io.square_indices`, `io.h5_totals` | `scripts/prepare_hd_same_run_offsets.py` | commit e1ff8d5 | Verbatim |
| `io.tissue_mask`, `io.published_nuclei`, `io.read_parquet` | `scripts/measure_hd_qc_reference.py` | as above | Verbatim, plus `.npz` input |
| `io.mtx_totals`, `io.load_counts`, `io.load_grid`, `methods.register` | — | — | New |
| `dip.local_dip`, `dip.dips`, `dip.depth_at` | — | — | New; the criterion of the manuscript's lung test (lower than all eight neighbouring shifts; depth against their median) |
| `dip.blocks_for`, `dip.block_sums`, `dip.resample`, `dip.assess` | — | — | New in 0.3.0; the 100 × 100 location block resampling (200 draws, seed 20261007) of the manuscript's bootstrap, with exact block sums by FFT; the local dip replaces the ring rule for choosing and accepting the shift |
| `dip.peak`, `dip.ring_mask` and the `fallback` option of `dip.assess` | — | — | New in 0.3.0 (experimental, off by default). From 0.4.0 the ring is used by default where the dip is not usable, as in the manuscript's method, and gives the manuscript's ring results (`docs/REGRESSION.md`); the peak of the nuclei (`fallback='peak'`) remains experimental |
| `dip.edge_distance_profile` | — | — | New, descriptive |
| `transform.*` | — | — | New; the manuscript's runs moved masks with their own scoring code. `sampling="archived-floor"` follows the pixel convention of those runs |
| C++ image registration | this repository's `native/` | commit e85a5de | Unchanged |
