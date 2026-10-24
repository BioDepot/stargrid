# Provenance

The STARgrid manuscript's registration results were computed with analysis scripts in the group's
development repository (private) and in this repository's `scripts/`. This package takes the
methods from them; the runs of those scripts remain the record of the published numbers. Each run
saved a copy of the scripts it ran, with their SHA-256. `docs/REGRESSION.md` shows the package
reproducing the results.

| Here | Taken from | Version used by the manuscript's runs | How |
|---|---|---|---|
| `dip.shift_surface`, `dip.contrast`, `dip.feature` | `scripts/measure_hd_qc_reference.py` | commit e1ff8d5, SHA-256 8225df02…26ad0143 | Verbatim, with a guard on the surface radius |
| `dip.regions` | `scripts/publication/fit_hd_mask_affine.py`, `measure_regions` | commit e1ff8d5, SHA-256 0f64edf0…3811b0ab | Verbatim; thresholds made arguments |
| `affine.design`, `affine.fit_once_with_rejection`, `affine.fit_regions` | the same file | the same | Verbatim; thresholds made arguments |
| `methods.prepare`, `slices`, `plant`, `ncc_surface`, `histogram_codes`, `mutual_information`, `mi_surface`, `optimum` | `hd_registration_methods.py` (development repository) | SHA-256 1576808f…dbdd10e8 | Verbatim |
| `other_methods.phase`, `other_methods.ecc`, `other_methods.sitk_mi`; `other_methods.FROZEN_2026_10_07` | the same file; settings from `registration_methods_frozen_20261007.json` (SHA-256 01d0a695…fb9ab6) | as above | Verbatim; kept for later work, not used by the command-line tool |
| `io.square_indices`, `io.h5_totals` | `scripts/prepare_hd_same_run_offsets.py` | commit e1ff8d5 | Verbatim |
| `io.tissue_mask`, `io.published_nuclei`, `io.read_parquet` | `scripts/measure_hd_qc_reference.py` | as above | Verbatim, plus `.npz` input |
| `io.mtx_totals`, `io.load_counts`, `io.load_grid`, `methods.register` | — | — | New |
| `dip.local_dip` | — | — | New; the criterion of the manuscript's lung test (lower than all eight neighbouring shifts; depth against their median) |
| `dip.edge_distance_profile` | — | — | New, descriptive |
| `transform.*` | — | — | New; the manuscript's runs moved masks with their own scoring code. `sampling="archived-floor"` follows the pixel convention of those runs |
| C++ image registration | this repository's `native/` | commit e85a5de | Unchanged |
