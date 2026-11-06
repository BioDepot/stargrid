# Regression against the manuscript's results

The package was run on the inputs of the accepted analysis runs of the STARgrid manuscript and its
output compared with their records. The inputs (count matrices, tissue flags, nucleus label images
and registrations) are not distributed here; the tests in `tests/` use synthetic slides.

The first results below are those of version 0.4.0, which takes the shift from the deepest stable
dip or, where the dip is not usable, from the highest stable peak of the ring (see the README).
The later ones are those of versions 0.1.0 and 0.2.0, which chose the shift by the contrast
against the shifts 3 to 5 locations away; from version 0.3.0 the shift is the deepest local dip,
accepted when it is stable under block resampling. The dip is usable on the whole ovarian slide
and in every region of the SPATCH slide, so the ring of 0.4.0 is not used there.

## The dip and the ring (version 0.4.0, 2026-11-06)

The package at its defaults was run on the inputs of the manuscript's analysis of the ring: the
tumour slides under each registration tested (12 placements), the four lung cores and the two
whole post-Xenium lung sections with nuclei from the H&E image and imaged by Xenium (12), and the
four public slides of other tissues (4). On each of the 448 surfaces that analysis measured (the
whole slide or section, and each region with at least 30,000 nucleus locations in tissue), it gave
the analysis's dip (whether there is one, its shift, its share of resamples within one location
and whether it is usable) and, where the dip was not usable, its ring of width 3 (the same for the
highest peak), and so the shift of the manuscript's rule: the dip on 327 surfaces, the ring on 27
and no shift on 94. There were no differences.

```sh
stargrid-register check --counts square_totals.npz --tissue tissue_positions.parquet \
    --nuclei grid_labels.npz --out check.json          # the ring is the default; --fallback none turns it off
```

## Ovarian slide (H1-YQJBZ7X/D1), whole slide (version 0.1.0, 2026-10-18)

Counts: STARgrid `hard` totals at 2 µm. Tissue: the 2 µm `tissue_positions.parquet` of the
dataset.

| Nucleus map | Shift (dy, dx) | Contrast | Mean count at the shift | Accepted record |
|---|---|---|---|---|
| Our Cellpose masks, our image registration | (−3, −3) | 0.14017 | 27.38177 | (−3, −3), 0.1402, 27.38177 |
| Space Ranger's nucleus map (`barcode_mappings.parquet`) | (0, 0) | 0.14962 | 26.82858 | (0, 0), 0.1496, 26.82858 |

```sh
stargrid-register check --counts square_totals.npz --tissue tissue_positions.parquet \
    --nuclei grid_labels.npz --nuclei-key nucleus_id --divisions 0 --out ov_cellpose.json
```

## SPATCH slide (H1-WP2FTZD/D1), regional fit (version 0.1.0, 2026-10-18)

Counts: STARgrid `hard` totals. Tissue: a Space Ranger 4.1.0 run on 100,000 reads. Nuclei: our
Cellpose masks under our image registration, before correction.

`stargrid-register fit` gave 18 usable regions, 9 of 9 held-out regions predicted within one
location, and an accepted model. The regional dips and the fitted coefficients are identical to
the accepted fit of 2026-10-04 (maximum difference in the coefficients 0.0).

```sh
stargrid-register fit --counts square_totals.npz --tissue tissue_positions.parquet \
    --nuclei grid_states.npz --nuclei-key nucleus --out spatch_fit.json
```

## Placements by Space Ranger's registrations, and the lung cores (2026-10-21)

Every case behind the manuscript's figure of the check was run again from the archived inputs:
our nucleus masks placed with `place` by each Space Ranger registration and scored with `check`,
and the local dips of the lung cores scored on the saved H&E nucleus grids. All eleven cases
reproduce the accepted records exactly (difference 0.0 in the depth of the dip; zero differences
in the placed labels; regional contrasts identical).

| Case | Shift (dy, dx) | Depth (contrast, or local depth for the lung cores) |
|---|---|---:|
| Ovarian, Cellpose masks, deposited Space Ranger 4.0.1 registration | (0, 0) | 0.144538 |
| Ovarian, StarDist masks, the same registration | (0, 0) | 0.156477 |
| Colorectal, StarDist masks, deposited Space Ranger 4.1.0 registration | (0, 0) | 0.064626 |
| SPATCH, Cellpose masks, Space Ranger 4.1.0 run on 100,000 reads (whole and every region) | (0, 0) | 0.080094 |
| SPATCH, Cellpose masks, Space Ranger 3.0.0 run on 100,000 reads (whole and every region) | (−5, −2) | 0.082403 |
| Examined colorectal, StarDist masks, Space Ranger 4.1.0 run on 100,000 reads (whole and every region) | (0, 0) | 0.047464 |
| Examined colorectal, StarDist masks, GEO deposit (Space Ranger 3.0.0, 8 µm positions) | (−1, −2) | 0.042711 |
| Lung core VUILD49LA, H&E nuclei | (−2, −3) | 0.021527 |
| Lung core TILD113LA, H&E nuclei | (−4, −2) | 0.039104 |
| Lung core TILD111LA, H&E nuclei | (−3, −3) | 0.015388 |
| Lung core VUHD049, H&E nuclei | (−5, −1) | 0.010513 |

Exact reproduction of the placements needs the pixel convention of the original runs:

```sh
stargrid-register place --tissue-positions tissue_positions.parquet --labels nucleus_labels.npy \
    --sampling archived-floor --image-to-labels frame.json --out-grid grid.npz --out place.json
```

With the default nearest-pixel sampling, the ovarian Cellpose masks gave the same shift and a
contrast of 0.144663, 0.000125 above the accepted value.
