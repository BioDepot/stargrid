# Result records of the STARgrid manuscript

The files of our analysis records that the manuscript's number files, figures and supplementary
tables are written from (`paper/README.md` says which script reads which). Each folder holds the
files the scripts read, not the whole record. Paths that pointed to our own folders are replaced by
placeholders such as `<runs>`, `<datasets>`, `<storage>` and `<remote>`.

| Folder | What it holds |
|---|---|
| `spatial_v1_9_5_20260915/` | the comparison of the counts with Space Ranger on the colorectal, SPATCH and ovarian slides: agreement at each scale and policy, read accounting, the molecule gain, the H&E and Xenium comparisons. The matrices behind it are byte-identical to those of STAR Suite 1.12.2 |
| `examined_agreement_20261027/` | the same comparison on slide H1-VM2JXXK/A1 |
| `release_gate_v1_12_2_20261105/` | the official spatial checks of STAR Suite 1.12.2 |
| `release_examination_v1_12_2_20261105/`, `release_examination_ovarian_v1_12_2_20261109/` | the read-by-read comparison with Space Ranger on slide H1-VM2JXXK/A1 and on the ovarian slide |
| `b1_location_effect_target_20261109/`, `b1_location_effect_ovarian_v1_12_2_20261109/` | where the molecules STARgrid counts beyond Space Ranger come from |
| `timing_20261108_cpp/` | the execution times of the two pipelines |
| `registration_check_20261018/` | the first tests of the registration check: profiles across the nucleus edge, lung on the same section |
| `paper_tests_20261021/` | tests of the check: far-shift background, sensitivity of the Xenium comparison |
| `registration_local_dip_20261106/`, `registration_ring_postxenium_20261106/` | the dip and the ring around the nuclei on every slide and region |
| `registration_correlation_ratio_20261108/` | the correlation ratio on every slide and region, and the sweep of the window of the local mean |
| `registration_cellpose_sam_20261110/`, `registration_cellpose_sam_lung_20261110/` | the registration with Cellpose-SAM nuclei on the tumour slides and on the lung |
| `spatch_rescore_ratio_20261109/` | the H&E tests of the added counts on the SPATCH slide, with its masks registered by the correlation ratio |
| `number_inputs/` | small files from our run folders that a few number generators read |
| `xenium_ovarian_adjacent_20251007/` | the Xenium dataset record (files, sizes, MD5) |
