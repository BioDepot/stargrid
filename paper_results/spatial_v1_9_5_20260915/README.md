# Accepted results, STAR Suite v1.9.5

Compact outputs of the analyses behind the manuscript: the full-slide primaries
for the colorectal Flex, SPATCH Flex (2020-A and 2024-A references) and ovarian
3′ slides (compatibility and annotated multimapper settings), and the matrix
analyses built on them. JSON outputs are named `summary.json`. Matrices and
large arrays are not included.

| Directory | Contents |
|---|---|
| `summary_*`, `supporting/` | read accounting and policy masses per primary; decoder check; timing |
| `crc_concordance`, `ovarian_concordance`, `spatch_concordance`, `spatch_2024a_concordance` | concordance with Space Ranger |
| `*_fields`, `*_gene_bin`, `flex_gene_bin_summary`, `ovarian_bin_totals`, `ovarian_nesting` | prepared fields, gene and bin residuals, policy nesting |
| `*_morphology`, `*_he_weights`, `flex_morphology_*`, `crc_vendor_compartments`, `crc_frozen_registration_sensitivity` | H&E scoring and registration sensitivity |
| `three_slide_summary` | molecule gain and shared/added decomposition |
| `xenium_*` | adjacent-section Xenium comparisons |
| `codex_*` | adjacent-section CODEX comparisons |
| `paper_number_export_20260927` | generated manuscript numbers (`numbers.tex`), old/new table and coverage |

H&E tables include the accepted October 2026 ovarian translation and SPATCH affine placement corrections. SPATCH bins with no supported 2 um child are excluded symmetrically from H&E metrics only; all counts and other results are preserved. The common-axis exclusion is 962 bins, holding 189,839 STAR hard and 181,448 Space Ranger counts. Full STAR-axis H&E excludes 1,278 bins. Fig. 4 and paper/numbers.tex use the corrected tables.
