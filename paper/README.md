# Results of the STARgrid manuscript

The number files, figures and supplementary tables of the manuscript (the version submitted to
Genome Research), the scripts that write them and, in `../paper_results/`, the result records they
read.

`paper/regenerate.sh` writes every file again into a new folder and compares it with the copy
here: number files without their comment lines, PNG figures byte for byte. Run it from anywhere:

```sh
paper/regenerate.sh                  # or: paper/regenerate.sh OUT_DIR
```

It needs Python 3.10 or later with numpy, pandas, scipy and matplotlib. A few files need data that
is not in this repository; `regenerate.sh` skips them unless it is given:

| Variable | What it is | Needed for |
|---|---|---|
| `VISIUM_HD_BARCODE_INPUTS` | the folder written by `scripts/make_visium_hd_contract.py` of [STAR Suite](https://github.com/morphic-bio/STAR-suite) from the slide-design file of a Space Ranger installation (not redistributed) | `design_numbers.tex`, `worked_case_numbers.tex`, Fig. 3 |
| `STAR_SUITE` | a clone of STAR Suite, whose source at the cited commit gives the length of the UMI in Read 1 | `read1_layout_numbers.tex` |
| `VISIUM_HD_AUDIT_LEDGER` | the read-level ledger of the examination of Space Ranger's locations on slide H1-VM2JXXK/A1 (not redistributed) | `audit_limit_numbers.tex` |

## Number files

`number_inputs.tex` reads them in this order; where two files define the same macro, the later
one is used. Each file records in its comments the results it came from.

| File | Written by | From |
|---|---|---|
| `numbers.tex` | `scripts/publication/export_spatial_paper_numbers.py`, not rerun here | the accepted result tables of the three slides; each macro names its result and run. Most values the text uses are replaced by the files below |
| `design_numbers.tex` | `figures/scripts/barcode_design_stats.py` | the slide-design oligo lists, and `number_inputs/barcode_contract_summary.json` |
| `audit_numbers.tex` | `figures/scripts/sr_audit_provenance.py` | `number_inputs/space_ranger_vm2jxxk_a1/` (the version, command line and metrics of our Space Ranger run) |
| `release_numbers.tex` | `figures/scripts/release_gate_numbers.py` | `release_gate_v1_12_2_20261105/` |
| `audit_limit_numbers.tex` | `figures/scripts/sr_audit_half_limit.py` | `number_inputs/` and the read-level ledger |
| `worked_case_numbers.tex` | `figures/scripts/worked_case_numbers.py` | `release_examination_v1_12_2_20261105/` and the slide-design oligo lists |
| `audit_outcome_numbers.tex` | `make_audit_outcome_numbers.py` | `number_inputs/` |
| `he_added_shared_numbers.tex` | `make_he_added_shared_numbers.py` | `spatch_rescore_ratio_20261109/` |
| `gain_range_numbers.tex` | `make_gain_range_numbers.py` | the number files |
| `decode_length_numbers.tex` | `make_decode_length_numbers.py` | the number files |
| `read1_layout_numbers.tex` | `make_read1_layout_numbers.py` | STAR Suite's source at the cited commit |
| `registration_numbers.tex` | `figures/scripts/registration_numbers.py` | the `registration_*`, `paper_tests_20261021` and `spatch_rescore_ratio_20261109` records |
| `examination_release_numbers.tex`, `ovarian_examination_numbers.tex` | `figures/scripts/examination_release_numbers.py` | `release_examination_v1_12_2_20261105/`, `release_examination_ovarian_v1_12_2_20261109/` |
| `timing_numbers.tex` | `figures/scripts/timing_numbers.py` | `timing_20261108_cpp/` |
| `source_split_numbers.tex` | `figures/scripts/source_split_numbers.py` | the `b1_location_effect_*` and `release_examination_*` records |
| `examined_agreement_numbers.tex` | `figures/scripts/examined_agreement_numbers.py` | `examined_agreement_20261027/` |

## Figures and supplementary tables

| Output | Written by |
|---|---|
| Fig. 1 | `figures/scripts/fig1_system.py` |
| Fig. 2 | `figures/scripts/fig2_agreement.py` |
| Fig. 3 | `figures/scripts/fig3_policy.py`, from `figures/data/` (written by `fig3_policy_data.py`) |
| Fig. 4 and Supplemental Fig. S2 (`fig_supp_pairdepth`) | `figures/scripts/fig_dipcheck.py` |
| Fig. 5 and Supplemental Figs S3 (`fig_supp_lung`) and S4 (`fig_supp_regsens`) | `figures/scripts/fig_registration.py` |
| Fig. 6 | `figures/scripts/fig_signal.py` |
| Supplemental Fig. S1 | `figures/scripts/fig_supp_dip.py` |
| Supplemental Tables S2, S6–S11, S13, S14 (`supplementary/supplementary_tables.tex`) | `supplementary/build_additional_file_1.py` |

The figure scripts read the number files through `figures/scripts/paper_macros.py`, so the figures
show the same values as the text, and write each figure at the width of the text block.

Paths in the records that pointed to our own folders are replaced by placeholders such as `<runs>`,
`<datasets>`, `<storage>` and `<remote>`.
