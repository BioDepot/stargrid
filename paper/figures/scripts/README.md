# Figure and table generators

Each script writes its figure to `paper/figures/` (or to a directory given as
its first argument) and reads only accepted result tables, the files in
`paper/figures/data/`, and the number macros through `paper_macros.py`.

| Output | Script |
|---|---|
| Fig. 1, system overview | `fig1_system.py` |
| Fig. 2, agreement with Space Ranger | `fig2_agreement.py` |
| Fig. 3, policy difference (audit) | `fig3_policy.py`; source data from `fig3_policy_data.py` |
| Fig. 4, H&E evaluation | `fig4_cellpose.py --table <flex_morphology_summary table> --out-dir <dir>` |
| Fig. 5, adjacent-section Xenium | `fig5_xenium.py` |
| Table 5, worked reads | `sr_audit_worked_cases.py` |
| `design_numbers.tex` | `barcode_design_stats.py` |
| `audit_numbers.tex` | `sr_audit_provenance.py` |
| `audit_limit_numbers.tex` | `sr_audit_half_limit.py` (uses `sr_audit_no_candidate_mates.py`) |
| `release_numbers.tex` | `release_gate_numbers.py` |

`spatial_paper_figstyle.py` holds the shared style; `figstyle.py` is historical.
