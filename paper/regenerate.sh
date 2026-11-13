#!/usr/bin/env bash
# Regenerate the manuscript's number files, figures and supplementary tables from paper_results/,
# and compare each with the copy in paper/. Run from anywhere; outputs go to a new folder.
#
#   paper/regenerate.sh [OUT_DIR]
#
# Optional inputs, for the few files that need data not in this repository:
#   VISIUM_HD_BARCODE_INPUTS  folder written by make_visium_hd_contract.py (STAR Suite) from the
#                             Space Ranger slide-design file: design_numbers.tex,
#                             worked_case_numbers.tex and Fig. 3 need its oligo lists
#   STAR_SUITE                a clone of STAR Suite: read1_layout_numbers.tex reads its source at
#                             the cited commit
#   VISIUM_HD_AUDIT_LEDGER    the read-level ledger of the examination of Space Ranger's
#                             locations (not redistributed): audit_limit_numbers.tex
# Without them those files are skipped and reported as such.
#
# Number files are compared without their comment lines, which record where they were written.
# PNG figures are compared byte for byte; PDF figures carry a creation date, so they are only
# written.
set -uo pipefail
cd "$(dirname "$0")/.."
out=${1:-$(mktemp -d)}
mkdir -p "$out/figures" "$out/data"
S=paper/figures/scripts
same=0
different=0
skipped=0
failed=0

compare_tex() {
    local name=$1
    if diff <(grep -v '^%' "paper/$name") <(grep -v '^%' "$out/$name") > /dev/null; then
        echo "same       $name"; same=$((same + 1))
    else
        echo "DIFFERENT  $name"; different=$((different + 1))
    fi
}
number() {        # number NAME COMMAND...: the command prints NAME to standard output
    local name=$1; shift
    if "$@" > "$out/$name" 2> "$out/$name.log"; then compare_tex "$name"
    else echo "FAILED     $name (see $out/$name.log)"; failed=$((failed + 1)); fi
}
written() {       # written NAME COMMAND...: the command writes $out/NAME itself
    local name=$1; shift
    if "$@" > "$out/$name.log" 2>&1; then compare_tex "$name"
    else echo "FAILED     $name (see $out/$name.log)"; failed=$((failed + 1)); fi
}
skip() {
    echo "skipped    $1 (set $2)"; skipped=$((skipped + 1))
}

number registration_numbers.tex python3 $S/registration_numbers.py
number release_numbers.tex python3 $S/release_gate_numbers.py paper_results/release_gate_v1_12_2_20261105/results/spatial_gate_rerun_v1122.json
number audit_numbers.tex python3 $S/sr_audit_provenance.py paper_results/number_inputs/space_ranger_vm2jxxk_a1
written examination_release_numbers.tex python3 $S/examination_release_numbers.py \
    --output "$out/examination_release_numbers.tex"
written ovarian_examination_numbers.tex python3 $S/examination_release_numbers.py \
    --record paper_results/release_examination_ovarian_v1_12_2_20261109/results.json \
    --catalogue paper_results/release_examination_ovarian_v1_12_2_20261109/macro_catalogue.json \
    --output "$out/ovarian_examination_numbers.tex"
written source_split_numbers.tex python3 $S/source_split_numbers.py "$out/source_split_numbers.tex"
written timing_numbers.tex python3 $S/timing_numbers.py --output "$out/timing_numbers.tex"
written examined_agreement_numbers.tex python3 $S/examined_agreement_numbers.py "$out/examined_agreement_numbers.tex"
number audit_outcome_numbers.tex python3 paper/make_audit_outcome_numbers.py
number he_added_shared_numbers.tex python3 paper/make_he_added_shared_numbers.py
number gain_range_numbers.tex python3 paper/make_gain_range_numbers.py
number decode_length_numbers.tex python3 paper/make_decode_length_numbers.py
if [[ -n "${VISIUM_HD_BARCODE_INPUTS:-}" ]]; then
    number design_numbers.tex python3 $S/barcode_design_stats.py
    number worked_case_numbers.tex python3 $S/worked_case_numbers.py
else
    skip design_numbers.tex VISIUM_HD_BARCODE_INPUTS
    skip worked_case_numbers.tex VISIUM_HD_BARCODE_INPUTS
fi
if [[ -n "${STAR_SUITE:-}" ]]; then
    number read1_layout_numbers.tex python3 paper/make_read1_layout_numbers.py
else
    skip read1_layout_numbers.tex STAR_SUITE
fi
if [[ -n "${VISIUM_HD_AUDIT_LEDGER:-}" ]]; then
    number audit_limit_numbers.tex python3 $S/sr_audit_half_limit.py
else
    skip audit_limit_numbers.tex VISIUM_HD_AUDIT_LEDGER
fi

# Supplemental tables S2, S6-S11, S13 and S14.
if python3 paper/supplementary/build_additional_file_1.py "$out/supplementary_tables.tex" > "$out/supplementary_tables.log" 2>&1; then
    if cmp -s paper/supplementary/supplementary_tables.tex "$out/supplementary_tables.tex"; then
        echo "same       supplementary/supplementary_tables.tex"; same=$((same + 1))
    else
        echo "DIFFERENT  supplementary/supplementary_tables.tex"; different=$((different + 1))
    fi
else
    echo "FAILED     supplementary/supplementary_tables.tex"; failed=$((failed + 1))
fi

# Figures, and the source data of Fig. 3.
figures=(fig1_system fig2_agreement fig_dipcheck fig_registration fig_signal fig_supp_dip)
if [[ -n "${VISIUM_HD_BARCODE_INPUTS:-}" ]]; then figures+=(fig3_policy); else skip "Fig. 3" VISIUM_HD_BARCODE_INPUTS; fi
for script in "${figures[@]}"; do
    python3 $S/$script.py "$out/figures" > "$out/figures/$script.log" 2>&1 \
        || { echo "FAILED     $script.py (see $out/figures/$script.log)"; failed=$((failed + 1)); }
done
python3 $S/fig3_policy_data.py "$out/data" > "$out/data/fig3_policy_data.log" 2>&1 \
    || { echo "FAILED     fig3_policy_data.py"; failed=$((failed + 1)); }
for file in "$out"/figures/*.png "$out"/data/*.tsv "$out"/data/*.json; do
    [[ -f "$file" ]] || continue
    case "$file" in */data/*) committed=paper/figures/data/$(basename "$file") ;; *) committed=paper/figures/$(basename "$file") ;; esac
    if cmp -s "$file" "$committed"; then
        echo "same       ${committed#paper/}"; same=$((same + 1))
    else
        echo "DIFFERENT  ${committed#paper/}"; different=$((different + 1))
    fi
done

echo
echo "$same the same, $different different, $failed failed, $skipped skipped; outputs in $out"
[[ $different -eq 0 && $failed -eq 0 ]]
