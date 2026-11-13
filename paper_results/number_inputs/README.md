# Small inputs of the number files

Files from our run folders that a few number generators read (see `paper/README.md`). Paths inside
them that pointed to our own folders are replaced by placeholders; `SOURCE_SHA256.txt` gives the
checksum of each file as it was before.

| File | What it is | Read by |
|---|---|---|
| `space_ranger_vm2jxxk_a1/` | the version (`_versions`), command line (`_cmdline`) and `outs/metrics_summary.csv` of our Space Ranger 4.0.1 run with BAM output on slide H1-VM2JXXK/A1, used for the read-by-read comparison | `sr_audit_provenance.py` |
| `candidate_preservation_summary.json` | for the reads of that run, how many have a best-matching location within the edit limits, and how many of those Space Ranger placed | `sr_audit_half_limit.py`, `make_audit_outcome_numbers.py` |
| `disagreement_taxonomy_summary.json` | the outcomes of the read-by-read comparison on that slide | `sr_audit_half_limit.py`, `make_audit_outcome_numbers.py` |
| `edit_strata.tsv` | the same reads by number of edits to the best-matching location and by ties | `make_audit_outcome_numbers.py` |
| `barcode_contract_summary.json` | our summary of the barcode layout: the Read 1 segments of the chemistry definition and the barcode lists of the slide-design file | `barcode_design_stats.py` |
