#!/usr/bin/env python3
"""Add shared-versus-STAR-excess fields to the frozen Xenium cancer analysis.

The upstream Spatial-suite scorer defines the expression-blind registration,
Xenium cancer labels, gene scopes, and 128-um patch cohort.  This wrapper pins
that implementation and augments each STAR policy at the scorer's native
gene-by-registered-patch unit:

    shared = min(STAR, Space Ranger)
    STAR excess = max(STAR - Space Ranger, 0)
    Space Ranger only = max(Space Ranger - STAR, 0)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


UPSTREAM_COMMIT = "6e073eeb0aa35c30411eaed616bd2a0348a27c98"
UPSTREAM_SCRIPT_SHA256 = (
    "5cdc122ad605e97be72a0e80c28d1da9d9d81277eafc5a2d68ffaf00ee7f75f4"
)


def parse_policy(value: str) -> str:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=MEX_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=MEX_DIR")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-worktree", type=Path,
                        help="Optional historical worktree; by default use the checksum-identical vendored scorer.")
    parser.add_argument("--sr-h5", required=True, type=Path)
    parser.add_argument("--policy-mex", action="append", type=parse_policy, required=True)
    parser.add_argument("--positions", required=True, type=Path)
    parser.add_argument("--scalefactors", required=True, type=Path)
    parser.add_argument("--xenium-h5", required=True, type=Path)
    parser.add_argument("--xenium-cells", required=True, type=Path)
    parser.add_argument("--xenium-clusters", required=True, type=Path)
    parser.add_argument("--xenium-diffexp", required=True, type=Path)
    parser.add_argument("--xenium-experiment", required=True, type=Path)
    parser.add_argument("--xenium-he-alignment", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--reference-panel-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--patch-um", type=float, default=128.0)
    parser.add_argument("--min-xenium-cells", type=int, default=10)
    parser.add_argument("--min-visium-tissue-bins", type=int, default=16)
    parser.add_argument("--cancer-rich-min-fraction", type=float, default=0.75)
    parser.add_argument("--cancer-poor-max-fraction", type=float, default=0.25)
    parser.add_argument("--cancer-clusters", default="1,2,3,5,6,8,9,14")
    parser.add_argument("--label-genes", default="MUC16,PAX8,EPCAM")
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def decompose_arrays(
    star: np.ndarray, space_ranger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if star.shape != space_ranger.shape:
        raise ValueError("STAR and Space Ranger gene-patch arrays differ")
    shared = np.minimum(star, space_ranger)
    star_excess = np.clip(star - space_ranger, 0.0, None)
    sr_only = np.clip(space_ranger - star, 0.0, None)
    if not np.allclose(shared + star_excess, star, rtol=0.0, atol=1e-10):
        raise ValueError("STAR Xenium components do not reconcile")
    if not np.allclose(shared + sr_only, space_ranger, rtol=0.0, atol=1e-10):
        raise ValueError("Space Ranger Xenium components do not reconcile")
    return shared, star_excess, sr_only


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    upstream_script = (
        args.upstream_worktree
        / "scripts"
        / "compare_ovarian_panel_in_xenium_cancer_cells.py"
    ) if args.upstream_worktree else Path(__file__).with_name("compare_ovarian_panel_in_xenium_cancer_cells.py")
    if not upstream_script.is_file():
        raise SystemExit(f"missing upstream scorer: {upstream_script}")
    sys.path.insert(0, str(upstream_script.parent))
    module = importlib.import_module(
        "compare_ovarian_panel_in_xenium_cancer_cells"
    )
    if module.sha256(upstream_script) != UPSTREAM_SCRIPT_SHA256:
        raise SystemExit("upstream Xenium cancer scorer differs from pinned source")
    labels = [value.split("=", 1)[0] for value in args.policy_mex]
    if set(labels) != {"hard", "soft_expected"}:
        raise SystemExit("this comparison requires hard and soft_expected policies")

    upstream_evaluate = module.evaluate_scope
    component_audit: dict[str, dict[str, float]] = {}

    def evaluate_with_components(
        scope: str,
        method_counts: dict[str, np.ndarray],
        xenium_counts: np.ndarray,
        gene_ids: list[str],
        gene_names: list[str],
        selected_genes: np.ndarray,
        cancer_rich: np.ndarray,
        cancer_poor: np.ndarray,
        bootstrap_seed: int,
        bootstrap_replicates: int,
    ):
        augmented = dict(method_counts)
        sr = method_counts["space_ranger"]
        for label in ("hard", "soft_expected"):
            shared, excess, sr_only = decompose_arrays(method_counts[label], sr)
            augmented[f"{label}_shared"] = shared
            augmented[f"{label}_star_excess"] = excess
            augmented[f"{label}_space_ranger_only"] = sr_only
            if label not in component_audit:
                component_audit[label] = {
                    "space_ranger_target_gene_patch_mass": float(sr.sum()),
                    "star_target_gene_patch_mass": float(method_counts[label].sum()),
                    "shared_target_gene_patch_mass": float(shared.sum()),
                    "star_excess_target_gene_patch_mass": float(excess.sum()),
                    "space_ranger_only_target_gene_patch_mass": float(sr_only.sum()),
                    "star_excess_cancer_rich_mass": float(excess[:, cancer_rich].sum()),
                    "star_excess_cancer_poor_mass": float(excess[:, cancer_poor].sum()),
                    "shared_cancer_rich_mass": float(shared[:, cancer_rich].sum()),
                    "shared_cancer_poor_mass": float(shared[:, cancer_poor].sum()),
                }
        return upstream_evaluate(
            scope,
            augmented,
            xenium_counts,
            gene_ids,
            gene_names,
            selected_genes,
            cancer_rich,
            cancer_poor,
            bootstrap_seed,
            bootstrap_replicates,
        )

    module.evaluate_scope = evaluate_with_components
    upstream_args = argparse.Namespace(**{
        key: value for key, value in vars(args).items() if key != "upstream_worktree"
    })
    result = module.run(upstream_args)

    audit_rows = []
    for label in ("hard", "soft_expected"):
        row = {"method": label, **component_audit[label]}
        row["net_star_minus_space_ranger_mass"] = (
            row["star_target_gene_patch_mass"]
            - row["space_ranger_target_gene_patch_mass"]
        )
        row["reconciled_net_from_components"] = (
            row["star_excess_target_gene_patch_mass"]
            - row["space_ranger_only_target_gene_patch_mass"]
        )
        audit_rows.append(row)
    audit_path = args.out_dir / "component_reconciliation.tsv"
    write_rows(audit_path, audit_rows)

    summary_path = args.out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["schema"] = (
        "visium_hd_processing.gex_xenium_cancer_shared_excess.v1"
    )
    summary["component_definition"] = (
        "Within each shared gene and registered 128-um patch: "
        "shared=min(STAR,SR), STAR excess=max(STAR-SR,0), "
        "SR only=max(SR-STAR,0)."
    )
    summary["component_unit"] = "shared gene by registered patch"
    summary["upstream_source"] = {
        "worktree": str(args.upstream_worktree.resolve()) if args.upstream_worktree else str(Path(__file__).resolve().parents[2]),
        "commit": UPSTREAM_COMMIT,
        "script": str(upstream_script.resolve()),
        "script_sha256": UPSTREAM_SCRIPT_SHA256,
    }
    summary["artifacts"]["component_reconciliation"] = audit_path.name
    summary["component_reconciliation"] = component_audit
    summary["invariants"] = {
        "hard_and_soft_reported_separately": True,
        "net_equals_star_excess_minus_space_ranger_only": True,
        "raw_mass_not_normalized": True,
        "xenium_not_used_for_assignment": True,
        "space_ranger_is_comparator_not_truth": True,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "axes": result["axes"],
        "component_methods": [
            f"{label}_{component}"
            for label in ("hard", "soft_expected")
            for component in ("shared", "star_excess", "space_ranger_only")
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
