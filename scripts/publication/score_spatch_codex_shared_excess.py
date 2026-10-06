#!/usr/bin/env python3
"""Compare shared STAR/SR and STAR-excess fields against adjacent CODEX.

This is a post-count analysis.  It reuses the frozen SPATCH coordinate
serialization and CODEX scoring implementation.  For each method, protein
marker, and CODEX evaluation grid, it decomposes raw RNA mass as

    shared = min(STAR, Space Ranger)
    STAR excess = max(STAR - Space Ranger, 0)
    Space Ranger only = max(Space Ranger - STAR, 0)

The whole-slide biological count comparison remains separate because the
CODEX component table is restricted to the direct RNA/protein marker panel and
the fixed CODEX evaluation universe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import score_hd_spatch_codex as base  # noqa: E402


def parse_method(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=MEX_DIR")
    return label, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-h5ad", type=Path, required=True)
    parser.add_argument("--codex-h5ad", type=Path, required=True)
    parser.add_argument("--probe-csv", type=Path, required=True)
    parser.add_argument("--sr-filtered-probe-set", type=Path, required=True)
    parser.add_argument(
        "--method", action="append", type=parse_method, required=True,
        metavar="LABEL=MEX_DIR",
    )
    parser.add_argument("--bin-size", action="append", type=int, default=[])
    parser.add_argument("--protein-positive-quantile", type=float, default=0.8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def project_method_to_codex(
    method: base.MethodData,
    codex: dict[int, tuple[np.ndarray, np.ndarray]],
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one method onto the complete frozen CODEX grid axis."""
    codex_keys, protein = codex[size]
    method_keys, method_values = method.grids[size]
    lookup = base.key_index(method_keys)
    rna = np.zeros_like(protein, dtype=np.float64)
    for target, key in enumerate(codex_keys):
        source = lookup.get((int(key[0]), int(key[1])))
        if source is not None:
            rna[target] = method_values[source]
    return codex_keys, rna


def decompose_arrays(
    star: np.ndarray, space_ranger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if star.shape != space_ranger.shape:
        raise ValueError("STAR and Space Ranger CODEX-grid arrays differ")
    shared = np.minimum(star, space_ranger)
    star_excess = np.clip(star - space_ranger, 0.0, None)
    sr_only = np.clip(space_ranger - star, 0.0, None)
    if not np.allclose(shared + star_excess, star, rtol=0.0, atol=1e-10):
        raise ValueError("STAR CODEX-grid components do not reconcile")
    if not np.allclose(shared + sr_only, space_ranger, rtol=0.0, atol=1e-10):
        raise ValueError("Space Ranger CODEX-grid components do not reconcile")
    return shared, star_excess, sr_only


def component_method(
    label: str,
    grids: dict[int, tuple[np.ndarray, np.ndarray]],
    coverage: list[dict[str, object]],
) -> base.MethodData:
    masses = {size: float(values.sum()) for size, (_, values) in grids.items()}
    total = float(next(iter(masses.values()))) if masses else 0.0
    return base.MethodData(
        label=label,
        grids=grids,
        full_mass=total,
        shared_axis_mass=total,
        evaluation_mass=masses,
        coverage=coverage,
        inputs=[],
    )


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: base.format_number(value) for key, value in row.items()})


def main() -> int:
    args = parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    labels = [label for label, _ in args.method]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate --method label")
    if set(labels) != {"hard", "soft_expected"}:
        raise SystemExit("this comparison requires hard and soft_expected methods")
    bin_sizes = sorted(set(args.bin_size or [100, 200, 300, 400, 500]))
    if any(size <= 0 for size in bin_sizes):
        raise SystemExit("bin sizes must be positive")
    if not 0 < args.protein_positive_quantile < 1:
        raise SystemExit("protein-positive quantile must be between zero and one")
    for path in (
        args.sr_h5ad,
        args.codex_h5ad,
        args.probe_csv,
        args.sr_filtered_probe_set,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    for _, root in args.method:
        if not root.is_dir():
            raise SystemExit(f"missing MEX directory: {root}")

    gene_to_symbol, _ = base.load_probe_symbols(args.probe_csv)
    space_ranger, coefficients, fit_diagnostics, sr_gene_ids = base.load_space_ranger(
        args.sr_h5ad, bin_sizes,
    )
    filtered_panel = base.load_filtered_probe_panel(args.sr_filtered_probe_set)
    eligible = set(filtered_panel.pop("eligible_genes"))
    if eligible != sr_gene_ids:
        raise SystemExit(
            "published Space Ranger H5AD does not equal the filtered probe-panel axis"
        )
    codex, codex_keys = base.load_codex(args.codex_h5ad, bin_sizes)
    star_methods = {
        label: base.load_mex_method(
            label,
            root,
            bin_sizes,
            coefficients,
            gene_to_symbol,
            sr_gene_ids,
            codex_keys,
            None,
        )
        for label, root in args.method
    }

    sr_projected = {
        size: project_method_to_codex(space_ranger, codex, size)
        for size in bin_sizes
    }
    components: list[base.MethodData] = []
    reconciliations: list[dict[str, object]] = []
    for label in ("hard", "soft_expected"):
        star = star_methods[label]
        component_grids: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {
            "shared": {},
            "star_excess": {},
            "space_ranger_only": {},
        }
        for size in bin_sizes:
            keys, sr_rna = sr_projected[size]
            _, star_rna = project_method_to_codex(star, codex, size)
            shared, excess, sr_only = decompose_arrays(star_rna, sr_rna)
            component_grids["shared"][size] = (keys, shared)
            component_grids["star_excess"][size] = (keys, excess)
            component_grids["space_ranger_only"][size] = (keys, sr_only)
            star_mass = float(star_rna.sum())
            sr_mass = float(sr_rna.sum())
            shared_mass = float(shared.sum())
            excess_mass = float(excess.sum())
            sr_only_mass = float(sr_only.sum())
            reconciliations.append({
                "method": label,
                "bin_size_um": size,
                "component_unit": "CODEX marker by evaluation grid",
                "space_ranger_direct_marker_mass": sr_mass,
                "star_direct_marker_mass": star_mass,
                "shared_direct_marker_mass": shared_mass,
                "star_excess_direct_marker_mass": excess_mass,
                "space_ranger_only_direct_marker_mass": sr_only_mass,
                "net_star_minus_space_ranger_direct_marker_mass": star_mass - sr_mass,
                "reconciled_net_from_components": excess_mass - sr_only_mass,
            })
        for component_label, grids in component_grids.items():
            components.append(component_method(
                f"{label}_{component_label}", grids, star.coverage,
            ))

    methods = [space_ranger, *components]
    marker_metrics, method_summary = base.score_methods(
        methods, codex, bin_sizes, args.protein_positive_quantile,
    )
    args.out_dir.mkdir(parents=True)
    marker_path = args.out_dir / "marker_metrics.tsv"
    method_path = args.out_dir / "method_summary.tsv"
    reconciliation_path = args.out_dir / "component_reconciliation.tsv"
    base.write_tsv(marker_path, marker_metrics, [
        "method", "bin_size_um", "protein", "genes", "mapping",
        "reference_marker_available", "evaluated_bins", "protein_positive_bins",
        "protein_positive_threshold", "rna_mass", "rna_detected_bins",
        "rna_detection_fraction", "protein_mass_covered_by_rna",
        "roc_auc_top_protein_quantile", "pearson_raw", "spearman_raw",
    ])
    base.write_tsv(method_path, method_summary, [
        "method", "bin_size_um", "markers_scored", "macro_roc_auc",
        "median_roc_auc", "direct_panel_rna_mass", "method_evaluation_mass",
    ])
    write_rows(reconciliation_path, reconciliations)

    outputs = [marker_path, method_path, reconciliation_path]
    payload = {
        "schema": "visium_hd_processing.spatch_codex_shared_excess.v1",
        "component_definition": (
            "Within each direct RNA/protein marker and fixed CODEX evaluation grid: "
            "shared=min(STAR,SR), STAR excess=max(STAR-SR,0), "
            "SR only=max(SR-STAR,0)."
        ),
        "component_unit": "CODEX marker by evaluation grid",
        "whole_slide_count_mass_included": False,
        "whole_slide_count_mass_reason": (
            "Whole-slide filtered-panel count accounting is reported separately; "
            "this table is restricted to direct CODEX marker genes and grids."
        ),
        "parameters": {
            "bin_sizes_um": bin_sizes,
            "protein_positive_quantile": args.protein_positive_quantile,
            "raw_mass_normalized": False,
        },
        "coordinate_serialization": {
            "coefficients": coefficients.tolist(),
            **fit_diagnostics,
        },
        "inputs": {
            "space_ranger_h5ad": str(args.sr_h5ad.resolve()),
            "codex_h5ad": str(args.codex_h5ad.resolve()),
            "probe_csv": str(args.probe_csv.resolve()),
            "space_ranger_filtered_probe_set": str(
                args.sr_filtered_probe_set.resolve()
            ),
            "star_methods": {
                label: str(root.resolve()) for label, root in args.method
            },
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": base.sha256(path)}
            for path in outputs
        },
        "invariants": {
            "hard_and_soft_reported_separately": True,
            "net_equals_star_excess_minus_space_ranger_only": True,
            "raw_mass_not_normalized": True,
            "codex_not_used_for_assignment": True,
            "space_ranger_is_comparator_not_truth": True,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (args.out_dir / "checksums.sha256").write_text(
        "".join(
            f"{base.sha256(path)}  {path.name}\n"
            for path in [*outputs, summary_path]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
