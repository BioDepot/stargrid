#!/usr/bin/env python3
"""Score Visium HD policy matrices against the fixed SPATCH CODEX transfer.

The retained SPATCH transcriptome object supplies only the published Visium-HD
to-CODEX coordinate serialization and the Space Ranger comparison matrix.  It
does not supply read, barcode, probe, UMI, molecule, or policy evidence to a
STAR method.  All methods are scored on the same CODEX grid cells, without
normalizing their RNA mass.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.stats import rankdata, spearmanr

from audit_space_ranger_probe_accounting import load_filtered_probe_panel


HD_BARCODE = re.compile(r"^s_008um_(\d+)_(\d+)(?:-\d+)?$")


@dataclass(frozen=True)
class Marker:
    protein: str
    genes: tuple[str, ...]
    mapping: str = "direct"


MARKERS = (
    Marker("CD8", ("CD8A",)),
    Marker("CD20", ("MS4A1",)),
    Marker("CD3e", ("CD3E",)),
    Marker("CD56", ("NCAM1",)),
    Marker("Pan-Cytokeratin", ("EPCAM", "KRT7", "KRT8", "KRT19"), "module"),
    Marker("CD4", ("CD4",)),
    Marker("CD34", ("CD34",)),
    Marker("SMA", ("ACTA2",)),
    Marker("FOXP3", ("FOXP3",)),
    Marker("CD163", ("CD163",)),
    Marker("HLA-A", ("HLA-A",)),
    Marker("CD11c", ("ITGAX",)),
    Marker("MPO", ("MPO",)),
    Marker("CD68", ("CD68",)),
    Marker("HLA-DR", ("HLA-DRA",)),
    Marker("IDO1", ("IDO1",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-h5ad", type=Path, required=True)
    parser.add_argument("--codex-h5ad", type=Path, required=True)
    parser.add_argument("--probe-csv", type=Path, required=True)
    parser.add_argument(
        "--sr-filtered-probe-set",
        type=Path,
        help=(
            "Space Ranger filtered probe reference used to require exact "
            "published-H5AD panel-axis accounting"
        ),
    )
    parser.add_argument(
        "--method", action="append", default=[], metavar="LABEL=MEX_DIR",
        help="STAR method label and one square_008um MEX directory",
    )
    parser.add_argument(
        "--reference-feature-list", action="append", default=[], metavar="LABEL=PATH",
        help="optional complete reference gene-ID axis for a STAR method",
    )
    parser.add_argument("--bin-size", action="append", type=int, default=[])
    parser.add_argument("--protein-positive-quantile", type=float, default=0.8)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def find_one(directory: Path, names: Iterable[str]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"missing required MEX file in {directory}: {list(names)}")


def parse_barcode(value: str) -> tuple[int, int]:
    found = HD_BARCODE.fullmatch(value)
    if found is None:
        raise ValueError(f"not a square_008um Visium HD barcode: {value}")
    return int(found.group(1)), int(found.group(2))


def fit_coordinate_serialization(
    barcodes: Iterable[str], spatial: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    row_col = np.asarray([parse_barcode(value) for value in barcodes], dtype=np.float64)
    if spatial.shape != row_col.shape or spatial.shape[1] != 2:
        raise ValueError("barcode and spatial coordinate shapes differ")
    design = np.column_stack((row_col, np.ones(row_col.shape[0], dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, spatial.astype(np.float64), rcond=None)
    if rank != 3:
        raise ValueError("published coordinate serialization is rank deficient")
    residual = np.linalg.norm(design @ coefficients - spatial, axis=1)
    diagnostics = {
        "rmse_um": float(np.sqrt(np.mean(residual * residual))),
        "max_um": float(np.max(residual)),
        "p99_um": float(np.quantile(residual, 0.99)),
    }
    return coefficients, diagnostics


def apply_coordinate_serialization(barcodes: Iterable[str], coefficients: np.ndarray) -> np.ndarray:
    row_col = np.asarray([parse_barcode(value) for value in barcodes], dtype=np.float64)
    design = np.column_stack((row_col, np.ones(row_col.shape[0], dtype=np.float64)))
    return design @ coefficients


def grid_coordinates(spatial: np.ndarray, bin_size: int) -> np.ndarray:
    if bin_size <= 0:
        raise ValueError("bin size must be positive")
    return np.floor(spatial / float(bin_size)).astype(np.int64)


def aggregate_rows(
    spatial: np.ndarray, values: np.ndarray, bin_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    keys, inverse = np.unique(grid_coordinates(spatial, bin_size), axis=0, return_inverse=True)
    aggregate = np.zeros((keys.shape[0], values.shape[1]), dtype=np.float64)
    np.add.at(aggregate, inverse, values)
    return keys, aggregate


def key_index(keys: np.ndarray) -> dict[tuple[int, int], int]:
    return {(int(row[0]), int(row[1])): index for index, row in enumerate(keys)}


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(spearmanr(left, right).statistic)


def load_probe_symbols(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    gene_to_symbol: dict[str, str] = {}
    symbol_to_genes: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = (line for line in handle if line.strip() and not line.startswith("#"))
        reader = csv.DictReader(lines)
        required = {"gene_id", "gene_name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"probe CSV lacks columns: {sorted(required)}")
        for row in reader:
            gene, symbol = row["gene_id"].split(".", 1)[0], row["gene_name"]
            if gene in gene_to_symbol and gene_to_symbol[gene] != symbol:
                raise ValueError(f"probe CSV maps {gene} to multiple symbols")
            gene_to_symbol[gene] = symbol
            symbol_to_genes.setdefault(symbol, set()).add(gene)
    return gene_to_symbol, symbol_to_genes


def marker_feature_map(
    feature_ids: list[str], gene_to_symbol: dict[str, str],
    reference_feature_ids: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    feature_marker = np.full(len(feature_ids), -1, dtype=np.int16)
    coverage: list[dict[str, object]] = []
    symbol_to_rows: dict[str, list[int]] = {}
    for index, feature in enumerate(feature_ids):
        symbol = gene_to_symbol.get(feature.split(".", 1)[0], "")
        if symbol:
            symbol_to_rows.setdefault(symbol, []).append(index)
    reference_symbols = {
        gene_to_symbol.get(feature.split(".", 1)[0], "")
        for feature in (reference_feature_ids if reference_feature_ids is not None else feature_ids)
    }
    for marker_index, marker in enumerate(MARKERS):
        present = []
        for gene in marker.genes:
            rows = symbol_to_rows.get(gene, [])
            present.extend(rows)
            for row in rows:
                if feature_marker[row] not in (-1, marker_index):
                    raise ValueError(f"feature assigned to multiple CODEX markers: {feature_ids[row]}")
                feature_marker[row] = marker_index
        coverage.append({
            "protein": marker.protein,
            "genes": ",".join(marker.genes),
            "mapping": marker.mapping,
            "reference_genes_present": sum(gene in reference_symbols for gene in marker.genes),
            "matrix_genes_present": sum(gene in symbol_to_rows for gene in marker.genes),
            "genes_required": len(marker.genes),
            "matrix_feature_rows": len(present),
        })
    return feature_marker, coverage


@dataclass
class MethodData:
    label: str
    grids: dict[int, tuple[np.ndarray, np.ndarray]]
    full_mass: float
    shared_axis_mass: float
    evaluation_mass: dict[int, float]
    coverage: list[dict[str, object]]
    inputs: list[Path]


def load_mex_method(
    label: str,
    directory: Path,
    bin_sizes: list[int],
    coefficients: np.ndarray,
    gene_to_symbol: dict[str, str],
    sr_gene_ids: set[str],
    codex_keys: dict[int, set[tuple[int, int]]],
    reference_feature_path: Path | None,
) -> MethodData:
    feature_path = find_one(directory, ("features.tsv", "features.tsv.gz"))
    barcode_path = find_one(directory, ("barcodes.tsv", "barcodes.tsv.gz"))
    matrix_path = find_one(directory, ("matrix.mtx", "matrix.mtx.gz"))
    with open_text(feature_path) as handle:
        feature_ids = [row[0].split(".", 1)[0] for row in csv.reader(handle, delimiter="\t") if row]
    with open_text(barcode_path) as handle:
        barcodes = [line.strip() for line in handle if line.strip()]
    if len(set(feature_ids)) != len(feature_ids) or len(set(barcodes)) != len(barcodes):
        raise ValueError(f"duplicate MEX axis entry in {directory}")
    matrix = mmread(matrix_path).tocoo(copy=False)
    if matrix.shape != (len(feature_ids), len(barcodes)):
        raise ValueError(f"MEX dimensions do not match axes in {directory}")
    data = np.asarray(matrix.data, dtype=np.float64)
    full_mass = float(data.sum())
    shared_feature = np.fromiter((gene in sr_gene_ids for gene in feature_ids), bool, len(feature_ids))
    shared_axis_mass = float(data[shared_feature[matrix.row]].sum())
    reference_feature_ids = feature_ids
    inputs = [feature_path, barcode_path, matrix_path]
    if reference_feature_path is not None:
        reference_feature_ids = [
            line.strip().split(".", 1)[0]
            for line in reference_feature_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not reference_feature_ids or len(set(reference_feature_ids)) != len(reference_feature_ids):
            raise ValueError(f"invalid complete reference feature list: {reference_feature_path}")
        inputs.append(reference_feature_path)
    feature_marker, coverage = marker_feature_map(
        feature_ids, gene_to_symbol, reference_feature_ids,
    )
    marker_index = feature_marker[matrix.row]
    selected = marker_index >= 0
    flat = marker_index[selected].astype(np.int64) * len(barcodes) + matrix.col[selected]
    marker_by_barcode = np.bincount(
        flat, weights=data[selected], minlength=len(MARKERS) * len(barcodes),
    ).reshape(len(MARKERS), len(barcodes)).T
    spatial = apply_coordinate_serialization(barcodes, coefficients)
    grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    evaluation_mass: dict[int, float] = {}
    for size in bin_sizes:
        keys, values = aggregate_rows(spatial, marker_by_barcode, size)
        grids[size] = keys, values
        method_keys = grid_coordinates(spatial, size)
        allowed = codex_keys[size]
        in_evaluation = np.fromiter(
            ((int(key[0]), int(key[1])) in allowed for key in method_keys), bool, len(barcodes),
        )
        evaluation_mass[size] = float(data[in_evaluation[matrix.col]].sum())
    return MethodData(
        label, grids, full_mass, shared_axis_mass, evaluation_mass, coverage,
        inputs,
    )


def load_space_ranger(
    path: Path, bin_sizes: list[int], coefficients_out: bool = True,
) -> tuple[MethodData, np.ndarray, dict[str, float], set[str]]:
    dataset = ad.read_h5ad(path, backed="r")
    try:
        if "spatial" not in dataset.obsm or "codex_common" not in dataset.obs:
            raise ValueError("SPATCH RNA H5AD lacks spatial/codex_common fields")
        spatial_all = np.asarray(dataset.obsm["spatial"], dtype=np.float64)
        coefficients, diagnostics = fit_coordinate_serialization(dataset.obs_names, spatial_all)
        common = np.asarray(dataset.obs["codex_common"], dtype=int) == 1
        common_index = np.flatnonzero(common)
        symbols = [str(value) for value in dataset.var_names]
        gene_ids = [str(value).split(".", 1)[0] for value in dataset.var["gene_ids"]]
        sr_gene_ids = set(gene_ids)
        symbol_rows: dict[str, list[int]] = {}
        for index, symbol in enumerate(symbols):
            symbol_rows.setdefault(symbol, []).append(index)
        selected_rows = sorted({row for marker in MARKERS for gene in marker.genes for row in symbol_rows.get(gene, [])})
        selected_position = {row: pos for pos, row in enumerate(selected_rows)}
        if selected_rows:
            subset = dataset[common_index, selected_rows].to_memory().X
            if sparse.issparse(subset):
                subset = subset.tocsr()
            else:
                subset = np.asarray(subset)
        else:
            subset = sparse.csr_matrix((len(common_index), 0), dtype=np.float64)
        marker_values = np.zeros((len(common_index), len(MARKERS)), dtype=np.float64)
        coverage = []
        for marker_index, marker in enumerate(MARKERS):
            rows = [row for gene in marker.genes for row in symbol_rows.get(gene, [])]
            positions = [selected_position[row] for row in rows]
            if positions:
                value = subset[:, positions].sum(axis=1)
                marker_values[:, marker_index] = np.asarray(value).ravel()
            coverage.append({
                "protein": marker.protein,
                "genes": ",".join(marker.genes),
                "mapping": marker.mapping,
                "reference_genes_present": sum(gene in symbol_rows for gene in marker.genes),
                "matrix_genes_present": sum(gene in symbol_rows for gene in marker.genes),
                "genes_required": len(marker.genes),
                "matrix_feature_rows": len(rows),
            })
        full_mass = 0.0
        common_mass = 0.0
        chunk = 25000
        for start in range(0, dataset.n_obs, chunk):
            stop = min(dataset.n_obs, start + chunk)
            block = dataset.X[start:stop, :]
            full_mass += float(block.sum())
            local = common[start:stop]
            if np.any(local):
                common_mass += float(block[local, :].sum())
        grids = {
            size: aggregate_rows(spatial_all[common], marker_values, size)
            for size in bin_sizes
        }
        method = MethodData(
            "space_ranger_2020a", grids, full_mass, full_mass,
            {size: common_mass for size in bin_sizes}, coverage, [path],
        )
        return method, coefficients, diagnostics, sr_gene_ids
    finally:
        dataset.file.close()


def load_codex(path: Path, bin_sizes: list[int]) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]], dict[int, set[tuple[int, int]]]
]:
    dataset = ad.read_h5ad(path)
    if "spatial" not in dataset.obsm or "codex_common" not in dataset.obs:
        raise ValueError("SPATCH CODEX H5AD lacks spatial/codex_common fields")
    common = np.asarray(dataset.obs["codex_common"], dtype=int) == 1
    names = [str(value) for value in dataset.var_names]
    if names != [marker.protein for marker in MARKERS]:
        raise ValueError("CODEX marker axis differs from the frozen SPATCH panel")
    values = dataset.X[common]
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values, dtype=np.float64)
    spatial = np.asarray(dataset.obsm["spatial"], dtype=np.float64)[common]
    grids = {size: aggregate_rows(spatial, values, size) for size in bin_sizes}
    key_sets = {size: set(key_index(grids[size][0])) for size in bin_sizes}
    return grids, key_sets


def format_number(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return "NA"
        return format(float(value), ".17g")
    return value


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_number(row.get(field, "")) for field in fields})


def score_methods(
    methods: list[MethodData],
    codex: dict[int, tuple[np.ndarray, np.ndarray]],
    bin_sizes: list[int],
    quantile: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for method in methods:
        for size in bin_sizes:
            protein_keys, protein_values = codex[size]
            method_keys, method_values = method.grids[size]
            method_lookup = key_index(method_keys)
            rna = np.zeros_like(protein_values, dtype=np.float64)
            for protein_row, key in enumerate(protein_keys):
                source = method_lookup.get((int(key[0]), int(key[1])))
                if source is not None:
                    rna[protein_row, :] = method_values[source, :]
            aucs = []
            for marker_index, marker in enumerate(MARKERS):
                protein = protein_values[:, marker_index]
                expression = rna[:, marker_index]
                threshold = float(np.quantile(protein, quantile))
                positive = protein >= threshold
                marker_coverage = method.coverage[marker_index]
                reference_available = (
                    marker_coverage["reference_genes_present"]
                    == marker_coverage["genes_required"]
                )
                auc = roc_auc(positive, expression) if reference_available else math.nan
                aucs.append(auc)
                rows.append({
                    "method": method.label,
                    "bin_size_um": size,
                    "protein": marker.protein,
                    "genes": ",".join(marker.genes),
                    "mapping": marker.mapping,
                    "reference_marker_available": reference_available,
                    "evaluated_bins": len(protein),
                    "protein_positive_bins": int(positive.sum()),
                    "protein_positive_threshold": threshold,
                    "rna_mass": float(expression.sum()),
                    "rna_detected_bins": int(np.count_nonzero(expression > 0)),
                    "rna_detection_fraction": float(np.mean(expression > 0)),
                    "protein_mass_covered_by_rna": float(
                        protein[expression > 0].sum() / protein.sum()
                    ) if protein.sum() > 0 else math.nan,
                    "roc_auc_top_protein_quantile": auc,
                    "pearson_raw": pearson(expression, protein) if reference_available else math.nan,
                    "spearman_raw": spearman(expression, protein) if reference_available else math.nan,
                })
            finite = [value for value in aucs if np.isfinite(value)]
            summaries.append({
                "method": method.label,
                "bin_size_um": size,
                "markers_scored": len(finite),
                "macro_roc_auc": float(np.mean(finite)) if finite else math.nan,
                "median_roc_auc": float(np.median(finite)) if finite else math.nan,
                "direct_panel_rna_mass": float(rna.sum()),
                "method_evaluation_mass": method.evaluation_mass[size],
            })
    return rows, summaries


def main() -> int:
    args = parse_args()
    bin_sizes = sorted(set(args.bin_size or [100, 200, 300, 400, 500]))
    if any(size <= 0 for size in bin_sizes):
        raise SystemExit("bin sizes must be positive")
    if not 0 < args.protein_positive_quantile < 1:
        raise SystemExit("protein positive quantile must be between zero and one")
    required_paths = [args.sr_h5ad, args.codex_h5ad, args.probe_csv]
    if args.sr_filtered_probe_set is not None:
        required_paths.append(args.sr_filtered_probe_set)
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.out_dir}")
    method_specs: list[tuple[str, Path]] = []
    for spec in args.method:
        if "=" not in spec:
            raise SystemExit(f"invalid --method, expected LABEL=MEX_DIR: {spec}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if not label or not path.is_dir():
            raise SystemExit(f"invalid method label/path: {spec}")
        method_specs.append((label, path))
    if len({label for label, _ in method_specs}) != len(method_specs):
        raise SystemExit("duplicate method label")
    reference_features: dict[str, Path] = {}
    for spec in args.reference_feature_list:
        if "=" not in spec:
            raise SystemExit(f"invalid --reference-feature-list: {spec}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if label not in {method_label for method_label, _ in method_specs} or not path.is_file():
            raise SystemExit(f"reference feature list has unknown label or missing path: {spec}")
        if label in reference_features:
            raise SystemExit(f"duplicate reference feature list label: {label}")
        reference_features[label] = path
    args.out_dir.mkdir(parents=True)

    gene_to_symbol, _ = load_probe_symbols(args.probe_csv)
    space_ranger, coefficients, fit_diagnostics, sr_gene_ids = load_space_ranger(
        args.sr_h5ad, bin_sizes,
    )
    space_ranger_accounting = None
    if args.sr_filtered_probe_set is not None:
        filtered_panel = load_filtered_probe_panel(args.sr_filtered_probe_set)
        filtered_panel_genes = set(filtered_panel.pop("eligible_genes"))
        missing = sorted(filtered_panel_genes - sr_gene_ids)
        diagnostic = sorted(sr_gene_ids - filtered_panel_genes)
        if missing or diagnostic:
            raise SystemExit(
                "published Space Ranger H5AD does not equal the filtered "
                f"probe-panel gene axis: missing={len(missing)}, "
                f"diagnostic={len(diagnostic)}"
            )
        space_ranger_accounting = {
            "h5ad_raw_mass": space_ranger.full_mass,
            "h5ad_gene_features": len(sr_gene_ids),
            "filtered_panel_gene_features": len(filtered_panel_genes),
            "diagnostic_non_panel_gene_features": len(diagnostic),
            "axis_exact": True,
            "mass_definition": (
                "published filtered-feature H5AD; diagnostic genes outside "
                "the filtered probe reference are absent"
            ),
            "filtered_probe_panel": filtered_panel,
        }
    codex, codex_keys = load_codex(args.codex_h5ad, bin_sizes)
    methods = [space_ranger]
    for label, directory in method_specs:
        methods.append(load_mex_method(
            label, directory, bin_sizes, coefficients, gene_to_symbol,
            sr_gene_ids, codex_keys, reference_features.get(label),
        ))
    metrics, method_summary = score_methods(
        methods, codex, bin_sizes, args.protein_positive_quantile,
    )
    metric_fields = [
        "method", "bin_size_um", "protein", "genes", "mapping",
        "reference_marker_available",
        "evaluated_bins", "protein_positive_bins", "protein_positive_threshold",
        "rna_mass", "rna_detected_bins", "rna_detection_fraction",
        "protein_mass_covered_by_rna", "roc_auc_top_protein_quantile",
        "pearson_raw", "spearman_raw",
    ]
    summary_fields = [
        "method", "bin_size_um", "markers_scored", "macro_roc_auc",
        "median_roc_auc", "direct_panel_rna_mass", "method_evaluation_mass",
    ]
    write_tsv(args.out_dir / "marker_metrics.tsv", metrics, metric_fields)
    write_tsv(args.out_dir / "method_summary.tsv", method_summary, summary_fields)
    mass_rows = [{
        "method": method.label,
        "full_raw_mass": method.full_mass,
        "space_ranger_2020a_shared_axis_mass": method.shared_axis_mass,
        **{f"evaluation_mass_{size}um": method.evaluation_mass[size] for size in bin_sizes},
    } for method in methods]
    mass_fields = [
        "method", "full_raw_mass", "space_ranger_2020a_shared_axis_mass",
        *(f"evaluation_mass_{size}um" for size in bin_sizes),
    ]
    write_tsv(args.out_dir / "raw_mass.tsv", mass_rows, mass_fields)
    coverage_rows = []
    for method in methods:
        for row in method.coverage:
            coverage_rows.append({"method": method.label, **row})
    write_tsv(
        args.out_dir / "marker_axis_coverage.tsv", coverage_rows,
        [
            "method", "protein", "genes", "mapping", "reference_genes_present",
            "matrix_genes_present", "genes_required", "matrix_feature_rows",
        ],
    )
    if space_ranger_accounting is not None:
        write_tsv(
            args.out_dir / "space_ranger_accounting.tsv",
            [{
                "method": "space_ranger_2020a",
                "h5ad_raw_mass": space_ranger_accounting["h5ad_raw_mass"],
                "h5ad_gene_features": space_ranger_accounting["h5ad_gene_features"],
                "filtered_panel_gene_features": (
                    space_ranger_accounting["filtered_panel_gene_features"]
                ),
                "diagnostic_non_panel_gene_features": 0,
                "filtered_panel_axis_exact": True,
                "mass_definition": space_ranger_accounting["mass_definition"],
            }],
            [
                "method", "h5ad_raw_mass", "h5ad_gene_features",
                "filtered_panel_gene_features", "diagnostic_non_panel_gene_features",
                "filtered_panel_axis_exact", "mass_definition",
            ],
        )
    output_files = [
        args.out_dir / "marker_metrics.tsv",
        args.out_dir / "method_summary.tsv",
        args.out_dir / "raw_mass.tsv",
        args.out_dir / "marker_axis_coverage.tsv",
    ]
    if space_ranger_accounting is not None:
        output_files.append(args.out_dir / "space_ranger_accounting.tsv")
    payload = {
        "schema": "visium_hd_processing.spatch_codex.v1",
        "method": "published fixed Visium-HD/CODEX transfer; common CODEX grid; raw RNA mass",
        "parameters": {
            "bin_sizes_um": bin_sizes,
            "protein_positive_quantile": args.protein_positive_quantile,
            "normalize_method_mass": False,
            "occupancy_jaccard": False,
            "primary_shape_metric": "ROC AUC for top-quantile CODEX protein intensity",
            "secondary_shape_metrics": ["raw Pearson", "raw Spearman"],
        },
        "coordinate_serialization": {
            "design": "[Visium 8um row, Visium 8um column, intercept] -> published SPATCH spatial microns",
            "coefficients": coefficients.tolist(),
            **fit_diagnostics,
            "role": "fixed coordinate serialization only; no expression enters the fit",
        },
        "inputs": {
            "space_ranger_h5ad": {"path": str(args.sr_h5ad.resolve()), "sha256": sha256(args.sr_h5ad)},
            "codex_h5ad": {"path": str(args.codex_h5ad.resolve()), "sha256": sha256(args.codex_h5ad)},
            "probe_csv": {"path": str(args.probe_csv.resolve()), "sha256": sha256(args.probe_csv)},
            "space_ranger_filtered_probe_set": (
                {
                    "path": str(args.sr_filtered_probe_set.resolve()),
                    "sha256": sha256(args.sr_filtered_probe_set),
                }
                if args.sr_filtered_probe_set is not None else None
            ),
            "star_methods": {
                method.label: [
                    {"path": str(path.resolve()), "sha256": sha256(path)} for path in method.inputs
                ] for method in methods if method.label != "space_ranger_2020a"
            },
        },
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_files
        },
        "interpretation": {
            "space_ranger_role": "compatibility comparator, not biological truth",
            "space_ranger_accounting": (
                "filtered-panel gene axis; diagnostic non-panel counts excluded"
                if space_ranger_accounting is not None
                else "not independently validated against a filtered probe reference"
            ),
            "raw_probe_filtered_probes_field": (
                "passed-gDNA-QC annotation; not the 10x included/target_sets "
                "off-target exclusion"
            ),
            "codex_role": "independent noisy protein and transferred-area oracle",
            "reference_arms": "reported separately; 2024-A is not projected onto the 2020-A Space Ranger axis",
        },
        "space_ranger_accounting": space_ranger_accounting,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in [*output_files, summary_path]),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
