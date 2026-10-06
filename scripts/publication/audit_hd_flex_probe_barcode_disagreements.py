#!/usr/bin/env python3
"""Audit probe and barcode evidence in asymmetric HD Flex read assignments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

COORDINATE = re.compile(r"^s_002um_(\d+)_(\d+)(?:-\d+)?$")
SEED_BOUNDS = ((0, 17), (17, 34), (34, 50))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_coordinate(value: str) -> tuple[int, int] | None:
    match = COORDINATE.fullmatch(value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def fastq_id(header: str) -> str:
    value = header.rstrip("\r\n").split()[0]
    if value.startswith("@"):
        value = value[1:]
    return value[:-2] if value.endswith(("/1", "/2")) else value


@dataclass(frozen=True)
class Probe:
    sequence: str
    gene_id: str
    probe_id: str


@dataclass(frozen=True)
class ProbeMatch:
    distance: int | None
    probe_count: int
    genes: tuple[str, ...]


class ProbeIndex:
    """Complete Hamming-distance <=2 lookup for 50-base probe sequences."""

    def __init__(self, probes: list[Probe]):
        if not probes or any(len(probe.sequence) != 50 for probe in probes):
            raise ValueError("ProbeIndex requires non-empty 50-base probes")
        self.probes = probes
        self.seeds: list[dict[str, list[int]]] = [defaultdict(list) for _ in SEED_BOUNDS]
        for index, probe in enumerate(probes):
            for seed_index, (begin, end) in enumerate(SEED_BOUNDS):
                self.seeds[seed_index][probe.sequence[begin:end]].append(index)

    def match(self, sequence: str) -> ProbeMatch:
        sequence = sequence[:50].upper()
        if len(sequence) != 50:
            return ProbeMatch(None, 0, ())
        candidates: set[int] = set()
        for seed_index, (begin, end) in enumerate(SEED_BOUNDS):
            candidates.update(self.seeds[seed_index].get(sequence[begin:end], ()))
        best = 3
        hits: list[Probe] = []
        for index in candidates:
            probe = self.probes[index]
            distance = sum(left != right for left, right in zip(sequence, probe.sequence, strict=True))
            if distance < best:
                best, hits = distance, [probe]
            elif distance == best:
                hits.append(probe)
        if best > 2:
            return ProbeMatch(None, 0, ())
        return ProbeMatch(best, len(hits), tuple(sorted({probe.gene_id for probe in hits})))


def load_probes(path: Path) -> list[Probe]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.startswith("#"):
                header = line
                break
        else:
            raise ValueError("probe set contains no CSV header")
        reader = csv.DictReader([header, *handle])
        required = {"gene_id", "probe_seq", "probe_id", "included"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("probe set is missing required columns")
        probes = [
            Probe(row["probe_seq"].upper(), row["gene_id"], row["probe_id"])
            for row in reader if row["included"].strip().lower() == "true"
        ]
    if len({probe.sequence for probe in probes}) != len(probes):
        raise ValueError("included probe sequences are not unique")
    return probes


def load_target_genes(path: Path) -> set[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        features = handle["matrix/features"]
        identifiers = [value.decode() for value in features["id"][:]]
        target_sets = features["target_sets"]
        if len(target_sets) != 1:
            raise ValueError("expected exactly one vendor target set")
        indexes = next(iter(target_sets.values()))[:]
    return {identifiers[int(index)] for index in indexes}


def load_open_reads(path: Path, umi_mode: str) -> dict[str, dict[str, object]]:
    reads: dict[str, dict[str, object]] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "umi_mode", "product", "feature_id", "corrected_umi",
            "unit_2um", "member_read_ids",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("open hard-molecule table has the wrong schema")
        for row in reader:
            if row["umi_mode"] != umi_mode or row["product"] != "postcollapse_hard":
                continue
            coordinate = parse_coordinate(row["unit_2um"])
            if coordinate is None:
                raise ValueError(f"invalid open coordinate: {row['unit_2um']}")
            value = {
                "gene": row["feature_id"], "umi": row["corrected_umi"],
                "coordinate": coordinate,
            }
            for read_id in row["member_read_ids"].split(";"):
                if read_id in reads and reads[read_id] != value:
                    raise ValueError(f"open read has conflicting hard assignments: {read_id}")
                reads[read_id] = value
    return reads


def load_cliques(path: Path) -> dict[str, dict[str, object]]:
    cliques: dict[str, dict[str, object]] = {}
    read_to_clique: dict[str, str] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_clique_id", "member_read_ids", "candidate", "posterior"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("read-clique table has the wrong schema")
        for row in reader:
            coordinate = parse_coordinate(row["candidate"])
            if coordinate is None:
                raise ValueError(f"invalid clique coordinate: {row['candidate']}")
            clique = cliques.setdefault(row["read_clique_id"], {"candidates": [], "reads": row["member_read_ids"].split(";")})
            clique["candidates"].append((coordinate, float(row["posterior"])))
    for clique_id, clique in cliques.items():
        candidates = clique["candidates"]
        if abs(sum(value for _, value in candidates) - 1.0) > 1e-9:
            raise ValueError(f"posterior does not normalize for {clique_id}")
        candidates.sort(key=lambda item: (-item[1], item[0]))
        for read_id in clique["reads"]:
            if read_id in read_to_clique:
                raise ValueError(f"read occurs in multiple cliques: {read_id}")
            read_to_clique[read_id] = clique_id
    return {read_id: cliques[clique_id] for read_id, clique_id in read_to_clique.items()}


def load_raw_candidates(path: Path) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = defaultdict(list)
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "row2", "col2", "min_tier", "bc1_edit", "bc2_edit", "candidate_count"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("raw candidate table has the wrong schema")
        for row in reader:
            result[row["read_id"]].append({
                "coordinate": (int(row["row2"]), int(row["col2"])),
                "min_tier": int(row["min_tier"]),
                "bc1_edit": int(row["bc1_edit"]),
                "bc2_edit": int(row["bc2_edit"]),
                "candidate_count": int(row["candidate_count"]),
            })
    for read_id, rows in result.items():
        declared = {int(row["candidate_count"]) for row in rows}
        if declared != {len(rows)} or len({row["coordinate"] for row in rows}) != len(rows):
            raise ValueError(f"raw candidates do not reconcile for {read_id}")
    return result


def load_star_features(path: Path) -> set[str]:
    import pysam

    features: dict[str, tuple[str, str]] = {}
    ambiguous: set[str] = set()
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_unmapped or record.is_secondary or record.is_supplementary:
                continue
            gene = str(record.get_tag("GX")) if record.has_tag("GX") else ""
            umi = str(record.get_tag("UR")) if record.has_tag("UR") else ""
            if not gene or gene == "-" or ";" in gene or "," in gene:
                continue
            value = (gene, umi)
            if record.query_name in features and features[record.query_name] != value:
                ambiguous.add(record.query_name)
            else:
                features[record.query_name] = value
    return set(features) - ambiguous


def load_star_feature_ledger(path: Path) -> set[str]:
    features: set[str] = set()
    seen: set[str] = set()
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"read_id", "action", "feature_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("STAR feature-lineage ledger has the wrong schema")
        for row in reader:
            read_id = row["read_id"]
            if read_id in seen:
                raise ValueError(f"duplicate STAR feature-lineage read: {read_id}")
            seen.add(read_id)
            if row["action"] == "keep":
                if not row["feature_id"]:
                    raise ValueError(f"STAR KEEP lacks a feature ID: {read_id}")
                features.add(read_id)
    return features


def load_vendor_reads(path: Path, target_genes: set[str]) -> dict[str, dict[str, object]]:
    import pysam

    reads: dict[str, dict[str, object]] = {}
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                continue
            if record.query_name in reads:
                raise ValueError(f"duplicate vendor primary read: {record.query_name}")
            coordinate = parse_coordinate(str(record.get_tag("sb"))) if record.has_tag("sb") else None
            gene = str(record.get_tag("GX")) if record.has_tag("GX") else ""
            umi = str(record.get_tag("UB")) if record.has_tag("UB") else ""
            xf = int(record.get_tag("xf")) if record.has_tag("xf") else 0
            reads[record.query_name] = {
                "coordinate": coordinate, "gene": gene, "umi": umi, "xf": xf,
                "target_gene": gene in target_genes,
                "eligible": bool(coordinate and umi and gene in target_genes and (xf & 1)),
            }
    return reads


def load_r2_sequences(paths: list[Path], requested: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="ascii") as handle:
            while True:
                header = handle.readline()
                if not header:
                    break
                sequence = handle.readline().strip().upper()
                plus, quality = handle.readline(), handle.readline().strip()
                if not plus or len(sequence) != len(quality):
                    raise ValueError(f"malformed FASTQ: {path}")
                read_id = fastq_id(header)
                if read_id in requested:
                    if read_id in result:
                        raise ValueError(f"duplicate R2 read ID: {read_id}")
                    result[read_id] = sequence
    if set(result) != requested:
        missing = sorted(requested - set(result))
        raise ValueError(f"R2 FASTQs are missing {len(missing)} requested reads; first={missing[:3]}")
    return result


def format_coordinate(value: tuple[int, int] | None) -> str:
    return "" if value is None else f"s_002um_{value[0]}_{value[1]}"


def ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-set", type=Path, required=True)
    parser.add_argument("--r2-fastq", type=Path, action="append", required=True)
    parser.add_argument("--open-candidates", type=Path, required=True)
    parser.add_argument("--open-read-cliques", type=Path, required=True)
    parser.add_argument("--open-hard-molecules", type=Path, required=True)
    star_feature_source = parser.add_mutually_exclusive_group(required=True)
    star_feature_source.add_argument("--star-feature-bam", type=Path)
    star_feature_source.add_argument("--star-feature-ledger", type=Path)
    parser.add_argument("--vendor-bam", type=Path, required=True)
    parser.add_argument("--vendor-target-h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--umi-mode", choices=("1mm_cr", "exact"), default="1mm_cr")
    parser.add_argument("--expected-raw-reads", type=int, default=100000)
    args = parser.parse_args()
    selected_star_feature_source = args.star_feature_ledger or args.star_feature_bam
    inputs = [
        args.probe_set, *args.r2_fastq, args.open_candidates, args.open_read_cliques,
        args.open_hard_molecules, selected_star_feature_source,
        args.vendor_bam, args.vendor_target_h5,
    ]
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.out_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    target_genes = load_target_genes(args.vendor_target_h5)
    open_reads = load_open_reads(args.open_hard_molecules, args.umi_mode)
    cliques = load_cliques(args.open_read_cliques)
    raw_candidates = load_raw_candidates(args.open_candidates)
    star_features = (
        load_star_feature_ledger(args.star_feature_ledger)
        if args.star_feature_ledger is not None
        else load_star_features(args.star_feature_bam)
    )
    vendor_reads = load_vendor_reads(args.vendor_bam, target_genes)
    if len(vendor_reads) != args.expected_raw_reads:
        raise SystemExit(f"vendor BAM primary universe differs from expectation: {len(vendor_reads)}")
    vendor_eligible = {read_id for read_id, row in vendor_reads.items() if row["eligible"]}
    open_set, vendor_set = set(open_reads), vendor_eligible
    star_only, vendor_only = open_set - vendor_set, vendor_set - open_set
    shared = open_set & vendor_set
    requested = open_set | vendor_set
    sequences = load_r2_sequences(args.r2_fastq, requested)
    probe_index = ProbeIndex(load_probes(args.probe_set))
    probe_matches = {read_id: probe_index.match(sequence) for read_id, sequence in sequences.items()}

    detail_fields = (
        "assignment_set", "read_id", "probe_min_hamming", "nearest_probe_count",
        "nearest_gene_count", "nearest_gene_ids", "assigned_gene", "assigned_gene_supported",
        "open_coordinate", "vendor_coordinate", "raw_candidate_count", "raw_min_tier",
        "open_coordinate_raw_supported", "vendor_coordinate_raw_supported",
        "open_candidate_posterior", "vendor_spatial", "vendor_ub", "vendor_target_gene",
        "vendor_xf_countable", "star_feature_available", "eligibility_explanation",
    )
    detail_rows: list[dict[str, object]] = []
    probe_counts: dict[str, Counter[str]] = defaultdict(Counter)
    barcode_counts: dict[str, Counter[str]] = defaultdict(Counter)
    probe_groups = (
        ("star_all", open_set, lambda read_id: str(open_reads[read_id]["gene"])),
        ("space_ranger_all", vendor_set, lambda read_id: str(vendor_reads[read_id]["gene"])),
        ("shared", shared, lambda read_id: str(open_reads[read_id]["gene"])),
        ("star_only", star_only, lambda read_id: str(open_reads[read_id]["gene"])),
        ("space_ranger_only", vendor_only, lambda read_id: str(vendor_reads[read_id]["gene"])),
    )
    for label, read_ids, assigned_gene_for in probe_groups:
        for read_id in read_ids:
            match = probe_matches[read_id]
            distance_label = f"hamming_{match.distance}" if match.distance is not None else "greater_than_2"
            probe_counts[label][distance_label] += 1
            probe_counts[label]["assigned_gene_supported"] += int(assigned_gene_for(read_id) in match.genes)
            probe_counts[label]["unique_nearest_gene"] += int(len(match.genes) == 1)
    for label, read_ids in (("star_only", star_only), ("space_ranger_only", vendor_only)):
        for read_id in sorted(read_ids):
            match = probe_matches[read_id]
            open_row = open_reads.get(read_id)
            vendor_row = vendor_reads[read_id]
            raw_rows = raw_candidates.get(read_id, [])
            raw_coordinates = {row["coordinate"] for row in raw_rows}
            open_coordinate = open_row["coordinate"] if open_row else None
            vendor_coordinate = vendor_row["coordinate"]
            assigned_gene = str(open_row["gene"] if open_row else vendor_row["gene"])
            barcode_counts[label]["reads"] += 1
            barcode_counts[label]["raw_candidate_present"] += int(bool(raw_rows))
            barcode_counts[label]["raw_candidate_ambiguous"] += int(len(raw_rows) > 1)
            barcode_counts[label]["open_coordinate_raw_supported"] += int(open_coordinate in raw_coordinates)
            barcode_counts[label]["vendor_coordinate_raw_supported"] += int(vendor_coordinate in raw_coordinates)
            barcode_counts[label]["star_feature_available"] += int(read_id in star_features)
            barcode_counts[label]["vendor_spatial"] += int(bool(vendor_coordinate))
            if label == "star_only" and vendor_coordinate and open_coordinate:
                barcode_counts[label]["spatial_comparable"] += 1
                for scale, factor in (("2um", 1), ("8um", 4), ("16um", 8)):
                    agrees = (
                        open_coordinate[0] // factor, open_coordinate[1] // factor
                    ) == (
                        vendor_coordinate[0] // factor, vendor_coordinate[1] // factor
                    )
                    barcode_counts[label][f"coordinate_agreement_{scale}"] += int(agrees)
            if raw_rows:
                barcode_counts[label][f"raw_min_tier_{min(int(row['min_tier']) for row in raw_rows)}"] += 1
            if label == "star_only":
                missing = []
                if not vendor_coordinate:
                    missing.append("no_sr_coordinate")
                if not vendor_row["umi"]:
                    missing.append("no_sr_corrected_umi")
                if not vendor_row["target_gene"]:
                    missing.append("no_sr_target_gene")
                if not (int(vendor_row["xf"]) & 1):
                    missing.append("not_sr_countable")
                explanation = ";".join(missing) or "unknown_sr_ineligibility"
            elif not raw_rows:
                explanation = "no_open_barcode_candidate"
            elif read_id not in star_features:
                explanation = "no_star_feature_assignment"
            else:
                explanation = "not_joined_or_resolved"
            barcode_counts[label][f"explanation:{explanation}"] += 1
            clique = cliques.get(read_id)
            posterior = ""
            if clique and open_coordinate is not None:
                posterior = next(
                    (value for coordinate, value in clique["candidates"] if coordinate == open_coordinate), ""
                )
            detail_rows.append({
                "assignment_set": label,
                "read_id": read_id,
                "probe_min_hamming": match.distance if match.distance is not None else ">2",
                "nearest_probe_count": match.probe_count,
                "nearest_gene_count": len(match.genes),
                "nearest_gene_ids": ";".join(match.genes),
                "assigned_gene": assigned_gene,
                "assigned_gene_supported": assigned_gene in match.genes,
                "open_coordinate": format_coordinate(open_coordinate),
                "vendor_coordinate": format_coordinate(vendor_coordinate),
                "raw_candidate_count": len(raw_rows),
                "raw_min_tier": min((int(row["min_tier"]) for row in raw_rows), default=""),
                "open_coordinate_raw_supported": open_coordinate in raw_coordinates,
                "vendor_coordinate_raw_supported": vendor_coordinate in raw_coordinates,
                "open_candidate_posterior": posterior,
                "vendor_spatial": bool(vendor_coordinate),
                "vendor_ub": bool(vendor_row["umi"]),
                "vendor_target_gene": bool(vendor_row["target_gene"]),
                "vendor_xf_countable": bool(int(vendor_row["xf"]) & 1),
                "star_feature_available": read_id in star_features,
                "eligibility_explanation": explanation,
            })

    detail_path = args.out_dir / "read_audit.tsv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(detail_rows)
    probe_rows = []
    probe_denominators = {
        "star_all": len(open_set), "space_ranger_all": len(vendor_set),
        "shared": len(shared), "star_only": len(star_only),
        "space_ranger_only": len(vendor_only),
    }
    for label in probe_denominators:
        denominator = probe_denominators[label]
        for metric in (
            "hamming_0", "hamming_1", "hamming_2", "greater_than_2",
            "assigned_gene_supported", "unique_nearest_gene",
        ):
            value = probe_counts[label][metric]
            probe_rows.append({"assignment_set": label, "metric": metric, "reads": value, "denominator": denominator, "fraction": ratio(value, denominator)})
    probe_path = args.out_dir / "probe_summary.tsv"
    with probe_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("assignment_set", "metric", "reads", "denominator", "fraction"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(probe_rows)
    barcode_rows = []
    for label, counter in sorted(barcode_counts.items()):
        denominator = counter["reads"]
        for metric, value in sorted(counter.items()):
            if metric == "reads":
                continue
            barcode_rows.append({"assignment_set": label, "metric": metric, "reads": value, "denominator": denominator, "fraction": ratio(value, denominator)})
    barcode_path = args.out_dir / "barcode_eligibility_summary.tsv"
    with barcode_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("assignment_set", "metric", "reads", "denominator", "fraction"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(barcode_rows)

    summary = {
        "schema": "visium_hd_processing.flex_probe_barcode_disagreement_audit.v1",
        "status": "pass",
        "parameters": {"umi_mode": args.umi_mode, "maximum_probe_hamming": 2},
        "inputs": {str(path): sha256(path) for path in inputs},
        "counts": {
            "raw_primary_reads": len(vendor_reads), "open_eligible_reads": len(open_set),
            "vendor_eligible_reads": len(vendor_set), "shared_eligible_reads": len(shared),
            "star_only_reads": len(star_only), "space_ranger_only_reads": len(vendor_only),
        },
        "probe_counts": {label: dict(sorted(counter.items())) for label, counter in sorted(probe_counts.items())},
        "barcode_counts": {label: dict(sorted(counter.items())) for label, counter in sorted(barcode_counts.items())},
        "invariants": {
            "raw_universe_reconciles": len(vendor_reads) == args.expected_raw_reads,
            "eligibility_partition_reconciles": len(shared) + len(star_only) + len(vendor_only) + len(set(vendor_reads) - open_set - vendor_set) == len(vendor_reads),
            "open_hard_assignments_raw_candidate_supported": barcode_counts["star_only"]["open_coordinate_raw_supported"] == len(star_only),
            "vendor_not_used_as_truth": True,
        },
    }
    if not all(summary["invariants"].values()):
        raise SystemExit(f"audit invariant failed: {summary['invariants']}")
    summary["outputs"] = {
        "read_audit": {"path": str(detail_path.resolve()), "sha256": sha256(detail_path)},
        "probe_summary": {"path": str(probe_path.resolve()), "sha256": sha256(probe_path)},
        "barcode_eligibility_summary": {"path": str(barcode_path.resolve()), "sha256": sha256(barcode_path)},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
