#!/usr/bin/env python3
"""Reads with a Space Ranger square but no square within two edits per half.

For each such read on slide H1-VM2JXXK/A1, asks whether its molecule (same
gene, Space Ranger corrected UMI and square) has another read whose own
candidate squares include that square. Reads the July 2026 candidate-
preservation shards (Space Ranger BAM tags and STAR candidates per read);
writes figures/data/sr_audit_no_candidate_mates.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

S = "<runs>/20260716_vm2jxxk_full_natural_candidate_reference_v1/candidate_preservation"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "sr_audit_no_candidate_mates.json"


def shard(part):
    sr = pd.read_csv(f"{S}/sr_read_shards/{part}", sep="\t", dtype={"sr_cb": str},
                     usecols=["read_id", "feature_id", "sr_corrected_umi", "sr_cb", "star_candidate_count"])
    has_cb = sr["sr_cb"].notna() & (sr["sr_cb"] != "")
    orphan = sr[has_cb & (sr["star_candidate_count"] == 0)]
    if orphan.empty:
        return 0, 0
    cand = pd.read_csv(f"{S}/candidate_shards/{part}", sep="\t", usecols=["read_id", "row2", "col2"])
    cb = sr.loc[has_cb, ["read_id", "feature_id", "sr_corrected_umi", "sr_cb"]]
    rc = cb["sr_cb"].str.extract(r"s_002um_(\d+)_(\d+)-1").astype(int)
    cb = cb.assign(row=rc[0].to_numpy(), col=rc[1].to_numpy())
    own = cand.merge(cb[["read_id", "row", "col"]], left_on=["read_id", "row2", "col2"],
                     right_on=["read_id", "row", "col"])
    cb["own_support"] = cb["read_id"].isin(set(own["read_id"]))
    key = ["feature_id", "sr_corrected_umi", "sr_cb"]
    mates = cb.groupby(key)["own_support"].sum().rename("supporting")
    joined = orphan.merge(mates, left_on=key, right_index=True, how="left")
    return len(joined), int((joined["supporting"] > 0).sum())


def main():
    parts = [f"part-{i:03d}.tsv.gz" for i in range(256)]
    with Pool(16) as pool:
        results = pool.map(shard, parts)
    record = {"source": S, "reads_with_square_but_no_candidate": sum(r[0] for r in results),
              "with_supporting_molecule_mate": sum(r[1] for r in results)}
    OUT.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
