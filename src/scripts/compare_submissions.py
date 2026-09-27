#!/usr/bin/env python3
"""
Compare two CASMI26 submission.csv files and quantify how many
molecule_id predictions actually changed.

Usage:
    python compare_submissions.py submission_A.csv submission_B.csv
    
Output:
    - How many molecule_ids have an identical top-25 list (no change at all)
    - How many changed rank-1 (the only thing that matters for MRR@25,
      since a change lower in the list only matters if the correct
      answer happens to live there)
    - How many changed anywhere in the top-25 but kept rank-1 the same
    - A CSV of the rank-1 changes so you can eyeball which molecules moved
    - Jaccard overlap of the full candidate sets, as a sanity check for
      "did the pool/ranker change a little or a lot"
"""

import sys
import csv
import argparse
from collections import Counter


def load_submission(path):
    """Return {molecule_id: [smiles, smiles, ...]} preserving rank order."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames}
        if "molecule_id" not in cols or "smiles" not in cols:
            raise ValueError(
                f"{path}: expected columns molecule_id,smiles — found {reader.fieldnames}"
            )
        mid_col, smi_col = cols["molecule_id"], cols["smiles"]
        for row in reader:
            mid = row[mid_col].strip()
            smi_field = row[smi_col]
            preds = [s.strip() for s in smi_field.split(";") if s.strip()]
            if mid in rows:
                raise ValueError(f"{path}: duplicate molecule_id {mid}")
            rows[mid] = preds
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file_a", help="baseline submission csv (e.g. the 0.334 anchor)")
    ap.add_argument("file_b", help="comparison submission csv (e.g. the 0.315 run)")
    ap.add_argument(
        "--out",
        default="rank1_changes.csv",
        help="where to write the rank-1-change detail csv (default: rank1_changes.csv)",
    )
    ap.add_argument(
        "--top-n-preview",
        type=int,
        default=20,
        help="how many changed molecules to print inline (default: 20)",
    )
    args = ap.parse_args()

    a = load_submission(args.file_a)
    b = load_submission(args.file_b)

    ids_a, ids_b = set(a), set(b)
    common = ids_a & ids_b
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a

    print(f"=== {args.file_a} ===  {len(ids_a)} molecule_ids")
    print(f"=== {args.file_b} ===  {len(ids_b)} molecule_ids")
    print(f"Common molecule_ids: {len(common)}")
    if only_a:
        print(f"  ⚠ {len(only_a)} molecule_ids only in A (missing from B)")
    if only_b:
        print(f"  ⚠ {len(only_b)} molecule_ids only in B (missing from A)")
    print()

    identical_full = 0
    rank1_changed = 0
    rank1_same_but_list_changed = 0
    both_empty_or_trivial = 0

    rank1_change_rows = []
    jaccard_scores = []

    for mid in sorted(common):
        preds_a = a[mid]
        preds_b = b[mid]

        if preds_a == preds_b:
            identical_full += 1
            continue

        top1_a = preds_a[0] if preds_a else None
        top1_b = preds_b[0] if preds_b else None

        set_a, set_b = set(preds_a), set(preds_b)
        union = set_a | set_b
        inter = set_a & set_b
        jaccard = (len(inter) / len(union)) if union else 1.0
        jaccard_scores.append(jaccard)

        if top1_a != top1_b:
            rank1_changed += 1
            rank1_change_rows.append(
                {
                    "molecule_id": mid,
                    "rank1_A": top1_a,
                    "rank1_B": top1_b,
                    "rank_of_A_top1_in_B": (preds_b.index(top1_a) + 1)
                    if top1_a in preds_b
                    else "not_in_B_top25",
                    "rank_of_B_top1_in_A": (preds_a.index(top1_b) + 1)
                    if top1_b in preds_a
                    else "not_in_A_top25",
                    "jaccard_top25": round(jaccard, 3),
                }
            )
        else:
            rank1_same_but_list_changed += 1

    n = len(common)
    print("=== Summary ===")
    print(f"Identical top-25 list (no change at all):      {identical_full:4d} / {n}")
    print(f"Rank-1 changed (this is what moves MRR@25):     {rank1_changed:4d} / {n}")
    print(f"Rank-1 same, but list below rank 1 changed:     {rank1_same_but_list_changed:4d} / {n}")
    print()

    if jaccard_scores:
        avg_j = sum(jaccard_scores) / len(jaccard_scores)
        print(f"Mean Jaccard overlap of top-25 sets (over changed molecules): {avg_j:.3f}")
        print("  (1.0 = same candidates just reordered; low = genuinely different candidate lists)")
        print()

    # Rough MRR delta estimate assuming rank-1-changed molecules flip
    # from "correct at rank1" to "incorrect" or vice versa is NOT knowable
    # without ground truth. This script only tells you *how many* rank-1
    # predictions differ, which upper-bounds how many molecules could have
    # flipped between "MRR contribution" and "0" between the two submissions.
    print(f"NOTE: {rank1_changed} molecules have a different top-1 guess between A and B.")
    print("This is the upper bound on how many molecules could have flipped score")
    print("(the true count of *scored* flips is <= this, since not every rank-1")
    print("swap changes correctness against the hidden ground truth).")
    print()

    if rank1_change_rows:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rank1_change_rows[0].keys()))
            writer.writeheader()
            writer.writerows(rank1_change_rows)
        print(f"Wrote {len(rank1_change_rows)} rank-1 changes to {args.out}")
        print()
        preview = rank1_change_rows[: args.top_n_preview]
        print(f"Preview (first {len(preview)} rank-1 changes):")
        for r in preview:
            print(
                f"  {r['molecule_id']}: "
                f"A='{r['rank1_A'][:40]}' -> B='{r['rank1_B'][:40]}' "
                f"(A's pick still in B's list at rank {r['rank_of_A_top1_in_B']}; "
                f"B's pick was in A's list at rank {r['rank_of_B_top1_in_A']})"
            )


if __name__ == "__main__":
    main()