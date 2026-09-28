#!/usr/bin/env python3
"""
Compare local_rank_*.csv outputs (from local_c2_rank_harness.py, run once per
notebook) to see whether the TRUE structure's rank actually moves between the
three pipeline configs, for the molecules where rank-1 is wrong.

Usage:
    python compare_local_ranks.py local_rank_refill_0334.csv \
                                   local_rank_v27_3_0320.csv \
                                   local_rank_v27_0315.csv
"""
import sys
import pandas as pd

def load(path):
    df = pd.read_csv(path)
    df = df.set_index('inchikey14')
    return df

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:4]
    names = [p.split('local_rank_')[-1].replace('.csv', '') for p in paths]
    dfs = [load(p) for p in paths]

    common = set(dfs[0].index) & set(dfs[1].index) & set(dfs[2].index)
    print(f"Common structures across all three: {len(common)}")

    rows = []
    for key in common:
        ranks = [df.loc[key, 'rank'] for df in dfs]
        rows.append((key,) + tuple(ranks))

    cmp_df = pd.DataFrame(rows, columns=['inchikey14'] + [f'rank_{n}' for n in names])

    print("\n=== Per-config rank-1 counts ===")
    for n in names:
        col = f'rank_{n}'
        r1 = (cmp_df[col] == 1).sum()
        found = cmp_df[col].notna().sum()
        print(f"  {n}: rank-1 = {r1}/{len(cmp_df)}, found in top-25 = {found}/{len(cmp_df)}")

    # Molecules where rank-1 is wrong in ALL three (the ones where reordering
    # among 2-25 is the only thing that could differ)
    cols = [f'rank_{n}' for n in names]
    never_rank1 = cmp_df[(cmp_df[cols] != 1).all(axis=1)]
    print(f"\nMolecules never at rank-1 in any config: {len(never_rank1)}/{len(cmp_df)}")

    moved = never_rank1[never_rank1[cols].nunique(axis=1) > 1]
    print(f"  ...of those, rank DIFFERS across configs: {len(moved)}")
    print(f"  ...of those, rank is IDENTICAL across all three: {len(never_rank1) - len(moved)}")

    if len(moved):
        print("\nSample of molecules whose true-answer rank moved between configs:")
        print(moved.head(20).to_string(index=False))

    out_path = "rank_comparison_detail.csv"
    cmp_df.to_csv(out_path, index=False)
    print(f"\nFull per-molecule comparison saved to {out_path}")

    print("\n=== Interpretation guide ===")
    print("If 'rank DIFFERS across configs' is small: the ranks-2-25 reordering you")
    print("found on the real submissions is mostly happening among WRONG candidates")
    print("and genuinely doesn't move MRR@25 much -- consistent with the LB gaps being")
    print("close to noise, as the strategy report argues.")
    print("If it's large: the reordering IS moving the true answer around meaningfully,")
    print("and a fusion/ensemble of these ranking signals could plausibly help -- worth")
    print("testing a reciprocal-rank-fusion combination on this same local set before")
    print("spending a submission on it.")


if __name__ == "__main__":
    main()