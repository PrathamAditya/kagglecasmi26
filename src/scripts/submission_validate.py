#!/usr/bin/env python3
"""
Pre-submission validator for CASMI26 submission.csv files.

Checks two tiers:
  A) HARD-REJECT conditions (per the competition's own rules) — these WILL
     cause Kaggle to reject the submission outright.
  B) SANITY WARNINGS — won't get rejected, but are red flags worth
     investigating before you spend a submission slot.

Usage:
    python validate_submission.py submission.csv sample_submission.csv

sample_submission.csv is used to check molecule_id coverage matches exactly.
If you don't have it handy, pass just the submission file and coverage
checks will be skipped (format/sanity checks still run).
"""
import sys
import csv
import argparse

# The competition pins RDKit at 2026.03.3 for its own tautomer-canonicalization
# scoring step (see Overview: "Matching"). A different version can canonicalize
# tautomers slightly differently, so parseability/canonicalization checks here
# are only guaranteed consistent with actual Kaggle scoring on this version.
EXPECTED_RDKIT_VERSION = "2026.03.3"

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit import rdBase
    RDLogger.DisableLog('rdApp.*')  # silence RDKit's own warnings, we handle them
    HAVE_RDKIT = True
    INSTALLED_RDKIT_VERSION = rdBase.rdkitVersion
except ImportError:
    HAVE_RDKIT = False
    INSTALLED_RDKIT_VERSION = None


def load(path):
    """Returns dict {molecule_id: [smiles, ...]} and list of raw rows for null/dup checks."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, [], "File is empty or has no header"
        cols = {c.lower(): c for c in reader.fieldnames}
        if "molecule_id" not in cols or "smiles" not in cols:
            return None, [], f"Missing required column(s). Found columns: {reader.fieldnames}"
        mid_col, smi_col = cols["molecule_id"], cols["smiles"]
        for row in reader:
            rows.append((row[mid_col], row[smi_col]))
    return mid_col, rows, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission", help="path to submission.csv to validate")
    ap.add_argument("sample", nargs="?", default=None,
                     help="path to sample_submission.csv (optional, enables coverage check)")
    ap.add_argument("--fallback-smiles", default="CCO",
                     help="the placeholder/fallback SMILES your pipeline uses when retrieval "
                          "fails (default: CCO). Rows using this are flagged as sanity warnings.")
    args = ap.parse_args()

    hard_fails = []
    warnings = []

    _, rows, err = load(args.submission)
    if err:
        print(f"❌ HARD FAIL: {err}")
        sys.exit(1)

    if not rows:
        hard_fails.append("Submission file has a header but zero data rows")

    seen_ids = set()
    dup_ids = set()
    null_rows = []
    over_25 = []
    empty_candidate_lists = []
    unparseable = []
    fallback_heavy_rows = []
    all_preds_per_row = {}

    for mid, smi_field in rows:
        mid = (mid or "").strip()
        if not mid or smi_field is None or smi_field.strip() == "":
            null_rows.append(mid or "<blank molecule_id>")
            continue

        if mid in seen_ids:
            dup_ids.add(mid)
        seen_ids.add(mid)

        preds = [s.strip() for s in smi_field.split(";") if s.strip()]
        all_preds_per_row[mid] = preds

        if not preds:
            empty_candidate_lists.append(mid)
            continue
        if len(preds) > 25:
            over_25.append((mid, len(preds)))

        if preds[0] == args.fallback_smiles:
            fallback_heavy_rows.append(mid)

        if HAVE_RDKIT:
            for smi in preds:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    unparseable.append((mid, smi))

    # --- Tier A: hard-reject conditions ---
    if null_rows:
        hard_fails.append(f"{len(null_rows)} row(s) with null/blank molecule_id or smiles")
    if dup_ids:
        hard_fails.append(f"{len(dup_ids)} duplicate molecule_id(s): {sorted(dup_ids)[:10]}"
                           f"{' ...' if len(dup_ids) > 10 else ''}")
    if over_25:
        hard_fails.append(f"{len(over_25)} row(s) exceed 25 candidates: "
                           f"{over_25[:5]}{' ...' if len(over_25) > 5 else ''}")
    if empty_candidate_lists:
        hard_fails.append(f"{len(empty_candidate_lists)} row(s) have an empty candidate list "
                           f"after splitting on ';'")

    if args.sample:
        _, sample_rows, serr = load(args.sample)
        if serr:
            warnings.append(f"Could not read sample file for coverage check: {serr}")
        else:
            sample_ids = {mid.strip() for mid, _ in sample_rows}
            sub_ids = seen_ids
            missing = sample_ids - sub_ids
            extra = sub_ids - sample_ids
            if missing:
                hard_fails.append(f"{len(missing)} molecule_id(s) from sample_submission missing "
                                   f"in your submission: {sorted(missing)[:10]}"
                                   f"{' ...' if len(missing) > 10 else ''}")
            if extra:
                warnings.append(f"{len(extra)} molecule_id(s) in your submission not present in "
                                 f"sample_submission (harmless unless it indicates a mismatch): "
                                 f"{sorted(extra)[:10]}{' ...' if len(extra) > 10 else ''}")

    # --- Tier B: sanity warnings ---
    if HAVE_RDKIT and unparseable:
        n_bad = len(unparseable)
        n_rows_affected = len({mid for mid, _ in unparseable})
        warnings.append(f"{n_bad} unparseable SMILES across {n_rows_affected} row(s) — these will "
                         f"score as wrong (not reject the file, just silently fail to match). "
                         f"Examples: {unparseable[:5]}")
    elif not HAVE_RDKIT:
        warnings.append(f"RDKit not installed — skipped SMILES-parseability check "
                         f"(pip install rdkit=={EXPECTED_RDKIT_VERSION} to match the "
                         f"competition's pinned scoring version)")

    if HAVE_RDKIT and INSTALLED_RDKIT_VERSION != EXPECTED_RDKIT_VERSION:
        warnings.append(f"Installed RDKit is {INSTALLED_RDKIT_VERSION}, but the competition "
                         f"pins {EXPECTED_RDKIT_VERSION} for its tautomer-canonicalization "
                         f"scoring step. Parseability checks above still work, but don't treat "
                         f"them as a guaranteed match to how Kaggle will score tautomer "
                         f"equivalence — run `pip install rdkit=={EXPECTED_RDKIT_VERSION}` "
                         f"if you want this validator's checks to line up exactly.")

    n_total_rows = len(seen_ids)
    if fallback_heavy_rows:
        pct = 100 * len(fallback_heavy_rows) / max(n_total_rows, 1)
        sev = "⚠️" if pct < 10 else "🚨"
        warnings.append(f"{sev} {len(fallback_heavy_rows)}/{n_total_rows} rows ({pct:.1f}%) have "
                         f"'{args.fallback_smiles}' as the TOP guess — likely candidate retrieval "
                         f"failed for these molecules. High % suggests a real pipeline problem, "
                         f"not just hard molecules.")

    short_lists = [mid for mid, preds in all_preds_per_row.items() if 0 < len(preds) < 5]
    if short_lists:
        warnings.append(f"{len(short_lists)} row(s) have fewer than 5 candidates — not a reject, "
                         f"but worth checking these aren't silently truncated.")

    # --- Report ---
    print(f"Rows checked: {n_total_rows}")
    print()
    if hard_fails:
        print("❌ HARD FAILS (would be rejected by Kaggle):")
        for f in hard_fails:
            print(f"   - {f}")
    else:
        print("✅ No hard-reject conditions found.")
    print()
    if warnings:
        print("⚠️  SANITY WARNINGS (won't reject, but worth checking):")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("✅ No sanity warnings.")
    print()

    if hard_fails:
        print("VERDICT: DO NOT SUBMIT — fix hard fails first.")
        sys.exit(1)
    elif warnings:
        print("VERDICT: Format is valid, but review the warnings above before submitting.")
    else:
        print("VERDICT: Looks clean.")


if __name__ == "__main__":
    main()