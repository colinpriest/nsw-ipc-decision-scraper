"""
Round 3 (§11): applicability semantics for the MONEY columns.

Round 2 refined the WPI columns and stopped there, so the 13 money columns were
still on bare `absent` and "FAIL on absent" could not be turned on for them —
it would have reported 539 rows of `Weekly Statutory Benefit` as extraction
failures when a damages determination has no reason to quantify a statutory
benefit at all. This finishes the migration.

Also §11.1: `Other Damages Heads` was the one money head shipped without a
`Status` companion, so a considered-and-refused other head and one that was
never in issue both collapsed to null, and 71% of the column read as missing
data when the values are zeros. The status column is added, and where the
accounting identity closes the blank is written as the zero it is.

Deterministic — re-derived from columns already on the row, so it is free to
re-run and cannot drift from the extraction.

    python backfill_round3_money_absence.py --dry-run
    python backfill_round3_money_absence.py

Then: python ctp_lump_sum_impairment.py
"""

import argparse
import json
from collections import Counter

from damages_extraction import MONEY_PROVENANCE_PAIRS, RECONCILE_TOLERANCE, damages_residual, to_float
from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CACHE_FILE,
    CSV_REPORT,
    WPI_VERSION,
    apply_money_absence_semantics,
    atomic_write_json,
    dataset_lock,
    regenerate_reports_from_cache,
)

TRACKED = [prov for _amount, prov in MONEY_PROVENANCE_PAIRS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--case-type", default="CTP", help="or 'all'")
    args = ap.parse_args()

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    before = {c: Counter() for c in TRACKED}
    after = {c: Counter() for c in TRACKED}
    misses, zeros, touched = [], 0, 0

    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if args.case_type != "all" and \
                str(row.get("Case Type", "") or "").strip() != args.case_type:
            continue
        for c in TRACKED:
            before[c][str(row.get(c) or "(empty)")] += 1

        residual, trustworthy = damages_residual(row)
        had_other = str(row.get("Other Damages Heads") or "").strip()
        apply_money_absence_semantics(row)
        touched += 1

        if not had_other and str(row.get("Other Damages Heads") or "").strip() == "0":
            zeros += 1
        if row.get("Other Damages Heads Provenance") == "absent" and trustworthy \
                and residual is not None and abs(residual) > 10 * RECONCILE_TOLERANCE:
            misses.append(((row.get("Case Name") or url).split("[")[0].strip()[:46],
                           residual, to_float(row.get("Total Damages Gross"))))

        for c in TRACKED:
            after[c][str(row.get(c) or "(empty)")] += 1

    print(f"Refined {touched} row(s) (Case Type = {args.case_type}).\n")
    for c in TRACKED:
        keys = sorted(set(before[c]) | set(after[c]))
        if all(before[c][k] == after[c][k] for k in keys):
            continue
        print(f"  {c}")
        for k in keys:
            b, a = before[c][k], after[c][k]
            mark = "" if b == a else f"  ({a - b:+d})"
            print(f"      {k:22} {b:>5} -> {a:<5}{mark}")
        print()

    print(f"`Other Damages Heads`: {zeros} blank(s) resolved to a derived zero "
          f"by the accounting identity.")
    if misses:
        print(f"\nEXTRACTION MISSES ({len(misses)}) — a head was awarded and not "
              f"captured. Residual is what is unaccounted for:")
        for name, residual, gross in sorted(misses, key=lambda x: -abs(x[1])):
            print(f"  {name:48} residual {residual:>12,.0f}  of gross {gross or 0:>12,.0f}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="backfill_round3_money_absence", wpi_version=WPI_VERSION)
    print("\nNext: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
