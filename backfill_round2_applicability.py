"""
Round 2 (§10): applicability semantics for the WPI columns.

The consumer's problem was not wrong values, it was unreadable absences. Every
empty cell said `absent`, so "there is no psychiatric injury to assess" and "we
failed to capture the figure" were the same word. On `WPI Psychiatric %` that
made 448 blanks indistinguishable, of which exactly one is a defect, and their
missingness check had to either flag all 448 or none.

This pass is entirely deterministic — it re-derives from columns already on the
row plus the classified mentions in `_wpi_resolution`, so it costs nothing and
can be re-run whenever the rules change:

  §10.1  `absent` splits into not_applicable / not_assessed / not_stated /
         absent, and only `absent` now means a defect.
  §10.2  the `Has Psychiatric Injury` gate is made self-consistent with the
         separately-stated psychiatric percentage.
  §10.4  `WPI Governing System` and `NEL Threshold Consistent` are added.

    python backfill_round2_applicability.py --dry-run
    python backfill_round2_applicability.py

Then: python ctp_lump_sum_impairment.py
"""

import argparse
import json
from collections import Counter

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CACHE_FILE,
    CSV_REPORT,
    SIDECAR_FILE,
    WPI_VERSION,
    apply_round2_semantics,
    atomic_write_json,
    dataset_lock,
    regenerate_reports_from_cache,
)

TRACKED = ("WPI Physical % Provenance", "WPI Psychiatric % Provenance",
           "WPI Provenance", "WPI Governing System", "NEL Threshold Consistent",
           "WPI Threshold Finding", "WPI Threshold Finding Basis")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--case-type", default="CTP",
                    help="restrict to one Case Type, or 'all'")
    args = ap.parse_args()

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    before = {c: Counter() for c in TRACKED}
    after = {c: Counter() for c in TRACKED}
    gate_fixes, touched = [], 0

    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if args.case_type != "all" and \
                str(row.get("Case Type", "") or "").strip() != args.case_type:
            continue
        for c in TRACKED:
            before[c][str(row.get(c) or "(empty)")] += 1

        psych_was = str(row.get("WPI Psychiatric %") or "").strip()
        flag_was = str(row.get("Has Psychiatric Injury") or "").strip()
        apply_round2_semantics(row)
        touched += 1

        if (str(row.get("WPI Psychiatric %") or "").strip() != psych_was
                or str(row.get("Has Psychiatric Injury") or "").strip() != flag_was):
            gate_fixes.append((
                (row.get("Case Name") or url).split("[")[0].strip()[:46],
                f"psych {psych_was or '-'} -> {row.get('WPI Psychiatric %') or '-'}",
                f"flag {flag_was} -> {row.get('Has Psychiatric Injury')}"))

        for c in TRACKED:
            after[c][str(row.get(c) or "(empty)")] += 1

    print(f"Applied round-2 semantics to {touched} row(s) "
          f"(Case Type = {args.case_type}).\n")
    for c in TRACKED:
        keys = sorted(set(before[c]) | set(after[c]))
        print(f"  {c}")
        for k in keys:
            b, a = before[c][k], after[c][k]
            if b == a:
                print(f"      {k:22} {b:>5}")
            else:
                print(f"      {k:22} {b:>5} -> {a:<5}  ({a - b:+d})")
        print()

    if gate_fixes:
        print(f"Gate contradictions resolved ({len(gate_fixes)}):")
        for name, a, b in gate_fixes:
            print(f"  {name:48} {a:<22} {b}")
        print()

    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0

    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        try:
            with open(SIDECAR_FILE, encoding="utf-8") as f:
                sidecar = json.load(f)
        except OSError:
            sidecar = {}
        if sidecar:
            for url, row in cache.items():
                if isinstance(row, dict) and url in sidecar:
                    sidecar[url]["wpi_version"] = row.get("_wpi_version", 0)
            atomic_write_json(SIDECAR_FILE, sidecar, ensure_ascii=False)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="backfill_round2_applicability", wpi_version=WPI_VERSION)
    print("Next: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
