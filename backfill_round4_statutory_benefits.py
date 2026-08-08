"""
Round 4 (§12): the last 62 `Statutory Benefits Paid` failures.

Two findings drove this, and they pull in opposite directions:

  §12.2  A stated s 3.40 deduction proves benefits WERE paid, but not that the
         decision quantifies the total PAID. The corroboration rule from round 3
         treated it as proof and so reported 62 rows as extraction failures.
         Where the decision describes only the deduction, `not_stated` is the
         honest value.
  §12.3  MACA-era language is 11x enriched among those failures — 16.1% of them
         against 1.4% of the successes. The predecessor scheme says s 83 /
         s 130 where MAIA says statutory benefits / s 3.40, so an extractor
         keyed on MAIA vocabulary misses them by construction. Those rows are
         RECOVERABLE, not reclassifiable.

So this pass reads each decision, recovers a stated benefits-paid total in
either scheme's wording, and reclassifies the remainder to `not_stated`.

What it will not do is derive Paid from Repaid, though that would close all 62
at a stroke. They are different fields: Paid is everything the claimant
received, Repaid is the deduction, which reaches only the recoverable
categories. They coincide on 96% of rows only because treatment and care is
98.1% not applicable, and asserting equality would understate benefits paid in
exactly the cases the field exists for.

    python backfill_round4_statutory_benefits.py --dry-run
    python backfill_round4_statutory_benefits.py

Then: python ctp_lump_sum_impairment.py
"""

import argparse
import html
import json
import os
import re

from damages_extraction import PROVENANCE_ABSENCE
from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CACHE_FILE,
    CSV_REPORT,
    DECISIONS_DIR,
    WPI_VERSION,
    atomic_write_json,
    dataset_lock,
    regenerate_reports_from_cache,
)
from statutory_benefits_recovery import (
    find_statutory_benefits_paid,
    uses_maca_language,
)


def decision_text(row):
    name = str(row.get("File Saved") or "").strip()
    if not name:
        return ""
    path = os.path.join(DECISIONS_DIR, name)
    if not os.path.exists(path):
        return ""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    recovered, reclassified, missing_file = [], 0, 0
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        # Any row with no figure, whatever the absence value currently says —
        # round 3's reclassification runs before this and would otherwise hide
        # every candidate behind `not_stated`, making the result depend on the
        # order the backfills happen to be run in.
        if str(row.get("Statutory Benefits Paid") or "").strip():
            continue
        if str(row.get("Statutory Benefits Paid Provenance") or "") not in PROVENANCE_ABSENCE:
            continue

        text = decision_text(row)
        if not text:
            missing_file += 1
            continue
        amount, quote = find_statutory_benefits_paid(text)
        name = (row.get("Case Name") or url).split("[")[0].strip()[:44]

        if amount is not None:
            recovered.append((name, amount, uses_maca_language(text),
                              str(row.get("Statutory Benefits Repaid") or "")))
            if not args.dry_run:
                row["Statutory Benefits Paid"] = f"{amount:g}"
                row["Statutory Benefits Paid Provenance"] = "stated"
                note = f"statutory benefits paid recovered from source: {quote[:120]}"
                row["Damages Notes"] = "; ".join(
                    x for x in [row.get("Damages Notes", ""), note] if x)[:1000]
            continue

        if str(row.get("Statutory Benefits Paid Provenance") or "") == "absent":
            reclassified += 1
            if not args.dry_run:
                # The decision describes a deduction and never a paid total.
                row["Statutory Benefits Paid Provenance"] = "not_stated"

    print(f"Triage of the `Statutory Benefits Paid` failures:\n")
    print(f"  recovered from source : {len(recovered)}  "
          f"({sum(1 for r in recovered if r[2])} in MACA-era wording)")
    print(f"  reclassified not_stated: {reclassified}  "
          f"(decision gives only the s 3.40 / s 130 deduction)")
    if missing_file:
        print(f"  source file unavailable: {missing_file}  (left as absent)")
    print()
    for name, amount, maca, repaid in sorted(recovered, key=lambda r: -r[1]):
        tag = "MACA" if maca else "    "
        differs = "" if not repaid or abs(amount - float(repaid)) <= 1 \
            else f"   repaid {float(repaid):,.2f}"
        print(f"  {name:46} {amount:>13,.2f}  {tag}{differs}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="backfill_round4_statutory_benefits", wpi_version=WPI_VERSION)
    print("\nNext: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
