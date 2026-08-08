"""
Withhold every WPI that its own row's award of non-economic loss proves wrong.

s 4.11 of the Motor Accident Injuries Act 2017 permits damages for non-economic
loss ONLY where whole person impairment exceeds 10%. A row carrying `Non-Economic
Loss Status = Awarded` and a WPI at or below 10 is therefore self-contradictory,
and on audit the WPI is nearly always the wrong half:

  * Washbourne [2025] NSWPIC 334 - "the shoulders were equally impaired ... at
    8% each and the cervical spine at 5%". The 8% is ONE SHOULDER; the combined
    total is around 20%. The decision never states the Medical Panel's own
    total, so there is nothing to promote - only something to withhold.
  * Young [2023] NSWPIC 473 - Dr Wallace assessed 6%, then the insurer conceded
    that with scarring and muscle atrophy the impairment "would likely to exceed
    the 10% threshold". 6% is real but superseded, and no revised figure exists.
  * Bond [2024] NSWPIC 468 - Dr Giles 7% (left lower limb only), Dr Lee 9%; "the
    parties agreed that entitlement to non-economic loss was enlivened". A
    settlement approval never had to certify a number.

Against one true exception, which this pass must NOT touch:

  * Silcocks [2023] NSWPIC 24 - 9% WPI, and the Member is explicit that there is
    no entitlement, approving $120,000 anyway as a compromise "where no legal
    obligation on insurer to make any allowance for non-economic loss". The data
    is right; the law simply allows an insurer to pay what it does not owe.

`quarantine_impossible_wpi` blanks the figure (preserving it in `WPI Candidates`),
flags the row `Needs Review`, and drops it out of the analysis-ready set under
`wpi_contradicts_nel_award` - so a WPI-conditional analysis no longer sees an 8%
row carrying $383,000 of non-economic loss.

    python backfill_wpi_nel_quarantine.py --dry-run   # show what would change
    python backfill_wpi_nel_quarantine.py             # apply, rewrite reports

Then: python ctp_lump_sum_impairment.py
"""

import argparse
import json

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CACHE_FILE,
    CSV_REPORT,
    MAI_ACT_CASE_TYPES,
    WPI_VERSION,
    atomic_write_json,
    dataset_lock,
    quarantine_impossible_wpi,
    regenerate_reports_from_cache,
    to_float_pct,
    wpi_award_is_ex_gratia,
)


def candidate_rows(cache):
    """Rows whose award contradicts their WPI, before any decision is made about
    which half is wrong. Motor accident rows only — s 4.11 does not govern
    workers compensation, where 10% WPI with non-economic loss is unremarkable."""
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if str(row.get("Case Type", "") or "").strip() not in MAI_ACT_CASE_TYPES:
            continue
        if str(row.get("Non-Economic Loss Status") or "").strip() != "Awarded":
            continue
        value = to_float_pct(row.get("Impairment % (Accepted)"))
        if value is None or value > 10:
            continue
        yield url, row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    quarantined, spared = [], []
    changed = 0
    for url, row in candidate_rows(cache):
        name = (row.get("Case Name") or url).split("[")[0].strip()[:50]
        wpi = str(row.get("Impairment % (Accepted)") or "").strip()
        nel = str(row.get("Non-Economic Loss") or "").strip()
        if args.dry_run:
            # Decide without mutating, so --dry-run is honest.
            (spared if wpi_award_is_ex_gratia(row) else quarantined).append(
                (name, wpi, nel))
            continue
        (quarantined if quarantine_impossible_wpi(row) else spared).append(
            (name, wpi, nel))
        # The ex gratia path withholds the WPI too but is not a quarantine, so
        # it returns False. Detect the mutation rather than the return value,
        # or that row is decided and then never written.
        if str(row.get("Impairment % (Accepted)") or "").strip() != wpi:
            changed += 1

    print(f"{len(quarantined) + len(spared)} row(s) award non-economic loss on a "
          f"WPI at or below 10%.\n")
    if quarantined:
        print(f"WITHHELD ({len(quarantined)}) - WPI contradicted by the award:")
        for name, wpi, nel in sorted(quarantined):
            print(f"  {name:52} WPI {wpi:>5}  NEL {nel:>12}")
    if spared:
        print(f"\nKEPT ({len(spared)}) - insurer paid without legal obligation "
              f"(s 4.11 exception):")
        for name, wpi, nel in sorted(spared):
            print(f"  {name:52} WPI {wpi:>5}  NEL {nel:>12}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if not changed:
        print("\nNo WPI value changed; cache unchanged.")
        return 0
    print(f"\n{changed} WPI value(s) withheld.")

    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="backfill_wpi_nel_quarantine", wpi_version=WPI_VERSION)
    print("\nNext: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
