"""
Re-apply the WPI resolution ladder WITHOUT re-calling the LLM.

The classified mentions from `backfill_wpi_resolution.py` are stored on each
cache row under `_wpi_resolution`, so a change to the deterministic ladder in
`wpi_resolution.resolve_wpi` can be replayed over them for free. Use this
whenever the LADDER changes but the CLASSIFICATION does not: re-classifying
costs money and re-rolls the model's judgement, which makes an unrelated ladder
fix look like a data change.

    python reresolve_wpi_offline.py            # replay, then rewrite reports
    python reresolve_wpi_offline.py --dry-run  # show what would change

Rows are restored to their pre-pass value (`_wpi_pre_value`) first, so the
ladder always decides from the same starting point.
"""

import argparse
import json
import logging
from types import SimpleNamespace

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CACHE_FILE,
    CSV_REPORT,
    SIDECAR_FILE,
    WPI_VERSION,
    atomic_write_json,
    dataset_lock,
    merge_wpi_resolution_into_record,
    regenerate_reports_from_cache,
)


def rebuild_parsed(entry, row=None):
    """Reconstruct a WpiResolution-shaped object from a stored record.

    Records written before the threshold finding was persisted fall back to the
    flat `WPI Threshold Finding` column, so the veto is not silently lost on
    replay."""
    row = row or {}
    mentions = [
        SimpleNamespace(
            value=m.get("value", ""),
            kind=m.get("kind", "other"),
            body_system=m.get("body_system", "unclear"),
            assessor=m.get("assessor", "unknown"),
            superseded=bool(m.get("superseded")),
            about_claimant=bool(m.get("about_claimant", True)),
            quote=m.get("quote", ""),
        )
        for m in (entry.get("mentions") or [])
    ]
    return SimpleNamespace(
        mentions=mentions,
        tribunal_selected_value=entry.get("tribunal_selected_value", ""),
        tribunal_selected_quote=entry.get("tribunal_selected_quote", ""),
        components_share_one_assessment=bool(entry.get("components_share_one_assessment")),
        totals_are_rival_assessments=bool(entry.get("totals_are_rival_assessments", True)),
        threshold_finding=(entry.get("threshold_finding")
                           or row.get("WPI Threshold Finding")
                           or "not determined"),
        settlement_approval_without_wpi=bool(entry.get("settlement_approval_without_wpi")),
        notes=entry.get("notes", ""),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)

    changes, replayed = [], 0
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        entry = row.get("_wpi_resolution") or {}
        if not entry.get("mentions") and not entry.get("tribunal_selected_value"):
            continue
        before = str(row.get("Impairment % (Accepted)") or "").strip()
        row["Impairment % (Accepted)"] = row.get("_wpi_pre_value", before)
        merge_wpi_resolution_into_record(row, rebuild_parsed(entry, row))
        replayed += 1
        after = str(row.get("Impairment % (Accepted)") or "").strip()
        if after != before:
            changes.append((row.get("Case Name", url)[:52], before or "(blank)",
                            after or "(blank)", (row.get("WPI Basis") or "")[:58]))

    print(f"Replayed the ladder over {replayed} classified rows; "
          f"{len(changes)} value(s) changed.\n")
    for name, before, after, basis in sorted(changes):
        print(f"  {name:54} {before:>7} -> {after:<7}  {basis}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        try:
            with open(SIDECAR_FILE, encoding="utf-8") as f:
                sidecar = json.load(f)
        except OSError:
            sidecar = {}
        for url, row in cache.items():
            if isinstance(row, dict) and url in sidecar:
                sidecar[url]["wpi_resolution"] = row.get("_wpi_resolution", {})
                sidecar[url]["wpi_version"] = row.get("_wpi_version", 0)
        if sidecar:
            atomic_write_json(SIDECAR_FILE, sidecar, ensure_ascii=False)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="reresolve_wpi_offline", wpi_version=WPI_VERSION)
    print("\nNext: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
