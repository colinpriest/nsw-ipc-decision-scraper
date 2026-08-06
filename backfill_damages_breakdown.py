"""
Backfill the damages breakdown over already-cached decisions.

Adds the columns requested by the downstream CTP simulator (spec 2026-07-27):
past economic loss, deductions and reductions, gross-vs-net, per-field
provenance, a reconciliation flag, and the P1 classification fields.

Reads each target row's locally saved HTML/PDF from `nsw_pic_decisions/` and
runs ONLY the damages pass — the existing extraction is not re-run, so the
eight high-value fields the consumer already relies on cannot regress.

Default target population mirrors the workbook: analysis-ready CTP rows with
a positive lump sum, whose `_damages_version` is below the current one. WPI is
NOT required — a decision that never states one is still a real award.

    python backfill_damages_breakdown.py                 # the workbook rows
    python backfill_damages_breakdown.py --all-ctp       # every CTP row
    python backfill_damages_breakdown.py --limit 25      # a costed sample
    python backfill_damages_breakdown.py --force         # redo done rows

After it finishes, re-run `python ctp_lump_sum_impairment.py` to refresh
`ctp_impairment_lump_sum.xlsx` (this script rewrites the CSVs, not the
workbook), then `python check_damages_acceptance.py`.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CSV_REPORT,
    DAMAGES_VERSION,
    DECISIONS_DIR,
    SCHEMA_VERSION,
    DecisionScraper,
    _is_quota_error,
    annotate_analysis_fields,
    cleanup_text,
    dataset_lock,
    generate_reports,
    get_worker_count,
    merge_damages_into_record,
    normalise_medical_costs,
    safe_decision_path,
)
from damages_extraction import to_float



def _positive(value):
    f = to_float(value)
    return f is not None and f > 0


def select_targets(cache, *, all_ctp=False, force=False, limit=None):
    """Rows needing the damages pass, in decision-date order (newest first).

    Pure over the cache dict so it can be exercised without the API.
    """
    targets = []
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if row.get("_schema_version") != SCHEMA_VERSION:
            continue
        if str(row.get("Case Type", "") or "").strip() != "CTP":
            continue
        if not force and int(row.get("_damages_version") or 0) >= DAMAGES_VERSION:
            continue
        if not all_ctp:
            annotated = annotate_analysis_fields(row)
            if annotated.get("Analysis Ready") != "Yes":
                continue
            if not _positive(row.get("Lump Sum")):
                continue
        targets.append((url, row))

    targets.sort(key=lambda t: str(t[1].get("Decision Date") or ""), reverse=True)
    if limit:
        targets = targets[:limit]
    return targets


def backfill_one(scraper, url, row):
    """Run the damages pass for one cached row and write it back. Returns
    'ok' / 'skipped' / 'error'."""
    title = row.get("Case Name") or url
    file_saved = row.get("File Saved") or ""
    if not file_saved:
        logging.warning(f"Skipping {url}: no File Saved entry")
        return "skipped"

    full_path = safe_decision_path(scraper.output_folder, file_saved)
    if not full_path or not os.path.exists(full_path):
        logging.warning(f"Skipping {url}: local file missing or unsafe ({file_saved!r})")
        return "skipped"

    try:
        with open(full_path, "rb") as f:
            content = f.read()
    except OSError as e:
        logging.error(f"Failed to read {full_path}: {e}")
        return "skipped"

    if file_saved.lower().endswith(".pdf"):
        raw_text = scraper.extractor.extract_text_from_pdf(content)
    else:
        raw_text = scraper.extractor.extract_text_from_html(content)
    decision_text = cleanup_text(raw_text)
    if len(decision_text) < 500:
        logging.warning(f"Skipping {url}: decision text too short ({len(decision_text)} chars)")
        return "skipped"

    if scraper.quota_breaker.is_aborted():
        return "skipped"

    parsed, usage, err = scraper.extractor.extract_damages(
        decision_text, context=f"damages backfill title={title[:60]}, url={url}",
    )

    with scraper.cache_lock:
        cached = scraper.cache.get(url)
        if not isinstance(cached, dict):
            return "skipped"
        updated = dict(cached)

    if usage is not None:
        usage_record = scraper.cost_tracker.record(usage)
        token_usage = dict(updated.get("_token_usage") or {})
        token_usage["damages_pass"] = usage_record
        updated["_token_usage"] = token_usage

    if err or parsed is None:
        if _is_quota_error(err):
            scraper.quota_breaker.record_quota_error()
        else:
            scraper.quota_breaker.record_non_quota_error()
        logging.error(f"Damages pass failed for {title[:60]}: {err or 'parse failed'}")
        # Leave the row's damages version untouched so a re-run retries it.
        return "error"

    scraper.quota_breaker.record_success()
    merge_damages_into_record(updated, parsed)
    # Same output-sentinel fix the live path applies (spec 4.1).
    updated["Medical Costs"] = normalise_medical_costs(updated.get("Medical Costs"))
    scraper.update_cache(url, annotate_analysis_fields(updated))
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-ctp", action="store_true",
                    help="every CTP row, not just the analysis-ready workbook population")
    ap.add_argument("--force", action="store_true",
                    help="re-run rows that already have the current damages version")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N rows (use for a costed sample first)")
    ap.add_argument("--no-reports", action="store_true",
                    help="skip CSV/manifest regeneration (for a sample run)")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        return 1

    scraper = DecisionScraper("https://www.austlii.edu.au", DECISIONS_DIR, api_key)

    with scraper.cache_lock:
        snapshot = dict(scraper.cache)
    targets = select_targets(snapshot, all_ctp=args.all_ctp, force=args.force,
                             limit=args.limit)
    logging.info(
        f"Damages backfill: {len(targets)} target rows "
        f"(cache={len(snapshot)}, damages schema v{DAMAGES_VERSION}, "
        f"{'all CTP' if args.all_ctp else 'workbook population'})."
    )
    if not targets:
        logging.info("Nothing to do.")
        return 0

    max_workers = get_worker_count()
    wall_t0 = time.monotonic()
    counts = {"ok": 0, "skipped": 0, "error": 0}
    aborted_logged = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(backfill_one, scraper, url, row): url
            for url, row in targets
        }
        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                counts[future.result()] += 1
            except Exception as e:  # noqa: BLE001
                counts["error"] += 1
                logging.error(f"Unhandled exception on {url}: {e}")
            completed += 1

            if scraper.quota_breaker.is_aborted() and not aborted_logged:
                aborted_logged = True
                logging.error(
                    f"QUOTA BREAKER TRIPPED at {completed}/{len(targets)} — cancelling "
                    f"remaining work. Top up and re-run; finished rows are preserved."
                )
                for f in future_to_url:
                    f.cancel()

            if completed % 25 == 0:
                with dataset_lock():
                    scraper._save_cache()
                elapsed = time.monotonic() - wall_t0
                rate = completed / max(elapsed, 1e-6)
                logging.info(
                    f"Progress: {completed}/{len(targets)}  elapsed {elapsed/60:.1f}m  "
                    f"rate {rate*60:.1f}/min  "
                    f"ETA {(len(targets)-completed)/max(rate,1e-6)/60:.1f}m  "
                    f"cost ${scraper.cost_tracker.total_cost():.2f}"
                )

    ct = scraper.cost_tracker
    logging.info("=" * 70)
    logging.info(f"Damages backfill done in {(time.monotonic()-wall_t0)/60:.1f}m: "
                 f"{counts['ok']} ok, {counts['skipped']} skipped, {counts['error']} errors")
    logging.info(f"  LLM calls {ct.calls}, cost ${ct.total_cost():.2f} "
                 f"(mean ${ct.total_cost()/max(ct.calls,1):.4f}/call)")

    with dataset_lock():
        scraper._save_cache()
        scraper._save_sidecar()
        if not args.no_reports:
            generate_reports(
                scraper, CSV_REPORT, ANALYSIS_READY_REPORT,
                script="backfill_damages_breakdown",
                manifest_extra={
                    "damages_version": DAMAGES_VERSION,
                    "damages_rows_processed": counts["ok"],
                    "damages_rows_failed": counts["error"],
                    "quota_aborted": bool(aborted_logged),
                    "llm_calls": ct.calls,
                },
            )

    logging.info("Next: python ctp_lump_sum_impairment.py && python check_damages_acceptance.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
