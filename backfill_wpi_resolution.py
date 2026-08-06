"""
Backfill WPI resolution over already-cached decisions.

Recovers the impairment percentage in the cases the strict single-token regex
deliberately refuses to touch, because the source holds MORE THAN ONE figure:

  * components of one assessment with a stated total  -> use the total
  * components with no stated total                   -> combine them (AMA
    Combined Values, not addition) and mark `derived`
  * competing assessments the tribunal never chose    -> central estimate,
    marked `inferred` so the consumer can exclude it
  * a genuine 0% assessment                           -> keep it; only the
    statutory 10% bar is a threshold recital
  * a settlement approval quoting no exact WPI        -> correctly left blank

An LLM classifies each mention (component / assessor total / MAS certificate /
tribunal finding / threshold recital / rejected, and physical vs psychiatric);
the deterministic ladder in `wpi_resolution.resolve_wpi` does the arithmetic.

    python backfill_wpi_resolution.py                  # rows needing it
    python backfill_wpi_resolution.py --limit 20       # a costed sample
    python backfill_wpi_resolution.py --all-ctp        # every CTP row
    python backfill_wpi_resolution.py --force          # redo done rows

Then re-run `python ctp_lump_sum_impairment.py` to refresh the workbook.
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
    DECISIONS_DIR,
    SCHEMA_VERSION,
    WPI_VERSION,
    DecisionScraper,
    _is_quota_error,
    annotate_analysis_fields,
    cleanup_text,
    dataset_lock,
    find_wpi_tokens,
    wpi_is_legally_impossible,
    generate_reports,
    get_worker_count,
    merge_wpi_resolution_into_record,
    safe_decision_path,
)
from damages_extraction import to_float


def _decision_text(scraper, row):
    file_saved = row.get("File Saved") or ""
    if not file_saved:
        return None
    path = safe_decision_path(scraper.output_folder, file_saved)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            content = f.read()
    except OSError:
        return None
    raw = (scraper.extractor.extract_text_from_pdf(content)
           if file_saved.lower().endswith(".pdf")
           else scraper.extractor.extract_text_from_html(content))
    text = cleanup_text(raw)
    return text if len(text) >= 500 else None


def needs_resolution(row, tokens):
    """A row is worth a call when the source holds ANY WPI-shaped figure and
    either the field is empty or the source holds several the regex would not
    choose between.

    Deliberately keyed on raw tokens rather than threshold-filtered candidates:
    rows whose only figures sit near threshold wording are exactly the ones the
    regex cannot judge, so they are the ones most worth asking about."""
    if not tokens:
        return False
    current = str(row.get("Impairment % (Accepted)") or "").strip()
    return (not current) or len(tokens) > 1 or wpi_is_legally_impossible(row)


def select_targets(scraper, *, all_ctp=False, force=False, limit=None):
    """Rows needing the pass. Reads the local decision text, so it is slower
    than a pure cache scan but avoids paying for calls that cannot help."""
    with scraper.cache_lock:
        snapshot = list(scraper.cache.items())

    targets, scanned = [], 0
    for url, row in snapshot:
        if not isinstance(row, dict) or row.get("_schema_version") != SCHEMA_VERSION:
            continue
        if str(row.get("Case Type", "") or "").strip() != "CTP":
            continue
        if not force and int(row.get("_wpi_version") or 0) >= WPI_VERSION:
            continue
        if not all_ctp:
            if annotate_analysis_fields(row).get("Analysis Ready") != "Yes":
                continue
            lump = to_float(row.get("Lump Sum"))
            if lump is None or lump <= 0:
                continue
        text = _decision_text(scraper, row)
        scanned += 1
        if not text:
            continue
        tokens = find_wpi_tokens(text)
        if needs_resolution(row, tokens):
            targets.append((url, row, sorted(tokens)))

    targets.sort(key=lambda t: str(t[1].get("Decision Date") or ""), reverse=True)
    logging.info(f"Scanned {scanned} candidate rows.")
    return targets[:limit] if limit else targets


def resolve_one(scraper, url, row, *, reset=False):
    text = _decision_text(scraper, row)
    if not text or scraper.quota_breaker.is_aborted():
        return "skipped"

    before = str(row.get("Impairment % (Accepted)") or "").strip()
    parsed, usage, err = scraper.extractor.extract_wpi_resolution(
        text, context=f"WPI resolution url={url}")

    with scraper.cache_lock:
        cached = scraper.cache.get(url)
        if not isinstance(cached, dict):
            return "skipped"
        updated = dict(cached)

    if reset and "_wpi_pre_value" in updated:
        # Re-running under a changed ladder: start from what the MAIN
        # extraction had, not from this pass's own earlier output.
        updated["Impairment % (Accepted)"] = updated["_wpi_pre_value"]
        before = str(updated["_wpi_pre_value"] or "").strip()

    if usage is not None:
        token_usage = dict(updated.get("_token_usage") or {})
        token_usage["wpi_resolution"] = scraper.cost_tracker.record(usage)
        updated["_token_usage"] = token_usage

    if err or parsed is None:
        if _is_quota_error(err):
            scraper.quota_breaker.record_quota_error()
        else:
            scraper.quota_breaker.record_non_quota_error()
        logging.error(f"WPI resolution failed for {url}: {err or 'parse failed'}")
        return "error"

    scraper.quota_breaker.record_success()
    merge_wpi_resolution_into_record(updated, parsed)
    scraper.update_cache(url, annotate_analysis_fields(updated))

    after = str(updated.get("Impairment % (Accepted)") or "").strip()
    if after and not before:
        logging.info(f"Recovered WPI {after} ({updated.get('WPI Basis')}) for "
                     f"{(updated.get('Case Name') or url)[:55]}")
        return "recovered"
    if after != before:
        logging.info(f"WPI changed {before} -> {after} ({updated.get('WPI Basis')}) for "
                     f"{(updated.get('Case Name') or url)[:55]}")
        return "changed"
    return "unchanged"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-ctp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-reports", action="store_true")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        return 1

    scraper = DecisionScraper("https://www.austlii.edu.au", DECISIONS_DIR, api_key)
    targets = select_targets(scraper, all_ctp=args.all_ctp, force=args.force,
                             limit=args.limit)
    logging.info(f"WPI resolution: {len(targets)} target rows (v{WPI_VERSION}).")
    if not targets:
        logging.info("Nothing to do.")
        return 0

    counts = {"recovered": 0, "changed": 0, "unchanged": 0, "skipped": 0, "error": 0}
    wall_t0 = time.monotonic()
    aborted_logged = False

    with ThreadPoolExecutor(max_workers=get_worker_count()) as executor:
        futures = {executor.submit(resolve_one, scraper, u, r, reset=args.force): u
                   for u, r, _ in targets}
        completed = 0
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception as e:  # noqa: BLE001
                counts["error"] += 1
                logging.error(f"Unhandled exception on {futures[future]}: {e}")
            completed += 1
            if scraper.quota_breaker.is_aborted() and not aborted_logged:
                aborted_logged = True
                logging.error("QUOTA BREAKER TRIPPED - cancelling remaining work.")
                for f in futures:
                    f.cancel()
            if completed % 25 == 0:
                with dataset_lock():
                    scraper._save_cache()
                logging.info(f"Progress: {completed}/{len(targets)}  "
                             f"cost ${scraper.cost_tracker.total_cost():.2f}")

    ct = scraper.cost_tracker
    logging.info("=" * 70)
    logging.info(f"WPI resolution done in {(time.monotonic()-wall_t0)/60:.1f}m: "
                 f"{counts['recovered']} recovered, {counts['changed']} changed, "
                 f"{counts['unchanged']} unchanged, {counts['skipped']} skipped, "
                 f"{counts['error']} errors")
    logging.info(f"  LLM calls {ct.calls}, cost ${ct.total_cost():.2f}")

    with dataset_lock():
        scraper._save_cache()
        scraper._save_sidecar()
        if not args.no_reports:
            generate_reports(
                scraper, CSV_REPORT, ANALYSIS_READY_REPORT,
                script="backfill_wpi_resolution",
                manifest_extra={"wpi_version": WPI_VERSION,
                                "wpi_rows_recovered": counts["recovered"],
                                "wpi_rows_changed": counts["changed"],
                                "quota_aborted": bool(aborted_logged)},
            )
    logging.info("Next: python ctp_lump_sum_impairment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
