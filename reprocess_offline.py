"""
Offline reprocess of cached decisions.

Reads each cache row's locally saved HTML/PDF from `nsw_pic_decisions/`,
runs the current LLM extraction prompt against it, and rewrites the cache
row at the current schema version. No AustLII fetches.

Use when AustLII is blocking us but we still want to reprocess everything
against an upgraded model or prompt.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CSV_REPORT,
    DECISIONS_DIR,
    DecisionScraper,
    SCHEMA_VERSION,
    _is_quota_error,
    cleanup_text,
    dataset_lock,
    generate_reports,
    get_worker_count,
    print_summary,
    record_austlii_data_error,
    safe_decision_path,
)


def reprocess_one(scraper, url, row):
    title = row.get("Case Name") or url
    file_saved = row.get("File Saved") or ""
    if not file_saved:
        logging.warning(f"Skipping {url}: no File Saved entry")
        return None

    full_path = safe_decision_path(scraper.output_folder, file_saved)  # ISSUE-012
    if not full_path:
        logging.warning(f"Skipping {url}: unsafe File Saved path {file_saved!r}")
        return None
    if not os.path.exists(full_path):
        logging.warning(f"Skipping {url}: local file missing ({full_path})")
        return None

    is_pdf = file_saved.lower().endswith(".pdf")
    try:
        with open(full_path, "rb") as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Failed to read {full_path}: {e}")
        return None

    if is_pdf:
        raw_text = scraper.extractor.extract_text_from_pdf(content)
    else:
        raw_text = scraper.extractor.extract_text_from_html(content)

    decision_text = cleanup_text(raw_text)
    if len(decision_text) < 500:
        logging.warning(f"Decision text too short for {title}; not caching")
        record_austlii_data_error(
            url,
            case_name=title,
            error_type="html_no_content",
            local_file=file_saved,
            extracted_chars=len(decision_text),
            notes=(
                "AustLII viewer rendered an empty body. Decision listed in index "
                "but body is not served; needs manual review."
            ),
        )
        return None

    log_title = (title[:75] + "...") if len(title) > 75 else title

    if scraper.quota_breaker.is_aborted():
        return None

    logging.info(f"Reprocessing: {log_title}")

    parsed, usage, llm_error = scraper.extractor.extract_combined(
        decision_text, context=f"offline title={log_title}, url={url}",
    )
    token_usage = scraper.cost_tracker.record(usage) if usage else {}

    # Cache write rule: ONLY on a valid parsed extraction. Failure paths
    # leave the cache untouched so the URL is retried on the next run.
    if llm_error or parsed is None:
        if _is_quota_error(llm_error):
            scraper.quota_breaker.record_quota_error()
        else:
            scraper.quota_breaker.record_non_quota_error()
            logging.error(f"LLM error for {log_title}: {llm_error or 'parse failed'}")
        return None

    scraper.quota_breaker.record_success()
    result_data = scraper._build_record_from_parsed(
        title=title, url=url, file_saved=file_saved,
        parsed=parsed, decision_text=decision_text,
        token_usage=token_usage,
    )
    scraper.update_cache(url, result_data)
    return result_data


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY not found in .env file.")
        return

    BASE_DOMAIN = "https://www.austlii.edu.au"
    OUTPUT_DIR = DECISIONS_DIR

    scraper = DecisionScraper(BASE_DOMAIN, OUTPUT_DIR, api_key)

    # Pick rows that need reprocessing (schema mismatch).
    with scraper.cache_lock:
        snapshot = list(scraper.cache.items())
    targets = [
        (url, row) for url, row in snapshot
        if not (isinstance(row, dict) and row.get("_schema_version") == SCHEMA_VERSION)
    ]
    already_current = len(snapshot) - len(targets)

    logging.info(
        f"Offline reprocess: {len(targets)} of {len(snapshot)} cache rows "
        f"need reprocessing ({already_current} already at schema v{SCHEMA_VERSION})."
    )

    if not targets:
        logging.info("Nothing to do — all cache rows already at current schema.")
        return

    max_workers = get_worker_count()  # validated/clamped (ISSUE-018)
    logging.info(f"Reprocessing with {max_workers} workers...")

    wall_t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(reprocess_one, scraper, url, row): url
            for url, row in targets
        }

        completed = 0
        aborted_logged = False
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"Unhandled exception while reprocessing {url}: {e}")
            completed += 1

            if scraper.quota_breaker.is_aborted() and not aborted_logged:
                aborted_logged = True
                logging.error(
                    f"QUOTA BREAKER TRIPPED at {completed}/{len(targets)} — "
                    f"cancelling remaining work. Top up OpenAI account and "
                    f"re-run; cached rows for unfinished URLs are preserved."
                )
                for f in future_to_url:
                    f.cancel()

            if completed % 25 == 0:
                with dataset_lock():
                    scraper._save_cache()
                elapsed = time.monotonic() - wall_t0
                rate = completed / max(elapsed, 1e-6)
                remaining = len(targets) - completed
                eta_s = remaining / max(rate, 1e-6)
                logging.info(
                    f"Progress: {completed}/{len(targets)}  "
                    f"elapsed {elapsed/60:.1f}m  "
                    f"rate {rate*60:.1f}/min  "
                    f"ETA {eta_s/60:.1f}m  "
                    f"cost ${scraper.cost_tracker.total_cost():.2f} "
                    f"({scraper.cost_tracker.calls} calls)"
                )

    wall_elapsed = time.monotonic() - wall_t0
    ct = scraper.cost_tracker

    logging.info("=" * 70)
    logging.info("LLM USAGE / COST")
    logging.info("=" * 70)
    logging.info(f"  Wall-clock:        {wall_elapsed:.1f}s ({wall_elapsed/60:.1f}m)")
    logging.info(f"  LLM calls:         {ct.calls}")
    logging.info(f"  Prompt tokens:     {ct.prompt_tokens:,}  (cached {ct.cached_tokens:,})")
    logging.info(f"  Completion tokens: {ct.completion_tokens:,}  (reasoning {ct.reasoning_tokens:,})")
    logging.info(f"  Total cost:        ${ct.total_cost():.2f}")
    if ct.calls:
        logging.info(f"  Mean per call:     ${ct.total_cost() / ct.calls:.4f}")

    # Persist cache + sidecar + reports as one locked snapshot (ISSUE-001/002/003/022).
    with dataset_lock():
        scraper._save_cache()
        scraper._save_sidecar()
        all_data, analysis_ready_data = generate_reports(
            scraper, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="reprocess_offline",
            manifest_extra={"quota_aborted": bool(aborted_logged), "llm_calls": ct.calls},
        )

    print_summary(all_data, analysis_ready_data)


if __name__ == "__main__":
    main()
