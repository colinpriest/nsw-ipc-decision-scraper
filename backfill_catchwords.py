"""
Backfill the 'Catchwords' field on existing cache rows by parsing the local
HTML/PDF files. No LLM, no AustLII fetch. Safe to run repeatedly.

Rows where extraction returns empty are left with Catchwords="" so the parse
gap is visible in the CSV without polluting the data.
"""

import json
import logging
import os

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    atomic_write_json,
    CACHE_FILE,
    cleanup_text,
    CSV_REPORT,
    dataset_lock,
    DECISIONS_DIR as OUTPUT_DIR,
    extract_catchwords,
    extract_html_with_paragraph_numbers,
    regenerate_reports_from_cache,
    safe_decision_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



def backfill():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    updated = unchanged = no_file = empty_catchwords = 0
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        fname = row.get("File Saved") or ""
        path = safe_decision_path(OUTPUT_DIR, fname)  # ISSUE-012
        if not path or not os.path.exists(path):
            if fname and not path:
                logging.warning(f"Rejected unsafe File Saved path: {fname!r}")
            no_file += 1
            continue

        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError as e:
            logging.warning(f"Could not read {path}: {e}")
            no_file += 1
            continue

        is_pdf = fname.lower().endswith(".pdf")
        if is_pdf:
            # extract_catchwords only handles cleaned text; PDFs would need a
            # different parse path. Skip PDFs for backfill; they re-extract on
            # next full-pipeline run via _build_record_from_parsed.
            continue

        text = cleanup_text(extract_html_with_paragraph_numbers(data))
        catchwords = extract_catchwords(text)

        prior = row.get("Catchwords", "")
        if prior == catchwords:
            unchanged += 1
        else:
            row["Catchwords"] = catchwords
            updated += 1
        if not catchwords:
            empty_catchwords += 1

    logging.info(f"Backfill complete: {updated} updated, {unchanged} unchanged, "
                 f"{no_file} skipped (missing file), {empty_catchwords} rows have empty catchwords.")

    # Persist cache + regenerate CSVs (atomic, always-write, manifest) under the
    # shared dataset lock (ISSUE-001/002/003).
    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT, script="backfill_catchwords")


if __name__ == "__main__":
    backfill()
