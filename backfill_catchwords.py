"""
Backfill the 'Catchwords' field on existing cache rows by parsing the local
HTML/PDF files. No LLM, no AustLII fetch. Safe to run repeatedly.

Rows where extraction returns empty are left with Catchwords="" so the parse
gap is visible in the CSV without polluting the data.
"""

import csv
import json
import logging
import os

from nsw_court_scraper import (
    RESULT_FIELDS,
    annotate_analysis_fields,
    cleanup_text,
    extract_catchwords,
    extract_html_with_paragraph_numbers,
    has_valid_iso_date,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_FILE = "processed_cache.json"
CSV_REPORT = "detailed_payout_summary.csv"
ANALYSIS_READY_REPORT = "analysis_ready_payout_summary.csv"
OUTPUT_DIR = "nsw_pic_decisions"


def backfill():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    updated = unchanged = no_file = empty_catchwords = 0
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        fname = row.get("File Saved") or ""
        if not fname:
            no_file += 1
            continue
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
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

    # Atomic write
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
    os.replace(tmp, CACHE_FILE)

    logging.info(f"Backfill complete: {updated} updated, {unchanged} unchanged, "
                 f"{no_file} skipped (missing file), {empty_catchwords} rows have empty catchwords.")

    # Regenerate CSVs so the new column is visible
    all_data = [annotate_analysis_fields(row) for row in cache.values()]
    analysis_ready = [r for r in all_data if r.get("Analysis Ready") == "Yes"]

    def sort_key(r):
        d = (r.get("Decision Date") or "").strip()
        return d if has_valid_iso_date(d) else "0000-00-00"

    all_data.sort(key=sort_key, reverse=True)
    analysis_ready.sort(key=sort_key, reverse=True)

    if all_data:
        with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_data)
        logging.info(f"Rewrote {CSV_REPORT} ({len(all_data)} rows)")

    if analysis_ready:
        with open(ANALYSIS_READY_REPORT, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(analysis_ready)
        logging.info(f"Rewrote {ANALYSIS_READY_REPORT} ({len(analysis_ready)} rows)")


if __name__ == "__main__":
    backfill()
