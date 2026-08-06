"""
Backfill Claimant Age for rows where the LLM left it empty but the decision
states a year of birth. We derive age = injury_year - birth_year.

Conservative — runs only on rows where:
  - Claimant Age is empty
  - Injury Date is a valid YYYY-MM-DD
  - Source text matches a year-of-birth pattern with a plausible year (1900..now)
  - Derived age is in [0, 120]
"""

import json
import logging
import os
import re

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    atomic_write_json,
    CACHE_FILE,
    cleanup_text,
    CSV_REPORT,
    dataset_lock,
    DECISIONS_DIR as OUTPUT_DIR,
    extract_html_with_paragraph_numbers,
    has_valid_iso_date,
    regenerate_reports_from_cache,
    safe_decision_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Match "born in 1995", "born early 1995", "born on 12 March 1995", "date of
# birth: 12 March 1995", etc. The LAST 4-digit number group is taken as the
# birth year. The patterns are ordered most-specific-first.
YOB_PATTERNS = [
    re.compile(r"born\s+(?:on\s+)?\d{1,2}\s+\w+\s+(\d{4})\b", re.IGNORECASE),
    re.compile(r"date\s+of\s+birth[:\s]+[^\n]*?(\d{4})\b", re.IGNORECASE),
    re.compile(r"d\.?o\.?b\.?[:\s]+[^\n]*?(\d{4})\b", re.IGNORECASE),
    re.compile(r"born\s+(?:on\s+|in\s+)?(?:early\s+|late\s+|mid[-\s]+)?(\d{4})\b", re.IGNORECASE),
]


def find_year_of_birth(text):
    for pat in YOB_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                y = int(m.group(1))
                if 1900 <= y <= 2030:
                    return y
            except ValueError:
                pass
    return None


def main():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    backfilled = 0
    skipped_no_yob = 0
    skipped_bad_age = 0
    sample = []

    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if (row.get("Claimant Age") or "").strip():
            continue
        if not has_valid_iso_date(row.get("Injury Date", "")):
            continue
        fname = row.get("File Saved") or ""
        if not fname or fname.lower().endswith(".pdf"):
            continue
        path = safe_decision_path(OUTPUT_DIR, fname)  # ISSUE-012
        if not path or not os.path.exists(path):
            if fname and not path:
                logging.warning(f"Rejected unsafe File Saved path: {fname!r}")
            continue

        with open(path, "rb") as f:
            data = f.read()
        text = cleanup_text(extract_html_with_paragraph_numbers(data))
        yob = find_year_of_birth(text)
        if yob is None:
            skipped_no_yob += 1
            continue

        injury_year = int(row["Injury Date"][:4])
        age = injury_year - yob
        if not (0 <= age <= 120):
            skipped_bad_age += 1
            continue

        row["Claimant Age"] = str(age)
        backfilled += 1
        if len(sample) < 8:
            sample.append((row.get("Case Name", "")[:55], yob, injury_year, age))

    logging.info(f"Backfilled Claimant Age from year-of-birth: {backfilled}")
    logging.info(f"  Skipped (no YOB found):         {skipped_no_yob}")
    logging.info(f"  Skipped (implausible derived):  {skipped_bad_age}")
    for name, yob, iy, age in sample:
        logging.info(f"    yob={yob}  injury_year={iy}  age={age}   {name}")

    # Save cache + regenerate CSVs (atomic, always-write, manifest) under lock.
    with dataset_lock():
        atomic_write_json(CACHE_FILE, cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="backfill_age_from_dob")


if __name__ == "__main__":
    main()
