"""
Backfill Claimant Age for rows where the LLM left it empty but the decision
states a year of birth. We derive age = injury_year - birth_year.

Conservative — runs only on rows where:
  - Claimant Age is empty
  - Injury Date is a valid YYYY-MM-DD
  - Source text matches a year-of-birth pattern with a plausible year (1900..now)
  - Derived age is in [0, 120]
"""

import csv
import json
import logging
import os
import re

from nsw_court_scraper import (
    RESULT_FIELDS,
    annotate_analysis_fields,
    cleanup_text,
    extract_html_with_paragraph_numbers,
    has_valid_iso_date,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_FILE = "processed_cache.json"
OUTPUT_DIR = "nsw_pic_decisions"

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
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
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

    # Save cache
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, default=str)
    os.replace(tmp, CACHE_FILE)

    # Regenerate CSVs
    all_data = [annotate_analysis_fields(row) for row in cache.values()]
    analysis_ready = [r for r in all_data if r.get("Analysis Ready") == "Yes"]

    def sk(r):
        d = (r.get("Decision Date") or "").strip()
        return d if has_valid_iso_date(d) else "0000-00-00"

    all_data.sort(key=sk, reverse=True)
    analysis_ready.sort(key=sk, reverse=True)

    for path, data in [
        ("detailed_payout_summary.csv", all_data),
        ("analysis_ready_payout_summary.csv", analysis_ready),
    ]:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(data)
        logging.info(f"Rewrote {path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
