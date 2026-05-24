"""
One-off targeted re-extraction of Accepted WPI for cases whose value was a
statutory-threshold false positive, plus the Zvedeniouk disputed-figure case.

Background
----------
extract_wpi_confident() used to return any lone non-zero "N% WPI" token as the
accepted impairment. Two failure modes surfaced:

  * Threshold framing: "does not exceed the statutory threshold of 10% whole
    person impairment" was read as a 10% finding (e.g. Quick [2024] NSWPIC 93).
    find_wpi_candidates() now drops threshold-context matches, so these cases
    re-run cleanly: regex returns None -> focused LLM.

  * Disputed/rejected figure: a claimant's tendered combined WPI read as the
    accepted basis even though the Member did not accept it (Gordian Runoff v
    Zvedeniouk [2024] NSWPIC 136 — 48% rejected, award based on ~0%). The regex
    cannot see "rejected", so these are force-routed to the focused LLM here.

This script clears the suspect values and re-derives them via the same pipeline
as backfill_wpi_accepted (new threshold-aware regex, then focused LLM), then
regenerates the CSV reports.
"""

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

from nsw_court_scraper import (
    RESULT_FIELDS,
    annotate_analysis_fields,
    cleanup_text,
    extract_html_with_paragraph_numbers,
    extract_wpi_confident,
    find_wpi_candidates,
    has_valid_iso_date,
    _WPI_FWD_RE,
    _WPI_REV_RE,
)
from backfill_wpi_accepted import (
    CACHE_FILE,
    CSV_REPORT,
    ANALYSIS_READY_REPORT,
    OUTPUT_DIR,
    call_focused_extraction,
    read_local_text,
    _save_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# URLs the regex cannot self-correct (rejected/disputed claimant figures).
FORCE_LLM_URLS = {
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2024/136.html",
}


def _isnum(x):
    return re.fullmatch(r"\d+(?:\.\d+)?", str(x).strip() or "x") is not None


def _old_nonzero_candidates(text):
    """The pre-fix candidate set (no threshold filtering), positive values only."""
    vals = set()
    for rgx in (_WPI_FWD_RE, _WPI_REV_RE):
        for m in rgx.finditer(text):
            v = float(m.group(1))
            if 0 < v <= 100:
                vals.add(v)
    return vals


def _clean_llm(s):
    s = (s or "").replace("%", "").strip()
    try:
        return s if float(s) > 0 else ""
    except ValueError:
        return ""


def find_threshold_false_positives(cache):
    """CTP cases whose stored Accepted WPI is a statutory-threshold artifact.

    Two sources of the bug, both caught here:
      (a) regex: the OLD regex returned the lone non-zero token (e.g. 10) that
          the NEW threshold-aware regex now drops; and
      (b) stage-2 LLM: it returned the bare threshold value 10 from "exceed/does
          not exceed the 10% threshold" phrasing — 10 never appears as a genuine
          non-threshold WPI token in the text.

    Both are confirmed re-running through the (now threshold-aware) regex and the
    (now threshold-hardened) focused LLM. We never override a value the strict
    main extractor itself recorded ('Impairment %' == accepted), as that was a
    deliberate in-proceeding finding.
    """
    out = []
    for url, row in cache.items():
        if not isinstance(row, dict) or row.get("Case Type") != "CTP":
            continue
        if not _isnum(row.get("Lump Sum")):
            continue
        acc = str(row.get("Impairment % (Accepted)") or "").strip()
        if not _isnum(acc):
            continue
        accf = float(acc)
        text = read_local_text(row.get("File Saved", ""))
        if text is None:
            continue
        new_nz = {v for v in find_wpi_candidates(text) if v > 0}
        if accf in new_nz:
            continue  # value survives as a genuine (non-threshold) assessment
        strict = str(row.get("Impairment %") or "").strip()
        if _isnum(strict) and float(strict) == accf:
            continue  # deliberate in-proceeding finding by the main extractor
        # (a) lone regex token dropped by threshold filter, or
        # (b) bare 10 from threshold framing (LLM-sourced; not a regex token)
        if _old_nonzero_candidates(text) == {accf} or accf == 10:
            out.append(url)
    return out


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    backup = f"{CACHE_FILE}.pre_threshold_wpi_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(CACHE_FILE, backup)
    logging.info(f"Backed up cache to {backup}")

    targets = set(find_threshold_false_positives(cache))
    targets |= {u for u in FORCE_LLM_URLS if u in cache}
    logging.info(f"Targets to re-extract: {len(targets)}")

    client = OpenAI(api_key=api_key)
    changes = []
    for url in sorted(targets):
        row = cache[url]
        old_acc = str(row.get("Impairment % (Accepted)") or "").strip()
        text = read_local_text(row.get("File Saved", ""))
        if text is None:
            logging.warning(f"No local text for {url}; skipping")
            continue

        # Clear and re-derive. Regex first (unless force-LLM), else focused LLM.
        new_acc = ""
        if url not in FORCE_LLM_URLS:
            val = extract_wpi_confident(text)
            if val is not None:
                new_acc = str(val) if val != int(val) else str(int(val))

        if not new_acc:
            parsed, usage, err = call_focused_extraction(client, text, context=url)
            if err or parsed is None:
                logging.error(f"LLM failed for {url}: {err}")
                # leave field cleared rather than keep the known-bad value
                new_acc = ""
                quote = "(LLM error)"
            else:
                new_acc = _clean_llm(parsed.impairment_percentage_accepted)
                quote = (parsed.wpi_evidence_quote or "").strip()[:120]
            time.sleep(0.2)
        else:
            quote = "(regex)"

        row["Impairment % (Accepted)"] = new_acc
        changes.append((url.split("/")[-1], old_acc, new_acc or "(empty)", quote))

    print("\n=== Proposed changes ===")
    for name, old, new, quote in changes:
        print(f"  {name:>12}: {old:>4} -> {new:<8}  {quote}")

    _save_cache(cache)
    logging.info(f"Saved cache with {len(changes)} updates.")

    # ---- Regenerate CSVs (mirrors backfill_wpi_accepted.main tail) ----
    all_data = [annotate_analysis_fields(row) for row in cache.values()]
    analysis_ready = [r for r in all_data if r.get("Analysis Ready") == "Yes"]

    def sort_key(r):
        d = (r.get("Decision Date") or "").strip()
        return d if has_valid_iso_date(d) else "0000-00-00"

    all_data.sort(key=sort_key, reverse=True)
    analysis_ready.sort(key=sort_key, reverse=True)

    import csv
    with open(CSV_REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_data)
    logging.info(f"Rewrote {CSV_REPORT} ({len(all_data)} rows)")

    with open(ANALYSIS_READY_REPORT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(analysis_ready)
    logging.info(f"Rewrote {ANALYSIS_READY_REPORT} ({len(analysis_ready)} rows)")


if __name__ == "__main__":
    main()
