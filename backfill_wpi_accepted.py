"""
Backfill the 'Impairment % (Accepted)' column for CTP cases.

Two stages:
  Stage 1 (regex, $0): when the local HTML contains exactly ONE distinct WPI
    number, use that. Validated at 97% precision vs LLM.
  Stage 2 (focused LLM, ~$0.03/call): for the remaining CTP cases with a
    Lump Sum but no Accepted WPI, run a small focused-schema LLM call that
    extracts only WPI fields. Much cheaper than the full extraction.

Also seeds 'Impairment % (Accepted)' from the existing strict 'Impairment %'
where set — any WPI the main extractor judged "made in this proceeding" is
necessarily also "accepted as basis of award".
"""

import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from pydantic import BaseModel, Field

from nsw_court_scraper import (
    MODEL,
    OPENAI_TIMEOUT,
    PRICE_INPUT_PER_M,
    PRICE_CACHED_INPUT_PER_M,
    PRICE_OUTPUT_PER_M,
    REASONING_EFFORT,
    QuotaCircuitBreaker,
    _is_quota_error,
    _is_transient_api_error,
    atomic_write_json,
    cleanup_text,
    dataset_lock,
    extract_html_with_paragraph_numbers,
    extract_wpi_confident,
    get_worker_count,
    regenerate_reports_from_cache,
    safe_decision_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CACHE_FILE = "processed_cache.json"
CSV_REPORT = "detailed_payout_summary.csv"
ANALYSIS_READY_REPORT = "analysis_ready_payout_summary.csv"
OUTPUT_DIR = "nsw_pic_decisions"


# ----------------------------------------------------------------------
# Focused WPI extraction schema
# ----------------------------------------------------------------------

class WPIFocusedSchema(BaseModel):
    impairment_percentage_made: str = Field(description=(
        "WPI percentage MADE IN THIS PROCEEDING (the Member's binding finding). "
        "EMPTY for CTP settlement approvals or damages assessments that merely "
        "accept a prior MAS certificate. Just digits and optional decimal point."
    ))
    impairment_percentage_accepted: str = Field(description=(
        "WPI percentage USED FOR THE AWARD, regardless of who assessed it. "
        "For settlement approvals and damages this is the WPI the lump sum "
        "is calibrated against (usually a prior MAS certificate). If multiple "
        "component WPIs are stated for different body parts and no combined "
        "value is given, use the HIGHEST single component. EMPTY only if no "
        "numeric WPI appears anywhere in the decision."
    ))
    wpi_evidence_quote: str = Field(description=(
        "A short verbatim quote (10-30 words) from the source text showing "
        "where the accepted WPI appears. Used for human spot-checking. "
        "EMPTY if no WPI found."
    ))


_FOCUSED_SYSTEM = """\
You are extracting Whole Person Impairment (WPI) information from a NSW
Personal Injury Commission decision. Return three fields per the schema.

Key distinction: in CTP matters under MAI Act 2017, settlement approvals
(s 6.23) and damages assessments (s 7.36) almost always RELY ON a prior
Medical Assessor's WPI certificate rather than MAKING the WPI finding in
this proceeding. Capture this in impairment_percentage_accepted.

If the source states multiple WPI numbers (e.g. 5% cervical spine, 7% lumbar
spine, combined 12%), prefer the COMBINED/TOTAL value. If only components
are given without a stated combined value, use the highest component.

CRITICAL — do NOT confuse the statutory threshold with a finding. The 10% WPI
bar gates non-economic loss under the MAI Act 2017, so phrases like "exceeds
the 10% whole person impairment threshold", "does not exceed 10% WPI", or
"the statutory threshold of 10%" describe the LEGISLATIVE BAR, not this
claimant's impairment. A "10%" that appears ONLY inside such threshold framing
is NOT the accepted WPI:
  * "injuries do/does NOT exceed 10%" -> claimant is below the bar; leave
    impairment_percentage_accepted EMPTY unless a specific assessed figure
    (e.g. "assessed at 4%") appears elsewhere.
  * "injuries EXCEED / exceeded the 10% threshold" -> claimant is above the
    bar but 10 is only the floor; return the actual assessed WPI if the
    decision states one, otherwise leave it EMPTY. Never return the bare 10.
Only return 10 when the decision states the claimant was actually ASSESSED at
10% (e.g. "assessed at 10% WPI", "certified 10% whole person impairment").
"""


# ----------------------------------------------------------------------
# Focused LLM caller with quota retry
# ----------------------------------------------------------------------

def call_focused_extraction(client, source_text, context=""):
    """Returns (parsed, usage, error). Retries on insufficient_quota and on
    transient timeout/connection errors (ISSUE-016)."""
    user = (
        "Source text of the decision follows. Extract WPI information.\n\n"
        "---\n"
        f"{source_text[:80000]}\n"
        "---\n"
    )
    backoff = [2, 5, 10, 20, 40, 80]
    for attempt in range(len(backoff) + 1):
        try:
            r = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": _FOCUSED_SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format=WPIFocusedSchema,
                reasoning_effort=REASONING_EFFORT,
            )
            return r.choices[0].message.parsed, r.usage, None
        except Exception as e:
            if (_is_quota_error(str(e)) or _is_transient_api_error(e)) and attempt < len(backoff):
                delay = backoff[attempt]
                kind = "insufficient_quota" if _is_quota_error(str(e)) else "timeout/connection"
                logging.warning(f"{kind} ({context}) - retry {attempt+1} in {delay}s")
                time.sleep(delay)
                continue
            logging.error(f"WPI focused extraction error ({context}): {e}")
            return None, None, str(e)
    return None, None, "exhausted retries"


def estimate_cost(usage):
    if usage is None:
        return 0.0
    prompt = usage.prompt_tokens or 0
    completion = usage.completion_tokens or 0
    cached = 0
    d = getattr(usage, "prompt_tokens_details", None)
    if d is not None:
        cached = getattr(d, "cached_tokens", 0) or 0
    non_cached = prompt - cached
    return (
        non_cached * PRICE_INPUT_PER_M / 1_000_000
        + cached * PRICE_CACHED_INPUT_PER_M / 1_000_000
        + completion * PRICE_OUTPUT_PER_M / 1_000_000
    )


# ----------------------------------------------------------------------
# Backfill
# ----------------------------------------------------------------------

def is_numeric(v):
    try:
        float(str(v).strip())
        return True
    except (ValueError, TypeError):
        return False


def _pdf_to_text(data):
    """Extract text from PDF bytes (ISSUE-007)."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        logging.warning(f"Unable to open PDF for WPI backfill: {e}")
        return ""
    pages = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append(t.strip())
    return "\n\n".join(pages).strip()


def read_local_text(file_saved):
    """Read a cached decision's text. Handles HTML AND PDF (ISSUE-007), and
    constrains the path to OUTPUT_DIR (ISSUE-012)."""
    path = safe_decision_path(OUTPUT_DIR, file_saved)
    if not path or not os.path.exists(path):
        if file_saved and not path:
            logging.warning(f"Rejected unsafe File Saved path: {file_saved!r}")
        return None
    with open(path, "rb") as f:
        data = f.read()
    if file_saved.lower().endswith(".pdf"):
        return cleanup_text(_pdf_to_text(data))
    return cleanup_text(extract_html_with_paragraph_numbers(data))


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # ---- Seed Impairment % (Accepted) from Impairment % where set ----
    # Skip seeds where strict == 0 — see note in extract_wpi_confident:
    # a true 0% finding wouldn't underpin a CTP lump sum, so '0' in the
    # strict field is almost always a misextraction and shouldn't propagate.
    seeded = 0
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if "Impairment % (Accepted)" not in row:
            row["Impairment % (Accepted)"] = ""
        strict_str = str(row.get("Impairment %") or "").strip()
        if not is_numeric(strict_str):
            continue
        try:
            if float(strict_str) <= 0:
                continue  # don't seed Accepted with 0 / negative
        except ValueError:
            continue
        if not row.get("Impairment % (Accepted)"):
            row["Impairment % (Accepted)"] = strict_str
            seeded += 1
    logging.info(f"Seeded Impairment % (Accepted) from Impairment % on {seeded} rows.")

    # ---- Identify CTP cases that still need Accepted WPI ----
    candidates = []
    for url, row in cache.items():
        if not isinstance(row, dict):
            continue
        if row.get("Case Type") != "CTP":
            continue
        if not is_numeric(row.get("Lump Sum")):
            continue  # not in our target population
        if is_numeric(row.get("Impairment % (Accepted)")):
            continue  # already filled (either by main extractor or seeding)
        candidates.append((url, row))

    logging.info(f"CTP cases needing Accepted WPI backfill: {len(candidates)}")

    # ---- Stage 1: regex backfill ----
    stage1_filled = 0
    stage2_needed = []
    for url, row in candidates:
        text = read_local_text(row.get("File Saved", ""))
        if text is None:
            continue
        val = extract_wpi_confident(text)
        if val is not None:
            row["Impairment % (Accepted)"] = str(val) if val != int(val) else str(int(val))
            stage1_filled += 1
        else:
            stage2_needed.append((url, row, text))

    logging.info(f"Stage 1 (regex) filled: {stage1_filled} / {len(candidates)}")
    logging.info(f"Stage 2 (focused LLM) candidates: {len(stage2_needed)}")

    # Save cache after stage 1 so a stage-2 failure doesn't lose stage-1 work
    with dataset_lock():
        _save_cache(cache)

    # ---- Stage 2: focused LLM ----
    if stage2_needed:
        # Explicit timeout + our own retry policy (ISSUE-016).
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT, max_retries=0)
        breaker = QuotaCircuitBreaker()  # abort on sustained quota failure (ISSUE-014)
        total_cost = 0.0
        stage2_filled = stage2_failed = 0
        aborted_logged = False

        def worker(url, row, text):
            if breaker.is_aborted():
                return url, row, None, None, "aborted"
            ctx = f"url={url}"
            parsed, usage, err = call_focused_extraction(client, text, context=ctx)
            return url, row, parsed, usage, err

        max_workers = get_worker_count()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(worker, u, r, t): (u, r) for u, r, t in stage2_needed}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    url, row, parsed, usage, err = fut.result()
                except Exception as e:
                    logging.error(f"Stage 2 worker exception: {e}")
                    stage2_failed += 1
                    continue
                total_cost += estimate_cost(usage)
                if err or parsed is None:
                    stage2_failed += 1
                    # Feed the quota breaker so a real billing cap aborts the
                    # backfill instead of grinding through every candidate.
                    if _is_quota_error(err):
                        breaker.record_quota_error()
                    elif err != "aborted":
                        breaker.record_non_quota_error()
                    if breaker.is_aborted() and not aborted_logged:
                        aborted_logged = True
                        logging.error(
                            f"QUOTA BREAKER TRIPPED at stage-2 item {i}/{len(stage2_needed)} — "
                            f"cancelling remaining WPI backfill calls. Top up and re-run.")
                        for f in futures:
                            f.cancel()
                    continue
                breaker.record_success()
                # Patch only the impairment fields; preserve everything else.
                # Treat "0" as unknown — see extract_wpi_confident comment.
                def _clean(s):
                    s = (s or "").replace("%", "").strip()
                    try:
                        return s if float(s) > 0 else ""
                    except ValueError:
                        return ""

                made = _clean(parsed.impairment_percentage_made)
                accepted = _clean(parsed.impairment_percentage_accepted)
                if made and not row.get("Impairment %"):
                    row["Impairment %"] = made
                if accepted:
                    row["Impairment % (Accepted)"] = accepted
                    stage2_filled += 1

                if i % 25 == 0:
                    logging.info(f"  stage 2 progress: {i}/{len(stage2_needed)}  cost ${total_cost:.2f}")
                    with dataset_lock():
                        _save_cache(cache)

        logging.info(f"Stage 2 filled: {stage2_filled}  failed: {stage2_failed}  cost ${total_cost:.2f}")

    # ---- Regenerate CSVs (always-write, atomic, manifest) ----
    with dataset_lock():
        _save_cache(cache)
        regenerate_reports_from_cache(
            cache, CSV_REPORT, ANALYSIS_READY_REPORT, script="backfill_wpi_accepted")


def _save_cache(cache):
    atomic_write_json(CACHE_FILE, cache)


if __name__ == "__main__":
    main()
