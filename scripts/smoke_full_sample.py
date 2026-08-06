"""
Live smoke check (NOT an automated test) — runs the real OpenAI extraction
against 5 cached cases and writes a human-readable dump.

Named outside pytest's `test_*.py` discovery on purpose (ISSUE-021): it needs a
real API key, the local cache, and local decision files, and is non-repeatable.
For automated, assertion-based regression coverage of _build_record_from_parsed
and the report/export pipeline, see test_extraction_fields.py.

Run:  python scripts/smoke_full_sample.py
"""
import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# Allow running from the scripts/ directory or the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nsw_court_scraper import (
    CACHE_FILE,
    cleanup_text,
    DECISIONS_DIR as DECISIONS_FOLDER,
    DecisionScraper,
    extract_html_with_paragraph_numbers,
    RESULT_FIELDS,
    safe_decision_path,
)

SAME_CASE_URLS = [
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2023/367.html",
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2024/650.html",
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2022/137.html",
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2022/380.html",
    "https://www.austlii.edu.au/cgi-bin/viewdoc/au/cases/nsw/NSWPIC/2022/379.html",
]

OUTPUT_FILE = "smoke_full_sample_output.txt"
SLICE_DISPLAY_CAP = 3000


def process_one(scraper, url, cached_row):
    title = cached_row.get("Case Name", "(unknown)")
    file_saved = cached_row.get("File Saved", "")
    path = safe_decision_path(DECISIONS_FOLDER, file_saved)  # ISSUE-012
    t0 = time.monotonic()
    if not path or not os.path.exists(path):
        return {"url": url, "title": title, "error": "source file missing/unsafe", "elapsed": 0.0}
    with open(path, "rb") as f:
        raw = f.read()
    source = cleanup_text(extract_html_with_paragraph_numbers(raw))
    parsed, usage, err = scraper.extractor.extract_combined(source, context=title)
    if usage:
        scraper.cost_tracker.record(usage)
    if err or parsed is None:
        return {"url": url, "title": title, "error": err or "no parse", "elapsed": time.monotonic() - t0}
    record = scraper._build_record_from_parsed(
        title=title, url=url, file_saved=file_saved,
        parsed=parsed, decision_text=source, token_usage={},
    )
    return {
        "url": url, "title": title, "error": None,
        "record": record, "source_text": source,
        "source_chars": len(source), "source_words": len(source.split()),
        "elapsed": time.monotonic() - t0,
    }


def render_case(idx, r):
    lines = []
    lines.append("=" * 100)
    lines.append(f"CASE {idx}/5: {r['title']}")
    lines.append(f"URL:  {r['url']}")
    if r.get("error"):
        lines.append(f"[error] {r['error']}")
        return "\n".join(lines)
    lines.append(f"Source: {r['source_chars']:,} chars / {r['source_words']:,} words")
    lines.append(f"Elapsed: {r['elapsed']:.1f}s")
    lines.append("=" * 100)
    record = r["record"]

    lines.append("\n--- FLAT FIELDS (CSV columns) ---")
    for k in RESULT_FIELDS:
        v = record.get(k, "")
        if k in ("Description", "Banded Description"):
            lines.append(f"  {k} ({len(str(v).split())} words):")
            lines.append(f"    {v}")
        else:
            lines.append(f"  {k:32} {v}")

    val = record.get("_banding_validation") or {}
    lines.append("\n--- BANDING VALIDATION ---")
    lines.append(f"  ok: {val.get('ok')}    tokens: "
                 + ", ".join(f"{k}={v}" for k, v in (val.get('tokens') or {}).items() if v))
    for issue in val.get("issues", []):
        sev = issue.get("severity", "?")
        typ = issue.get("type", "?")
        match = issue.get("match")
        detail = issue.get("detail")
        suffix = f" -> {match!r}" if match else (f" -- {detail}" if detail else "")
        lines.append(f"  [{sev}] {typ}{suffix}")

    lines.append("\n--- NARRATIVE SUB-FIELDS ---")
    for k, v in (record.get("_narrative") or {}).items():
        lines.append(f"\n  [{k}] ({len(str(v).split())} words)")
        lines.append(f"  {v}")

    lines.append("\n--- VERBATIM SLICES ---")
    for k, info in (record.get("_slices") or {}).items():
        lines.append(f"\n  [{k}] present={info.get('present')}")
        if info.get("resolution_error"):
            lines.append(f"    [resolution failed: {info['resolution_error']}]")
            continue
        txt = info.get("text", "")
        if not txt:
            continue
        lines.append(f"    ({len(txt):,} chars / {len(txt.split())} words)")
        display = txt if len(txt) <= SLICE_DISPLAY_CAP else (
            txt[: SLICE_DISPLAY_CAP // 2]
            + f"\n[... {len(txt) - SLICE_DISPLAY_CAP:,} chars elided ...]\n"
            + txt[-SLICE_DISPLAY_CAP // 2:]
        )
        for line in display.splitlines():
            lines.append(f"    | {line}")

    lines.append("\n--- KEY PARAGRAPHS ---")
    kps = record.get("_key_paragraphs") or []
    total_words = sum(len((kp.get("text") or "").split()) for kp in kps)
    lines.append(f"  {len(kps)} paragraphs / {total_words} words total")
    for kp in kps:
        lines.append(f"\n  [para {kp.get('paragraph_number')}] ({len((kp.get('text') or '').split())} words) "
                     f"-- {kp.get('rationale')}")
        for line in (kp.get("text") or "(unresolved)").splitlines():
            lines.append(f"    | {line}")

    lines.append("\n--- EVENT HISTORY ---")
    events = record.get("_event_history") or []
    for i, ev in enumerate(events, start=1):
        lines.append(f"  Event {i}: <date: {ev.get('date')}><actor: {ev.get('actor')}><tag: {ev.get('tag')}>")
    lines.append(f"  Total: {len(events)} events")

    return "\n".join(lines)


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY missing")
        sys.exit(1)
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    scraper = DecisionScraper("https://www.austlii.edu.au", DECISIONS_FOLDER, api_key)
    t0 = time.monotonic()

    results = [None] * len(SAME_CASE_URLS)
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(process_one, scraper, u, cache.get(u, {})): i
                   for i, u in enumerate(SAME_CASE_URLS)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
    wall = time.monotonic() - t0

    blocks = [render_case(i, r) for i, r in enumerate(results, start=1)]

    ct = scraper.cost_tracker
    summary = [
        "",
        "=" * 100,
        "RUN SUMMARY",
        "=" * 100,
        f"  Wall-clock (parallel x5): {wall:.1f}s",
        f"  LLM calls:                {ct.calls}",
        f"  Prompt tokens:            {ct.prompt_tokens:,} (cached {ct.cached_tokens:,})",
        f"  Completion tokens:        {ct.completion_tokens:,} (reasoning {ct.reasoning_tokens:,})",
        f"  Total cost:               ${ct.total_cost():.4f}",
    ]

    output = "\n".join(blocks) + "\n" + "\n".join(summary) + "\n"
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Wrote {OUTPUT_FILE} ({len(output):,} chars)")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
