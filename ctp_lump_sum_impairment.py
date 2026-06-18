"""
CTP-only export: rows with both accepted WPI and lump sum, enriched with every
structured and semi-structured field we extract.

Joins the flat CSV (RESULT_FIELDS) with processed_sidecar.json by URL and
flattens nested fields (narrative sub-fields, verbatim slices, key paragraphs,
event history) into Excel-friendly columns.

All executable logic lives in main() (ISSUE-004): importing this module has no
side effects, so helpers can be reused/tested without reading production files
or overwriting the workbook.
"""

import json
import os

import pandas as pd

from nsw_court_scraper import coerce_leading_number

INPUT_CSV_CANDIDATES = [
    "analysis_ready_payout_summary.csv",
    "detailed_payout_summary.csv",
]
SIDECAR_FILE = "processed_sidecar.json"
OUTPUT_XLSX = "ctp_impairment_lump_sum.xlsx"

# Excel hard-limits cell text at 32,767 chars. Cap long serialised fields.
CELL_CHAR_CAP = 32_000


# ----------------------------------------------------------------------
# Pure helpers (no I/O)
# ----------------------------------------------------------------------

def is_numeric(series):
    def _check(val):
        val = str(val).strip()
        if not val:
            return False
        try:
            float(val)
            return True
        except ValueError:
            return False
    return series.apply(_check)


def is_positive_numeric(series):
    def _check(val):
        val = str(val).strip()
        if not val:
            return False
        try:
            return float(val) > 0
        except ValueError:
            return False
    return series.apply(_check)


def is_analysis_ready(dataframe):
    if "Analysis Ready" in dataframe.columns:
        return dataframe["Analysis Ready"].astype(str).str.strip().eq("Yes")

    status_ok = (
        dataframe["Status"].astype(str).str.strip().eq("ok")
        if "Status" in dataframe.columns
        else pd.Series(True, index=dataframe.index)
    )
    llm_ok = (
        dataframe["LLM Error"].astype(str).str.strip().eq("")
        if "LLM Error" in dataframe.columns
        else pd.Series(True, index=dataframe.index)
    )
    has_decision_date = (
        dataframe["Decision Date"].astype(str).str.strip().str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    )
    return status_ok & llm_ok & has_decision_date


NARRATIVE_FIELDS = [
    ("claimant_profile",                     "Narrative: Claimant Profile"),
    ("accident_or_injury_mechanism",         "Narrative: Accident or Injury Mechanism"),
    ("injuries_and_diagnoses",               "Narrative: Injuries and Diagnoses"),
    ("treatment_history",                    "Narrative: Treatment History"),
    ("functional_impact_and_work_capacity",  "Narrative: Functional Impact and Work Capacity"),
    ("medical_evidence_summary",             "Narrative: Medical Evidence Summary"),
    ("previous_insurer_actions_and_offers",  "Narrative: Previous Insurer Actions and Offers"),
    ("claimant_submissions",                 "Narrative: Claimant Submissions"),
    ("insurer_submissions",                  "Narrative: Insurer Submissions"),
    ("legal_issues_and_reasoning",           "Narrative: Legal Issues and Reasoning"),
]

# LLM-marked verbatim slices from the source decision. The "Catchwords" column
# already in RESULT_FIELDS comes from the deterministic regex parser; this
# slice is the LLM-marked equivalent (useful for cross-validation).
SLICE_FIELDS = [
    ("catchwords",              "Slice (LLM): Catchwords"),
    ("determinations_or_orders","Slice (LLM): Determinations or Orders"),
    ("introduction",            "Slice (LLM): Introduction"),
]

NUMERIC_COERCE_COLUMNS = [
    "Lump Sum",
    "Impairment %",
    "Impairment % (Accepted)",
    "Weekly Benefit",
    "Non-Economic Loss",
    "Future Economic Loss",
    "Claimant Age",
    "Claimant Weekly Income",
    # Ordinal scores — already integers in the source, but coerce defensively.
    "Injury Burden Intensity",
    "Psychological Injury Emphasis",
    "Liability Clarity",
    "Causation Complexity",
    "Treatment Burden",
    "Work Impact Severity",
    "Pre-existing Condition Salience",
    "Legal Procedural Complexity",
]


def _truncate(text):
    if text is None:
        return ""
    s = str(text)
    if len(s) <= CELL_CHAR_CAP:
        return s
    return s[: CELL_CHAR_CAP - 50] + "\n...[TRUNCATED for Excel cell limit]"


def format_key_paragraphs(items):
    """Render key_paragraphs list as one block of multiline text."""
    if not items:
        return ""
    out = []
    for kp in items:
        n = kp.get("paragraph_number")
        rationale = (kp.get("rationale") or "").strip()
        text = (kp.get("text") or "").strip()
        out.append(f"[para {n}] {rationale}")
        if text:
            out.append(text)
        out.append("")  # blank line between paragraphs
    return _truncate("\n".join(out).rstrip())


def format_event_history(events):
    """Render event_history list as one pipe-delimited line per event."""
    if not events:
        return ""
    lines = []
    for ev in events:
        date = (ev.get("date") or "").strip() or "?"
        actor = (ev.get("actor") or "").strip() or "?"
        tag = (ev.get("tag") or "").strip() or "?"
        lines.append(f"{date} | {actor} | {tag}")
    return _truncate("\n".join(lines))


def enrich_row(url, sidecar):
    """Return a dict of flattened sidecar columns for the given URL."""
    entry = sidecar.get(url) or {}
    enriched = {}

    narrative = entry.get("narrative") or {}
    for key, col in NARRATIVE_FIELDS:
        enriched[col] = _truncate((narrative.get(key) or "").strip())

    slices = entry.get("slices") or {}
    for key, col in SLICE_FIELDS:
        slc = slices.get(key) or {}
        enriched[col] = _truncate((slc.get("text") or "").strip())

    enriched["Key Paragraphs"] = format_key_paragraphs(entry.get("key_paragraphs") or [])
    enriched["Event History"] = format_event_history(entry.get("event_history") or [])

    return enriched


def _to_number(val):
    """Tolerant numeric coercion shared with nsw_court_scraper so this script
    benefits from any future schema changes there."""
    s = coerce_leading_number(val)
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def build_workbook(df, sidecar):
    """Filter to the CTP payout-vs-WPI rows and build the enriched output frame.
    Pure: takes the loaded CSV frame + sidecar dict, returns the output frame.

    ISSUE-005: the workbook is the payout-vs-WPI analysis input, so it MUST
    require a positive numeric accepted WPI as well as a positive lump sum —
    not just Case Type == CTP + lump sum."""
    filtered = df[
        is_analysis_ready(df)
        & (df["Case Type"] == "CTP")
        & is_positive_numeric(df["Lump Sum"])
        & is_positive_numeric(df["Impairment % (Accepted)"])
    ].copy()

    if filtered.empty:
        raise SystemExit(
            "No analysis-ready CTP rows with positive numeric Lump Sum AND "
            "Impairment % (Accepted)."
        )

    enriched_records = [enrich_row(url, sidecar) for url in filtered["URL"]]
    enriched_df = pd.DataFrame(enriched_records, index=filtered.index).fillna("")
    out_df = pd.concat([filtered, enriched_df], axis=1)

    new_cols = (
        [col for _, col in NARRATIVE_FIELDS]
        + [col for _, col in SLICE_FIELDS]
        + ["Key Paragraphs", "Event History"]
    )
    flat_cols = [c for c in out_df.columns if c not in new_cols]
    out_df = out_df[flat_cols + new_cols]

    for col in NUMERIC_COERCE_COLUMNS:
        if col in out_df.columns:
            out_df[col] = out_df[col].apply(_to_number)

    # The analysis-useful WPI is Impairment % (Accepted), populated for every
    # filtered row. Drop the strict Impairment % (sparse, misleading), rename
    # Accepted -> "WPI %", and move it next to Lump Sum.
    if "Impairment %" in out_df.columns:
        out_df = out_df.drop(columns=["Impairment %"])
    out_df = out_df.rename(columns={"Impairment % (Accepted)": "WPI %"})
    if "WPI %" in out_df.columns and "Lump Sum" in out_df.columns:
        cols = list(out_df.columns)
        cols.remove("WPI %")
        cols.insert(cols.index("Lump Sum") + 1, "WPI %")
        out_df = out_df[cols]

    return out_df, len(new_cols)


def main():
    input_csv = next((p for p in INPUT_CSV_CANDIDATES if os.path.exists(p)), None)
    if not input_csv:
        raise FileNotFoundError(
            f"Input CSV not found. Checked: {', '.join(INPUT_CSV_CANDIDATES)}"
        )

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    required_columns = {"Case Type", "Impairment % (Accepted)", "Lump Sum", "URL"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    sidecar = {}
    if os.path.exists(SIDECAR_FILE):
        with open(SIDECAR_FILE, "r", encoding="utf-8") as f:
            sidecar = json.load(f)
    else:
        print(f"WARNING: {SIDECAR_FILE} not present — semi-structured fields will be empty.")

    out_df, new_col_count = build_workbook(df, sidecar)
    out_df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Wrote {len(out_df)} rows x {len(out_df.columns)} cols to {OUTPUT_XLSX}")
    print(f"  Source CSV: {input_csv}")
    print(f"  Sidecar:    {'loaded ' + str(len(sidecar)) + ' entries' if sidecar else 'NOT FOUND'}")
    print(f"  New cols added: {new_col_count} (narrative={len(NARRATIVE_FIELDS)}, "
          f"slices={len(SLICE_FIELDS)}, key_paragraphs=1, event_history=1)")


if __name__ == "__main__":
    main()
