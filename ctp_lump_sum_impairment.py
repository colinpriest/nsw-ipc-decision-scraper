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

from nsw_court_scraper import (
    ANALYSIS_READY_REPORT,
    CSV_REPORT,
    SIDECAR_FILE,
    WORKBOOK_FILE,
    coerce_leading_number,
    normalise_medical_costs,
)
from damages_extraction import DAMAGES_NUMERIC_FIELDS, DAMAGES_SIGNED_FIELDS, to_float

INPUT_CSV_CANDIDATES = [ANALYSIS_READY_REPORT, CSV_REPORT]
OUTPUT_XLSX = WORKBOOK_FILE

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
] + list(DAMAGES_NUMERIC_FIELDS)


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

    Population rule: analysis-ready CTP rows with a **positive lump sum**.

    An earlier rule also required a positive accepted WPI (ISSUE-005), on the
    reasoning that a payout-vs-WPI workbook needs both axes. That silently
    excluded ~250 real awards whose decision simply never states a WPI, which
    is a fact about the decision, not a defect in the row — and it made the
    absence invisible rather than modellable. WPI now stays BLANK on those rows
    with `WPI % Provenance = absent`, so a consumer can filter them out for a
    WPI-conditional analysis and keep them for everything else."""
    filtered = df[
        is_analysis_ready(df)
        & (df["Case Type"] == "CTP")
        & is_positive_numeric(df["Lump Sum"])
    ].copy()

    if filtered.empty:
        raise SystemExit("No analysis-ready CTP rows with a positive numeric Lump Sum.")

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
            # Residuals are signed; everything else uses the shared coercion,
            # which rejects negatives on purpose.
            coerce = to_float if col in DAMAGES_SIGNED_FIELDS else _to_number
            out_df[col] = out_df[col].apply(coerce)

    # The analysis-useful WPI is Impairment % (Accepted). Drop the strict
    # Impairment % (sparse, misleading), rename Accepted -> "WPI %", and move it
    # next to Lump Sum.
    if "Impairment %" in out_df.columns:
        out_df = out_df.drop(columns=["Impairment %"])

    # Spec 4.1 asked us to populate or drop three empty columns. Statutory
    # Benefits and Medical Costs are now populated (the latter was only ever
    # *reading* as empty — see normalise_medical_costs). Weekly Benefit is
    # genuinely absent: a CTP damages assessment or settlement approval almost
    # never states a weekly statutory-benefit rate, so it is dropped here
    # rather than left implying data we do not have. The damages pass's
    # `Weekly Statutory Benefit` stays, because it carries a provenance value
    # that distinguishes "not stated" from "not looked for".
    if "Weekly Benefit" in out_df.columns:
        out_df = out_df.drop(columns=["Weekly Benefit"])
    out_df = out_df.rename(columns={"Impairment % (Accepted)": "WPI %"})

    # WPI is no longer guaranteed present, so it needs the same provenance
    # discipline as every other extracted figure: a blank must be readable as
    # "the decision does not state one", not as a missing cell, and a value we
    # computed or estimated must not read as one the decision stated.
    out_df = out_df.rename(columns={
        "WPI Provenance": "WPI % Provenance",
        "WPI Basis": "WPI % Basis",
        "WPI Candidates": "WPI % Candidates",
    })
    if "WPI %" in out_df.columns:
        # Rows predating the WPI-resolution pass carry no provenance (a missing
        # column, or a blank cell). Fall back to value-present == stated, so the
        # column is never partially populated — a blank provenance would be
        # indistinguishable from a blank value.
        if "WPI % Provenance" not in out_df.columns:
            out_df["WPI % Provenance"] = ""
        fallback = out_df["WPI %"].map(lambda v: "absent" if pd.isna(v) else "stated")
        blank = out_df["WPI % Provenance"].isna() | \
            out_df["WPI % Provenance"].astype(str).str.strip().eq("")
        out_df.loc[blank, "WPI % Provenance"] = fallback[blank]
        wpi_cols = [c for c in ("WPI %", "WPI % Provenance", "WPI % Basis",
                                "WPI % Candidates") if c in out_df.columns]
        cols = [c for c in out_df.columns if c not in wpi_cols]
        anchor = cols.index("Lump Sum") + 1 if "Lump Sum" in cols else 0
        cols[anchor:anchor] = wpi_cols
        out_df = out_df[cols]

    return out_df, len(new_cols)


def main():
    input_csv = next((p for p in INPUT_CSV_CANDIDATES if os.path.exists(p)), None)
    if not input_csv:
        raise FileNotFoundError(
            f"Input CSV not found. Checked: {', '.join(INPUT_CSV_CANDIDATES)}"
        )

    # keep_default_na=False so sentinels that look like nulls to pandas —
    # notably Medical Costs' "N/A" — survive the round trip instead of
    # reaching the consumer as an empty column (spec 4.1).
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False).fillna("")
    if "Medical Costs" in df.columns:
        df["Medical Costs"] = df["Medical Costs"].apply(normalise_medical_costs)
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
