import os
import pandas as pd

INPUT_CSV_CANDIDATES = [
    "analysis_ready_payout_summary.csv",
    "detailed_payout_summary.csv",
]
OUTPUT_XLSX = "ctp_impairment_lump_sum.xlsx"

INPUT_CSV = next((path for path in INPUT_CSV_CANDIDATES if os.path.exists(path)), None)
if not INPUT_CSV:
    raise FileNotFoundError(
        f"Input CSV not found. Checked: {', '.join(INPUT_CSV_CANDIDATES)}"
    )

df = pd.read_csv(INPUT_CSV, dtype=str)
df = df.fillna("")

required_columns = {"Case Type", "Impairment %", "Lump Sum"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")


def is_numeric(series):
    """Return boolean mask for values that are valid numbers."""
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


def is_analysis_ready(dataframe):
    """Return boolean mask for rows suitable for downstream analysis."""
    if "Analysis Ready" in dataframe.columns:
        return dataframe["Analysis Ready"].astype(str).str.strip().eq("Yes")

    if "Status" in dataframe.columns:
        status_ok = dataframe["Status"].astype(str).str.strip().eq("ok")
    else:
        status_ok = pd.Series(True, index=dataframe.index)

    if "LLM Error" in dataframe.columns:
        llm_ok = dataframe["LLM Error"].astype(str).str.strip().eq("")
    else:
        llm_ok = pd.Series(True, index=dataframe.index)

    has_decision_date = dataframe["Decision Date"].astype(str).str.strip().str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    return status_ok & llm_ok & has_decision_date


filtered = df[
    is_analysis_ready(df)
    & (df["Case Type"] == "CTP")
    & is_numeric(df["Impairment %"])
    & is_numeric(df["Lump Sum"])
]

filtered.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote {len(filtered)} rows to {OUTPUT_XLSX}")
