import pandas as pd

INPUT_CSV = "detailed_payout_summary.csv"
OUTPUT_XLSX = "ctp_impairment_lump_sum.xlsx"

df = pd.read_csv(INPUT_CSV, dtype=str)
df = df.fillna("")


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


filtered = df[
    (df["Case Type"] == "CTP")
    & is_numeric(df["Impairment %"])
    & is_numeric(df["Lump Sum"])
]

filtered.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote {len(filtered)} rows to {OUTPUT_XLSX}")
