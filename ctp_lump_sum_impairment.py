import pandas as pd

INPUT_CSV = "detailed_payout_summary.csv"
OUTPUT_XLSX = "ctp_impairment_lump_sum.xlsx"

df = pd.read_csv(INPUT_CSV, dtype=str)
df = df.fillna("")

NON_NUMERIC = {"", "n/a", "unknown", "none", "nan"}

filtered = df[
    (df["Case Type"] == "CTP")
    & (~df["Impairment %"].str.strip().str.lower().isin(NON_NUMERIC))
    & (~df["Lump Sum"].str.strip().str.lower().isin(NON_NUMERIC))
]

filtered.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote {len(filtered)} rows to {OUTPUT_XLSX}")
