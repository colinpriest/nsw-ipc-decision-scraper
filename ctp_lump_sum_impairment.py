import pandas as pd

INPUT_CSV = "detailed_payout_summary.csv"
OUTPUT_XLSX = "ctp_impairment_lump_sum.xlsx"

df = pd.read_csv(INPUT_CSV)

filtered = df[
    (df["Case Type"] == "CTP")
    & (df["Impairment %"].notna())
    & (df["Impairment %"].astype(str).str.strip() != "")
    & (df["Lump Sum"].notna())
    & (df["Lump Sum"].astype(str).str.strip() != "")
]

filtered.to_excel(OUTPUT_XLSX, index=False)
print(f"Wrote {len(filtered)} rows to {OUTPUT_XLSX}")
