"""
Acceptance checks for the damages-breakdown delivery (downstream spec section 6).

Run against the delivered workbook:

    python check_damages_acceptance.py [ctp_impairment_lump_sum.xlsx]

Implements the consumer's five criteria verbatim, plus the gross-vs-net
confirmation they asked for in section 3.3. It reports what the data says; it
does not tune anything to pass — criterion 2 in particular is the honest
measure of extractor accuracy against columns we already trust, so a failure
there is information, not a bug to paper over.
"""

import sys

import pandas as pd

from nsw_court_scraper import WORKBOOK_FILE

TOLERANCE = 1000.0

HEADS = ["Non-Economic Loss", "Past Economic Loss", "Future Economic Loss"]
DEDUCTIONS = ["Contributory Negligence Amount", "Statutory Benefits Repaid", "Other Deductions"]

MONEY_PROVENANCE_PAIRS = [
    ("Lump Sum", "Lump Sum Provenance"),
    ("Net Sum Payable", "Net Sum Payable Provenance"),
    ("Non-Economic Loss", "Non-Economic Loss Provenance"),
    ("Future Economic Loss", "Future Economic Loss Provenance"),
    ("Past Economic Loss", "Past Economic Loss Provenance"),
    ("Total Damages Gross", "Total Damages Gross Provenance"),
    ("Contributory Negligence Percent", "Contributory Negligence Percent Provenance"),
    ("Contributory Negligence Amount", "Contributory Negligence Amount Provenance"),
    ("Statutory Benefits Repaid", "Statutory Benefits Repaid Provenance"),
    ("Other Deductions", "Other Deductions Provenance"),
    ("Buffer Amount", "Buffer Amount Provenance"),
    ("Other Damages Heads", "Other Damages Heads Provenance"),
    ("Statutory Benefits Paid", "Statutory Benefits Paid Provenance"),
    ("Treatment And Care Paid", "Treatment And Care Paid Provenance"),
    ("Weekly Statutory Benefit", "Weekly Statutory Benefit Provenance"),
]

STATUS_AMOUNT_PAIRS = [
    ("Past Economic Loss Status", "Past Economic Loss"),
    ("Non-Economic Loss Status", "Non-Economic Loss"),
    ("Future Economic Loss Status", "Future Economic Loss"),
]


def num(series):
    return pd.to_numeric(series, errors="coerce")


def head_value(df, head):
    """A head's contribution to the identity: its amount, with 'Nil' and
    'Not addressed' counting as a known zero, and an 'Awarded' head with no
    amount counting as unknown (NaN)."""
    amount = num(df[head])
    status = df.get(f"{head} Status", pd.Series("", index=df.index)).astype(str).str.strip()
    known_zero = status.isin(["Nil", "Not addressed"])
    return amount.where(~known_zero, 0.0)


def _result(name, ok, detail):
    return {"criterion": name, "pass": ok, "detail": detail}


def check_reconciliation(df):
    """Criterion 1: both identities within $1,000 for >=90% of eligible rows."""
    out = []

    stated = pd.Series(True, index=df.index)
    for head in HEADS + ["Total Damages Gross"]:
        col = f"{head} Provenance"
        stated &= df.get(col, pd.Series("", index=df.index)).astype(str).str.strip().eq("stated")
    sub = df[stated]
    if len(sub) == 0:
        out.append(_result("1a heads identity", False,
                           "0 rows have all four figures 'stated' — cannot evaluate"))
    else:
        gross = num(sub["Total Damages Gross"])
        total = sum(head_value(sub, h) for h in HEADS)
        for extra in ("Buffer Amount", "Other Damages Heads"):
            if extra in sub.columns:
                total = total + num(sub[extra]).fillna(0.0)
        resid = (gross - total).abs()
        rate = (resid <= TOLERANCE).mean()
        out.append(_result(
            "1a heads identity",
            rate >= 0.90,
            f"{rate:.1%} of {len(sub)} all-stated rows reconcile within ${TOLERANCE:,.0f} "
            f"(median |residual| ${resid.median():,.0f})",
        ))

    # 1b, as the spec words it: Lump Sum ~= Total Damages Gross - deductions.
    # Rows whose gross was derived FROM the lump sum would close by
    # construction and are excluded.
    gross = num(df["Total Damages Gross"])
    ded = sum(num(df[c]).fillna(0.0) for c in DEDUCTIONS if c in df.columns)
    derivation = df.get("Damages Gross Derivation", pd.Series("", index=df.index)).astype(str)
    for label, col in (("1b net identity (Lump Sum)", "Lump Sum"),
                       ("1b net identity (Net Sum Payable)", "Net Sum Payable")):
        if col not in df.columns:
            continue
        payable = num(df[col])
        eligible = payable.notna() & gross.notna() & ~derivation.str.startswith("net")
        if eligible.sum() == 0:
            out.append(_result(label, False, "0 eligible rows — cannot evaluate"))
            continue
        resid = (payable[eligible] - (gross[eligible] - ded[eligible])).abs()
        rate = (resid <= TOLERANCE).mean()
        out.append(_result(
            label, rate >= 0.90,
            f"{rate:.1%} of {int(eligible.sum())} rows satisfy {col} = Gross - deductions "
            f"within ${TOLERANCE:,.0f} (median |residual| ${resid.median():,.0f})",
        ))
    return out


def check_regression(df):
    """Criterion 2: the re-extracted NEL/FEL must agree with the trusted
    columns within $1,000 on >=95% of rows where both are non-null. THIS IS
    THE KEY CHECK — it is what tells the consumer whether to believe past
    economic loss, where no ground truth exists."""
    out = []
    for head in ("Non-Economic Loss", "Future Economic Loss"):
        recheck_col = f"{head} (Recheck)"
        if recheck_col not in df.columns:
            out.append(_result(f"2 {head} regression", False, f"{recheck_col} missing"))
            continue
        a, b = num(df[head]), num(df[recheck_col])
        both = a.notna() & b.notna()
        if both.sum() == 0:
            out.append(_result(f"2 {head} regression", False, "no comparable rows"))
            continue
        agree = ((a[both] - b[both]).abs() <= TOLERANCE)
        rate = agree.mean()
        out.append(_result(
            f"2 {head} regression",
            rate >= 0.95,
            f"{rate:.1%} of {int(both.sum())} comparable rows agree within "
            f"${TOLERANCE:,.0f} ({int((~agree).sum())} disagreements)",
        ))
    return out


def check_coverage(df):
    """Criterion 3: status populated for 100% of rows; amount 'stated' for
    >=60%."""
    out = []
    status = df.get("Past Economic Loss Status", pd.Series("", index=df.index))
    populated = status.astype(str).str.strip().isin(["Awarded", "Nil", "Not addressed"]).mean()
    out.append(_result("3a past EL status coverage", populated >= 1.0,
                       f"{populated:.1%} of {len(df)} rows carry a valid status"))

    prov = df.get("Past Economic Loss Provenance", pd.Series("", index=df.index))
    stated = prov.astype(str).str.strip().eq("stated").mean()
    out.append(_result("3b past EL amount stated", stated >= 0.60,
                       f"{stated:.1%} of rows have a 'stated' past economic loss "
                       f"(target >=60%, observed mention rate 66.5%)"))
    return out


def check_status_discipline(df):
    """Criterion 4: no 'Not addressed' row carries an amount; no 'Nil' row
    carries a non-zero amount."""
    violations = []
    for status_col, amount_col in STATUS_AMOUNT_PAIRS:
        if status_col not in df.columns or amount_col not in df.columns:
            continue
        status = df[status_col].astype(str).str.strip()
        amount = num(df[amount_col])
        bad_na = int((status.eq("Not addressed") & amount.notna()).sum())
        bad_nil = int((status.eq("Nil") & amount.fillna(0).ne(0)).sum())
        if bad_na:
            violations.append(f"{amount_col}: {bad_na} 'Not addressed' rows carry an amount")
        if bad_nil:
            violations.append(f"{amount_col}: {bad_nil} 'Nil' rows carry a non-zero amount")
    return [_result("4 status discipline", not violations,
                    "; ".join(violations) if violations else "no violations")]


def check_provenance(df):
    """Criterion 5: every money field has a non-null provenance value."""
    valid = {"stated", "derived", "inferred", "absent"}
    missing = []
    for amount_col, prov_col in MONEY_PROVENANCE_PAIRS:
        if amount_col not in df.columns:
            continue
        if prov_col not in df.columns:
            missing.append(f"{prov_col}: column absent")
            continue
        bad = int((~df[prov_col].astype(str).str.strip().isin(valid)).sum())
        if bad:
            missing.append(f"{prov_col}: {bad} rows without a valid value")
    return [_result("5 provenance completeness", not missing,
                    "; ".join(missing) if missing else
                    f"all {len(MONEY_PROVENANCE_PAIRS)} money fields carry a provenance value")]


def report_context(df):
    """Section 3.3: confirm or correct the consumer's assumption that Lump Sum
    is net, plus the residual picture that motivated the request."""
    lines = []
    if "Lump Sum Basis" in df.columns:
        counts = df["Lump Sum Basis"].astype(str).str.strip().value_counts()
        lines.append("Lump Sum Basis: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # The direct answer to "is Lump Sum net?": test both readings on the rows
    # where the gross figure was independently stated.
    gross = num(df["Total Damages Gross"])
    ded = sum(num(df[c]).fillna(0.0) for c in DEDUCTIONS if c in df.columns)
    derivation = df.get("Damages Gross Derivation", pd.Series("", index=df.index)).astype(str)
    stated_gross = derivation.eq("stated") & gross.notna()
    has_ded = ded > 0
    scope = stated_gross & has_ded
    if scope.sum():
        lump = num(df["Lump Sum"])
        as_gross = ((lump[scope] - gross[scope]).abs() <= TOLERANCE).mean()
        as_net = ((lump[scope] - (gross[scope] - ded[scope])).abs() <= TOLERANCE).mean()
        lines.append(
            f"Gross-vs-net test on {int(scope.sum())} rows with a stated gross AND a "
            f"non-zero deduction: Lump Sum matches GROSS {as_gross:.1%}, "
            f"matches NET {as_net:.1%}"
        )
    else:
        lines.append("Gross-vs-net test: no rows with both a stated gross and a deduction")

    lump = num(df["Lump Sum"])
    resid = lump - sum(head_value(df, h) for h in HEADS)
    ok = resid.notna()
    if ok.any():
        r = resid[ok]
        lines.append(
            f"Lump Sum - (NEL + past EL + future EL): "
            f"{(r.abs() <= TOLERANCE).mean():.1%} reconcile, "
            f"{(r > TOLERANCE).mean():.1%} positive residual, "
            f"{(r < -TOLERANCE).mean():.1%} negative; "
            f"median ${r.median():,.0f}, p95 ${r.quantile(0.95):,.0f}"
        )
        lines.append("  (before this change the same statistic omitted past EL entirely: "
                     f"{( (lump - head_value(df,'Non-Economic Loss') - head_value(df,'Future Economic Loss')) > TOLERANCE).mean():.1%} "
                     "positive residual)")

    for col in ("Damages Reconciled", "Net Reconciled", "Accident Mechanism",
                "Claimant Road Role", "Has Psychiatric Injury", "WC Overlap",
                "Damages Extraction Status"):
        if col in df.columns:
            counts = df[col].astype(str).str.strip().value_counts().head(8)
            lines.append(f"{col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    if "Damages Reconciled" in df.columns:
        rec = df["Damages Reconciled"].astype(str).str.strip()
        lines.append(
            f"Heads identity over ALL rows (not just all-stated): "
            f"{rec.eq('yes').mean():.1%} yes, {rec.eq('no').mean():.1%} no, "
            f"{rec.eq('insufficient data').mean():.1%} insufficient"
        )

    for col in ("Statutory Benefits Paid", "Treatment And Care Paid", "Weekly Benefit",
                "Medical Costs", "WPI Physical %", "WPI Psychiatric %",
                "Past Economic Loss", "Total Damages Gross", "Net Sum Payable"):
        if col in df.columns:
            filled = df[col].notna() & df[col].astype(str).str.strip().ne("")
            lines.append(f"populated {col}: {filled.mean():.1%}")

    for col in ("Description", "Description With Figures"):
        if col in df.columns:
            rate = df[col].astype(str).str.contains(r"\$", regex=True).mean()
            lines.append(f"{col} containing a $ figure: {rate:.1%}")
    return lines


def main(path=None):
    path = path or WORKBOOK_FILE
    df = pd.read_excel(path, keep_default_na=False, na_values=[""])
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols from {path}\n")

    results = []
    results += check_reconciliation(df)
    results += check_regression(df)
    results += check_coverage(df)
    results += check_status_discipline(df)
    results += check_provenance(df)

    width = max(len(r["criterion"]) for r in results)
    print("ACCEPTANCE CRITERIA (spec section 6)")
    print("=" * 78)
    for r in results:
        flag = "PASS" if r["pass"] else "FAIL"
        print(f"  [{flag}] {r['criterion']:<{width}}  {r['detail']}")

    print("\nCONTEXT / SECTION 3.3 CONFIRMATION")
    print("=" * 78)
    for line in report_context(df):
        print(f"  {line}")

    failed = [r for r in results if not r["pass"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} criteria pass.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
