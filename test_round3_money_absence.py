"""
Round 3 (§11) — applicability semantics for the money columns.

Fixtures are real workbook rows. The residual cases are pinned hardest, because
the triage of the 17 high-residual rows found three different faults wearing the
same symptom, and only one of them is what the symptom looked like.

    python test_round3_money_absence.py     # or: pytest
"""

import damages_extraction as dx
import nsw_court_scraper as ns


def _row(**over):
    row = ns.build_result_record("Case", "http://x/1",
                                 status="ok", **{"Decision Date": "2024-01-01"})
    row.update({"Case Type": "CTP", "Damages Extraction Status": "ok"})
    row.update(over)
    return row


# ----------------------------------------------------------------------
# §11 — the four kinds of money column
# ----------------------------------------------------------------------

def test_a_statutory_benefit_is_not_applicable_to_a_damages_determination():
    """539 rows of `Weekly Statutory Benefit` and 530 of `Treatment And Care
    Paid`. Under MAIA these are statutory benefits, not damages, so a damages
    determination that does not quantify one is complete, not deficient —
    exactly the same category as a claimant with no psychiatric injury."""
    for column in ("Weekly Statutory Benefit", "Treatment And Care Paid",
                   "Statutory Benefits Paid"):
        assert dx.classify_money_absence(column) == "not_applicable", column


def test_an_event_that_did_not_happen_is_not_applicable():
    """No buffer awarded, no contributory negligence found, no deduction made."""
    for column in ("Buffer Amount", "Other Deductions", "Statutory Benefits Repaid",
                   "Contributory Negligence Percent", "Contributory Negligence Amount"):
        assert dx.classify_money_absence(column) == "not_applicable", column


def test_a_head_never_in_issue_is_not_applicable_but_one_allowed_is_not_stated():
    assert dx.classify_money_absence(
        "Future Economic Loss", status="Not addressed") == "not_applicable"
    # Allowed and never broken out — a global settlement figure, or economic
    # loss assessed as an undifferentiated buffer.
    assert dx.classify_money_absence(
        "Future Economic Loss", status="Awarded") == "not_stated"


def test_a_column_that_always_applies_is_never_not_applicable():
    """Every award has a net sum payable whether or not the decision states one."""
    assert dx.classify_money_absence("Net Sum Payable") == "not_stated"


def test_corroboration_is_the_only_route_to_absent():
    for column in ("Weekly Statutory Benefit", "Buffer Amount", "Net Sum Payable"):
        assert dx.classify_money_absence(column, corroborated=True) == "absent", column
    # And it outranks the head status: a head recorded as never in issue that
    # still leaves a hole in the accounting identity is a miss.
    assert dx.classify_money_absence(
        "Other Damages Heads", status="Not addressed", corroborated=True) == "absent"


def test_repaid_benefits_show_the_precondition_arose_but_not_the_total():
    """Superseded by round 4 §12.2. This originally read a stated repayment as
    proof the paid TOTAL was recoverable, and so reported 62 rows as extraction
    failures. A repayment proves benefits were paid — the precondition plainly
    arises, so `not_applicable` would deny a fact the row records — but it does
    not show the decision put a total on them. Paid and Repaid are different
    amounts whenever there is a treatment-and-care component."""
    row = _row(**{"Statutory Benefits Repaid": "34899.12",
                  "Statutory Benefits Repaid Provenance": "stated"})
    dx.refine_money_absence(row)
    assert row["Statutory Benefits Paid Provenance"] == "not_stated"

    # Nothing on the row suggests benefits were paid at all.
    row = _row()
    dx.refine_money_absence(row)
    assert row["Statutory Benefits Paid Provenance"] == "not_applicable"


def test_a_positive_figure_is_never_reclassified():
    row = _row(**{"Buffer Amount": "50000", "Buffer Amount Provenance": "stated",
                  "Non-Economic Loss": "100000", "Non-Economic Loss Provenance": "derived"})
    dx.refine_money_absence(row)
    assert row["Buffer Amount Provenance"] == "stated"
    assert row["Non-Economic Loss Provenance"] == "derived"


def test_a_nil_head_is_repaired_to_a_genuine_zero():
    """9 NEL rows carried Status `Nil` with provenance `absent` — a
    contradiction, since a refusal finding IS the zero."""
    row = _row(**{"Non-Economic Loss": "", "Non-Economic Loss Status": "Nil",
                  "Non-Economic Loss Provenance": "absent"})
    dx.refine_money_absence(row)
    assert row["Non-Economic Loss"] == "0"
    assert row["Non-Economic Loss Provenance"] == "derived"


# ----------------------------------------------------------------------
# §11.1 — Other Damages Heads
# ----------------------------------------------------------------------

def _apportioned(other_gross, **over):
    row = _row(**{"Total Damages Gross": str(other_gross),
                  "Total Damages Gross Provenance": "stated",
                  "Non-Economic Loss": "400000", "Non-Economic Loss Provenance": "stated",
                  "Past Economic Loss": "138867.32", "Past Economic Loss Provenance": "stated",
                  "Future Economic Loss": "180000", "Future Economic Loss Provenance": "stated"})
    row.update(over)
    return row


def test_the_identity_closing_means_there_is_no_other_head():
    """302 of the 385 blanks. Not one of the 155 observed values is zero, so the
    column is only ever written when an other head exists — a blank where the
    identity closes is a zero, not missing data."""
    row = _apportioned(718867.32)
    residual, trustworthy = dx.damages_residual(row)
    assert trustworthy and abs(residual) < 1
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads"] == "0"
    assert row["Other Damages Heads Provenance"] == "derived"
    assert row["Other Damages Heads Status"] == "Nil"


def test_macdonald_a_genuine_uncaptured_head():
    """Macdonald (No 2): NEL 400,000 + PEL 125,105.60 + past super 13,761.72 +
    FEL 180,000 + future super 26,244 = 745,111.32. `Future Economic Loss`
    excludes future superannuation by definition, so the 26,244 is a real other
    head that was never captured.

    Round 6 §14.3 added a second requirement: the residual alone does not say
    WHICH column is wrong, so `absent` also needs the other head itemised in
    the decision. Macdonald's breakdown itemises it."""
    row = _apportioned(745111.32, **{
        "Award Breakdown": ("future economic loss at $180,000 plus future "
                            "superannuation of $26,244, a subtotal of $745,111.32")})
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] == "absent"


def test_taaga_an_unapportioned_settlement_is_not_an_other_head():
    """Taaga: "The parties agreed total damages at $1,900,000 ... The decision
    did not apportion the $1,900,000 between heads of damage." The whole gross
    shows up as residual, and it says nothing whatever about other heads.
    Reading it as one would invent a $1.9m head that does not exist."""
    row = _row(**{"Total Damages Gross": "1900000",
                  "Total Damages Gross Provenance": "stated",
                  "Non-Economic Loss Status": "Not addressed",
                  "Past Economic Loss Status": "Not addressed",
                  "Future Economic Loss Status": "Not addressed"})
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] != "absent"
    # And the named heads were ALLOWED, just never broken out — so they are
    # `not_stated`, not `not_applicable`. A downstream `fel_applies` gate that
    # read `not_applicable` here would drop a live head.
    assert row["Future Economic Loss Provenance"] == "not_stated"
    assert row["Non-Economic Loss Provenance"] == "not_stated"


def test_javed_a_nil_head_still_counts_as_unapportioned():
    """Javed: "$165,000, representing damages for past and future economic
    loss; the decision did not apportion it and found no entitlement to
    non-economic loss". NEL is a quantified zero, so counting populated cells
    would call this apportioned. What matters is that none of the gross was
    allocated."""
    row = _row(**{"Total Damages Gross": "165000",
                  "Total Damages Gross Provenance": "stated",
                  "Non-Economic Loss": "0", "Non-Economic Loss Status": "Nil",
                  "Non-Economic Loss Provenance": "derived",
                  "Past Economic Loss Status": "Not addressed",
                  "Future Economic Loss Status": "Not addressed"})
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] != "absent"


def test_pantelis_a_named_head_holding_the_net_is_reported_as_such():
    """Pantelis: "settlement comprising non-economic loss of $275,000.
    Contributory negligence of 20% reduced damages by $55,000, resulting in a
    payable sum of $220,000" — and 220,000 was recorded as the head. The
    residual equals the reduction exactly. That is a defect in `Non-Economic
    Loss`, and attributing it to other heads would hide it."""
    row = _row(**{"Total Damages Gross": "275000",
                  "Total Damages Gross Provenance": "stated",
                  "Non-Economic Loss": "220000", "Non-Economic Loss Provenance": "stated",
                  "Past Economic Loss Status": "Not addressed",
                  "Future Economic Loss Status": "Not addressed",
                  "Contributory Negligence Amount": "55000"})
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] != "absent"
    assert "net figure" in row["Damages Notes"]


def test_a_derived_gross_cannot_corroborate_anything():
    """If the gross was itself computed from the heads the identity closes by
    construction, so the residual carries no information either way."""
    row = _apportioned(745111.32, **{"Total Damages Gross Provenance": "derived"})
    _residual, trustworthy = dx.damages_residual(row)
    assert trustworthy is False


def test_an_unknown_head_makes_the_residual_unattributable():
    row = _apportioned(745111.32, **{"Past Economic Loss": "",
                                     "Past Economic Loss Provenance": "not_stated",
                                     "Past Economic Loss Status": "Awarded"})
    _residual, trustworthy = dx.damages_residual(row)
    assert trustworthy is False


def test_other_damages_heads_now_has_a_status_column():
    """§11.1: the only money head that shipped without one, so a
    considered-and-refused head and one never in issue both collapsed to null."""
    assert "Other Damages Heads Status" in ns.RESULT_FIELDS
    assert ("Other Damages Heads Status", "Other Damages Heads") in dx.STATUS_AMOUNT_PAIRS
    row = _row(**{"Other Damages Heads": "12430",
                  "Other Damages Heads Provenance": "stated"})
    dx.refine_money_absence(row)
    assert row["Other Damages Heads Status"] == "Awarded"


# ----------------------------------------------------------------------
# Round 2 regressions, and the ex gratia decision
# ----------------------------------------------------------------------

def test_the_pass_is_idempotent():
    row = _apportioned(745111.32)
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    first = dict(row)
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    for _amount, prov in dx.MONEY_PROVENANCE_PAIRS:
        assert row[prov] == first[prov], prov


def test_silcocks_an_ex_gratia_payment_withholds_the_wpi():
    """Operator decision, 2026-08-07: on an ex gratia payment the WPI is
    withheld. The figure is correct, but publishing it makes every downstream
    s 4.11 check read the row as an impossible combination — a checker
    comparing WPI to 10 cannot see that the payment was never made under
    s 4.11 at all. The figure stays in `WPI % Candidates`."""
    row = _row(**{
        "Non-Economic Loss Status": "Awarded", "Non-Economic Loss": "120000",
        "Impairment % (Accepted)": "9",
        "Catchwords": ("assessments of whole person impairment 9%; notwithstanding, "
                       "no entitlement non-economic loss offer of settlement by insurer; "
                       "appropriate compromise having regard to serious injury sustained "
                       "and where no legal obligation on insurer to make any allowance "
                       "for non-economic loss"),
    })
    ns.quarantine_impossible_wpi(row)
    ns.apply_round2_semantics(row)
    assert row["Impairment % (Accepted)"] == ""
    assert row["WPI Candidates"] == "9"
    assert row["WPI Provenance"] == "not_applicable"      # not a defect
    assert row["_wpi_ex_gratia"] is True
    # The row is no longer an s 4.11 violation, because the rule never applied.
    assert row["NEL Threshold Consistent"] == "cannot determine"
    assert ns.annotate_analysis_fields(row)["Analysis Ready"] == "Yes"


def _run():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
