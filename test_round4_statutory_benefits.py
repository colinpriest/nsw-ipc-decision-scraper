"""
Round 4 (§12) — Statutory Benefits Paid, and the MACA-era vocabulary gap.

Quotes are verbatim from the decisions named.

    python test_round4_statutory_benefits.py     # or: pytest
"""

import damages_extraction as dx
import statutory_benefits_recovery as sbr


# ----------------------------------------------------------------------
# §12.1 — Paid and Repaid are different fields
# ----------------------------------------------------------------------

def test_paid_is_never_derived_from_repaid():
    """The single most tempting shortcut: deriving Paid from Repaid closes all
    62 rows at a stroke. It is wrong. On row 43 the difference between them is
    $34,893.12, which is that row's `Treatment And Care Paid` to the cent —
    Paid is everything the claimant received, Repaid is the s 3.40 deduction,
    which reaches only the recoverable categories. They agree on 96% of rows
    only because treatment and care is 98.1% not applicable, so asserting
    equality understates benefits paid in exactly the cases the field exists
    for."""
    row = {"Statutory Benefits Repaid": "49568.97",
           "Statutory Benefits Repaid Provenance": "stated",
           "Statutory Benefits Paid": "",
           "Statutory Benefits Paid Provenance": "absent"}
    dx.refine_money_absence(row)
    assert row["Statutory Benefits Paid"] == ""
    assert row["Statutory Benefits Paid Provenance"] != "stated"


def test_a_described_deduction_is_not_a_missed_paid_total():
    """§12.2. Round 3 treated a stated repayment as proof the paid total was
    recoverable, which reported 62 rows as extraction failures. A deduction
    proves benefits were paid, not that the decision quantifies the total."""
    row = {"Statutory Benefits Repaid": "9840.46",
           "Statutory Benefits Repaid Provenance": "stated",
           "Statutory Benefits Paid": "",
           "Statutory Benefits Paid Provenance": "absent"}
    dx.refine_money_absence(row)
    assert row["Statutory Benefits Paid Provenance"] == "not_stated"


# ----------------------------------------------------------------------
# §12.3 — MACA-era wording
# ----------------------------------------------------------------------

def test_greer_the_maca_case_the_request_cites():
    """Greer [2026] NSWPIC 279 — "s 130 MACA credit for s 83 payments". The
    same concept as a s 3.40 statutory-benefits deduction, in Motor Accidents
    Compensation Act 1999 terms."""
    text = ("the insurer is to have credit for the following payments in "
            "accordance with s 130. Section 83 payments $825.10. COSTS AND "
            "DISBURSEMENTS")
    amount, quote = sbr.find_statutory_benefits_paid(text)
    assert amount == 825.10
    assert quote


def test_maca_payment_statements_in_several_shapes():
    for text, expected in [
        ("The insurer confirmed the s 83 payments amount to $12,369.", 12369.0),
        ("The insurer has paid s 83 expenses in the sum of $9,774.75.", 9774.75),
        ("Section 83 expenses amount to $880 not including a non-", 880.0),
        ("the following payments: S 83 payments $3,021.81 LEGAL COSTS", 3021.81),
    ]:
        amount, _ = sbr.find_statutory_benefits_paid(text)
        assert amount == expected, text


def test_a_section_130_credit_heading_does_not_disqualify_a_section_83_total():
    """s 83 is the obligation to PAY; s 130 is the separate right to recover.
    So an itemised "Section 83 payments $X" states an amount paid even when it
    sits under a s 130 credit heading — which is how Pearson and Obeid read."""
    text = ("the insurer is to have credit for the following payments made in "
            "accordance with s 130 of the Act: Section 83 payments $3,073.70")
    assert sbr.find_statutory_benefits_paid(text)[0] == 3073.70


def test_maia_wording_is_accepted_only_when_it_states_an_amount_paid():
    paid = ("The insurer confirmed that the claimant has been paid weekly "
            "payments of statutory benefits in the amount of $68,144.90.")
    assert sbr.find_statutory_benefits_paid(paid)[0] == 68144.90

    # The same words framed as the deduction state `Repaid`, not `Paid`.
    for deduction in [
        "the insurer seeks a deduction for weekly payments of statutory "
        "benefits paid of $2,395.17",
        "the only deduction to be made from the proposed settlement was the "
        "sum of $23,245.66 representing weekly payments of statutory benefits",
        "The total amount proposed is $22,000 less deduction of statutory "
        "benefits paid in the sum of $1,810.60",
        "the insurer is entitled to deduct the sum of $45,034.22 from the "
        "proposed settlement by way of recovery for weekly payments of "
        "statutory benefits paid to the claimant under Division 3.3",
    ]:
        assert sbr.find_statutory_benefits_paid(deduction)[0] is None, deduction


def test_a_stated_total_counts_even_when_it_equals_the_deduction():
    """Mepani states a paid total that happens to match the repayment exactly.
    Filtering equal values as duplicates would hide genuinely stated figures —
    the point is where the figure came from, not what it equals."""
    text = ("On a commercial basis the insurer has allowed the sum of "
            "$212,752.78 in benefits paid to date and $34,297 in taxation")
    assert sbr.find_statutory_benefits_paid(text)[0] == 212752.78


def test_lookalikes_that_are_not_benefits_totals():
    for text in [
        # Fox v Wood: tax on benefits, not the benefits.
        "(c) tax paid on statutory benefits $16,259.00 (d) past superannuation",
        "(iv) tax paid on statutory benefits $868 (b) future economic loss",
        # A parenthetical inside another head.
        "Past economic loss (incl s 83 payments) $222,124 Future economic loss",
        "Past treatment (incl. s 83 payments) $225,373.98 Future treatment",
        # A buffer.
        "damages for past economic loss limited to statutory benefits paid by "
        "insurer and a $150,000 buffer for future loss",
    ]:
        assert sbr.find_statutory_benefits_paid(text)[0] is None, text


def test_maca_language_detection():
    assert sbr.uses_maca_language("s 130 MACA credit for s 83 payments")
    assert sbr.uses_maca_language("under the Motor Accidents Compensation Act 1999")
    assert sbr.uses_maca_language("", "Section 83 payments $825.10")
    assert not sbr.uses_maca_language("s 3.40(1)(b) statutory benefits deduction")
    assert not sbr.uses_maca_language("", None)


def test_maca_vocabulary_reaches_the_context_window():
    """The structural half of §12.3. `_DAMAGES_KEYWORDS` selects which part of
    a long decision the model ever sees; a MACA quantum section that says only
    "s 83" was being trimmed out before extraction, which is why the miss was
    "by construction" rather than a model failure."""
    for term in ("s 83", "section 83", "s 130", "maca",
                 "motor accidents compensation act"):
        assert term in dx._DAMAGES_KEYWORDS, term


def test_the_schema_warns_against_copying_repaid_into_paid():
    field = dx.DamagesSchema.model_fields["statutory_benefits_paid"]
    description = field.description
    assert "s 83" in description
    assert "NEVER copy one into the other" in description


def test_empty_and_junk_input():
    assert sbr.find_statutory_benefits_paid("")[0] is None
    assert sbr.find_statutory_benefits_paid(None)[0] is None
    # Below the plausibility floor for a benefits total.
    assert sbr.find_statutory_benefits_paid("Section 83 payments $12")[0] is None


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
