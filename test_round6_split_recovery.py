"""
Round 6 (§14) — filling the split columns from the resolution, and the review
panel supersession rule that reconciling Tiwari exposed.

    python test_round6_split_recovery.py     # or: pytest
"""

from types import SimpleNamespace

import damages_extraction as dx
import nsw_court_scraper as ns
import wpi_resolution as w


def _m(value, system, kind="MAS certificate", assessor="Dr A", **over):
    m = dict(value=str(value), kind=kind, body_system=system, assessor=assessor,
             superseded=False, about_claimant=True, quote="")
    m.update(over)
    return SimpleNamespace(**m)


def _parsed(mentions, **over):
    base = dict(mentions=mentions, tribunal_selected_value="",
                tribunal_selected_quote="", components_share_one_assessment=True,
                totals_are_rival_assessments=True, threshold_finding="not determined",
                settlement_approval_without_wpi=False, notes="")
    base.update(over)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# §14.2 — Tiwari, and the rule it exposed
# ----------------------------------------------------------------------

def test_tiwari_a_review_panel_supersedes_the_certificate_it_reviewed():
    """Tiwari [2026] NSWPIC 251 (row 14), reconciled against source.

    MAS Oates certified 0% for the lumbar spine; a Review Panel then certified
    12%; MAS Roberts certified 7% for PTSD. Averaging Oates and the Panel gave
    6%, which made psychiatric (7%) look like the governing system — and that
    is why the label said `psychiatric` while the total said 12.

    A review panel does not offer a rival opinion: under s 7.26 it may revoke
    the certificate it reviewed and issue its own. So physical is 12, not 6,
    psychiatric is 7, and PHYSICAL governs — consistent with the total of 12.
    The consumer's proposed fill (psychiatric = 12) would have been wrong.
    """
    out = w.resolve_wpi(_parsed([
        _m(0, "physical", assessor="MAS Oates"),
        _m(12, "physical", assessor="Review Panel (Garvey)"),
        _m(7, "psychiatric", assessor="MAS Roberts"),
    ]))
    assert out["Impairment % (Accepted)"] == "12"
    assert out["WPI Governing System"] == "physical"
    assert out["_wpi_systems"]["physical"][0] == "12"
    assert out["_wpi_systems"]["psychiatric"][0] == "7"
    assert "supersedes" in out["WPI Basis"]


def test_supersession_only_applies_to_rival_certificates():
    """Quigley: MAS Curtin 4% (scarring, nerve) and a Review Panel 8% (brain
    injury, shoulder) cover DIFFERENT injuries, so they combine to 12. A panel
    that reviewed one certificate does not supersede a separate certificate
    about other injuries."""
    out = w.resolve_wpi(_parsed([
        _m(4, "physical", assessor="MAS Curtin"),
        _m(8, "physical", assessor="Review Panel"),
    ], totals_are_rival_assessments=False))
    assert out["Impairment % (Accepted)"] == "12"
    assert "different injuries" in out["WPI Basis"]


def test_rival_certificates_without_a_panel_are_still_averaged():
    out = w.resolve_wpi(_parsed([
        _m(14, "physical", assessor="MAS A"),
        _m(20, "physical", assessor="MAS B"),
    ]))
    assert out["Impairment % (Accepted)"] == "17"
    assert out["WPI Provenance"] == "inferred"


def test_a_panel_with_no_other_certificate_to_supersede_changes_nothing():
    out = w.resolve_wpi(_parsed([_m(9, "physical", assessor="Review Panel")]))
    assert out["Impairment % (Accepted)"] == "9"
    assert "supersedes" not in out["WPI Basis"]


# ----------------------------------------------------------------------
# §14.1 — a governing label implies both components were compared
# ----------------------------------------------------------------------

def test_the_split_columns_are_filled_from_the_resolution():
    """A governing-system label asserts a COMPARISON happened, so the two
    components cannot simultaneously be `absent`. That contradiction stood on
    8 rows; the components are now carried from the same reduction that named
    the winner."""
    row = ns.build_result_record("Case", "http://x/1", status="ok",
                                 **{"Decision Date": "2024-01-01"})
    row["Case Type"] = "CTP"
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(5, "physical", assessor="MAS Herald"),
        _m(16, "psychiatric", assessor="MAS Sidorov"),
    ]))
    assert row["WPI Physical %"] == "5"
    assert row["WPI Psychiatric %"] == "16"
    assert row["WPI Governing System"] == "psychiatric"
    assert row["WPI Physical % Provenance"] == "stated"
    assert "carried from the resolution" in row["WPI Resolution Notes"]


def test_a_captured_component_is_never_overwritten():
    row = ns.build_result_record("Case", "http://x/1", status="ok",
                                 **{"Decision Date": "2024-01-01"})
    row.update({"Case Type": "CTP", "WPI Physical %": "9",
                "WPI Physical % Provenance": "stated"})
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(5, "physical"), _m(16, "psychiatric", assessor="MAS S")]))
    assert row["WPI Physical %"] == "9"


def test_one_system_alone_does_not_fill_the_split_columns():
    """`WPI Physical %` is defined as populated only where the decision states
    physical and psychiatric SEPARATELY, so a single-system resolution must not
    write into it — and the governing system stays `not determined`."""
    row = ns.build_result_record("Case", "http://x/1", status="ok",
                                 **{"Decision Date": "2024-01-01"})
    row["Case Type"] = "CTP"
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(14, "physical", assessor="MAS A")]))
    assert row["WPI Physical %"] == ""
    assert row["WPI Governing System"] == "not determined"


# ----------------------------------------------------------------------
# §14.3 — Other Damages Heads
# ----------------------------------------------------------------------

def test_an_itemised_other_head_is_required_for_absent():
    """A residual localises a problem without identifying WHICH column is
    wrong, and on the real rows it usually indicts a different one. `absent`
    now needs an itemised other head visibly in the decision."""
    assert dx.has_itemised_other_head(
        "future economic loss of $180,000 plus future superannuation of $26,244")
    assert dx.has_itemised_other_head(
        "past superannuation of $11,000 and Fox v Wood of $728")
    # A residual explained by a DIFFERENT column's miss is not an other head.
    assert not dx.has_itemised_other_head(
        "past economic loss of $50,000 and future economic loss of $110,000, "
        "discounted 50% to $80,000")
    assert not dx.has_itemised_other_head(
        "A 30% reduction for contributory negligence was applied, yielding a "
        "payable sum of $160,000")
    assert not dx.has_itemised_other_head("", None)


def test_macdonald_stays_a_genuine_miss():
    """Future superannuation of $26,244, which `Future Economic Loss` excludes
    by definition — an other head that is itemised and was not captured."""
    row = {"Total Damages Gross": "745111.32",
           "Total Damages Gross Provenance": "stated",
           "Non-Economic Loss": "400000", "Non-Economic Loss Provenance": "stated",
           "Past Economic Loss": "138867.32", "Past Economic Loss Provenance": "stated",
           "Future Economic Loss": "180000", "Future Economic Loss Provenance": "stated",
           "Other Damages Heads": "", "Other Damages Heads Provenance": "absent",
           "Award Breakdown": ("future economic loss at $180,000 plus future "
                               "superannuation of $26,244, a subtotal of $745,111.32")}
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] == "absent"


def test_an_uncaptured_future_economic_loss_is_not_an_other_head():
    """"leaving $100,000 for future economic loss" — the residual is real but
    it indicts `Future Economic Loss`, and booking it against other heads would
    hide the actual defect."""
    row = {"Total Damages Gross": "315000",
           "Total Damages Gross Provenance": "stated",
           "Non-Economic Loss": "200000", "Non-Economic Loss Provenance": "stated",
           "Past Economic Loss": "15000", "Past Economic Loss Provenance": "stated",
           "Future Economic Loss": "", "Future Economic Loss Status": "Not addressed",
           "Future Economic Loss Provenance": "not_applicable",
           "Other Damages Heads": "", "Other Damages Heads Provenance": "absent",
           "Award Breakdown": ("Non-economic loss was $200,000 and past economic "
                               "loss was $15,000, leaving $100,000 for future "
                               "economic loss")}
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads Provenance"] == "not_stated"


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
