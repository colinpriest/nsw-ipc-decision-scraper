"""
Round 7 (§15) — the last `absent` values.

The target is `absent = 0`: every remaining blank carries a reason that is a
fact about the source rather than a defect in the extraction.

    python test_round7_last_absent.py     # or: pytest
"""

import damages_extraction as dx
import nsw_court_scraper as ns


def _row(**over):
    row = ns.build_result_record("Case", "http://x/1", status="ok",
                                 **{"Decision Date": "2024-01-01"})
    row.update({"Case Type": "CTP", "Damages Extraction Status": "ok"})
    row.update(over)
    return row


def _mention(value, system, kind="MAS certificate", **over):
    m = {"value": str(value), "body_system": system, "kind": kind,
         "assessor": "MAS A", "superseded": False, "about_claimant": True,
         "quote": ""}
    m.update(over)
    return m


# ----------------------------------------------------------------------
# §15.1 — a quarantine is a decision, not a failure
# ----------------------------------------------------------------------

def test_a_quarantined_wpi_is_not_stated_not_absent():
    """Washbourne (row 97). The 8% was DELIBERATELY not used — it is one
    shoulder, not the governing total. Recording that as `absent` asserted the
    opposite: that the figure was recoverable and we failed to capture it. The
    decision assessed impairment and never states the governing total, which is
    exactly `not_stated`."""
    row = _row(**{"Non-Economic Loss Status": "Awarded",
                  "Non-Economic Loss": "383000",
                  "Impairment % (Accepted)": "8"})
    assert ns.quarantine_impossible_wpi(row) is True
    assert row["Impairment % (Accepted)"] == ""
    assert row["WPI Provenance"] == "not_stated"
    assert row["WPI Candidates"] == "8"          # still auditable
    assert row["_wpi_quarantined"] == "8"


def test_a_zero_psychiatric_cannot_govern_a_total_above_ten():
    """Row 97's other contradiction: `governing = psychiatric` beside a
    psychiatric figure of 0.0 and a threshold of `above 10%`. With no physical
    component to compare against, the honest label is `not determined`."""
    row = _row(**{"Impairment % (Accepted)": "", "WPI Psychiatric %": "0",
                  "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "psychiatric | upper limb",
                  "WPI Governing System": "psychiatric",   # stale
                  "WPI Threshold Finding": "above 10%"})
    ns.apply_round2_semantics(row)
    assert row["WPI Governing System"] == "not determined"


# ----------------------------------------------------------------------
# §15.2 — the quarantined rows against their comparator
# ----------------------------------------------------------------------

def test_a_stated_component_makes_a_blank_total_not_stated():
    """Row 500 (Judges) carries a physical 18% and read `not_assessed`, which
    denied an assessment the row itself records. A stated component IS an
    assessment; what is missing is the total."""
    row = _row(**{"Impairment % (Accepted)": "", "WPI Physical %": "18",
                  "WPI Provenance": "absent"})
    ns.apply_round2_semantics(row)
    assert row["WPI Provenance"] == "not_stated"


def test_nothing_assessed_anywhere_is_still_not_assessed():
    """The distinction has to survive: `not_assessed` means no impairment
    figure exists anywhere on the row."""
    row = _row(**{"Impairment % (Accepted)": "", "WPI Provenance": "absent"})
    ns.apply_round2_semantics(row)
    assert row["WPI Provenance"] == "not_assessed"


# ----------------------------------------------------------------------
# §15.3 — the psychiatric contract differs from the physical one
# ----------------------------------------------------------------------

def test_psychiatric_fills_from_a_lone_resolution_but_physical_does_not():
    """The two columns are specified differently, and round 7 turned on it.
    `WPI Psychiatric %` is "only if separately stated", which a psychiatric
    assessment satisfies alone. `WPI Physical %` is "only if the decision
    states physical and psychiatric SEPARATELY", so it needs both. Requiring
    both for each left 4 rows `absent` with their psychiatric figure resolved
    and unused."""
    from types import SimpleNamespace
    parsed = SimpleNamespace(
        mentions=[SimpleNamespace(value="4", kind="MAS certificate",
                                  body_system="psychiatric", assessor="MAS Fukui",
                                  superseded=False, about_claimant=True, quote="")],
        tribunal_selected_value="", tribunal_selected_quote="",
        components_share_one_assessment=True, totals_are_rival_assessments=True,
        threshold_finding="not above 10%", settlement_approval_without_wpi=False,
        notes="")
    row = _row()
    ns.merge_wpi_resolution_into_record(row, parsed)
    assert row["WPI Psychiatric %"] == "4"
    assert row["WPI Physical %"] == ""          # no counterpart to split from


# ----------------------------------------------------------------------
# §15.4 — capturing the other-head figures
# ----------------------------------------------------------------------

def _apportioned(gross, **over):
    row = _row(**{"Total Damages Gross": str(gross),
                  "Total Damages Gross Provenance": "stated",
                  "Non-Economic Loss": "400000", "Non-Economic Loss Provenance": "stated",
                  "Past Economic Loss": "138867.32", "Past Economic Loss Provenance": "stated",
                  "Future Economic Loss": "180000", "Future Economic Loss Provenance": "stated"})
    row.update(over)
    return row


def test_a_positive_itemised_residual_is_captured():
    """Macdonald: future superannuation of $26,244, which `Future Economic
    Loss` excludes by definition. The identity gives the amount and the
    breakdown names the head, so there is nothing left to flag."""
    row = _apportioned(745111.32, **{
        "Award Breakdown": ("future economic loss at $180,000 plus future "
                            "superannuation of $26,244")})
    residual, trustworthy = dx.damages_residual(row)
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads"] == "26244"
    assert row["Other Damages Heads Provenance"] == "derived"
    assert row["Other Damages Heads Status"] == "Awarded"


def test_a_negative_residual_is_never_captured():
    """Row 155: past superannuation and Fox v Wood are folded INTO past
    economic loss, so the named heads overshoot the gross. The composition is
    ambiguous, not the amount — there is no separate figure to state, and a
    negative other-heads total would be nonsense."""
    row = _apportioned(700000, **{
        "Award Breakdown": ("past economic loss of $100,000, past superannuation "
                            "of $11,000 and Fox v Wood of $728")})
    residual, trustworthy = dx.damages_residual(row)
    assert residual < 0
    dx.refine_money_absence(row, residual=residual, residual_trustworthy=trustworthy)
    assert row["Other Damages Heads"] == ""
    assert row["Other Damages Heads Provenance"] == "not_stated"


# ----------------------------------------------------------------------
# §15.5 / §15.6 — the target, and what must not regress
# ----------------------------------------------------------------------

def test_absent_remains_reachable_for_a_genuine_defect():
    """`absent = 0` must be an outcome, not a vocabulary change. The value has
    to stay reachable, or the check it feeds means nothing."""
    assert dx.classify_money_absence("Buffer Amount", corroborated=True) == "absent"
    assert "absent" not in dx.PROVENANCE_NON_DEFECT


def test_the_pass_is_idempotent():
    row = _row(**{"Impairment % (Accepted)": "", "WPI Physical %": "18",
                  "WPI Provenance": "absent", "WPI Governing System": "physical"})
    ns.apply_round2_semantics(row)
    first = dict(row)
    ns.apply_round2_semantics(row)
    for key in ("WPI Provenance", "WPI Governing System", "WPI Physical %",
                "WPI Psychiatric %", "WPI Threshold Finding"):
        assert row[key] == first[key], key


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
