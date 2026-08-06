"""
Golden-fixture tests for WPI resolution — the deterministic half of the pass.

The LLM classifies each mention; everything below decides what to DO with the
classification, so it is all testable without the API: the AMA Combined Values
arithmetic, the resolution ladder, physical-vs-psychiatric handling, provenance
honesty, and the cases that must stay blank.

Fixtures are built from real corpus cases (named in each test).

    python test_wpi_resolution.py     # or: pytest test_wpi_resolution.py
"""

from types import SimpleNamespace

import nsw_court_scraper as ns
import wpi_resolution as w


def _m(value, kind, *, system="physical", assessor="Dr A",
       superseded=False, about_claimant=True, quote=""):
    return SimpleNamespace(value=str(value), kind=kind, body_system=system,
                           assessor=assessor, superseded=superseded,
                           about_claimant=about_claimant, quote=quote)


def _parsed(mentions, *, selected="", share_one=True, settlement=False, notes="",
            rival=True, threshold="not determined"):
    return SimpleNamespace(
        mentions=mentions, tribunal_selected_value=selected,
        tribunal_selected_quote="", components_share_one_assessment=share_one,
        totals_are_rival_assessments=rival, threshold_finding=threshold,
        settlement_approval_without_wpi=settlement, notes=notes)


# ----------------------------------------------------------------------
# AMA Combined Values arithmetic
# ----------------------------------------------------------------------

def test_combined_values_reproduces_the_assessors_own_total():
    """Vanzanella [2022] NSWPIC 524: neck 5%, back 5%, ankle 3%+3%+1%, which
    the assessor himself combined to 16%. Plain addition gives 17 — which is
    exactly the figure the Member queried in the decision."""
    assert w.combine_wpi([5, 5, 3, 3, 1]) == 16
    assert sum([5, 5, 3, 3, 1]) == 17


def test_components_combine_they_do_not_add():
    # Each impairment applies to the capacity remaining after the previous one.
    assert w.combine_wpi([20] * 5) == 67          # sum would be 100
    assert w.combine_wpi([50, 50]) == 75          # sum would be 100
    assert w.combine_wpi([10]) == 10
    assert w.combine_wpi([]) is None
    assert w.combine_wpi([0, 0]) is None


def test_combined_values_is_order_independent():
    assert w.combine_wpi([3, 5, 1, 5, 3]) == w.combine_wpi([5, 5, 3, 3, 1])


def test_central_estimate():
    assert w.central_estimate([14, 19]) == 16.5      # median == mean for two
    assert w.central_estimate([6, 11, 22]) == 11.0   # median resists the outlier
    assert w.central_estimate([]) is None


def test_to_pct_rejects_implausible_values():
    assert w.to_pct("16%") == 16.0
    assert w.to_pct("7.5") == 7.5
    assert w.to_pct("120") is None
    assert w.to_pct("") is None


# ----------------------------------------------------------------------
# Ladder rung 1 — the tribunal chose
# ----------------------------------------------------------------------

def test_tribunal_selection_wins_over_everything():
    out = w.resolve_wpi(_parsed([
        _m(14, "assessor total", assessor="Dr Doig"),
        _m(19, "assessor total", assessor="Dr Bodel"),
    ], selected="19"))
    assert out["Impairment % (Accepted)"] == "19"
    assert out["WPI Provenance"] == "stated"
    assert out["WPI Basis"] == "tribunal selected"


def test_threshold_recitals_never_become_the_value():
    """The s 4.11 statutory bar is not a finding about the claimant."""
    out = w.resolve_wpi(_parsed([
        _m(10, "threshold recital", about_claimant=False),
        _m(6, "assessor total"),
    ]))
    assert out["Impairment % (Accepted)"] == "6"


def test_rejected_and_superseded_figures_are_excluded():
    out = w.resolve_wpi(_parsed([
        _m(30, "claimed or rejected"),
        _m(15, "assessor total", superseded=True),   # pre-deduction
        _m(10, "assessor total"),
    ]))
    assert out["Impairment % (Accepted)"] == "10"
    assert out["WPI Provenance"] == "stated"


# ----------------------------------------------------------------------
# Ladder rung 2/3 — totals and components
# ----------------------------------------------------------------------

def test_a_stated_total_is_used_verbatim():
    """Javed [2024]: 'a WPI of 5%, giving a total WPI of 11%'."""
    out = w.resolve_wpi(_parsed([
        _m(5, "component"),
        _m(11, "assessor total"),
    ]))
    assert out["Impairment % (Accepted)"] == "11"
    assert out["WPI Provenance"] == "stated"


def test_components_without_a_total_are_combined_and_marked_derived():
    """Seaman: 3% foot/ankle, 2% wrist, 2% scarring, no total stated."""
    out = w.resolve_wpi(_parsed([
        _m(3, "component"), _m(2, "component"), _m(2, "component"),
    ]))
    assert out["Impairment % (Accepted)"] == "7"
    assert out["WPI Provenance"] == "derived"
    assert "Combined Values" in out["WPI Basis"]


def test_mas_certificate_outranks_a_party_report():
    out = w.resolve_wpi(_parsed([
        _m(19, "assessor total", assessor="Dr Bodel"),
        _m(11, "MAS certificate", assessor="MAS McGlynn"),
    ]))
    assert out["Impairment % (Accepted)"] == "11"
    assert out["WPI Basis"].endswith("MAS certificate")


# ----------------------------------------------------------------------
# Ladder rung 4 — competing assessments
# ----------------------------------------------------------------------

def test_competing_assessments_use_a_central_estimate_marked_inferred():
    """Ladhani: Dr Doig 14%, Dr Bodel 19%, no selection by the Member.
    A central estimate beats a blank, but must never masquerade as stated."""
    out = w.resolve_wpi(_parsed([
        _m(14, "assessor total", assessor="Dr Doig"),
        _m(19, "assessor total", assessor="Dr Bodel"),
    ]))
    assert out["Impairment % (Accepted)"] == "16.5"
    assert out["WPI Provenance"] == "inferred"
    assert "competing" in out["WPI Basis"]


def test_rival_component_sets_are_combined_per_assessor_then_averaged():
    """Components from different doctors must not be pooled into one total."""
    out = w.resolve_wpi(_parsed([
        _m(5, "component", assessor="Dr A"), _m(5, "component", assessor="Dr A"),
        _m(10, "component", assessor="Dr B"), _m(10, "component", assessor="Dr B"),
    ], share_one=False))
    # Dr A -> 10, Dr B -> 19; median 14.5
    assert out["Impairment % (Accepted)"] == "14.5"
    assert out["WPI Provenance"] == "inferred"


# ----------------------------------------------------------------------
# Physical vs psychiatric
# ----------------------------------------------------------------------

def test_physical_and_psychiatric_are_never_combined_with_each_other():
    """MAIA assesses them separately; the higher governs. Measured on the
    corpus: accepted WPI == max(physical, psychiatric) on 88% of rows that
    state both, == their sum on 23%."""
    out = w.resolve_wpi(_parsed([
        _m(4, "assessor total", system="physical", assessor="MAS Menogue"),
        _m(6, "assessor total", system="psychiatric", assessor="MAS Shen"),
    ]))
    assert out["Impairment % (Accepted)"] == "6"     # not 10, not 9.8
    assert "higher of 2 body systems" in out["WPI Basis"]


def test_components_combine_within_a_system_before_systems_are_compared():
    out = w.resolve_wpi(_parsed([
        _m(5, "component", system="physical"),
        _m(5, "component", system="physical"),
        _m(8, "assessor total", system="psychiatric", assessor="Dr P"),
    ]))
    # physical combines to 10, psychiatric is 8 -> physical governs
    assert out["Impairment % (Accepted)"] == "10"
    assert out["WPI Provenance"] == "derived"


# ----------------------------------------------------------------------
# Genuine zero, and the cases that stay blank
# ----------------------------------------------------------------------

def test_a_genuine_zero_assessment_is_kept():
    """Ulkin/Fuchs: 'He assessed 0% permanent impairment' is a real finding of
    no permanent impairment, not statutory-threshold framing."""
    out = w.resolve_wpi(_parsed([_m(0, "assessor total", quote="assessed 0% WPI")]))
    assert out["Impairment % (Accepted)"] == "0"
    assert out["WPI Provenance"] == "stated"


def test_settlement_approval_quoting_no_wpi_stays_blank():
    out = w.resolve_wpi(_parsed([
        _m(10, "threshold recital", about_claimant=False),
    ], settlement=True))
    assert out["Impairment % (Accepted)"] == ""
    assert out["WPI Provenance"] == "absent"
    assert "settlement approval" in out["WPI Basis"]


def test_no_wpi_anywhere_stays_blank():
    out = w.resolve_wpi(_parsed([]))
    assert out["Impairment % (Accepted)"] == ""
    assert out["WPI Provenance"] == "absent"


def test_candidates_column_records_everything_considered():
    out = w.resolve_wpi(_parsed([
        _m(10, "threshold recital", about_claimant=False),
        _m(5, "component"), _m(11, "assessor total"),
    ]))
    assert out["WPI Candidates"] == "5 | 10 | 11"


# ----------------------------------------------------------------------
# Integration with the flat record
# ----------------------------------------------------------------------

def test_wpi_fields_are_in_result_fields():
    for field in w.WPI_FIELDS:
        assert field in ns.RESULT_FIELDS, field


def test_merge_never_blanks_a_wpi_the_main_pass_captured():
    row = ns.build_result_record("Case", "http://example/1")
    row["Impairment % (Accepted)"] = "14"
    ns.merge_wpi_resolution_into_record(row, _parsed([], settlement=True))
    assert row["Impairment % (Accepted)"] == "14"
    assert row["WPI Provenance"] == "stated"
    assert row["WPI Basis"] == "retained from main extraction"


def test_merge_fills_and_records_provenance():
    row = ns.build_result_record("Case", "http://example/1")
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(3, "component"), _m(2, "component"), _m(2, "component")]))
    assert row["Impairment % (Accepted)"] == "7"
    assert row["WPI Provenance"] == "derived"
    assert row["_wpi_version"] == ns.WPI_VERSION
    assert len(row["_wpi_resolution"]["mentions"]) == 3


def test_blank_record_defaults():
    row = ns.build_result_record("Case", "http://example/1")
    assert row["WPI Provenance"] == "absent"
    assert row["WPI Basis"] == ""


def test_gate_skips_rows_with_nothing_to_resolve():
    row = ns.build_result_record("Case", "http://example/1")
    row["Impairment % (Accepted)"] = "14"
    # A single figure already captured: no call needed.
    assert ns.wpi_resolution_applies(row, "assessed at 14% whole person impairment") is False
    # No figure anywhere: nothing to reason about.
    assert ns.wpi_resolution_applies(row, "no impairment figures here") is False
    # Several figures: the regex refuses to choose, so the pass earns its call.
    assert ns.wpi_resolution_applies(
        row, "5% whole person impairment for the neck and 11% WPI in total") is True


# ----------------------------------------------------------------------
# Rival vs combining certificates, and the threshold constraint
# ----------------------------------------------------------------------

def test_certificates_covering_different_injuries_are_combined_not_averaged():
    """Quigley: MAS Curtin certified 4% (scarring, left-leg nerve injury) and a
    Review Panel certified 8% (brain injury, shoulder). Different injuries, so
    they combine to 12 — averaging them to 6 would be a category error."""
    out = w.resolve_wpi(_parsed([
        _m(4, "MAS certificate", assessor="MAS Curtin"),
        _m(8, "MAS certificate", assessor="Review Panel"),
    ], rival=False))
    assert out["Impairment % (Accepted)"] == "12"
    assert out["WPI Provenance"] == "derived"
    assert "different injuries" in out["WPI Basis"]


def test_rival_certificates_of_the_same_injury_are_averaged():
    out = w.resolve_wpi(_parsed([
        _m(14, "MAS certificate", assessor="MAS A"),
        _m(19, "MAS certificate", assessor="MAS B"),
    ], rival=True))
    assert out["Impairment % (Accepted)"] == "16.5"
    assert out["WPI Provenance"] == "inferred"


def test_estimate_contradicting_the_threshold_is_never_published_as_is():
    """Sengsavang: Dr Keller 0%, Dr Curtin 19%, insurer CONCEDED >10%. The
    median 9.5 is refuted by the document, so it is never published — see
    test_average_below_threshold_is_repaired_using_the_assessments_above_it
    for what replaces it."""
    out = w.resolve_wpi(_parsed([
        _m(0, "assessor total", assessor="Dr Keller"),
        _m(19, "assessor total", assessor="Dr Curtin"),
    ], rival=True, threshold="above 10%"))
    assert out["Impairment % (Accepted)"] != "9.5"


def test_threshold_constraint_vetoes_even_a_stated_value():
    """CXA: the ladder resolved a quoted 'total of 9% WPI' on a decision that
    awarded non-economic loss — which s 4.11 only permits above 10%. A stated
    figure contradicting the threshold means we picked the WRONG quoted figure,
    so it is withheld rather than published."""
    out = w.resolve_wpi(_parsed([
        _m(9, "MAS certificate", assessor="MAS A"),
    ], threshold="above 10%"))
    assert out["Impairment % (Accepted)"] == ""
    assert out["WPI Provenance"] == "absent"
    assert "contradicts" in out["WPI Basis"]


def test_threshold_constraint_passes_a_consistent_value():
    out = w.resolve_wpi(_parsed([
        _m(14, "MAS certificate", assessor="MAS A"),
    ], threshold="above 10%"))
    assert out["Impairment % (Accepted)"] == "14"
    assert out["WPI Provenance"] == "stated"


def test_threshold_finding_is_carried_as_a_column():
    out = w.resolve_wpi(_parsed([_m(6, "assessor total")], threshold="not above 10%"))
    assert out["WPI Threshold Finding"] == "not above 10%"
    out = w.resolve_wpi(_parsed([]))
    assert out["WPI Threshold Finding"] == "not determined"


# ----------------------------------------------------------------------
# The overwrite guard
# ----------------------------------------------------------------------

def test_an_inferred_estimate_never_displaces_a_captured_value():
    """A median of rival reports is a fallback for an EMPTY field, not an
    improvement on a figure the main extraction already captured."""
    row = ns.build_result_record("Case", "http://example/1")
    row["Impairment % (Accepted)"] = "19"
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(0, "assessor total", assessor="Dr Keller"),
        _m(19, "assessor total", assessor="Dr Curtin"),
    ], rival=True))
    assert row["Impairment % (Accepted)"] == "19"
    assert row["WPI Provenance"] == "stated"
    assert "kept extracted 19" in row["WPI Resolution Notes"]


def test_a_derived_value_never_displaces_a_captured_value():
    row = ns.build_result_record("Case", "http://example/1")
    row["Impairment % (Accepted)"] = "12"
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(3, "component"), _m(2, "component")], share_one=True))
    assert row["Impairment % (Accepted)"] == "12"


def test_no_resolution_ever_overwrites_a_captured_value():
    """FILL ONLY, whatever the provenance. An audit of the corrections this
    pass proposed found a third of them wrong: Transport Accident Commission v
    George dropped 25% to 10% because the insurer's CONCESSION of 25% was
    classified as a rejected claim, and Obeid preferred one doctor's 0% over
    another's assessment. The pass sees classified mentions; the main
    extraction read the Member's reasoning. It does not get to relitigate."""
    for mentions in (
        [_m(10, "assessor total", assessor="Dr Economos"),
         _m(25, "claimed or rejected", assessor="the insurer")],
        [_m(3, "MAS certificate", system="physical"),
         _m(6, "MAS certificate", system="psychiatric", assessor="MAS Shen")],
    ):
        row = ns.build_result_record("Case", "http://example/1")
        row["Impairment % (Accepted)"] = "25"
        ns.merge_wpi_resolution_into_record(row, _parsed(mentions))
        assert row["Impairment % (Accepted)"] == "25"
        assert row["WPI Basis"] == "retained from main extraction"


def test_empty_field_takes_the_estimate():
    row = ns.build_result_record("Case", "http://example/1")
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(14, "assessor total", assessor="Dr A"),
        _m(19, "assessor total", assessor="Dr B"),
    ], rival=True))
    assert row["Impairment % (Accepted)"] == "16.5"
    assert row["WPI Provenance"] == "inferred"


def test_an_itemising_assessor_is_not_discarded_for_one_who_stated_a_total():
    """Seaman: Dr Bentivoglio itemised 3% + 2% + 2% (no total); Dr Wallace
    stated 2%. Preferring 'any stated total' threw Bentivoglio away entirely.
    Each assessor is reduced to one figure FIRST, then they are compared."""
    out = w.resolve_wpi(_parsed([
        _m(3, "component", assessor="Dr Bentivoglio"),
        _m(2, "component", assessor="Dr Bentivoglio"),
        _m(2, "component", assessor="Dr Bentivoglio"),
        _m(2, "assessor total", assessor="Dr Wallace"),
    ], share_one=False, rival=True))
    # Bentivoglio combines to 7, Wallace states 2 -> median 4.5
    assert out["Impairment % (Accepted)"] == "4.5"
    assert out["WPI Provenance"] == "inferred"


def test_one_assessor_itemising_still_combines_cleanly():
    out = w.resolve_wpi(_parsed([
        _m(3, "component", assessor="Dr B"), _m(2, "component", assessor="Dr B"),
        _m(2, "component", assessor="Dr B"),
    ], share_one=True))
    assert out["Impairment % (Accepted)"] == "7"
    assert out["WPI Provenance"] == "derived"


def test_agreeing_assessors_are_not_collapsed_into_one_vote():
    """Vanzanella: Dr Conrad 16%, Dr Dryson 16%, Dr Ugwu (ankle only) 6%.
    Deduplicating the values first makes the median 11; one vote per ASSESSOR
    makes it 16, which is what both agreeing assessors actually said."""
    out = w.resolve_wpi(_parsed([
        _m(16, "assessor total", assessor="Dr Conrad"),
        _m(16, "assessor total", assessor="Dr Dryson"),
        _m(6, "component", assessor="Dr Ugwu"),
        _m(17, "claimed or rejected", assessor="the Member"),
        _m(10, "threshold recital", about_claimant=False),
    ], share_one=False, rival=True))
    assert out["Impairment % (Accepted)"] == "16"


# ----------------------------------------------------------------------
# Threshold repair: averaging only the assessments the threshold allows
# ----------------------------------------------------------------------

def test_average_below_threshold_is_repaired_using_the_assessments_above_it():
    """Sengsavang: Dr Keller 0%, Dr Curtin 19%, insurer conceded >10%. The
    median 9.5 is impossible, so the assessments the threshold rules out are
    dropped and what remains is averaged -> 19."""
    out = w.resolve_wpi(_parsed([
        _m(0, "assessor total", assessor="Dr Keller"),
        _m(19, "assessor total", assessor="Dr Curtin"),
    ], rival=True, threshold="above 10%"))
    assert out["Impairment % (Accepted)"] == "19"
    assert out["WPI Provenance"] == "inferred"
    assert "above the 10% threshold" in out["WPI Basis"]


def test_an_awarded_nel_establishes_the_threshold_by_itself():
    """s 4.11 permits non-economic loss ONLY above 10%, so an award is harder
    evidence than the model's own reading."""
    out = w.resolve_wpi(_parsed([
        _m(4, "assessor total", assessor="Dr A"),
        _m(14, "assessor total", assessor="Dr B"),
    ], rival=True, threshold="not determined"), nel_status="Awarded")
    assert out["Impairment % (Accepted)"] == "14"   # median 9 is impossible
    assert out["WPI Provenance"] == "inferred"


def test_future_economic_loss_is_NOT_evidence_of_the_threshold():
    """27 of the 33 decisions assessing 0% WPI still awarded future economic
    loss — economic loss carries no impairment threshold, so it must never
    raise the WPI."""
    out = w.resolve_wpi(_parsed([
        _m(0, "assessor total", assessor="Dr A"),
        _m(8, "assessor total", assessor="Dr B"),
    ], rival=True, threshold="not determined"), nel_status="Not addressed")
    assert out["Impairment % (Accepted)"] == "4"      # plain median, no uplift


def test_repair_withholds_when_no_assessment_clears_the_threshold():
    out = w.resolve_wpi(_parsed([
        _m(6, "assessor total", assessor="Dr A"),
        _m(8, "assessor total", assessor="Dr B"),
    ], rival=True, threshold="above 10%"))
    assert out["Impairment % (Accepted)"] == ""
    assert "no assessment exceeds it" in out["WPI Basis"]


def test_repair_works_in_the_other_direction_too():
    out = w.resolve_wpi(_parsed([
        _m(8, "assessor total", assessor="Dr A"),
        _m(30, "assessor total", assessor="Dr B"),
    ], rival=True, threshold="not above 10%"))
    assert out["Impairment % (Accepted)"] == "8"
    assert "at or below the 10% threshold" in out["WPI Basis"]


def test_nel_awarded_with_wpi_at_or_below_ten_is_flagged_impossible():
    row = ns.build_result_record("Case", "http://x/1")
    row.update({"Non-Economic Loss Status": "Awarded", "Impairment % (Accepted)": "8"})
    assert ns.wpi_is_legally_impossible(row) is True
    row["Impairment % (Accepted)"] = "14"
    assert ns.wpi_is_legally_impossible(row) is False
    row.update({"Non-Economic Loss Status": "Nil", "Impairment % (Accepted)": "8"})
    assert ns.wpi_is_legally_impossible(row) is False


def test_a_legally_impossible_value_is_the_one_exception_to_fill_only():
    row = ns.build_result_record("Case", "http://x/1")
    row.update({"Non-Economic Loss Status": "Awarded", "Impairment % (Accepted)": "8"})
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(8, "component", assessor="Dr A"),
        _m(18, "assessor total", assessor="Dr A"),
    ]))
    assert row["Impairment % (Accepted)"] == "18"


def test_the_exception_does_not_fire_when_the_ladder_agrees_it_is_low():
    """If the ladder also lands at or below 10, the captured value stands —
    we do not manufacture a number to satisfy the statute."""
    row = ns.build_result_record("Case", "http://x/1")
    row.update({"Non-Economic Loss Status": "Awarded", "Impairment % (Accepted)": "8"})
    ns.merge_wpi_resolution_into_record(row, _parsed([
        _m(8, "assessor total", assessor="Dr A")]))
    assert row["Impairment % (Accepted)"] == "8"


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
