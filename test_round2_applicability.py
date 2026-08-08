"""
Round 2 (§10) — applicability semantics.

Every fixture is a real workbook row, named, because the round-2 request was
argued from specific rows and three of them turned out to mean the opposite of
what the row numbers suggested. Those three are pinned hardest.

    python test_round2_applicability.py     # or: pytest
"""

import damages_extraction as dx
import nsw_court_scraper as ns
import wpi_resolution as w


def _row(**over):
    row = ns.build_result_record("Case", "http://x/1",
                                 status="ok", **{"Decision Date": "2024-01-01"})
    row["Case Type"] = "CTP"
    row.update(over)
    return row


def _mention(value, system, kind="assessor total", **over):
    m = {"value": str(value), "body_system": system, "kind": kind,
         "assessor": "Dr A", "superseded": False, "about_claimant": True,
         "quote": ""}
    m.update(over)
    return m


# ----------------------------------------------------------------------
# §10.1 — the vocabulary itself
# ----------------------------------------------------------------------

def test_the_two_provenance_enums_stay_in_step():
    """`WpiProvenanceEnum` and `ProvenanceEnum` are parallel by design so
    neither module imports the other. If one gains a value, so must the other,
    or the same blank means different things in different columns."""
    assert {e.value for e in w.WpiProvenanceEnum} == \
           {e.value for e in dx.ProvenanceEnum}


def test_absent_is_the_only_defect_value():
    assert "absent" not in dx.PROVENANCE_NON_DEFECT
    for value in ("stated", "derived", "inferred",
                  "not_applicable", "not_assessed", "not_stated"):
        assert value in dx.PROVENANCE_NON_DEFECT, value


def test_no_psychiatric_injury_makes_psychiatric_wpi_not_applicable():
    """310 of the 448 blanks. Not a gap — there is nothing to assess."""
    assert w.classify_split_wpi_absence(
        system="psychiatric", has_psychiatric=False,
        total_present=True, mentions=[_mention(14, "physical")]) == "not_applicable"


def test_a_total_with_no_split_is_not_stated():
    """64 of the blanks: the impairment WAS assessed, the decision just gives
    one combined figure."""
    assert w.classify_split_wpi_absence(
        system="psychiatric", has_psychiatric=True,
        total_present=True, mentions=[]) == "not_stated"


def test_nothing_quantified_anywhere_is_not_assessed():
    """74 of the blanks: a psychiatric injury, but no MAS certificate."""
    assert w.classify_split_wpi_absence(
        system="psychiatric", has_psychiatric=True,
        total_present=False, mentions=[]) == "not_assessed"


def test_an_uncaptured_psychiatric_certificate_is_the_one_real_defect():
    """1 of the 448 — the row whose notes evidence a psychiatric certificate
    that governed, with no figure carried. This is what a data-quality check
    should fail on, and the only thing it should fail on."""
    assert w.classify_split_wpi_absence(
        system="psychiatric", has_psychiatric=True, total_present=True,
        mentions=[_mention(6, "psychiatric", "MAS certificate")]) == "absent"


def test_physical_is_not_a_defect_without_a_counterpart_to_split_from():
    """`WPI Physical %` is defined as populated ONLY where the decision states
    physical and psychiatric separately. Physical figures on a decision that
    never quantified psychiatric are behaving exactly as specified — reading
    them as defects put 91 rows in `absent`."""
    assert w.classify_split_wpi_absence(
        system="physical", has_psychiatric=False, total_present=True,
        mentions=[_mention(9, "physical")]) == "not_stated"
    # With both systems quantified, a missing physical half IS a miss.
    assert w.classify_split_wpi_absence(
        system="physical", has_psychiatric=True, total_present=True,
        mentions=[_mention(9, "physical"),
                  _mention(6, "psychiatric")]) == "absent"


def test_threshold_recitals_and_rejected_figures_are_not_evidence_of_a_miss():
    for kind in ("threshold recital", "claimed or rejected", "other"):
        assert w.classify_split_wpi_absence(
            system="psychiatric", has_psychiatric=True, total_present=False,
            mentions=[_mention(10, "psychiatric", kind)]) == "not_assessed", kind
    assert w.classify_split_wpi_absence(
        system="psychiatric", has_psychiatric=True, total_present=False,
        mentions=[_mention(6, "psychiatric", superseded=True)]) == "not_assessed"


def test_a_damages_head_never_put_in_issue_is_not_assessed():
    _a, prov, _s, _i = dx.apply_head_status("", "absent", "Not addressed")
    assert prov == "not_assessed"


# ----------------------------------------------------------------------
# §10.2 — the psychiatric gate
# ----------------------------------------------------------------------

def test_mason_the_psychiatric_percentage_is_a_brain_injury_component():
    """Allianz v Mason [2024] NSWPIC 348 (workbook row 214), the row the
    request cites as proof the FLAG is wrong. It is the other way round.

    Professor Cameron assessed a traumatic brain injury: mental status 0%,
    emotional and behavioural functioning 6%, left shoulder 7%, pelvis 0%,
    nasal 0% — and certified 13% overall. He could only reach 13 by COMBINING
    the 6 with the 7, and psychiatric impairment is never combined with
    physical under the Motor Accident Guidelines. So the 6% is a neurological
    component of the brain injury, not a psychiatric injury. `Injury
    Categories` agrees: brain injury | head or facial | lower limb | upper
    limb, with no psychiatric entry.

    The fix is to withdraw the misattributed percentage, NOT to flip the flag.
    """
    row = _row(**{
        "Has Psychiatric Injury": "No",
        "Injury Categories": "brain injury | head or facial | lower limb | upper limb",
        "Impairment % (Accepted)": "13",
        "WPI Physical %": "7",
        "WPI Psychiatric %": "6",
    })
    ns.apply_round2_semantics(row)
    assert row["Has Psychiatric Injury"] == "No"
    assert row["WPI Psychiatric %"] == ""
    assert row["WPI Psychiatric % Provenance"] == "not_applicable"
    assert "withdrawn" in row["WPI Resolution Notes"]
    # Once the misattribution is withdrawn only ONE component remains, so the
    # comparison cannot be made and the honest label is `not determined`
    # (round 5 §13). The 13% total is unaffected.
    assert row["WPI Governing System"] == "not determined"


def test_the_flag_yields_when_an_independent_signal_says_psychiatric():
    """The mirror image: where `Injury Categories` independently records a
    psychiatric injury, the lone `No` is the outlier and gets corrected."""
    row = _row(**{
        "Has Psychiatric Injury": "No",
        "Injury Categories": "spinal | psychiatric",
        "Impairment % (Accepted)": "14",
        "WPI Psychiatric %": "14",
    })
    ns.apply_round2_semantics(row)
    assert row["Has Psychiatric Injury"] == "Yes"
    assert row["WPI Psychiatric %"] == "14"
    assert "corrected to Yes" in row["WPI Resolution Notes"]


def test_taylor_substantial_emphasis_with_no_psychiatric_injury_is_normal():
    """NRMA v Taylor [2024] NSWPIC 301 (row 220), cited as "under any reading
    these should not co-occur". They can, and this is the reading the request
    itself proposes in §10.2(a).

    Ms Taylor's driving anxiety and lost confidence occupy much of the
    decision — hence emphasis 2 — but the Member expressly held "there was not
    a need to refer Ms Taylor for WPI assessment of her psychological
    symptoms". Nothing was diagnosed, assessed or accepted. Emphasis measures
    narrative weight; the flag records an accepted injury. No contradiction,
    and nothing to fix.
    """
    row = _row(**{
        "Has Psychiatric Injury": "No",
        "Psychological Injury Emphasis": "2",
        "Injury Categories": "chest or torso | spinal",
        "Impairment % (Accepted)": "0",
    })
    ns.apply_round2_semantics(row)
    assert row["Has Psychiatric Injury"] == "No"
    assert row["WPI Psychiatric % Provenance"] == "not_applicable"


def test_zero_means_assessed_at_zero_and_survives():
    """Washbourne (row 97): MAS Samuell diagnosed an adjustment disorder and
    assessed it at 0%. That is data, not a null — it must not be swept into an
    absence class."""
    row = _row(**{
        "Has Psychiatric Injury": "Yes",
        "Injury Categories": "psychiatric | upper limb",
        "WPI Psychiatric %": "0",
    })
    ns.apply_round2_semantics(row)
    assert row["WPI Psychiatric %"] == "0"
    assert row["WPI Psychiatric % Provenance"] == "stated"


# ----------------------------------------------------------------------
# §10.4 rule 1 — the governing body system
# ----------------------------------------------------------------------

def test_antonio_the_greater_system_governs():
    """Antonio [2026] NSWPIC 213 (row 22). A Review Panel certified 3% for the
    PHYSICAL injuries; MAS Sidorov separately certified 6% for Major
    Depressive Disorder. Both figures are right; the total took the lower."""
    assert w.governing_system("3", "6", "3") == "psychiatric"


def test_slyney_the_greater_system_governs():
    """Slyney [2024] NSWPIC 293 (row 222). Dr Home 3% physical, psychiatrist
    Dr George 5%."""
    assert w.governing_system("3", "5", "3") == "psychiatric"


def test_cowper_both_systems_assessed_at_zero():
    """Cowper [2025] NSWPIC 596 (row 67). Medical Assessors Fitzsimons and
    Jeyasingam each found 0% impairment. Both components stated, so the
    governing system is determinate even though the total is missing."""
    assert w.governing_system("0", "0", None) == "combined"

    # And the missing total is closed by the greater-governs rule. 0 here is an
    # assessment, not a null — "a finding of nil permanent impairment can still
    # indicate that the accident caused an injury", as the Member put it.
    row = _row(**{"Has Psychiatric Injury": "Yes",
                  "Injury Categories": "brain injury | psychiatric",
                  "WPI Physical %": "0", "WPI Psychiatric %": "0"})
    ns.apply_round2_semantics(row)
    assert row["Impairment % (Accepted)"] == "0"
    assert row["WPI Provenance"] == "derived"
    assert "greater governs" in row["WPI Basis"]


def test_filling_a_missing_total_never_overwrites_one_that_exists():
    """Antonio (row 22) keeps its 3.0 even though psychiatric 6% governs: the
    disagreement is REPORTED via `WPI Governing System`, not resolved by
    rewriting a captured value. Neither figure crosses 10, so the s 4.11
    answer is unaffected either way."""
    row = _row(**{"Has Psychiatric Injury": "Yes",
                  "Injury Categories": "upper limb | psychiatric",
                  "Impairment % (Accepted)": "3",
                  "WPI Physical %": "3", "WPI Psychiatric %": "6",
                  "WPI Threshold Finding": "not above 10%"})
    ns.apply_round2_semantics(row)
    assert row["Impairment % (Accepted)"] == "3"
    assert row["WPI Governing System"] == "psychiatric"
    assert row["WPI Threshold Finding"] == "not above 10%"


def test_governing_system_when_only_one_side_is_stated():
    """Superseded by round 5 §13. Naming the captured component made the
    column a tautology on 145 rows — "the system we happen to hold is the
    system that governs" — and it was demonstrably wrong wherever the missing
    component was the larger. Which system governs is a COMPARISON."""
    assert w.governing_system("9", "", None) == "not determined"
    assert w.governing_system("", "9", None) == "not determined"
    assert w.governing_system("", "", None) == "not stated"


def test_a_total_above_both_components_means_they_were_combined():
    assert w.governing_system("7", "6", "13") == "combined"


# ----------------------------------------------------------------------
# §10.4 rule 2 — non-economic loss against the threshold
# ----------------------------------------------------------------------

def test_nel_awarded_above_the_threshold_is_consistent():
    assert w.nel_threshold_consistency(
        nel_status="Awarded", threshold_finding="above 10%", wpi="14") == "yes"


def test_silcocks_the_disagreement_is_reported_not_resolved():
    """Silcocks [2023] NSWPIC 24 (row 407): $120,000 of non-economic loss on a
    recorded finding of `not above 10%`. Genuinely inconsistent with s 4.11 —
    the insurer paid what it did not owe — so it is reported as `no`. Nothing
    is edited on the strength of it."""
    assert w.nel_threshold_consistency(
        nel_status="Awarded", threshold_finding="not above 10%", wpi="9") == "no"


def test_no_award_means_the_rule_has_nothing_to_say():
    for status in ("Nil", "Not addressed", ""):
        assert w.nel_threshold_consistency(
            nel_status=status, threshold_finding="not above 10%",
            wpi="9") == "cannot determine", status


def test_the_finding_outranks_the_percentage():
    """Physical and psychiatric are assessed separately, so a `WPI %` holding
    one body system cannot be compared to 10 directly. An explicit finding wins."""
    assert w.nel_threshold_consistency(
        nel_status="Awarded", threshold_finding="above 10%", wpi="6") == "yes"
    assert w.nel_threshold_consistency(
        nel_status="Awarded", threshold_finding="", wpi="6") == "no"
    assert w.nel_threshold_consistency(
        nel_status="Awarded", threshold_finding="", wpi="") == "cannot determine"


# ----------------------------------------------------------------------
# §10.3 — threshold coverage
# ----------------------------------------------------------------------

def test_an_award_of_non_economic_loss_implies_the_threshold():
    finding, basis = w.derive_threshold_finding(nel_status="Awarded", wpi="")
    assert (finding, basis) == ("above 10%", "implied by non-economic loss award")


def test_the_ex_gratia_row_is_not_used_to_imply_a_threshold():
    """Silcocks again: the one lawful way non-economic loss is paid below the
    threshold, so the award implies nothing about impairment."""
    finding, basis = w.derive_threshold_finding(
        nel_status="Awarded", wpi="", ex_gratia=True)
    assert finding == "not determined"


def test_a_stated_wpi_implies_the_threshold_either_way():
    assert w.derive_threshold_finding(nel_status="Nil", wpi="14")[0] == "above 10%"
    assert w.derive_threshold_finding(nel_status="Nil", wpi="10")[0] == "not above 10%"


def test_nothing_is_deduced_from_a_refused_head_alone():
    """A head can be refused for many reasons besides the threshold."""
    finding, basis = w.derive_threshold_finding(nel_status="Nil", wpi="")
    assert (finding, basis) == ("not determined", "not determined")


def test_a_decision_finding_is_never_overwritten_by_a_deduction():
    row = _row(**{"WPI Threshold Finding": "not above 10%",
                  "Non-Economic Loss Status": "Awarded",
                  "Non-Economic Loss": "120000",
                  "Impairment % (Accepted)": "9"})
    ns.apply_round2_semantics(row)
    assert row["WPI Threshold Finding"] == "not above 10%"
    assert row["WPI Threshold Finding Basis"] == "decision"
    # And the disagreement is still visible rather than deduced away.
    assert row["NEL Threshold Consistent"] == "no"


def test_consistency_is_judged_before_the_gap_is_filled():
    """Otherwise deducing `above 10%` from the award and then checking the
    award against it would make every row agree with itself."""
    row = _row(**{"Non-Economic Loss Status": "Awarded",
                  "Non-Economic Loss": "200000",
                  "Impairment % (Accepted)": ""})
    ns.apply_round2_semantics(row)
    assert row["WPI Threshold Finding"] == "above 10%"
    assert row["WPI Threshold Finding Basis"] == "implied by non-economic loss award"
    assert row["NEL Threshold Consistent"] == "cannot determine"


def test_every_round2_column_is_in_result_fields():
    for field in ("WPI Governing System", "NEL Threshold Consistent",
                  "WPI Threshold Finding Basis"):
        assert field in ns.RESULT_FIELDS, field


def test_the_pass_is_idempotent():
    row = _row(**{"Has Psychiatric Injury": "No",
                  "Injury Categories": "brain injury",
                  "Impairment % (Accepted)": "13",
                  "WPI Physical %": "7", "WPI Psychiatric %": "6"})
    ns.apply_round2_semantics(row)
    first = dict(row)
    ns.apply_round2_semantics(row)
    for key in ("WPI Psychiatric %", "WPI Psychiatric % Provenance",
                "WPI Governing System", "NEL Threshold Consistent",
                "WPI Threshold Finding", "WPI Threshold Finding Basis",
                "Has Psychiatric Injury"):
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
