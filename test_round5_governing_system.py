"""
Round 5 (§13) — the governing-system column, and recovering the physical split.

    python test_round5_governing_system.py     # or: pytest
"""

import nsw_court_scraper as ns
import wpi_resolution as w


def _row(**over):
    row = ns.build_result_record("Case", "http://x/1",
                                 status="ok", **{"Decision Date": "2024-01-01"})
    row["Case Type"] = "CTP"
    row.update(over)
    return row


def _m(value, system, kind="assessor total", assessor="Dr A", **over):
    from types import SimpleNamespace
    m = dict(value=str(value), kind=kind, body_system=system, assessor=assessor,
             superseded=False, about_claimant=True, quote="")
    m.update(over)
    return SimpleNamespace(**m)


def _parsed(mentions, **over):
    from types import SimpleNamespace
    base = dict(mentions=mentions, tribunal_selected_value="",
                tribunal_selected_quote="", components_share_one_assessment=True,
                totals_are_rival_assessments=True, threshold_finding="not determined",
                settlement_approval_without_wpi=False, notes="")
    base.update(over)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# §13 — the column must not name the survivor
# ----------------------------------------------------------------------

def test_one_component_cannot_answer_which_system_governs():
    """The tautology: with a single component captured the old derivation could
    only ever return that component, so the column read "the system we happen
    to hold is the system that governs" on 145 rows."""
    assert w.governing_system("9", "", None) == "not determined"
    assert w.governing_system("", "9", None) == "not determined"


def test_neither_component_is_distinct_from_one_component():
    """`not stated` means nothing was quantified; `not determined` means one
    system was and the comparison still cannot be made. Collapsing them would
    lose the distinction the column exists for."""
    assert w.governing_system("", "", None) == "not stated"
    assert w.governing_system("9", "", None) == "not determined"


def test_both_components_still_compare_normally():
    assert w.governing_system("3", "6", "3") == "psychiatric"
    assert w.governing_system("15", "5", "15") == "physical"
    assert w.governing_system("7", "6", "13") == "combined"


def test_the_resolution_is_the_source_of_the_column():
    """§13's core ask: the column comes from the same place as the notes.
    `resolve_wpi` compares the per-system figures and writes "higher of 2 body
    systems (physical)"; the column must carry that, not a downstream re-guess
    from which cells happen to be populated."""
    out = w.resolve_wpi(_parsed([
        _m(4, "physical", "MAS certificate", assessor="MAS Curtin"),
        _m(1, "psychiatric", "MAS certificate", assessor="MAS Ng"),
    ]))
    assert "higher of 2 body systems (physical)" in out["WPI Basis"]
    assert out["WPI Governing System"] == "physical"


def test_a_single_system_resolution_reports_not_determined():
    out = w.resolve_wpi(_parsed([_m(14, "physical", "MAS certificate")]))
    assert out["WPI Governing System"] == "not determined"


def test_the_resolution_beats_the_flat_columns():
    """Quigley [2026] NSWPIC 280 (row 7). The column read `psychiatric` — the
    only component captured — while the resolution's own notes said "higher of
    2 body systems (physical)". The notes were right.

    Round 7 tightened what counts as "the resolution said so": the EVIDENCE
    must be on the row, not just a label in the cell. Keying on the stored
    value let 78 rows keep a label written before round 5, when the derivation
    was still circular — a stale answer is indistinguishable from a computed
    one. So the fixture now carries the mentions the comparison was made from.
    """
    row = _row(**{
        "Impairment % (Accepted)": "12",
        "WPI Psychiatric %": "1",
        "WPI Governing System": "physical",       # as the resolution wrote it
        "WPI Resolution Notes": "higher of 2 body systems (physical)",
        "_wpi_resolution": {"mentions": [
            {"value": "12", "body_system": "physical", "kind": "MAS certificate",
             "assessor": "Review Panel", "superseded": False, "about_claimant": True},
            {"value": "1", "body_system": "psychiatric", "kind": "MAS certificate",
             "assessor": "MAS Ng", "superseded": False, "about_claimant": True},
        ]},
    })
    ns.apply_round2_semantics(row)
    assert row["WPI Governing System"] == "physical"


def test_a_stale_label_with_no_evidence_behind_it_is_recomputed():
    """The other half of the same fix. Row 500 named `physical` on a row whose
    resolution holds no mentions at all — a label left over from before round 5
    that survived every rerun because the guard trusted the cell."""
    row = _row(**{
        "Impairment % (Accepted)": "",
        "WPI Physical %": "18",
        "WPI Governing System": "physical",       # stale
    })
    ns.apply_round2_semantics(row)
    assert row["WPI Governing System"] == "not determined"


# ----------------------------------------------------------------------
# §13.1 A1 — recovering the physical component from the total
# ----------------------------------------------------------------------

def test_quigley_the_total_is_the_physical_figure():
    """Row 7, confirmed from source. MAS Curtin certified 4% (scarring, nerve
    injury) and a Review Panel 8% (brain injury, shoulder) — DIFFERENT
    injuries, so they combine to 12 rather than competing, and MAS Lahz's
    combined certificate independently certifies "greater than 10%".
    Psychiatric is 1%. The total of 12 therefore IS the physical figure.

    Note the resolution ladder had these as rival assessments and took the
    median, 6 — which is why the recovery works from the TOTAL and not from
    the ladder's per-system arithmetic."""
    row = _row(**{"Impairment % (Accepted)": "12", "WPI Psychiatric %": "1",
                  "WPI Provenance": "stated", "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "brain injury | psychiatric"})
    ns.apply_round2_semantics(row)
    assert row["WPI Physical %"] == "12"
    assert row["WPI Physical % Provenance"] == "stated"
    assert "greater body system governs" in row["WPI Resolution Notes"]
    # And with both components present the comparison is now possible.
    assert row["WPI Governing System"] == "physical"


def test_the_recovered_component_inherits_the_totals_provenance():
    """A recovered figure is exactly as good as the total it came from — an
    inferred total cannot yield a stated component."""
    row = _row(**{"Impairment % (Accepted)": "16.5", "WPI Psychiatric %": "6",
                  "WPI Provenance": "inferred", "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "spinal | psychiatric"})
    ns.apply_round2_semantics(row)
    assert row["WPI Physical %"] == "16.5"
    assert row["WPI Physical % Provenance"] == "inferred"


def test_nothing_is_recovered_when_the_total_only_equals_psychiatric():
    """Group A2. The total equalling psychiatric means psychiatric governs and
    physical is bounded above by it — known to be ≤, not known to be equal."""
    row = _row(**{"Impairment % (Accepted)": "6", "WPI Psychiatric %": "6",
                  "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "spinal | psychiatric"})
    ns.apply_round2_semantics(row)
    assert row["WPI Physical %"] == ""
    assert row["WPI Governing System"] == "not determined"


def test_mason_the_rule_must_not_run_backwards():
    """The counter-example that stopped this being symmetric. Mason [2024]
    NSWPIC 348: 7% shoulder and 6% emotional/behavioural COMBINE to 13% inside
    one brain-injury assessment. A total above the stated physical figure is
    usually further physical components, not a larger psychiatric one — so
    recovering psychiatric from the total would invent a 13% psychiatric
    impairment on a claimant with no psychiatric injury at all."""
    row = _row(**{
        "Has Psychiatric Injury": "No",
        "Injury Categories": "brain injury | head or facial | upper limb",
        "Impairment % (Accepted)": "13",
        "WPI Physical %": "7",
    })
    ns.apply_round2_semantics(row)
    assert row["WPI Psychiatric %"] == ""
    assert row["WPI Psychiatric % Provenance"] == "not_applicable"


def test_recovery_does_not_fire_without_a_total():
    row = _row(**{"Impairment % (Accepted)": "", "WPI Psychiatric %": "6",
                  "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "psychiatric"})
    ns.apply_round2_semantics(row)
    assert row["WPI Physical %"] == ""


def test_recovery_never_overwrites_a_captured_physical_figure():
    row = _row(**{"Impairment % (Accepted)": "20", "WPI Psychiatric %": "5",
                  "WPI Physical %": "18", "WPI Physical % Provenance": "stated",
                  "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "spinal | psychiatric"})
    ns.apply_round2_semantics(row)
    assert row["WPI Physical %"] == "18"


def test_the_pass_is_idempotent():
    row = _row(**{"Impairment % (Accepted)": "12", "WPI Psychiatric %": "1",
                  "WPI Provenance": "stated", "Has Psychiatric Injury": "Yes",
                  "Injury Categories": "brain injury | psychiatric"})
    ns.apply_round2_semantics(row)
    first = dict(row)
    ns.apply_round2_semantics(row)
    for key in ("WPI Physical %", "WPI Physical % Provenance",
                "WPI Governing System", "WPI Psychiatric %"):
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
