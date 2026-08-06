"""
Golden-fixture regression tests for the damages-breakdown helpers.

Everything the downstream spec depends on that does NOT need the LLM: money
coercion, the three-valued status contract, the fatality pathway, buffer and
other-head accounting, gross derivation, both reconciliation identities,
provenance defaults, the damages context window, and the composed
Description With Figures.

Run standalone (no pytest needed):
    python test_damages_extraction.py
Or with pytest:
    pytest test_damages_extraction.py
"""

from types import SimpleNamespace

import damages_extraction as dx
import nsw_court_scraper as ns


# ----------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------

def _money(amount="", provenance="absent", quote=""):
    return SimpleNamespace(amount=amount, provenance=provenance, quote=quote)


def _stated(amount, quote="as stated"):
    return _money(amount, "stated", quote)


def _parsed(**kw):
    """A DamagesSchema-shaped object with everything empty/not-addressed."""
    base = {
        "past_economic_loss": _money(),
        "past_economic_loss_status": "Not addressed",
        "non_economic_loss": _money(),
        "non_economic_loss_status": "Not addressed",
        "future_economic_loss": _money(),
        "future_economic_loss_status": "Not addressed",
        "buffer_amount": _money(),
        "buffer_basis": "",
        "other_damages_heads": _money(),
        "other_damages_heads_basis": "",
        "total_damages_gross": _money(),
        "lump_sum_net": _money(),
        "lump_sum_basis": "unclear",
        "contributory_negligence_percent": _money(),
        "contributory_negligence_amount": _money(),
        "statutory_benefits_repaid": _money(),
        "other_deductions": _money(),
        "deductions_basis": "",
        "statutory_benefits_paid": _money(),
        "treatment_and_care_paid": _money(),
        "weekly_statutory_benefit": _money(),
        "award_breakdown_sentences": "",
        "accident_mechanism": "unclear",
        "claimant_road_role": "other",
        "injury_categories": [],
        "primary_injury_category": "other",
        "has_psychiatric_injury": False,
        "wpi_physical_percent": "",
        "wpi_psychiatric_percent": "",
        "wc_overlap": 0,
        "is_fatality_or_dependency_claim": False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


# A worked example modelled on the shape the spec quotes: three heads, a
# s 3.40 statutory-benefits deduction, and a stated total.
def _worked_example(**kw):
    fields = {
        "non_economic_loss": _stated("180000", "assessed non-economic loss at $180,000"),
        "non_economic_loss_status": "Awarded",
        "past_economic_loss": _stated("64500", "past economic loss of $64,500"),
        "past_economic_loss_status": "Awarded",
        "future_economic_loss": _stated("250000", "future economic loss of $250,000"),
        "future_economic_loss_status": "Awarded",
        "total_damages_gross": _stated("494500", "assessed total damages at $494,500"),
        "lump_sum_net": _stated("470829.43", "settlement sum of $470,829.43"),
        "lump_sum_basis": "net of deductions",
        "statutory_benefits_repaid": _stated(
            "23670.57",
            "Pursuant to s 3.40(1)(b) of the MAI Act, the insurer is entitled to "
            "deduct the sum of $23,670.57",
        ),
        "deductions_basis": "s 3.40(1)(b) statutory benefits",
    }
    fields.update(kw)
    return _parsed(**fields)


# ----------------------------------------------------------------------
# Money coercion
# ----------------------------------------------------------------------

def test_to_money():
    assert dx.to_money("$300,000") == "300000"
    assert dx.to_money("23,670.57") == "23670.57"
    assert dx.to_money("Nil") == ""
    assert dx.to_money("Not addressed") == ""
    assert dx.to_money("") == ""
    assert dx.to_money(None) == ""
    assert dx.to_money("0") == "0"


def test_to_money_agrees_with_scraper_coercion():
    """The two coercions live in different modules on purpose (one-way import);
    they must not drift apart on the values they both see."""
    for value in ("$300,000", "1,134.68 PIAWE", "Nil", "", "0", "476407",
                  "Not addressed", "$1,000,000.00"):
        assert dx.to_money(value) == ns.coerce_money(value), value


def test_to_float():
    assert dx.to_float("$1,000") == 1000.0
    assert dx.to_float("Nil") is None


# ----------------------------------------------------------------------
# Status discipline (acceptance criterion 4)
# ----------------------------------------------------------------------

def test_not_addressed_never_carries_an_amount():
    amount, prov, status, issues = dx.apply_head_status("50000", "stated", "Not addressed")
    assert amount == ""
    assert prov == "absent"
    assert status == "Not addressed"
    assert issues


def test_nil_is_a_genuine_zero():
    amount, prov, status, _ = dx.apply_head_status("", "absent", "Nil")
    assert (amount, status) == ("0", "Nil")
    assert prov == "derived"


def test_nil_with_a_non_zero_amount_is_forced_to_zero():
    amount, _, status, issues = dx.apply_head_status("40000", "stated", "Nil")
    assert (amount, status) == ("0", "Nil")
    assert issues


def test_awarded_zero_is_reclassified_as_nil():
    amount, _, status, issues = dx.apply_head_status("0", "stated", "Awarded")
    assert (amount, status) == ("0", "Nil")
    assert issues


def test_awarded_without_an_amount_is_reported_not_invented():
    amount, prov, status, issues = dx.apply_head_status("", "absent", "Awarded")
    assert amount == ""
    assert prov == "absent"
    assert status == "Awarded"
    assert issues


def test_fatality_claim_uses_not_addressed_rather_than_nil():
    _, _, status, issues = dx.apply_head_status("", "absent", "Nil", fatality=True)
    assert status == "Not addressed"
    assert issues


# ----------------------------------------------------------------------
# Normalisation end-to-end
# ----------------------------------------------------------------------

def test_worked_example_reconciles_both_identities():
    flat, sidecar, issues = dx.normalise_damages(
        _worked_example(), existing={"Lump Sum": "470829.43"})
    assert flat["Past Economic Loss"] == "64500"
    assert flat["Past Economic Loss Status"] == "Awarded"
    assert flat["Past Economic Loss Provenance"] == "stated"
    assert flat["Statutory Benefits Repaid"] == "23670.57"
    assert flat["Total Damages Gross"] == "494500"
    assert flat["Damages Reconciled"] == "yes"
    assert flat["Damages Residual"] == "0"
    assert flat["Net Reconciled"] == "yes"
    assert flat["Lump Sum Basis"] == "net of deductions"
    assert sidecar["gross_derivation"] == "stated"
    assert not issues


def test_residual_is_signed_and_reported_when_it_does_not_close():
    parsed = _worked_example(total_damages_gross=_stated("600000"))
    flat, _, _ = dx.normalise_damages(parsed, existing={"Lump Sum": "470829.43"})
    assert flat["Damages Reconciled"] == "no"
    assert float(flat["Damages Residual"]) == 105500.0


def test_negative_residual_survives_the_workbook_coercion():
    """The shared coerce_leading_number rejects negatives on purpose, which
    would blank exactly the rows where the heads exceed the stated gross."""
    parsed = _worked_example(total_damages_gross=_stated("400000"))
    flat, _, _ = dx.normalise_damages(parsed, existing={})
    assert flat["Damages Residual"] == "-94500"
    assert dx.to_float(flat["Damages Residual"]) == -94500.0
    assert ns.coerce_leading_number(flat["Damages Residual"]) != "-94500"  # why we need our own
    for col in ("Damages Residual", "Net Residual"):
        assert col in dx.DAMAGES_SIGNED_FIELDS


def test_gross_derived_from_heads_never_counts_as_reconciled():
    """Deriving gross from the heads would close the identity by construction,
    which would make the consumer's criterion 1 meaningless."""
    parsed = _worked_example(total_damages_gross=_money(), lump_sum_net=_money())
    flat, _, _ = dx.normalise_damages(parsed, existing={})
    assert flat["Damages Gross Derivation"] == "sum of heads"
    assert flat["Total Damages Gross Provenance"] == "derived"
    assert flat["Damages Reconciled"] == "insufficient data"
    assert flat["Damages Residual"] == ""


def test_gross_derived_from_net_plus_deductions():
    parsed = _worked_example(total_damages_gross=_money())
    flat, _, _ = dx.normalise_damages(parsed, existing={"Lump Sum": "470829.43"})
    assert flat["Damages Gross Derivation"] == "net plus deductions"
    assert float(flat["Total Damages Gross"]) == 494500.0
    assert flat["Net Reconciled"] == "insufficient data"  # closes by construction


def test_awarded_head_without_an_amount_blocks_reconciliation():
    parsed = _worked_example(past_economic_loss=_money(), past_economic_loss_status="Awarded")
    flat, _, _ = dx.normalise_damages(parsed, existing={"Lump Sum": "470829.43"})
    assert flat["Damages Reconciled"] == "insufficient data"


def test_not_addressed_heads_count_as_zero_in_the_identity():
    parsed = _parsed(
        non_economic_loss=_stated("100000"), non_economic_loss_status="Awarded",
        total_damages_gross=_stated("100000"),
    )
    flat, _, _ = dx.normalise_damages(parsed, existing={"Lump Sum": "100000"})
    assert flat["Past Economic Loss Status"] == "Not addressed"
    assert flat["Past Economic Loss"] == ""
    assert flat["Damages Reconciled"] == "yes"


def test_buffer_and_other_heads_enter_the_identity():
    parsed = _worked_example(
        total_damages_gross=_stated("569500"),
        buffer_amount=_stated("75000", "a buffer of $75,000 for future treatment"),
        buffer_basis="future treatment",
    )
    flat, _, _ = dx.normalise_damages(parsed, existing={})
    assert flat["Buffer Amount"] == "75000"
    assert flat["Damages Reconciled"] == "yes"


def test_zero_deductions_are_reported_as_none_not_as_a_quantified_zero():
    parsed = _worked_example(
        contributory_negligence_percent=_stated("0"),
        statutory_benefits_repaid=_stated("0"),
    )
    flat, _, _ = dx.normalise_damages(parsed, existing={})
    assert flat["Contributory Negligence Percent"] == ""
    assert flat["Contributory Negligence Percent Provenance"] == "absent"
    assert flat["Statutory Benefits Repaid"] == ""
    assert flat["Statutory Benefits Repaid Provenance"] == "absent"


def test_contributory_negligence_percent_out_of_range_is_dropped():
    parsed = _worked_example(contributory_negligence_percent=_stated("2000"))
    flat, _, issues = dx.normalise_damages(parsed, existing={})
    assert flat["Contributory Negligence Percent"] == ""
    assert flat["Contributory Negligence Percent Provenance"] == "absent"
    assert any("contributory negligence" in i for i in issues)


def test_disagreement_with_trusted_columns_is_recorded_not_resolved():
    flat, _, issues = dx.normalise_damages(
        _worked_example(), existing={"Lump Sum": "470829.43", "Non-Economic Loss": "120000"})
    # The trusted column is untouched; the pass's own reading is kept separately.
    assert flat["Non-Economic Loss (Recheck)"] == "180000"
    assert "Non-Economic Loss" not in flat
    assert any("Non-Economic Loss disagreement" in i for i in issues)


def test_amount_with_absent_provenance_is_downgraded_to_inferred():
    parsed = _worked_example(past_economic_loss=_money("64500", "absent", ""))
    flat, _, _ = dx.normalise_damages(parsed, existing={})
    assert flat["Past Economic Loss Provenance"] == "inferred"


def test_every_money_field_carries_a_provenance_value():
    """Acceptance criterion 5, on the emptiest possible row."""
    flat, _, _ = dx.normalise_damages(_parsed(), existing={})
    row = dx.empty_damages_row()
    row.update(flat)
    valid = {"stated", "derived", "inferred", "absent"}
    for _amount_col, prov_col in dx.MONEY_PROVENANCE_PAIRS:
        assert row.get(prov_col) in valid, prov_col


def test_split_wpi_is_not_invented():
    flat, _, _ = dx.normalise_damages(_parsed(), existing={})
    assert flat["WPI Physical %"] == ""
    assert flat["WPI Physical % Provenance"] == "absent"

    flat, _, _ = dx.normalise_damages(
        _parsed(wpi_physical_percent="12", wpi_psychiatric_percent="8"), existing={})
    assert (flat["WPI Physical %"], flat["WPI Psychiatric %"]) == ("12", "8")
    assert flat["WPI Physical % Provenance"] == "stated"


def test_multi_label_injury_is_deduplicated_and_ordered():
    flat, _, _ = dx.normalise_damages(
        _parsed(injury_categories=["spinal", "psychiatric", "spinal"],
                primary_injury_category="spinal", has_psychiatric_injury=True),
        existing={})
    assert flat["Injury Categories"] == "psychiatric | spinal"
    assert flat["Primary Injury Category"] == "spinal"
    assert flat["Has Psychiatric Injury"] == "Yes"


# ----------------------------------------------------------------------
# Description with figures (spec 4.2)
# ----------------------------------------------------------------------

def test_compose_description_with_figures():
    out = dx.compose_description_with_figures(
        "The claimant was injured in a rear-end collision.",
        "Non-economic loss was assessed at $180,000.")
    assert out == ("The claimant was injured in a rear-end collision. "
                   "Non-economic loss was assessed at $180,000.")
    assert "$" in out


def test_compose_description_handles_missing_pieces():
    assert dx.compose_description_with_figures("Only a description.", "") == "Only a description."
    assert dx.compose_description_with_figures("", "Only a breakdown.") == "Only a breakdown."
    assert dx.compose_description_with_figures("No terminator", "Added.") == "No terminator. Added."


def test_damages_row_composes_description_with_figures():
    row, _, _ = dx.damages_row_from_parsed(
        _worked_example(award_breakdown_sentences="Total damages were $494,500."),
        existing={"Lump Sum": "470829.43"},
        description="A rear-end collision.")
    assert row["Description With Figures"] == "A rear-end collision. Total damages were $494,500."
    assert row["Damages Extraction Status"] == "ok"


# ----------------------------------------------------------------------
# Context window
# ----------------------------------------------------------------------

def test_short_source_is_sent_whole():
    text = "Short decision about past economic loss."
    assert dx.build_damages_context(text, cap=10_000) == text


def test_long_source_keeps_the_orders_and_the_quantum_section():
    filler = "Filler paragraph about procedural history. " * 400          # ~17k
    submissions = "The claimant SEEKS past economic loss of $900,000. "    # early, claimed
    determination = ("I assess past economic loss at $64,500 and future economic "
                     "loss at $250,000. ORDERS: judgment for the claimant.")
    text = submissions + filler + "middle " * 2000 + filler + determination
    out = dx.build_damages_context(text, cap=8_000)
    assert len(out) <= 8_000 + 200          # plus section-break markers
    assert "I assess past economic loss at $64,500" in out
    assert "ORDERS: judgment for the claimant." in out


# ----------------------------------------------------------------------
# Integration with the scraper's flat record
# ----------------------------------------------------------------------

def test_damages_fields_are_all_in_result_fields():
    for field in dx.DAMAGES_FIELDS:
        assert field in ns.RESULT_FIELDS, field


def test_blank_record_defaults_are_honest_about_not_knowing():
    row = ns.build_result_record("Case", "http://example/1")
    assert row["Past Economic Loss Status"] == "Not addressed"
    assert row["Past Economic Loss"] == ""
    assert row["Damages Reconciled"] == "insufficient data"
    assert row["Past Economic Loss Provenance"] == "absent"
    assert row["Damages Extraction Status"] == "not run"


def test_merge_damages_into_record_protects_the_trusted_columns():
    row = ns.build_result_record("Case", "http://example/1")
    row.update({
        "Description": "A rear-end collision.",
        "Lump Sum": "470829.43",
        "Non-Economic Loss": "120000",
        "Non-Economic Loss Status": "Awarded",
        "Future Economic Loss": "250000",
        "Future Economic Loss Status": "Awarded",
        "Weekly Benefit": "",
    })
    parsed = _worked_example(
        award_breakdown_sentences="Total damages were $494,500.",
        weekly_statutory_benefit=_stated("522.84"),
    )
    ns.merge_damages_into_record(row, parsed)

    # Load-bearing columns untouched...
    assert row["Lump Sum"] == "470829.43"
    assert row["Non-Economic Loss"] == "120000"
    assert row["Future Economic Loss"] == "250000"
    # ...the pass's independent reading kept alongside...
    assert row["Non-Economic Loss (Recheck)"] == "180000"
    # ...new heads populated...
    assert row["Past Economic Loss"] == "64500"
    # ...the previously-empty Weekly Benefit filled...
    assert row["Weekly Benefit"] == "522.84"
    # ...and the figures reach the prose.
    assert "$494,500" in row["Description With Figures"]
    assert row["_damages_version"] == ns.DAMAGES_VERSION


def test_merge_damages_does_not_overwrite_an_existing_weekly_benefit():
    row = ns.build_result_record("Case", "http://example/1")
    row["Weekly Benefit"] = "600"
    ns.merge_damages_into_record(row, _worked_example(weekly_statutory_benefit=_stated("522.84")))
    assert row["Weekly Benefit"] == "600"


def test_medical_costs_sentinel_survives_a_pandas_round_trip():
    """'N/A' is in pandas' default na_values, which is why the consumer sees
    Medical Costs as 0% populated (spec 4.1)."""
    assert ns.normalise_medical_costs("N/A") == "Not addressed"
    assert ns.normalise_medical_costs("") == "Not addressed"
    assert ns.normalise_medical_costs("Yes") == "Yes"
    assert ns.normalise_medical_costs("No") == "No"


def test_workbook_population_requires_lump_sum_only():
    """A decision that never states a WPI is still a real award. Blank WPI must
    keep the row and be marked `absent`, not silently drop it."""
    import pandas as pd

    import ctp_lump_sum_impairment as ctp

    base = {f: "" for f in ns.RESULT_FIELDS}
    base.update({"Case Type": "CTP", "Analysis Ready": "Yes", "Decision Date": "2025-01-01"})
    rows = [
        {**base, "URL": "u1", "Lump Sum": "100000", "Impairment % (Accepted)": "14"},
        {**base, "URL": "u2", "Lump Sum": "250000", "Impairment % (Accepted)": ""},   # no WPI
        {**base, "URL": "u3", "Lump Sum": "", "Impairment % (Accepted)": "9"},        # no award
        {**base, "URL": "u4", "Lump Sum": "50000", "Impairment % (Accepted)": "",
         "Case Type": "Workers Compensation"},                                        # not CTP
    ]
    out, _ = ctp.build_workbook(pd.DataFrame(rows), {})

    assert sorted(out["URL"]) == ["u1", "u2"], "blank-WPI CTP award must be kept"
    assert out.loc[out["URL"].eq("u1"), "WPI % Provenance"].item() == "stated"
    assert out.loc[out["URL"].eq("u2"), "WPI % Provenance"].item() == "absent"
    assert pd.isna(out.loc[out["URL"].eq("u2"), "WPI %"].item())
    # Provenance sits next to the value it describes.
    cols = list(out.columns)
    assert cols[cols.index("WPI %") + 1] == "WPI % Provenance"


def test_outputs_are_confined_to_the_output_folder():
    import os
    for path in (ns.CACHE_FILE, ns.SIDECAR_FILE, ns.CSV_REPORT,
                 ns.ANALYSIS_READY_REPORT, ns.WORKBOOK_FILE, ns.LOG_FILE,
                 ns.RUN_MANIFEST_FILE, ns.AUSTLII_ERROR_LOG):
        assert os.path.dirname(path) == ns.OUTPUT_ROOT, path
    # Scraped source stays put — it is input, not output.
    assert os.path.dirname(ns.DECISIONS_DIR) == ""


def test_legacy_output_migration_never_overwrites(tmp_path=None):
    """The cache is thousands of dollars of extraction; a migration that
    clobbered a newer copy would be unrecoverable."""
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            os.makedirs("out")
            with open("thing.json", "w") as f:
                f.write("legacy")
            with open(os.path.join("out", "thing.json"), "w") as f:
                f.write("current")
            moved = ns.migrate_legacy_outputs(root="out", names=("thing.json",))
            assert moved == []
            assert open(os.path.join("out", "thing.json")).read() == "current"
            assert os.path.exists("thing.json"), "legacy copy kept for reconciliation"

            os.remove(os.path.join("out", "thing.json"))
            assert ns.migrate_legacy_outputs(root="out", names=("thing.json",)) == ["thing.json"]
            assert open(os.path.join("out", "thing.json")).read() == "legacy"
            assert not os.path.exists("thing.json")
        finally:
            os.chdir(cwd)


def test_damages_pass_gating():
    row = ns.build_result_record("Case", "http://example/1")
    row["Case Type"] = "CTP"
    assert ns.damages_pass_applies(row) is True
    row["Case Type"] = "Workers Compensation"
    assert ns.damages_pass_applies(row) is False


# ----------------------------------------------------------------------
# Standalone runner
# ----------------------------------------------------------------------

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
