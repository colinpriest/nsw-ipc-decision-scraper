"""
The liability_posture adjudication draw.

The other reference sets stratify on the model's own label, which answers "how
often is the model wrong". This one is drawn from the rule-vs-LLM disagreement
cells instead, because the cross-tab over the full corpus already showed the
disagreement is directional 4:1 and concentrated where a consequential
condition is claimed. Its job is to test whether the category needs a third
value, not to pick a winner.

Two properties matter enough to pin down: the allocation must stay
deliberately non-proportional (proportional starves the reverse cell), and no
column revealing either extractor's answer may sit ahead of the human's column.

    python test_wc_reference_sets.py      # or: pytest
"""

import re
from collections import Counter

import pandas as pd
import pytest

import wc_case_extract as wc

# The dominant cell's real dispute mix: impairment 152, medical 129,
# liability 124, other 89. Liability is the diagnostic stratum and the
# scarcest of the three that matter.
DOMINANT_NATURES = (["Permanent Impairment"] * 152 + ["Medical Dispute"] * 129
                    + ["Liability Dispute"] * 124 + ["Statutory Benefits Dispute"] * 89)


def _extract(denied_quantum=494, quantum_denied=129, procedural=77, agreeing=1674):
    """A frame with the full-corpus disagreement structure, in miniature."""
    rows = []

    def add(count, llm, rule, consequential="No", basis="explicit_concession",
            natures=None):
        for i in range(count):
            n = len(rows)
            rows.append({
                "case_id": f"C{n:05d}",
                "case_name": f"Worker v Employer No {n}",
                "source_html_file": f"decision_{n}.html",
                "nature_of_case": (natures[i % len(natures)] if natures
                                   else "Medical Dispute"),
                "catchwords": "WORKERS COMPENSATION - permanent impairment",
                "liability_posture": llm,
                "liability_posture_rule": rule,
                "primary_injury": ("back_spine", "psychological", "upper_limb")[n % 3],
                "liability_posture_basis": basis,
                "liability_posture_evidence": "the respondent disputes the consequential condition",
                # Roughly the measured 48% within the dominant cell.
                "consequential_condition_claimed": consequential if i % 2 else "No",
            })

    add(denied_quantum, "liability_denied", "quantum_or_entitlement_only", "Yes",
        natures=DOMINANT_NATURES)
    # 88 of the real 494 rest on the Nature field alone rather than concession
    # language: a weak-rule failure, not an under-specified category.
    for row in rows[:88]:
        row["liability_posture_basis"] = "nature_field"
    add(quantum_denied, "quantum_or_entitlement_only", "liability_denied")
    add(procedural, "not_applicable_procedural", "quantum_or_entitlement_only")
    add(agreeing, "liability_denied", "liability_denied")
    return pd.DataFrame(rows)


def test_the_allocation_is_fixed_not_proportional():
    """Proportional would spend 35 of 50 on the dominant cell; that is the bug."""
    sample = wc.build_liability_adjudication_sample(_extract(), size=50)
    counts = sample["_cell"].value_counts()
    assert len(sample) == 50
    assert counts["llm_denied__rule_quantum"] == 25
    assert counts["llm_quantum__rule_denied"] == 12
    assert counts["procedural_either_side"] == 13


def test_agreeing_cases_are_never_drawn():
    sample = wc.build_liability_adjudication_sample(_extract(), size=50)
    assert (sample["liability_posture"] != sample["liability_posture_rule"]).all()


def test_the_dominant_cell_is_crossed_on_both_competing_explanations():
    """Under-specified category and weak rule separate on this cross, not on
    either axis alone, so both must survive the draw."""
    sample = wc.build_liability_adjudication_sample(_extract(), size=50)
    dominant = sample[sample["_cell"] == "llm_denied__rule_quantum"]
    bases = {"concession" if b in wc.CONCESSION_BASES else "nature_field_only"
             for b in dominant["liability_posture_basis"]}
    assert bases == {"concession", "nature_field_only"}


def test_the_adjudication_sheet_is_sized_for_two_questions_not_one():
    """The other sheets estimate one accuracy each. This one also has to say
    whether partial denial is localised, and at 50 the diagnostic stratum lands
    6 cases -- too few to tell 'no mechanism' from 'no power'."""
    sheets = wc.build_reference_worksheets(_extract(), size=50)
    assert len(sheets["liability_posture"]) == 80
    assert len(sheets["primary_injury"]) == 50


def test_the_diagnostic_dispute_type_survives_the_draw():
    """Liability disputes are where the outcome lift says partial denial is the
    real cause (x2.26 against x1.15 in impairment). Unstratified, chance gave
    them 3 of 25 and loaded the sample with the types whose answer is likely
    'rule failure' -- which would then be read back as a finding."""
    sample = wc.build_liability_adjudication_sample(_extract(), size=50)
    dominant = sample[sample["_cell"] == "llm_denied__rule_quantum"]
    groups = Counter(wc.dispute_group(n) for n in dominant["nature_of_case"])
    assert groups["liability"] >= 5, groups
    assert groups["impairment"] >= 3, groups
    assert groups["medical"] >= 3, groups


def test_the_near_balanced_axis_is_left_to_chance_and_survives_anyway():
    """consequential sits near 50/50 inside the cell, so a random draw carries
    it; the scarce diagnostic variable is the one that needs the stratum. Three
    axes do not fit in 25 slots and this is the trade."""
    sample = wc.build_liability_adjudication_sample(_extract(), size=50)
    dominant = sample[sample["_cell"] == "llm_denied__rule_quantum"]
    consequential = Counter(dominant["consequential_condition_claimed"].astype(str))
    assert consequential["Yes"] >= 5 and consequential["No"] >= 5, consequential


def test_a_starved_cell_does_not_shrink_the_sample():
    """If a cell holds fewer rows than its share, the draw refills elsewhere
    rather than silently returning a short sample."""
    sample = wc.build_liability_adjudication_sample(
        _extract(denied_quantum=494, quantum_denied=3, procedural=77), size=50)
    assert len(sample) == 50
    assert (sample["_cell"] == "llm_quantum__rule_denied").sum() == 3


def test_the_draw_is_reproducible_for_a_given_seed():
    first = wc.build_liability_adjudication_sample(_extract(), size=50, seed=7)
    second = wc.build_liability_adjudication_sample(_extract(), size=50, seed=7)
    assert list(first["case_id"]) == list(second["case_id"])


def test_nothing_revealing_either_answer_precedes_the_human_column():
    """The labeller must reach a verdict before seeing what either extractor
    said. Anchoring inflates measured accuracy and defeats the exercise --
    doubly so here, where the LLM's evidence quote argues one side of the very
    disagreement under adjudication."""
    sheets = wc.build_reference_worksheets(_extract(), size=50)
    columns = list(sheets["liability_posture"].columns)
    human = columns.index("HUMAN_liability_posture")
    revealing = ("MODEL_liability_posture", "RULE_liability_posture", "_cell",
                 "liability_posture_evidence", "liability_posture_basis",
                 "consequential_condition_claimed")
    for name in revealing:
        assert columns.index(name) > human, f"{name} anchors the labeller"


def test_the_guidance_offers_the_third_value_the_schema_cannot_return():
    guidance = dict(wc.REFERENCE_SET_FIELDS)["liability_posture"]
    assert "liability_denied_in_part" in guidance


def test_scoring_refuses_to_pass_the_adjudication_off_as_corpus_accuracy(tmp_path):
    """Accuracy here is conditioned on the two methods already disagreeing, so
    it understates the field badly. The design must be reported alongside it."""
    sheets = wc.build_reference_worksheets(_extract(), size=50)
    sheet = sheets["liability_posture"].copy()
    # Hand-label as the partial-denial hypothesis predicts: the dominant cell
    # resolves to the third value, the reverse cell to the rule's reading.
    sheet["HUMAN_liability_posture"] = [
        "liability_denied_in_part" if cell == "llm_denied__rule_quantum"
        else "liability_denied" if cell == "llm_quantum__rule_denied"
        else "not_applicable_procedural"
        for cell in sheet["_cell"]]
    path = tmp_path / "sets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="liability_posture", index=False)

    report = wc.score_reference_set(path)
    row = report[report["field"] == "liability_posture"].iloc[0]
    assert "NOT a corpus accuracy" in row["design"]
    assert "partial-denial verdicts 40/80" in row["note"]
    assert "llm_denied__rule_quantum: n=40" in row["note"]


def test_blank_labels_are_skipped_rather_than_scored_as_errors(tmp_path):
    sheets = wc.build_reference_worksheets(_extract(), size=50)
    sheet = sheets["liability_posture"].copy()
    sheet["HUMAN_liability_posture"] = ""
    sheet.loc[sheet.index[:4], "HUMAN_liability_posture"] = "liability_denied"
    path = tmp_path / "sets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="liability_posture", index=False)

    row = wc.score_reference_set(path).iloc[0]
    assert row["labelled"] == 4


def _outcomes(boundary=240, flips=19, agreeing=2126):
    """The full-corpus shape: a disagreement set that is almost entirely the
    partial-success population, plus a small tail of genuine flips."""
    rows = []
    for i in range(boundary):
        rows.append({"case_id": f"B{i}", "outcome": "mixed",
                     "outcome_rule": "claimant" if i % 2 else "insurer",
                     "outcome_agreement": "differs", "nature_of_case": "Permanent Impairment"})
    for i in range(flips):
        rows.append({"case_id": f"F{i}", "outcome": "claimant", "outcome_rule": "insurer",
                     "outcome_agreement": "differs", "nature_of_case": "Liability Dispute"})
    for i in range(agreeing):
        rows.append({"case_id": f"A{i}", "outcome": "claimant", "outcome_rule": "claimant",
                     "outcome_agreement": "same", "nature_of_case": "Medical Dispute"})
    return pd.DataFrame(rows)


def test_clean_flips_are_marked_not_dropped():
    """A boundary-characterisation set whose premise is 'nobody is wrong' must
    not quietly carry cases where somebody is."""
    frozen = wc.freeze_definitional_set(_outcomes())
    assert len(frozen) == 259
    kinds = frozen["disagreement_kind"].value_counts()
    assert kinds["boundary_partial_success"] == 240
    assert kinds["clean_flip"] == 19


def test_the_whole_disagreement_set_is_kept():
    """259 cases, not a curated draw from them."""
    frozen = wc.freeze_definitional_set(_outcomes())
    assert set(frozen["case_id"]) == set(_outcomes().query("outcome_agreement == 'differs'")
                                         ["case_id"])


def test_the_frozen_set_carries_what_the_boundary_is_characterised_by():
    extract = _outcomes()
    extract["liability_posture"] = "liability_denied"
    extract["primary_injury"] = "back_spine"
    frozen = wc.freeze_definitional_set(extract)
    for column in ("nature_of_case", "liability_posture", "primary_injury"):
        assert column in frozen.columns


def test_the_metric_shift_is_reported_not_left_as_an_assertion():
    shift = wc.summarise_definitional_shift(_outcomes())
    overall = shift[shift["scope"] == "ALL"].iloc[0]
    # 240 mixed of 2385 -> the swing between counting mixed as loss vs win.
    assert overall["definitional_swing_points"] == 10.1
    assert round(overall["worker_success_generous_%"]
                 - overall["worker_success_strict_%"], 1) == overall["definitional_swing_points"]
    # Broken out by dispute type, because the boundary is not evenly spread.
    assert "Permanent Impairment" in set(shift["scope"])


def _score_a_labelled_sheet(tmp_path):
    """Label the adjudication sheet as the partial-denial hypothesis predicts,
    score it, and return the note. Labels vary by arm so the arm comparison has
    something to report."""
    sheets = wc.build_reference_worksheets(_extract(), size=50)
    sheet = sheets["liability_posture"].copy()
    labels = []
    for cell, basis in zip(sheet["_cell"], sheet["liability_posture_basis"]):
        if cell != "llm_denied__rule_quantum":
            labels.append("liability_denied")
        elif str(basis) in wc.CONCESSION_BASES:
            labels.append("liability_denied_in_part")
        else:
            # The weak-rule arm: the human sides with the LLM instead.
            labels.append("liability_denied")
    sheet["HUMAN_liability_posture"] = labels
    path = tmp_path / "sets.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="liability_posture", index=False)
    return wc.score_reference_set(path).iloc[0]["note"]


def test_the_dominant_cell_is_scored_by_arm_not_as_one_number(tmp_path):
    """If the human sides with the rule at a different rate where the rule had
    only the Nature field, that is two mechanisms, not one story."""
    note = _score_a_labelled_sheet(tmp_path)
    assert "arm concession:" in note
    assert "arm nature_field_only:" in note
    # And the localisation test: pooling dispute types would let medical and
    # impairment -- half the disagreements, and the ones whose answer is likely
    # 'rule failure' -- speak for liability disputes.
    assert "dispute liability:" in note
    assert "dispute impairment:" in note


def test_every_small_group_rate_carries_an_interval(tmp_path):
    """A point estimate on n=9 is misleading on its own, and 'significance' is
    the wrong frame at these counts -- direction and size are what the labels
    can support."""
    note = _score_a_labelled_sheet(tmp_path)
    # count/total = rate [low-high] on the group and arm lines.
    assert re.search(r"dispute liability: partial \d+/\d+ = \d+% \[\d+-\d+\]", note), note
    assert re.search(r"arm concession: sides with rule \d+/\d+ = \d+% \[\d+-\d+\]", note), note
    assert "arm difference:" in note and "points" in note
    assert "dispute spread:" in note


def test_the_arm_caveat_travels_with_the_arm_number(tmp_path):
    """The workbook outlives the conversation that hedged it. Whoever opens it
    next quotes the comparison without the exchange that qualified it."""
    note = _score_a_labelled_sheet(tmp_path)
    assert "NOT evidence the two arms are one story" in note
    assert "not significance" in note


def test_the_localisation_caveat_names_the_right_instrument(tmp_path):
    """These labels answer whether the category is real. Localisation runs at
    corpus scale, and reading it off 40 cases is the conflation to prevent."""
    note = _score_a_labelled_sheet(tmp_path)
    assert "not the instrument for localisation" in note
    assert "2,385" in note


def test_wilson_interval_behaves_at_the_extremes():
    """Normal approximation would run outside [0, 100] at exactly the counts
    this report lives at."""
    assert wc.wilson_interval(0, 0) == (0, 0)
    low, high = wc.wilson_interval(9, 9)
    assert 0 <= low <= 100 and high == 100
    low, high = wc.wilson_interval(0, 9)
    assert low == 0 and 0 <= high <= 100
    low, high = wc.wilson_interval(7, 9)
    assert low < 78 < high


def test_the_stale_disagreement_figure_is_corrected_in_source():
    """21% was measured pre-full-run and makes the field look marginal; the
    commit that recorded it cannot be edited, so the correction lives here."""
    assert "29.8%" in wc.build_liability_adjudication_sample.__doc__
    assert "003a899" in wc.build_liability_adjudication_sample.__doc__


def test_the_gaming_vector_is_tracked_source_not_just_prose():
    examples = {e["example"] for e in wc.WORKED_EXAMPLES}
    assert any("Partial denial" in name for name in examples)
    partial = next(e for e in wc.WORKED_EXAMPLES if "Partial denial" in e["example"])
    # The point is the channel, not merely the disagreement.
    assert "cooperative category" in partial["why_it_matters"]
    assert set(partial) >= {"what_happened", "why_it_was_wrong", "why_it_matters",
                            "how_it_was_caught", "the_fix", "lesson"}


def test_no_worked_example_still_quotes_the_hundred_case_sample():
    """Sample-derived base rates were the thing the sample could not supply."""
    for example in wc.WORKED_EXAMPLES:
        if example["example"].startswith("The sample"):
            continue
        assert "66%" not in example.get("what_happened", "")


def test_a_run_will_not_overwrite_hand_entered_labels(tmp_path):
    """Every other output is regenerable; hand labels are not. The conduct run
    regenerates this workbook by default, and labelling starts before it."""
    path = tmp_path / "sets.xlsx"
    extract = _extract()
    wc.write_reference_sets(path, extract, size=50)          # first write: fine
    wc.write_reference_sets(path, extract, size=50)          # unlabelled: still fine

    sheet = pd.read_excel(pd.ExcelFile(path), "liability_posture")
    sheet["HUMAN_liability_posture"] = sheet["HUMAN_liability_posture"].astype(object)
    sheet.loc[sheet.index[0], "HUMAN_liability_posture"] = "liability_denied_in_part"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="liability_posture", index=False)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        wc.write_reference_sets(path, extract, size=50)


def test_the_label_guard_fails_open_on_an_unreadable_file(tmp_path):
    """Blocking the pipeline over a file nobody has labelled would be the wrong
    failure direction."""
    path = tmp_path / "not-a-workbook.xlsx"
    path.write_text("garbage", encoding="utf-8")
    assert wc.count_human_labels(path) == 0
    assert wc.count_human_labels(tmp_path / "absent.xlsx") == 0


def test_a_frame_without_the_rule_column_yields_no_sheet_rather_than_a_bad_one():
    frame = _extract().drop(columns=["liability_posture_rule"])
    assert wc.build_liability_adjudication_sample(frame, size=50).empty
    assert "liability_posture" not in wc.build_reference_worksheets(frame, size=50)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
