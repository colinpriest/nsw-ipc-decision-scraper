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

import pandas as pd

import wc_case_extract as wc


def _extract(denied_quantum=494, quantum_denied=129, procedural=77, agreeing=1674):
    """A frame with the full-corpus disagreement structure, in miniature."""
    rows = []

    def add(count, llm, rule, consequential="No", basis="explicit_concession"):
        for i in range(count):
            n = len(rows)
            rows.append({
                "case_id": f"C{n:05d}",
                "case_name": f"Worker v Employer No {n}",
                "source_html_file": f"decision_{n}.html",
                "nature_of_case": "Claim for permanent impairment compensation",
                "catchwords": "WORKERS COMPENSATION - permanent impairment",
                "liability_posture": llm,
                "liability_posture_rule": rule,
                "liability_posture_basis": basis,
                "liability_posture_evidence": "the respondent disputes the consequential condition",
                # Roughly the measured 48% within the dominant cell.
                "consequential_condition_claimed": consequential if i % 2 else "No",
            })

    add(denied_quantum, "liability_denied", "quantum_or_entitlement_only", "Yes")
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
    assert set(dominant["consequential_condition_claimed"]) >= {"Yes", "No"}


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
    assert "partial-denial verdicts 25/50" in row["note"]
    assert "llm_denied__rule_quantum: n=25" in row["note"]


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


def test_a_frame_without_the_rule_column_yields_no_sheet_rather_than_a_bad_one():
    frame = _extract().drop(columns=["liability_posture_rule"])
    assert wc.build_liability_adjudication_sample(frame, size=50).empty
    assert "liability_posture" not in wc.build_reference_worksheets(frame, size=50)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
