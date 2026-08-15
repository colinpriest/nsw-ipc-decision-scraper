"""
The additive second pass: denial scope, success granularity, conduct findings.

The whole value of this pass is that it is versioned APART from the main
schema. Bumping WC_SCHEMA_VERSION re-extracts all 191 fields with a model
measured at 91% run-to-run stability, which would move every figure already
written into the data dictionary and the worked examples. So the tests that
matter most here are the ones that stop the two passes silently recoupling —
which they can do through three separate doors: the cache loader, the main
pass's write, and the version gate.

    python test_wc_conduct_pass.py       # or: pytest
"""

import json

import pytest

import wc_case_extract as wc


def _minimal(model):
    """A valid instance of a model whose fields are all required.

    Both schemas declare every field required with no defaults -- the file's
    convention, and what strict structured output wants -- so a fixture cannot
    just omit the fields it does not care about.
    """
    import enum
    import typing
    values = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        origin = typing.get_origin(annotation)
        if origin is list:
            values[name] = []
        elif isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values[name] = list(annotation)[0]
        elif annotation is int:
            values[name] = 0
        elif annotation is float:
            values[name] = 0.0
        elif annotation is bool:
            values[name] = False
        else:
            values[name] = ""
    return values


def _main_payload(**over):
    payload = {k: (v.value if hasattr(v, "value") else v)
               for k, v in _minimal(wc.WCCaseSchema).items()}
    payload.update(over)
    return payload


def _conduct(**over):
    values = dict(
        denial_scope=[wc.DenialScopeEnum.consequential_condition],
        denial_scope_evidence="liability is admitted save for the consequential condition",
        heads_claimed=4,
        heads_succeeded=3,
        conduct_finding=wc.ConductFindingEnum.criticism_made,
        conduct_scope=[wc.ConductScopeEnum.timeliness],
        conduct_evidence="the insurer's delay in determining the claim was unexplained",
    )
    values.update(over)
    return wc.WCConductSchema(**values)


def _cache_entry(main_version=wc.WC_SCHEMA_VERSION, conduct_version=wc.WC_CONDUCT_VERSION):
    entry = _main_payload(outcome="claimant")
    entry["_wc_schema_version"] = main_version
    entry["_conduct"] = _conduct().model_dump(mode="json")
    entry["_conduct_version"] = conduct_version
    return entry


# ----------------------------------------------------------------------
# The passes must not recouple
# ----------------------------------------------------------------------

def test_a_stale_main_record_does_not_evict_current_conduct_data(tmp_path):
    """The load-time filter used to drop the whole entry on a main-schema
    mismatch, which would have taken the conduct data with it and made the
    additive pass cost full price every time the main schema moved."""
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"u1": _cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1)}),
                    encoding="utf-8")
    cache = wc.load_llm_cache(path)
    assert "u1" in cache
    parsed, _error, was_cached = wc.cached_conduct_extract(None, cache, "u1", "text")
    assert was_cached and parsed.heads_claimed == 4


def test_a_stale_main_record_is_still_not_reused_as_if_current(tmp_path):
    """Keeping the entry must not mean trusting it: the version gate moved into
    the reader, it did not disappear."""
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"u1": _cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1)}),
                    encoding="utf-8")
    cache = wc.load_llm_cache(path)
    parsed, error, was_cached = wc.cached_llm_extract(None, cache, "u1", "text")
    assert parsed is None and not was_cached and error == "llm disabled"


def _force_main_reextraction(cache, url="u1"):
    """Drive cached_llm_extract down its WRITE path.

    The entry has to be STALE for this to exercise anything: with a current
    entry the call returns from cache and never writes, which makes any
    assertion about what the write preserves pass vacuously.
    """
    def _fake(extractor, text, context=None, seed=None):
        return wc.WCCaseSchema(**_main_payload(outcome="insurer")), None, None

    original = wc.extract_wc_case_llm
    wc.extract_wc_case_llm = _fake
    try:
        return wc.cached_llm_extract(object(), cache, url, "text")
    finally:
        wc.extract_wc_case_llm = original


def test_re_extracting_the_main_pass_preserves_the_conduct_data():
    """cache[url] = record would wipe it -- the same coupling through a
    different door."""
    cache = {"u1": _cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1)}
    parsed, _error, was_cached = _force_main_reextraction(cache)
    assert parsed is not None and not was_cached  # the write path really ran

    assert cache["u1"]["_conduct_version"] == wc.WC_CONDUCT_VERSION
    assert cache["u1"]["_conduct"]["heads_claimed"] == 4
    assert cache["u1"]["_wc_schema_version"] == wc.WC_SCHEMA_VERSION
    assert cache["u1"]["outcome"] == "insurer"  # and the main record did update


def test_stale_votes_are_dropped_on_re_extraction():
    """Unlike conduct data, the stability votes ARE values of main-schema
    fields, so a re-extraction makes them stale rather than reusable."""
    cache = {"u1": dict(_cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1),
                        _votes={"1": {"primary_injury": "spinal"}})}
    _force_main_reextraction(cache)
    assert "_votes" not in cache["u1"]
    assert "_conduct" in cache["u1"]


def test_conduct_data_is_invisible_to_the_main_schema(tmp_path):
    """It is nested under an underscore key precisely because the main record is
    reconstructed from every non-underscore key in the entry."""
    cache = {"u1": _cache_entry()}
    parsed, _error, was_cached = wc.cached_llm_extract(None, cache, "u1", "text")
    assert was_cached and parsed is not None
    assert not hasattr(parsed, "conduct_finding")


def test_a_conduct_version_bump_re_runs_only_the_conduct_pass(tmp_path):
    cache = {"u1": _cache_entry(conduct_version=wc.WC_CONDUCT_VERSION - 1)}
    parsed, error, was_cached = wc.cached_conduct_extract(None, cache, "u1", "text")
    assert parsed is None and not was_cached and error == "llm disabled"
    # ...while the main record is untouched and still served from cache.
    main, _error, main_cached = wc.cached_llm_extract(None, cache, "u1", "text")
    assert main_cached and main is not None


def test_cache_pass_counts_reports_each_pass_separately():
    cache = {"a": _cache_entry(),
             "b": _cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1),
             "c": _cache_entry(conduct_version=wc.WC_CONDUCT_VERSION - 1)}
    counts = wc.cache_pass_counts(cache)
    assert counts == {"entries": 3, "main_current": 2, "conduct_current": 2}


# ----------------------------------------------------------------------
# Field semantics
# ----------------------------------------------------------------------

def test_absence_is_recorded_as_a_finding_not_a_blank():
    """'The Member said nothing' and 'the Member expressly did not criticise'
    are opposite evidence for convergent validity."""
    row = wc.apply_conduct({}, _conduct(conduct_finding=wc.ConductFindingEnum.not_addressed,
                                        conduct_scope=[], conduct_evidence=""))
    assert row["conduct_finding"] == "not_addressed"
    assert row["conduct_status"] == "ok"

    not_run = wc.apply_conduct({}, None)
    assert not_run["conduct_status"] == "not run"
    assert not_run["conduct_finding"] is None


def test_success_degree_is_a_ratio_not_a_bucket():
    """The point of the field: four-of-five and one-of-five are currently the
    same value."""
    four_of_five = wc.apply_conduct({}, _conduct(heads_claimed=5, heads_succeeded=4))
    one_of_five = wc.apply_conduct({}, _conduct(heads_claimed=5, heads_succeeded=1))
    assert four_of_five["claimant_success_degree"] == 0.8
    assert one_of_five["claimant_success_degree"] == 0.2


def test_impossible_arithmetic_is_flagged_not_published():
    row = wc.apply_conduct({}, _conduct(heads_claimed=2, heads_succeeded=5))
    assert row["conduct_status"] == "heads_inconsistent"
    assert row["claimant_success_degree"] is None


def test_no_heads_means_null_not_zero():
    """0/0 is undeterminable, and a 0.0 would read as 'lost everything'."""
    row = wc.apply_conduct({}, _conduct(heads_claimed=0, heads_succeeded=0))
    assert row["claimant_success_degree"] is None


def test_multi_valued_scopes_survive_as_readable_text():
    row = wc.apply_conduct({}, _conduct(
        denial_scope=[wc.DenialScopeEnum.consequential_condition,
                      wc.DenialScopeEnum.specific_treatment],
        conduct_scope=[wc.ConductScopeEnum.surveillance,
                       wc.ConductScopeEnum.reasons_adequacy]))
    assert row["denial_scope"] == "consequential_condition;specific_treatment"
    assert row["conduct_scope"] == "surveillance;reasons_adequacy"


def test_surveillance_and_defective_notice_are_scope_values_not_fields():
    """Absorbing them into the enumeration simplifies the schema rather than
    extending it."""
    values = {member.value for member in wc.ConductScopeEnum}
    assert {"surveillance", "defective_notice"} <= values
    columns = {column for _attribute, column in wc.CONDUCT_OVERLAY}
    assert "surveillance" not in columns and "defective_notice" not in columns


def test_the_conduct_vocabulary_is_the_regulators():
    """Standards-relevant conduct is a far stronger validation target than
    whether a Member was unimpressed."""
    values = {member.value for member in wc.ConductScopeEnum}
    assert {"reasons_adequacy", "timeliness", "investigation_proportionality",
            "failure_to_consider_evidence"} <= values


def test_a_new_side_pass_is_carried_across_a_main_re_extraction():
    """The carry-over is generic, not a named list: a pass added later must not
    be silently dropped by a write that predates it."""
    cache = {"u1": dict(_cache_entry(main_version=wc.WC_SCHEMA_VERSION - 1),
                        _relief_votes={"1": "partly_granted"},
                        _relief_version=wc.WC_RELIEF_VERSION)}
    _force_main_reextraction(cache)
    assert cache["u1"]["_relief_votes"] == {"1": "partly_granted"}
    assert cache["u1"]["_conduct_version"] == wc.WC_CONDUCT_VERSION


def test_the_pilot_reports_the_two_failure_modes_it_exists_to_catch():
    """A field that piles up cannot carry a curve however accurate; a field
    that moves between runs makes the curve out of noise."""
    piled_and_stable = [["fully_granted"] * 3 for _ in range(20)]
    report = wc.summarise_relief_pilot(piled_and_stable)
    assert report["largest_bucket_%"] == 100.0 and report["distinct_values"] == 1
    assert report["unanimous_%"] == 100.0

    spread_but_unstable = [["fully_granted", "partly_granted", "refused"] for _ in range(10)]
    report = wc.summarise_relief_pilot(spread_but_unstable)
    assert report["no_majority_%"] == 100.0


def test_the_pilot_ignores_cases_that_could_not_be_voted():
    report = wc.summarise_relief_pilot([[], ["fully_granted"], ["refused", "refused"]])
    assert report["cases"] == 1


def test_the_relief_pilot_is_versioned_apart_from_the_conduct_pass():
    """It is still being piloted; it must not be able to invalidate the $46 of
    conduct data that works."""
    cache = {"u1": _cache_entry()}
    # A stale relief version must leave the conduct record entirely alone.
    cache["u1"]["_relief_version"] = wc.WC_RELIEF_VERSION - 1
    parsed, _error, was_cached = wc.cached_conduct_extract(None, cache, "u1", "text")
    assert was_cached and parsed is not None
    # ...and vice versa: the two version keys are independent, not aliases.
    stale_conduct = {"u1": _cache_entry(conduct_version=wc.WC_CONDUCT_VERSION - 1)}
    stale_conduct["u1"]["_relief_votes"] = {"1": "refused"}
    assert wc.cached_relief_votes(None, stale_conduct, "u1", "text",
                                  seeds=(1,)) == ["refused"]


def test_the_endogeneity_caveat_is_recorded_where_a_consumer_will_hit_it():
    """A validation target whose limitation lives only in a conversation is a
    validation target whose limitation will not be stated."""
    entry = next(row for row in wc.FIELD_DOCS if row[0] == "conduct_finding")
    definition = entry[-1]
    assert "ENDOGENEITY" in definition
    assert "same document by the same person" in definition


def test_the_relief_pass_does_not_reuse_pilot_votes_for_the_delivered_value():
    """The votes hold the ordinal but no evidence quote. Reusing them would
    leave those rows with a degree and no way to audit it."""
    cache = {"u1": {"_relief_votes": {str(wc.RELIEF_PILOT_SEEDS[0]): "partly_granted"},
                    "_relief_version": wc.WC_RELIEF_VERSION}}
    parsed, error, was_cached = wc.cached_relief_extract(None, cache, "u1", "text")
    assert not was_cached and parsed is None and error == "llm disabled"


def test_a_stored_relief_record_is_served_from_cache():
    cache = {"u1": {"_relief": wc.WCReliefSchema(
        relief_granted_degree=wc.ReliefGrantedEnum.partly_granted,
        relief_granted_evidence="allowed for eight months").model_dump(mode="json"),
        "_relief_version": wc.WC_RELIEF_VERSION}}
    parsed, _error, was_cached = wc.cached_relief_extract(None, cache, "u1", "text")
    assert was_cached and parsed.relief_granted_degree.value == "partly_granted"


def test_the_single_vote_choice_is_recorded_on_every_row():
    """A field carrying ~11% soft values must say so where a consumer reads it,
    not only in a commit message."""
    row = wc.apply_relief({}, wc.WCReliefSchema(
        relief_granted_degree=wc.ReliefGrantedEnum.substantially_granted,
        relief_granted_evidence="the great majority of the period claimed"))
    assert row["relief_granted_degree"] == "substantially_granted"
    assert "single_vote" in row["relief_stability"]
    assert row["relief_status"] == "ok"

    not_run = wc.apply_relief({}, None)
    assert not_run["relief_status"] == "not run"
    assert not_run["relief_granted_degree"] is None


def test_one_opinion_is_never_counted_as_two():
    """The production pass and the pilot's first vote share a seed. Appending
    them would manufacture unanimity out of a single opinion on the 100 pilot
    cases."""
    seed = str(wc.RELIEF_PILOT_SEEDS[0])
    entry = {"_relief": {"relief_granted_degree": "partly_granted",
                         "relief_granted_evidence": "q"},
             "_relief_votes": {seed: "partly_granted"}}
    value, agreement, votes = wc.resolve_relief_votes(entry)
    assert votes == ["partly_granted"]
    assert agreement == "single_vote" and value == "partly_granted"


def test_revotes_resolve_by_majority_and_flag_a_three_way_split():
    seeds = [str(s) for s in wc.RELIEF_PILOT_SEEDS]
    base = {"_relief": {"relief_granted_degree": "partly_granted", "relief_granted_evidence": ""}}

    unanimous = dict(base, _relief_votes={seeds[1]: "partly_granted",
                                          seeds[2]: "partly_granted"})
    assert wc.resolve_relief_votes(unanimous)[1] == "unanimous"

    majority = dict(base, _relief_votes={seeds[1]: "substantially_granted",
                                         seeds[2]: "partly_granted"})
    value, agreement, _ = wc.resolve_relief_votes(majority)
    assert agreement == "majority" and value == "partly_granted"

    split = dict(base, _relief_votes={seeds[1]: "substantially_granted",
                                      seeds[2]: "minimally_granted"})
    value, agreement, _ = wc.resolve_relief_votes(split)
    # Keeps the delivered value rather than pretending a tie was broken.
    assert agreement == "no_majority" and value == "partly_granted"


def test_revoting_targets_the_middle_and_samples_the_extremes():
    cache = {}
    for i in range(40):
        degree = "partly_granted" if i < 10 else "fully_granted"
        cache[f"u{i}"] = {"_relief": {"relief_granted_degree": degree,
                                      "relief_granted_evidence": ""}}
    middle, extreme = wc.relief_revote_targets(cache, extremes=5, seed=1)
    assert len(middle) == 10
    assert len(extreme) == 5
    assert not set(middle) & set(extreme)
    # The bound is opt-out, not mandatory.
    assert wc.relief_revote_targets(cache, extremes=0)[1] == []


def test_the_resolved_value_replaces_the_single_vote_on_the_row():
    parsed = wc.WCReliefSchema(relief_granted_degree=wc.ReliefGrantedEnum.partly_granted,
                               relief_granted_evidence="q")
    row = wc.apply_relief({}, parsed, resolution=(
        "substantially_granted", "majority",
        ["partly_granted", "substantially_granted", "substantially_granted"]))
    assert row["relief_granted_degree"] == "substantially_granted"
    assert row["relief_agreement"] == "majority"
    assert row["relief_votes"].count(",") == 2
    assert row["relief_stability"] == "revoted_majority"


def test_every_new_column_is_documented():
    """The dictionary asserts coverage; a new column with no entry comes back
    as UNDOCUMENTED rather than being silently omitted."""
    columns = [column for _attribute, column in wc.CONDUCT_OVERLAY]
    columns += [column for _attribute, column in wc.RELIEF_OVERLAY]
    columns += ["relief_agreement", "relief_votes"]
    columns += ["claimant_success_degree", "conduct_status", "relief_status", "relief_stability"]
    documented = wc.build_dictionary(columns=columns)
    undocumented = documented[documented["group"] == "UNDOCUMENTED"]["field"].tolist()
    assert not [c for c in columns if c in undocumented], undocumented


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
