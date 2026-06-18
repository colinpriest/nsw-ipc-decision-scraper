"""
Golden-fixture regression tests for the deterministic extraction helpers (C1).

These cover the field coercion, WPI reconciliation, age/DOB cross-check, the
field-loss gate, the focused-pass merge, and truncation keyword anchoring — i.e.
everything that protects the eight high-value fields WITHOUT needing the LLM.
Run them whenever the prompt or SCHEMA_VERSION changes.

Run standalone (no pytest needed):
    python test_extraction_fields.py
Or with pytest:
    pytest test_extraction_fields.py
"""

from types import SimpleNamespace

import nsw_court_scraper as ns


# ---- enum-like stand-ins for FocusedFields fields (have a .value) ----
def _enum(value):
    return SimpleNamespace(value=value)


def _focused(**kw):
    """Build a FocusedFields-like object with sensible empty defaults."""
    base = {
        "wpi_percent": "",
        "non_economic_loss": "",
        "non_economic_loss_status": _enum("Not addressed"),
        "claimant_weekly_income": "",
        "claimant_weekly_income_basis": "",
        "future_economic_loss": "",
        "future_economic_loss_status": _enum("Not addressed"),
        "claimant_age": "",
        "claimant_gender": _enum("Not stated"),
        "claimant_occupation": "",
        "location": "",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _blank_record(**kw):
    row = {f: "" for f in ns.RESULT_FIELDS}
    row["Non-Economic Loss Status"] = "Not addressed"
    row["Future Economic Loss Status"] = "Not addressed"
    row["Claimant Gender"] = "Not stated"
    row.update(kw)
    return row


# ----------------------------------------------------------------------
# coerce_money / coerce_leading_number
# ----------------------------------------------------------------------

def test_coerce_money():
    assert ns.coerce_money("$300,000") == "300000"
    assert ns.coerce_money("1,134.68 PIAWE") == "1134.68"
    assert ns.coerce_money("Nil") == ""
    assert ns.coerce_money("Not addressed") == ""
    assert ns.coerce_money("Not stated") == ""
    assert ns.coerce_money("") == ""
    assert ns.coerce_money("300000 for non-economic loss") == "300000"


def test_value_present():
    assert ns._value_present("300000")
    assert ns._value_present("0")  # a real Nil amount mapped to 0 counts as present
    assert not ns._value_present("")
    assert not ns._value_present("Not stated")
    assert not ns._value_present("Nil")
    assert not ns._value_present("N/A")


# ----------------------------------------------------------------------
# WPI cleaning + reconciliation (B1) — the #1 error source
# ----------------------------------------------------------------------

def test_clean_wpi_value():
    assert ns._clean_wpi_value("14") == "14"
    assert ns._clean_wpi_value("14.0") == "14"
    assert ns._clean_wpi_value("11%") == "11"
    assert ns._clean_wpi_value("0") == ""      # lone zero suppressed
    assert ns._clean_wpi_value("150") == ""    # implausible
    assert ns._clean_wpi_value("Not stated") == ""
    assert ns._clean_wpi_value("") == ""


def test_reconcile_wpi_seeds_accepted_from_strict():
    strict, accepted, issues = ns.reconcile_wpi("15", "", "irrelevant text")
    assert strict == "15"
    assert accepted == "15"


def test_reconcile_wpi_regex_backfill():
    text = "The claimant was assessed at 14% whole person impairment by the MAS."
    strict, accepted, issues = ns.reconcile_wpi("", "", text)
    assert accepted == "14"
    assert any(i["type"] == "wpi_regex_backfill" for i in issues)


def test_reconcile_wpi_threshold_is_not_a_finding():
    # MAI Act statutory threshold framing must NOT become a WPI value.
    text = ("The claim does not exceed the statutory threshold of 10% whole "
            "person impairment required for non-economic loss.")
    strict, accepted, issues = ns.reconcile_wpi("", "", text)
    assert accepted == ""   # 10% threshold is not a finding
    assert ns.find_wpi_candidates(text) == set()


def test_reconcile_wpi_mismatch_flagged():
    text = "A single assessment of 8% whole person impairment is recorded."
    strict, accepted, issues = ns.reconcile_wpi("", "20", text)
    assert accepted == "20"  # we keep the LLM value...
    assert any(i["type"] == "wpi_mismatch" for i in issues)  # ...but flag it


def test_extract_wpi_confident_single_nonzero():
    text = "WPI of 7% was certified."
    assert ns.extract_wpi_confident(text) == 7.0


# ----------------------------------------------------------------------
# Age / DOB cross-check (B4)
# ----------------------------------------------------------------------

def test_derive_age_from_dob():
    assert ns.derive_age_from_dob("1980", "2020-05-01") == 40
    assert ns.derive_age_from_dob("12 March 1995", "2023-01-01") == 28
    assert ns.derive_age_from_dob("", "2020-01-01") is None
    assert ns.derive_age_from_dob("1700", "2020-01-01") is None  # implausible


def test_reconcile_ages():
    # Wilkie: "now 31" in 2021, injury 2018, no DOB -> age at injury = 28
    ai, ad = ns.reconcile_ages("", "31", "2018-01-05", "2021-06-01")
    assert ai == "28" and ad == "31"
    # reverse: age at injury known, fill age at decision
    ai, ad = ns.reconcile_ages("28", "", "2018-01-05", "2021-06-01")
    assert ai == "28" and ad == "31"
    # never overwrite an existing value
    ai, ad = ns.reconcile_ages("40", "31", "2018", "2021")
    assert ai == "40" and ad == "31"
    # same year -> equal
    ai, ad = ns.reconcile_ages("", "50", "2022-03-01", "2022-09-01")
    assert ai == "50"
    # not derivable without a year -> unchanged blanks
    ai, ad = ns.reconcile_ages("", "31", "Unknown", "2021-06-01")
    assert ai == "" and ad == "31"
    # implausible result rejected
    ai, ad = ns.reconcile_ages("", "2", "1990", "2021")
    assert ai == ""


def test_check_age_consistency():
    # stated 45 but DOB implies 40 -> mismatch
    issue = ns.check_age_consistency("45", "1980", "2020-01-01")
    assert issue and issue["type"] == "age_dob_mismatch"
    # consistent within a year -> no issue
    assert ns.check_age_consistency("40", "1980", "2020-01-01") is None
    # no DOB -> no issue
    assert ns.check_age_consistency("40", "", "2020-01-01") is None


# ----------------------------------------------------------------------
# Field-loss gate (A1)
# ----------------------------------------------------------------------

def test_wpi_detection_broad_but_escalation_precise():
    rec = _blank_record()
    # DETECTION stays broad: a WPI token in the text flags (so the 2nd pass runs)
    text = "The Member accepted the assessment of 14% whole person impairment."
    wpi_losses = [i for i in ns.detect_field_losses(rec, text, {})
                  if i["field"] == "Impairment % (Accepted)"]
    assert wpi_losses, "broad detection should flag a WPI token for second-pass recovery"
    # ESCALATION is precise: no quote -> not confirmed (no Needs Review)
    assert ns.confirmed_high_losses(wpi_losses, {}) == []
    # clean value in the quote -> confirmed
    assert ns.confirmed_high_losses(
        wpi_losses, {"wpi_quote": "assessed at 14% whole person impairment"})
    # threshold framing in the quote -> not confirmed (no recoverable value)
    assert ns.confirmed_high_losses(
        wpi_losses, {"wpi_quote": "permanent impairment of greater than 10%"}) == []


def test_loss_gate_nel_only_when_awarded_but_empty():
    # Idea 1: a damages head is a loss ONLY when Awarded with a missing amount.
    rec = _blank_record()
    rec["Non-Economic Loss Status"] = "Awarded"  # amount empty
    text = "The claimant was awarded non-economic loss of $300,000 for pain and suffering."
    assert any(i["field"] == "Non-Economic Loss" and i["severity"] == "high"
               for i in ns.detect_field_losses(rec, text, {}))
    # Not addressed / Nil are deliberate dispositions -> never a loss
    for status in ("Not addressed", "Nil"):
        rec2 = _blank_record()
        rec2["Non-Economic Loss Status"] = status
        assert not any(i["field"] == "Non-Economic Loss"
                       for i in ns.detect_field_losses(rec2, text, {}))


def test_loss_gate_flags_income_and_age():
    rec = _blank_record()
    text = "The claimant, aged 45, had a PIAWE of $1,200 per week before the injury."
    issues = ns.detect_field_losses(rec, text, {})
    fields = {i["field"] for i in issues}
    assert "Claimant Weekly Income" in fields
    assert "Claimant Age" in fields


def test_age_gate_skips_when_age_at_decision_present():
    # Age-at-injury empty but age-at-decision captured -> NOT a loss (age is
    # recorded; at-injury just isn't derivable without an injury year).
    rec = _blank_record(**{"Claimant Age At Decision": "55"})
    text = "The claimant is now 55 years of age. PIAWE was not in issue."
    prov = {"age_quote": "the claimant is now 55 years of age"}
    assert not any(i["field"] == "Claimant Age" for i in ns.detect_field_losses(rec, text, prov))
    # With NEITHER age field populated, the age signal still flags.
    rec2 = _blank_record()
    assert any(i["field"] == "Claimant Age" for i in ns.detect_field_losses(rec2, text, prov))


def test_loss_gate_provenance_drives_gender_and_location():
    rec = _blank_record()
    prov = {"gender_quote": "the claimant, a 52 year old woman",
            "location_quote": "the intersection of George and King Streets"}
    issues = ns.detect_field_losses(rec, "no obvious signals here", prov)
    fields = {i["field"] for i in issues}
    assert "Claimant Gender" in fields
    assert "Accident/Injury Location" in fields


def test_loss_gate_clean_when_all_present():
    rec = _blank_record(**{
        "Impairment % (Accepted)": "14",
        "Non-Economic Loss": "300000",
        "Non-Economic Loss Status": "Awarded",
        "Future Economic Loss": "50000",
        "Future Economic Loss Status": "Awarded",
        "Claimant Weekly Income": "1200",
        "Claimant Age": "45",
        "Claimant Gender": "Female",
        "Claimant Occupation": "registered nurse",
        "Accident/Injury Location": "Parramatta",
    })
    text = ("aged 45, PIAWE $1,200 per week, non-economic loss of $300,000, "
            "future economic loss $50,000, 14% whole person impairment.")
    assert ns.detect_field_losses(rec, text, {}) == []


# ----------------------------------------------------------------------
# worth_second_pass cost guard
# ----------------------------------------------------------------------

def test_worth_second_pass():
    high = [{"field": "Claimant Age", "severity": "high", "detail": "x"}]
    med = [{"field": "Claimant Occupation", "severity": "medium", "detail": "x"}]
    med_q = [{"field": "Claimant Gender", "severity": "medium", "detail": "x"}]
    assert ns.worth_second_pass(high, {})
    assert not ns.worth_second_pass(med, {})
    assert ns.worth_second_pass(med_q, {"gender_quote": "she was the claimant"})


def test_confirmed_high_losses_requires_quote():
    # regex-only high loss (no provenance quote) must NOT escalate to Needs Review
    regex_only = [{"field": "Impairment % (Accepted)", "severity": "high", "detail": "x"}]
    assert ns.confirmed_high_losses(regex_only, {}) == []
    assert ns.confirmed_high_losses(regex_only, {"wpi_quote": ""}) == []
    # model self-contradiction (quoted the value but left field empty) -> confirmed
    confirmed = ns.confirmed_high_losses(
        regex_only, {"wpi_quote": "a 2% whole person impairment was assessed"})
    assert len(confirmed) == 1
    # medium losses never escalate regardless of quote
    med = [{"field": "Claimant Occupation", "severity": "medium", "detail": "x"}]
    assert ns.confirmed_high_losses(med, {"occupation_quote": "worked as a fitter"}) == []


def test_absence_quotes_do_not_count():
    # Idea 3: quotes that prove absence must not corroborate a loss
    assert ns._quote_indicates_absence("the claim is confined to past economic loss")
    assert ns._quote_indicates_absence("he was unemployed and on a Disability Support Pension")
    assert ns._quote_indicates_absence("future economic loss was not claimed")
    assert not ns._quote_indicates_absence("non-economic loss of $300,000 was awarded")
    # an absence quote does not make _quote_signal true
    assert not ns._quote_signal({"weekly_income_quote": "the claimant was unemployed"}, "weekly_income_quote")
    assert ns._quote_signal({"weekly_income_quote": "PIAWE of $1,200 per week"}, "weekly_income_quote")
    # and therefore does not confirm a high loss
    rem = [{"field": "Claimant Weekly Income", "severity": "high", "detail": "x"}]
    assert ns.confirmed_high_losses(rem, {"weekly_income_quote": "he was unemployed"}) == []


# ----------------------------------------------------------------------
# Focused-pass merge (A6)
# ----------------------------------------------------------------------

def test_merge_focused_recovers_nel_with_status():
    rec = _blank_record()
    foc = _focused(non_economic_loss="300000", non_economic_loss_status=_enum("Awarded"))
    recovered = ns.merge_focused_into_record(rec, foc, ["Non-Economic Loss"])
    assert recovered == ["Non-Economic Loss"]
    assert rec["Non-Economic Loss"] == "300000"
    assert rec["Non-Economic Loss Status"] == "Awarded"


def test_merge_focused_income_sets_basis():
    rec = _blank_record()
    foc = _focused(claimant_weekly_income="1200", claimant_weekly_income_basis="PIAWE gross weekly")
    recovered = ns.merge_focused_into_record(rec, foc, ["Claimant Weekly Income"])
    assert rec["Claimant Weekly Income"] == "1200"
    assert rec["Claimant Weekly Income Basis"] == "PIAWE gross weekly"


def test_merge_focused_does_not_override_existing():
    rec = _blank_record(**{"Claimant Age": "45"})
    foc = _focused(claimant_age="99")
    recovered = ns.merge_focused_into_record(rec, foc, ["Claimant Age"])
    assert recovered == []
    assert rec["Claimant Age"] == "45"


def test_merge_focused_wpi_suppresses_zero():
    rec = _blank_record()
    foc = _focused(wpi_percent="0")
    recovered = ns.merge_focused_into_record(rec, foc, ["Impairment % (Accepted)"])
    assert recovered == []
    assert rec["Impairment % (Accepted)"] == ""


# ----------------------------------------------------------------------
# Banding bands (sanity — boundaries go to the lower band)
# ----------------------------------------------------------------------

def test_band_boundaries():
    assert ns._band_for(11, ns.WPI_BANDS) == "11-15%"
    assert ns._band_for(300_000, ns.NEL_BANDS) == "$150k-$300k"
    assert ns._band_for(1200, ns.INCOME_WEEKLY_BANDS) == "$1000-$1500"


# ----------------------------------------------------------------------
# Truncation keyword anchoring (A3) — quantum section must survive
# ----------------------------------------------------------------------

def test_truncate_keeps_quantum_section():
    saved = ns.SINGLE_PASS_LIMIT_CHARS
    try:
        ns.SINGLE_PASS_LIMIT_CHARS = 35000
        filler = "the tribunal considered the evidence carefully. " * 800  # ~38k chars
        marker = " The claimant was awarded non-economic loss of $300,000. "
        text = filler + marker + ("conclusion. " * 200)
        assert len(text) > 35000
        out = ns._narrative_truncate(text)
        assert "non-economic loss of $300,000" in out
    finally:
        ns.SINGLE_PASS_LIMIT_CHARS = saved


def test_truncate_passthrough_when_short():
    text = "short decision text"
    assert ns._narrative_truncate(text) == text


def test_truncate_prioritises_quantum_over_generic_headings():
    # ISSUE-017: a long doc with many early generic headings, quantum only at the
    # very end. The late quantum window must survive (priority over structure).
    saved = ns.SINGLE_PASS_LIMIT_CHARS
    try:
        ns.SINGLE_PASS_LIMIT_CHARS = 40000
        generic = ("Background. Facts. History. Reasons. Submissions. Discussion. "
                   "Findings. Conclusion. Orders. " * 700)  # ~45k of generic headings
        marker = " The claimant was awarded non-economic loss of $250,000 and a 14% WPI. "
        text = generic + marker + ("end. " * 100)
        assert len(text) > 40000
        out = ns._narrative_truncate(text)
        assert "non-economic loss of $250,000" in out
        assert "14% WPI" in out
    finally:
        ns.SINGLE_PASS_LIMIT_CHARS = saved


# ----------------------------------------------------------------------
# Money coercion on Lump Sum / Weekly Benefit (ISSUE-006)
# ----------------------------------------------------------------------

def test_money_coercion_strips_trailing_text():
    assert ns.coerce_money("$300,000 inclusive of costs") == "300000"
    assert ns.coerce_money("Weekly $522.84 ongoing") == "522.84"
    assert ns.coerce_money("$1,234,567.89") == "1234567.89"
    assert ns.coerce_money("Not stated") == ""


# ----------------------------------------------------------------------
# Safe path resolution (ISSUE-012)
# ----------------------------------------------------------------------

def test_safe_decision_path():
    base = "nsw_pic_decisions"
    ok = ns.safe_decision_path(base, "Foo_2024_1.html")
    assert ok and ok.endswith("Foo_2024_1.html")
    # traversal / absolute / drive / alt-separator / tilde -> rejected
    assert ns.safe_decision_path(base, "../.env") is None
    assert ns.safe_decision_path(base, "..\\..\\secrets.txt") is None
    assert ns.safe_decision_path(base, "/etc/passwd") is None
    assert ns.safe_decision_path(base, "C:\\Windows\\system.ini") is None
    assert ns.safe_decision_path(base, "~/secret") is None
    assert ns.safe_decision_path(base, "") is None
    assert ns.safe_decision_path(base, None) is None


# ----------------------------------------------------------------------
# Worker-count validation (ISSUE-018)
# ----------------------------------------------------------------------

def test_get_worker_count():
    import os
    saved = os.environ.get("EXTRACTION_WORKERS")
    try:
        os.environ["EXTRACTION_WORKERS"] = "8"
        assert ns.get_worker_count() == 8
        os.environ["EXTRACTION_WORKERS"] = "0"
        assert ns.get_worker_count() == 1
        os.environ["EXTRACTION_WORKERS"] = "-3"
        assert ns.get_worker_count() == 1
        os.environ["EXTRACTION_WORKERS"] = "abc"
        assert ns.get_worker_count(default=7) == 7
        os.environ["EXTRACTION_WORKERS"] = str(ns.MAX_WORKERS + 100)
        assert ns.get_worker_count() == ns.MAX_WORKERS
    finally:
        if saved is None:
            os.environ.pop("EXTRACTION_WORKERS", None)
        else:
            os.environ["EXTRACTION_WORKERS"] = saved


# ----------------------------------------------------------------------
# Privacy transforms (ISSUE-011) — default keeps everything
# ----------------------------------------------------------------------

def test_privacy_default_is_noop():
    assert not ns.privacy_active()
    row = {"Applicant": "Jane Smith", "Respondent": "QBE", "Employer Name": "Acme"}
    assert ns.apply_privacy_to_row(row) == row  # default: unchanged
    sc = {"provenance": {"wpi_quote": "x"}, "field_review": {"date_of_birth": "1980"}}
    assert ns.apply_privacy_to_sidecar(sc) == sc


def test_privacy_redact_and_drop():
    # Toggle module-level flags directly (they're read at import from env).
    saved = (ns.PRIVACY_DROP_IDENTITY, ns.PRIVACY_DROP_DOB, ns.PRIVACY_DROP_PROVENANCE,
             ns.PRIVACY_NAME_MODE)
    try:
        ns.PRIVACY_NAME_MODE = "redact"
        assert ns.privacy_active()
        row = {"Applicant": "Jane Smith", "Respondent": "QBE", "Employer Name": ""}
        out = ns.apply_privacy_to_row(row)
        assert out["Applicant"] == "[REDACTED]" and out["Respondent"] == "[REDACTED]"
        assert out["Employer Name"] == ""  # empty stays empty
        # hash mode is stable
        ns.PRIVACY_NAME_MODE = "hash"
        h1 = ns.apply_privacy_to_row({"Applicant": "Jane Smith"})["Applicant"]
        h2 = ns.apply_privacy_to_row({"Applicant": "Jane Smith"})["Applicant"]
        assert h1 == h2 and h1.startswith("name_") and "Jane" not in h1
        # drop DOB + provenance in sidecar
        ns.PRIVACY_NAME_MODE = "keep"
        ns.PRIVACY_DROP_DOB = True
        ns.PRIVACY_DROP_PROVENANCE = True
        sc = ns.apply_privacy_to_sidecar(
            {"provenance": {"wpi_quote": "x"}, "field_review": {"date_of_birth": "1980-01-01"}})
        assert sc["provenance"] == {} and sc["field_review"]["date_of_birth"] == ""
    finally:
        (ns.PRIVACY_DROP_IDENTITY, ns.PRIVACY_DROP_DOB, ns.PRIVACY_DROP_PROVENANCE,
         ns.PRIVACY_NAME_MODE) = saved


# ----------------------------------------------------------------------
# Integration: report generation + Excel filter (ISSUE-019/001/005)
# ----------------------------------------------------------------------

def _seed_cache_row(url, **over):
    return ns.build_result_record(f"Case {url}", url, status="ok", **over)


def test_regenerate_reports_always_writes_and_filters_schema():
    import os, csv, json, tempfile
    d = tempfile.mkdtemp()
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cache = {}
        # current-schema, analysis-ready CTP row
        cache["u1"] = _seed_cache_row("u1", **{
            "Case Type": "CTP", "Decision Date": "2024-03-01",
            "Lump Sum": "200000", "Impairment % (Accepted)": "14"})
        # current-schema but not analysis-ready (bad date)
        cache["u2"] = _seed_cache_row("u2", **{"Decision Date": "Unknown"})
        # stale schema -> excluded entirely
        stale = _seed_cache_row("u3", **{"Decision Date": "2024-01-01"})
        stale["_schema_version"] = 1
        cache["u3"] = stale

        det, rdy = "d.csv", "r.csv"
        all_data, ready = ns.regenerate_reports_from_cache(cache, det, rdy, script="test")
        assert os.path.exists(det) and os.path.exists(rdy)
        urls = {r["URL"] for r in all_data}
        assert urls == {"u1", "u2"}          # stale u3 excluded
        assert {r["URL"] for r in ready} == {"u1"}  # only the analysis-ready one
        # manifest written with counts
        man = json.load(open("run_manifest.json", encoding="utf-8"))
        assert man["total_rows"] == 2 and man["analysis_ready_rows"] == 1
        assert man["stale_rows_excluded"] == 1

        # Empty cache -> header-only CSVs (ISSUE-001), no crash
        ns.regenerate_reports_from_cache({}, det, rdy, script="test")
        with open(rdy, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 and rows[0][0] == "Case Name"  # header only
    finally:
        os.chdir(cwd)


def test_ctp_excel_requires_accepted_wpi():
    import pandas as pd
    import ctp_lump_sum_impairment as ctp
    df = pd.DataFrame([
        {"URL": "a", "Case Type": "CTP", "Analysis Ready": "Yes",
         "Lump Sum": "200000", "Impairment % (Accepted)": "14"},   # kept
        {"URL": "b", "Case Type": "CTP", "Analysis Ready": "Yes",
         "Lump Sum": "150000", "Impairment % (Accepted)": ""},     # dropped: no WPI
    ])
    out, _ = ctp.build_workbook(df, {})
    assert list(out["URL"]) == ["a"]          # row b excluded (ISSUE-005)
    assert "WPI %" in out.columns and "Impairment % (Accepted)" not in out.columns


# ----------------------------------------------------------------------
# Mocked LLM retry / quota / timeout (ISSUE-020)
# ----------------------------------------------------------------------

class _APITimeoutError(Exception):
    pass


def _fake_completion(parsed_obj):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed_obj))],
        usage=None,
    )


class _FakeParseAPI:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = 0

    def parse(self, **kw):
        b = self.behaviors[min(self.calls, len(self.behaviors) - 1)]
        self.calls += 1
        if isinstance(b, Exception):
            raise b
        return _fake_completion(b)


def _extractor_with(behaviors, monkeypatch_sleep=True):
    ex = ns.LLMExtractor.__new__(ns.LLMExtractor)  # bypass real OpenAI init
    ex.client = SimpleNamespace(beta=SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeParseAPI(behaviors))))
    return ex


def test_parse_retries_quota_then_succeeds():
    saved = ns.time.sleep
    ns.time.sleep = lambda *a, **k: None
    try:
        quota = Exception("Error code: 429 insufficient_quota: exceeded your current quota")
        ex = _extractor_with([quota, quota, "OK"])
        parsed, usage, err = ex._parse_with_retry("sys", "user", str)
        assert parsed == "OK" and err is None
        assert ex.client.beta.chat.completions.calls == 3
    finally:
        ns.time.sleep = saved


def test_parse_retries_timeout_then_succeeds():
    saved = ns.time.sleep
    ns.time.sleep = lambda *a, **k: None
    try:
        ex = _extractor_with([_APITimeoutError("request timed out"), "OK"])
        parsed, usage, err = ex._parse_with_retry("sys", "user", str)
        assert parsed == "OK" and err is None
    finally:
        ns.time.sleep = saved


def test_parse_non_retryable_returns_error_immediately():
    ex = _extractor_with([Exception("schema validation failed: bad field")])
    parsed, usage, err = ex._parse_with_retry("sys", "user", str)
    assert parsed is None and "schema validation" in err
    assert ex.client.beta.chat.completions.calls == 1  # no retry


def test_quota_breaker_trips_after_threshold_when_cold():
    b = ns.QuotaCircuitBreaker(threshold=2, cold_window_seconds=0)
    assert not b.is_aborted()
    b.record_quota_error()
    assert not b.is_aborted()       # below threshold
    b.record_quota_error()
    assert b.is_aborted()           # threshold reached + cold window 0
    b2 = ns.QuotaCircuitBreaker(threshold=2, cold_window_seconds=0)
    b2.record_quota_error()
    b2.record_success()             # success resets the consecutive count
    b2.record_quota_error()
    assert not b2.is_aborted()


def test_is_transient_api_error():
    assert ns._is_transient_api_error(_APITimeoutError("x"))
    assert ns._is_transient_api_error(Exception("Connection error."))
    assert not ns._is_transient_api_error(Exception("bad schema"))


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
