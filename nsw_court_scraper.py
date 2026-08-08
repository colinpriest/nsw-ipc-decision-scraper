import os
import time
import csv
import io
import logging
import json
import re
import shutil
from enum import Enum
from typing import List, Literal
import requests
from curl_cffi import requests as cf_requests
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import datetime
import random
from threading import Lock
from collections import defaultdict
import hashlib

# Damages-breakdown pass (downstream spec 2026-07-27). Lives in its own module
# and imports nothing from here, so the dependency runs one way only.
from wpi_resolution import (
    WPI_FIELDS,
    to_pct as to_float_pct,
    WPI_RESOLUTION_ENABLED,
    WPI_RESOLUTION_REASONING_EFFORT,
    WPI_SYSTEM_INSTRUCTION,
    WpiResolution,
    empty_wpi_row,
    classify_split_wpi_absence,
    derive_threshold_finding,
    governing_system,
    nel_paid_without_entitlement,
    nel_threshold_consistency,
    resolve_wpi,
    BodySystemEnum,
    NelConsistencyEnum,
    GoverningSystemEnum,
    to_pct,
    ThresholdBasisEnum,
    ThresholdFindingEnum,
    WpiProvenanceEnum,
)
from damages_extraction import (
    PROVENANCE_ABSENCE,
    damages_residual,
    refine_money_absence,
    DAMAGES_CASE_TYPES,
    DAMAGES_FIELDS,
    DAMAGES_PASS_ENABLED,
    DAMAGES_REASONING_EFFORT,
    DAMAGES_SYSTEM_INSTRUCTION,
    DamagesSchema,
    build_damages_context,
    compose_description_with_figures,
    damages_row_from_parsed,
    empty_damages_row,
)

# ----------------------------------------------------------------------
# Output layout
# ----------------------------------------------------------------------
#
# Everything this pipeline GENERATES lives under one folder so the repo root
# holds source only. Source inputs (the downloaded decisions) stay where they
# are — they are scraped material, not output, and moving 6,000+ files buys
# nothing. Override the location with NSW_OUTPUT_DIR.
OUTPUT_ROOT = os.getenv("NSW_OUTPUT_DIR", "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

DECISIONS_DIR = os.getenv("NSW_DECISIONS_DIR", "nsw_pic_decisions")

CACHE_FILE = os.path.join(OUTPUT_ROOT, "processed_cache.json")
SIDECAR_FILE = os.path.join(OUTPUT_ROOT, "processed_sidecar.json")
CSV_REPORT = os.path.join(OUTPUT_ROOT, "detailed_payout_summary.csv")
ANALYSIS_READY_REPORT = os.path.join(OUTPUT_ROOT, "analysis_ready_payout_summary.csv")
WORKBOOK_FILE = os.path.join(OUTPUT_ROOT, "ctp_impairment_lump_sum.xlsx")
LOG_FILE = os.path.join(OUTPUT_ROOT, "scraper.log")

# Artifacts written before the output folder existed. Moved on import rather
# than left behind: the cache represents thousands of dollars of extraction, and
# silently starting from an empty one because the path moved would be the worst
# possible failure mode. Runs BEFORE logging is configured, because the log file
# is one of the things being moved.
_LEGACY_OUTPUT_FILES = (
    "processed_cache.json", "processed_sidecar.json",
    "detailed_payout_summary.csv", "analysis_ready_payout_summary.csv",
    "ctp_impairment_lump_sum.xlsx", "run_manifest.json",
    "austlii_data_errors.json", "scraper.log",
)


def migrate_legacy_outputs(root=OUTPUT_ROOT, names=_LEGACY_OUTPUT_FILES):
    """Move pre-`output/` artifacts from the working directory into `root`.

    Never overwrites: if the file already exists in `root` the legacy copy is
    left alone for the operator to reconcile. Returns the list moved.
    """
    moved = []
    for name in names:
        dest = os.path.join(root, name)
        if os.path.exists(name) and not os.path.exists(dest):
            try:
                shutil.move(name, dest)
                moved.append(name)
            except OSError as e:  # pragma: no cover - filesystem dependent
                print(f"WARNING: could not move {name} to {dest}: {e}")
    return moved


_MIGRATED_OUTPUTS = migrate_legacy_outputs()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

if _MIGRATED_OUTPUTS:
    logging.info(f"Moved {len(_MIGRATED_OUTPUTS)} legacy output file(s) into "
                 f"{OUTPUT_ROOT}/: {', '.join(_MIGRATED_OUTPUTS)}")

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Bump SCHEMA_VERSION when the cached extraction shape changes. Rows with a
# mismatched _schema_version are re-processed.
#
# v3: gender enum; age split into at-injury / at-decision + DOB cross-check;
#     weekly-income basis + numeric coercion; NEL/FEL status enums; location
#     locality/state; per-field provenance; live WPI reconciliation; field-loss
#     gate with focused second pass; Needs Review.
SCHEMA_VERSION = 3

RESULT_FIELDS = [
    # Identity
    "Case Name", "URL", "File Saved",
    # Decision metadata
    "Jurisdiction", "Case Type", "Decision Date", "Injury Date",
    "Applicant", "Respondent",
    # Claimant info
    "Claimant Age", "Claimant Age At Decision",
    "Claimant Gender", "Claimant Occupation",
    "Claimant Weekly Income", "Claimant Weekly Income Basis",
    "Employer Name", "Accident/Injury Location",
    "Location Locality", "Location State",
    # Outcome
    "Claimant Outcome",
    "Impairment %", "Lump Sum", "Weekly Benefit",
    "Non-Economic Loss", "Non-Economic Loss Status",
    "Future Economic Loss", "Future Economic Loss Status",
    "Statutory Benefits",
    "Medical Costs",
    "Nature", "Result", "Description", "Banded Description", "Catchwords",
    "Impairment % (Accepted)",
    # Ordinal scores
    "Injury Burden Intensity", "Psychological Injury Emphasis",
    "Liability Clarity", "Causation Complexity", "Treatment Burden",
    "Work Impact Severity", "Pre-existing Condition Salience",
    "Legal Procedural Complexity",
    # Regulatory
    "Regulatory Sections",
    # Status
    "Status", "LLM Error",
    "Needs Review", "Review Notes",
    "Analysis Ready", "Analysis Exclusion Reason",
] + list(WPI_FIELDS) + list(DAMAGES_FIELDS)

# The eight high-value target fields the extraction must capture reliably.
# Used by the field-loss gate, the focused second pass, and coverage metrics.
KEY_TARGET_FIELDS = (
    "Impairment % (Accepted)",
    "Non-Economic Loss",
    "Claimant Weekly Income",
    "Future Economic Loss",
    "Claimant Age",
    "Claimant Gender",
    "Claimant Occupation",
    "Accident/Injury Location",
)

# Keys stored on cached rows but excluded from the flat CSV output.
SIDECAR_KEYS = (
    "_narrative",            # dict of narrative sub-fields (incl. submissions)
    "_slices",               # dict of catchwords / determinations / introduction
    "_key_paragraphs",       # list of {paragraph_number, rationale, text}
    "_event_history",        # list of {date, actor, tag}
    "_schema_version",
    "_token_usage",          # last extraction's token usage
    "_banding_validation",   # banded_case_description validation result
    "_provenance",           # per-field verbatim source quotes (A2)
    "_field_review",         # field-loss gate issues + second-pass record (A1/A6)
    "_damages",              # damages-pass quotes/issues/derivation detail
    "_wpi_resolution",       # classified WPI mentions + resolution ladder detail
    "_wpi_version",          # WPI-resolution pass version
    "_wpi_pre_value",        # WPI before the resolution pass first ran
    "_damages_version",      # damages-pass schema version (see DAMAGES_VERSION)
)

# Version of the damages-breakdown pass. Bumped independently of
# SCHEMA_VERSION so adding/changing damages fields does NOT invalidate every
# cached row (and so does not empty the reports, which only carry
# current-SCHEMA_VERSION rows). backfill_damages_breakdown.py targets rows whose
# _damages_version is below this.
DAMAGES_VERSION = 1

# Version of the WPI-resolution pass (see wpi_resolution.py). Independent of
# SCHEMA_VERSION for the same reason as DAMAGES_VERSION.
WPI_VERSION = 2

MODEL = "gpt-5"
REASONING_EFFORT = "low"
# Reasoning effort for the focused second pass (A6) — bumped above the main
# pass because it only fires on the handful of fields the first pass missed.
FOCUSED_REASONING_EFFORT = os.getenv("NSW_FOCUSED_REASONING_EFFORT", "medium")
# Toggle the focused second pass (A6). Default on; set to 0 to disable (e.g. to
# bound cost on a large backfill).
FOCUSED_SECOND_PASS_ENABLED = os.getenv("NSW_FOCUSED_SECOND_PASS", "1") not in ("0", "false", "False", "")
DEFAULT_WORKERS = 25
# Single-pass char budget. gpt-5's context is far larger than this; the cap
# exists only to bound cost on rare very long decisions. Raised from 100k and
# made env-configurable so most decisions are sent whole (no truncation loss).
SINGLE_PASS_LIMIT_CHARS = int(os.getenv("NSW_SINGLE_PASS_LIMIT_CHARS", "200000"))

# Pricing (USD per 1M tokens) — override via env if needed
PRICE_INPUT_PER_M = float(os.getenv("GPT5_PRICE_INPUT_PER_M", "1.25"))
PRICE_CACHED_INPUT_PER_M = float(os.getenv("GPT5_PRICE_CACHED_INPUT_PER_M", "0.125"))
PRICE_OUTPUT_PER_M = float(os.getenv("GPT5_PRICE_OUTPUT_PER_M", "10.00"))

# Application-level timeout (seconds) for each OpenAI request (ISSUE-016). The
# SDK's own retries are disabled (max_retries=0) so our _parse_with_retry owns
# the retry/backoff policy; a stalled call fails after this deadline instead of
# occupying a worker thread indefinitely.
OPENAI_TIMEOUT = float(os.getenv("NSW_OPENAI_TIMEOUT", "120"))

# Worker-count bounds (ISSUE-018).
MAX_WORKERS = int(os.getenv("NSW_MAX_WORKERS", "64"))

# Operational artifacts.
RUN_MANIFEST_FILE = os.getenv("NSW_RUN_MANIFEST", os.path.join(OUTPUT_ROOT, "run_manifest.json"))
DATASET_LOCK_PATH = os.getenv("NSW_DATASET_LOCK", os.path.join(OUTPUT_ROOT, ".nsw_dataset.lock"))


# ----------------------------------------------------------------------
# Privacy / data-handling configuration (ISSUE-011)
# ----------------------------------------------------------------------
#
# DEFAULTS RETAIN EVERYTHING. This pipeline processes PUBLIC NSW Personal Injury
# Commission decisions, and DOB, per-field provenance quotes, and party identity
# are VITAL to the payout-vs-WPI use case — so by default nothing is dropped,
# hashed, or redacted. The knobs below let a privacy-sensitive deployment
# minimise the DERIVED OUTPUTS (CSV reports, sidecar, Excel) without changing
# default behaviour; the local working cache always retains full data.
#
# THIRD-PARTY PROCESSING NOTICE: decision text is sent to the OpenAI API for
# extraction. See the "Privacy & data handling" section of README.md.
def _flag(name, default="0"):
    return os.getenv(name, default).strip().lower() not in ("", "0", "false", "no", "off")

PRIVACY_DROP_IDENTITY = _flag("NSW_PRIVACY_DROP_IDENTITY")      # blank Applicant/Respondent/Employer in outputs
PRIVACY_DROP_DOB = _flag("NSW_PRIVACY_DROP_DOB")               # blank date-of-birth in sidecar outputs
PRIVACY_DROP_PROVENANCE = _flag("NSW_PRIVACY_DROP_PROVENANCE")  # drop verbatim provenance quotes in sidecar outputs
PRIVACY_NAME_MODE = os.getenv("NSW_PRIVACY_NAME_MODE", "keep").strip().lower()  # keep | hash | redact
PRIVACY_HASH_SALT = os.getenv("NSW_PRIVACY_HASH_SALT", "")

IDENTITY_NAME_FIELDS = ("Applicant", "Respondent", "Employer Name")


def privacy_active():
    return (PRIVACY_DROP_IDENTITY or PRIVACY_DROP_DOB or PRIVACY_DROP_PROVENANCE
            or PRIVACY_NAME_MODE in ("hash", "redact"))


def privacy_summary():
    return {
        "drop_identity": PRIVACY_DROP_IDENTITY,
        "drop_dob": PRIVACY_DROP_DOB,
        "drop_provenance": PRIVACY_DROP_PROVENANCE,
        "name_mode": PRIVACY_NAME_MODE,
    }


def _transform_name(value):
    s = str(value or "")
    if not s.strip() or PRIVACY_NAME_MODE == "keep":
        return s
    if PRIVACY_NAME_MODE == "redact":
        return "[REDACTED]"
    if PRIVACY_NAME_MODE == "hash":
        h = hashlib.sha256((PRIVACY_HASH_SALT + s).encode("utf-8")).hexdigest()[:12]
        return f"name_{h}"
    return s


def apply_privacy_to_row(row):
    """Return a (possibly transformed) flat CSV row per the privacy config.
    No-op unless a privacy knob is set; defaults keep all fields."""
    if not privacy_active():
        return row
    r = dict(row)
    for f in IDENTITY_NAME_FIELDS:
        if r.get(f):
            r[f] = "" if PRIVACY_DROP_IDENTITY else _transform_name(r[f])
    return r


def apply_privacy_to_sidecar(entry):
    """Return a (possibly transformed) sidecar entry per the privacy config."""
    if not privacy_active():
        return entry
    e = dict(entry)
    if PRIVACY_DROP_PROVENANCE:
        e["provenance"] = {}
        # The damages pass stores verbatim source snippets too.
        dmg = dict(e.get("damages") or {})
        if "quotes" in dmg:
            dmg["quotes"] = {}
        e["damages"] = dmg
    if PRIVACY_DROP_DOB:
        fr = dict(e.get("field_review") or {})
        if "date_of_birth" in fr:
            fr["date_of_birth"] = ""
        e["field_review"] = fr
    if PRIVACY_DROP_IDENTITY and e.get("Case Name"):
        e["Case Name"] = _transform_name(e["Case Name"])
    return e


# ----------------------------------------------------------------------
# Safe local-file path resolution (ISSUE-012)
# ----------------------------------------------------------------------

def safe_decision_path(output_dir, file_saved):
    """Resolve a cached `File Saved` value to a path INSIDE output_dir, or None.

    Rejects empty values, absolute paths, drive-qualified paths, and any value
    that escapes output_dir via traversal or alternate separators. The cache is
    a local trust boundary: a corrupted/edited/poisoned row must never let a
    script read (and then send to the LLM) files outside the decisions dir."""
    if not file_saved or not isinstance(file_saved, str):
        return None
    fs = file_saved.strip().replace("\\", "/")
    if not fs or fs.startswith("/") or ":" in fs or fs.startswith("~"):
        return None
    base = os.path.abspath(output_dir)
    candidate = os.path.abspath(os.path.join(base, fs))
    if candidate != base and not candidate.startswith(base + os.sep):
        return None
    return candidate


# ----------------------------------------------------------------------
# Worker-count validation (ISSUE-018)
# ----------------------------------------------------------------------

def get_worker_count(env_var="EXTRACTION_WORKERS", default=None):
    """Parse a worker count from env, clamped to [1, MAX_WORKERS]. Invalid or
    out-of-range values log a warning and fall back to the default."""
    if default is None:
        default = DEFAULT_WORKERS
    raw = os.getenv(env_var)
    if raw is None or str(raw).strip() == "":
        return max(1, min(default, MAX_WORKERS))
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        logging.warning(f"{env_var}={raw!r} is not an integer; using default {default}")
        return max(1, min(default, MAX_WORKERS))
    if n < 1:
        logging.warning(f"{env_var}={n} < 1; using 1")
        return 1
    if n > MAX_WORKERS:
        logging.warning(f"{env_var}={n} exceeds MAX_WORKERS={MAX_WORKERS}; clamping")
        return MAX_WORKERS
    return n


# ----------------------------------------------------------------------
# Cross-process dataset lock + atomic writes + run manifest (ISSUE-002/003/022)
# ----------------------------------------------------------------------

class _FileLock:
    """Best-effort advisory cross-process lock via an O_EXCL lock file.

    NOT reentrant — callers must not nest acquisitions in the same process.
    Save/report helpers therefore do NOT self-lock; call sites wrap a whole
    save+report batch in a single `with dataset_lock():`."""

    def __init__(self, lock_path, timeout=600, poll=0.5, stale_after=7200):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll = poll
        self.stale_after = stale_after

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"pid={os.getpid()} at={datetime.datetime.now().isoformat()}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock_path) > self.stale_after:
                        logging.warning(f"Removing stale dataset lock: {self.lock_path}")
                        os.remove(self.lock_path)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Could not acquire dataset lock {self.lock_path} within {self.timeout}s "
                        f"(another scraper/backfill run may be active)")
                time.sleep(self.poll)

    def __exit__(self, *exc):
        try:
            os.remove(self.lock_path)
        except OSError:
            pass
        return False


def dataset_lock(timeout=600):
    """Context manager: cross-process exclusion around shared cache/sidecar/
    report writes (ISSUE-003)."""
    return _FileLock(DATASET_LOCK_PATH, timeout=timeout)


def _per_proc_tmp(path):
    return f"{path}.tmp.{os.getpid()}"


def atomic_write_json(path, obj, **dump_kwargs):
    """Write JSON via a per-process temp file + os.replace (ISSUE-002/003)."""
    tmp = _per_proc_tmp(path)
    dump_kwargs.setdefault("indent", 2)
    dump_kwargs.setdefault("default", str)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **dump_kwargs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_csv(path, fieldnames, rows):
    """Write a CSV via a per-process temp file + os.replace. ALWAYS writes the
    header, even for an empty row set (ISSUE-001)."""
    tmp = _per_proc_tmp(path)
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows or [])
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def write_run_manifest(*, script, total_rows, analysis_ready_rows, needs_review_rows,
                       path=RUN_MANIFEST_FILE, **extra):
    """Write a small run-metadata manifest so consumers can detect stale/empty
    outputs (ISSUE-001/002/015/022)."""
    manifest = {
        "script": script,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total_rows": total_rows,
        "analysis_ready_rows": analysis_ready_rows,
        "needs_review_rows": needs_review_rows,
        "privacy": privacy_summary(),
    }
    manifest.update(extra)
    try:
        atomic_write_json(path, manifest)
        logging.info(f"Run manifest written to {path}: {total_rows} rows, "
                     f"{analysis_ready_rows} analysis-ready, {needs_review_rows} needs-review")
    except OSError as e:
        logging.error(f"Failed to write run manifest {path}: {e}")


def has_valid_iso_date(value):
    if not isinstance(value, str):
        return False
    return bool(ISO_DATE_PATTERN.fullmatch(value.strip()))


# Rows flagged Needs Review (a high-value field the gate believes is present in
# the source but the extraction did not capture, even after the focused second
# pass) are held out of the analysis-ready set so probable losses are triaged
# rather than silently consumed. Toggle with NSW_EXCLUDE_NEEDS_REVIEW=0.
EXCLUDE_NEEDS_REVIEW = os.getenv("NSW_EXCLUDE_NEEDS_REVIEW", "1") not in ("0", "false", "False", "")


def get_analysis_exclusion_reasons(row):
    reasons = []
    status = str(row.get("Status", "") or "").strip()
    llm_error = str(row.get("LLM Error", "") or "").strip()

    if status and status != "ok":
        reasons.append(status)
    elif llm_error:
        reasons.append("llm_error")

    if not has_valid_iso_date(row.get("Decision Date", "")):
        reasons.append("missing_decision_date")

    if EXCLUDE_NEEDS_REVIEW and str(row.get("Needs Review", "") or "").strip() == "Yes":
        reasons.append("needs_review")

    # NOTE: `_wpi_quarantined` is deliberately NOT an exclusion reason. A row
    # whose WPI was withheld under s 4.11 keeps its place in the analysis set
    # with `WPI %` blank — see quarantine_impossible_wpi.
    return reasons


def annotate_analysis_fields(row):
    annotated = dict(row)
    reasons = get_analysis_exclusion_reasons(annotated)
    annotated["Analysis Ready"] = "Yes" if not reasons else "No"
    annotated["Analysis Exclusion Reason"] = "; ".join(reasons)
    return annotated


def ensure_damages_defaults(row):
    """Fill the damages columns on a row extracted before the damages pass
    existed. A blank in `Past Economic Loss` would read as a genuine zero; the
    defaults say 'Not addressed' / 'absent' / 'not run' instead, which is what
    is actually true of those rows."""
    if row.get("Damages Extraction Status"):
        return row
    filled = empty_damages_row(status="not run")
    filled.update(empty_wpi_row(row.get("Impairment % (Accepted)", "")))
    filled.update({k: v for k, v in row.items() if v not in (None, "")})
    return filled


def build_result_record(title, url, file_saved="", status="", llm_error="", **overrides):
    row = {field: "" for field in RESULT_FIELDS}
    row.update({
        "Case Name": title,
        "URL": url,
        "File Saved": file_saved,
        "Status": status,
        "LLM Error": llm_error,
        "Needs Review": "No",
    })
    # Sidecar defaults
    row["_schema_version"] = SCHEMA_VERSION
    row["_narrative"] = {}
    row["_slices"] = {}
    row["_key_paragraphs"] = []
    row["_event_history"] = []
    row["_token_usage"] = {}
    row["_provenance"] = {}
    row["_field_review"] = {}
    row["_damages"] = {}
    row["_damages_version"] = 0
    # Damages columns default to the honest "we have not looked" state rather
    # than blanks that read as genuine zeros.
    row.update(empty_damages_row(status="not run"))
    row.update(empty_wpi_row())
    row.update(overrides)
    return annotate_analysis_fields(row)

class JurisdictionEnum(str, Enum):
    NSW = "NSW"
    VIC = "VIC"
    QLD = "QLD"
    WA = "WA"
    SA = "SA"
    TAS = "TAS"
    ACT = "ACT"
    NT = "NT"
    FEDERAL = "Federal"

class ClaimantOutcomeEnum(str, Enum):
    FOR_CLAIMANT = "For Claimant"
    AGAINST_CLAIMANT = "Against Claimant"

class CaseCategoryEnum(str, Enum):
    WORKERS_COMPENSATION = "Workers Compensation"
    CTP = "CTP"
    OTHER = "Other"

class MedicalCostsEnum(str, Enum):
    YES = "Yes"
    NO = "No"
    NA = "N/A"

class GenderEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"
    NOT_STATED = "Not stated"

class QuantumStatusEnum(str, Enum):
    """Disposition of a damages head, kept separate from its dollar amount so
    'explicitly denied' (Nil) is never conflated with 'not dealt with in this
    decision' (Not addressed)."""
    AWARDED = "Awarded"
    NIL = "Nil"
    NOT_ADDRESSED = "Not addressed"


class QuantumProvenance(BaseModel):
    """Per-field verbatim source snippets supporting the eight high-value
    target fields. Used to (a) audit the extraction, (b) detect silent losses
    (a non-empty quote alongside an empty value is a contradiction), and
    (c) make the income normalisation checkable. Each quote is a VERBATIM
    substring copied from the source (<=200 chars) that ESTABLISHES the value;
    empty string when the field is genuinely 'Not stated' / Nil / Not addressed
    (do NOT quote text that merely proves the field is absent)."""
    wpi_quote: str = Field(description="Verbatim source text stating the WPI %, e.g. '...assessed at 14% whole person impairment...'. Empty if no WPI is stated anywhere.")
    non_economic_loss_quote: str = Field(description="Verbatim source text stating non-economic loss / general damages / pain and suffering. Empty if not addressed.")
    weekly_income_quote: str = Field(description="Verbatim source text stating the income figure used (e.g. PIAWE, weekly wage, salary). Empty if no income figure appears.")
    future_economic_loss_quote: str = Field(description="Verbatim source text stating future economic loss / buffer / loss of future earning capacity. Empty if not addressed.")
    age_quote: str = Field(description="Verbatim source text stating the claimant's age or date of birth. Empty if neither appears.")
    gender_quote: str = Field(description="Verbatim source text indicating the claimant's gender (explicit statement or clear gendered reference). Empty if not determinable.")
    occupation_quote: str = Field(description="Verbatim source text stating the claimant's occupation. Empty if not stated.")
    location_quote: str = Field(description="Verbatim source text stating where the injury/accident occurred. Empty if not stated.")


class SliceLocator(BaseModel):
    present: bool = Field(description="True if this section is identifiable in the source.")
    start_marker: str = Field(description=(
        "30-80 char VERBATIM substring from the start of this section, copied "
        "exactly. Empty if present=False."
    ))
    end_marker: str = Field(description=(
        "30-80 char VERBATIM substring from the end of this section, copied "
        "exactly. Empty if present=False."
    ))


class KeyParagraph(BaseModel):
    paragraph_number: int = Field(description=(
        "Numbered paragraph in the source decision (e.g. 64 for the paragraph "
        "that begins '64. '). Do NOT return paragraph text — only the integer "
        "number."
    ))
    rationale: str = Field(description="One short sentence on why this paragraph is analytically important.")


class HistoryEvent(BaseModel):
    date: str = Field(description=(
        "Date of the event in DD-Mon-YYYY format (e.g. '25-Apr-2024'). Use "
        "'Unknown' if the source gives only a vague reference."
    ))
    actor: str = Field(description=(
        "Who initiated or performed this event. Short lowercase string naming "
        "the most specific accurate stakeholder. Examples: 'claimant', "
        "'insurer', 'employer', 'Commission', 'Member', 'treating GP', "
        "'treating surgeon', 'treating psychiatrist', 'IME doctor', 'Medical "
        "Assessor', 'claimant solicitor', 'insurer solicitor', 'police', "
        "'ambulance', 'hospital', 'regulator'. If unclear, use 'unknown'."
    ))
    tag: str = Field(description=(
        "Short lowercase tag describing the event. Examples: 'accident', "
        "'injury', 'GP visit', 'hospital admission', 'surgery', 'imaging', "
        "'IME', 'WPI assessment', 'return to work', 'claim lodged', "
        "'liability notice', 'internal review', 'insurer offers settlement', "
        "'insurer denies liability', 'application filed', 'reply filed', "
        "'submission filed', 'teleconference', 'hearing', 'decision'."
    ))


class CombinedSchema(BaseModel):
    # ---- Structured facts ----
    applicant_name: str
    respondent_name: str
    claimant_outcome: ClaimantOutcomeEnum
    case_type: CaseCategoryEnum
    claimant_age: str = Field(description=(
        "Claimant's age AT THE TIME OF INJURY/ACCIDENT, as a number. "
        "Examples: '47', '21'. DERIVE it whenever possible:\n"
        "  - From year of birth: age = injury_year - birth_year.\n"
        "  - From a stated CURRENT age (age at the decision/assessment): "
        "age_at_injury = current_age - (decision_year - injury_year). E.g. 'the "
        "claimant is now 31' in a 2021 decision for a 2018 injury => '28'.\n"
        "'Not stated' ONLY if there is no age, no year-of-birth, and no current "
        "age anywhere in the decision."
    ))
    claimant_age_at_decision: str = Field(description=(
        "Claimant's age AT THE TIME OF THE DECISION/ASSESSMENT, as a number. Use "
        "a stated current age directly; otherwise derive it (from date-of-birth "
        "and the decision date, or as age_at_injury + (decision_year - "
        "injury_year)). 'Not stated' if it cannot be determined. May equal "
        "claimant_age when injury and decision fall in the same year."
    ))
    claimant_date_of_birth: str = Field(description=(
        "Claimant's date of birth as YYYY-MM-DD if a full date is stated; or the "
        "4-digit year alone (e.g. '1995') if only the year is given. 'Not stated' "
        "if no birth date or birth year appears. Used to cross-check the ages."
    ))
    claimant_gender: GenderEnum = Field(description=(
        "Claimant's gender. Choose Male/Female/Other from an explicit statement "
        "OR from unambiguous gendered references in the decision (consistent use "
        "of he/his or she/her for the claimant, 'Mr'/'Ms'/'Mrs'). Use 'Not "
        "stated' only when gender is genuinely indeterminable."
    ))
    claimant_occupation: str = Field(description="Claimant's occupation at time of injury (e.g. 'bus driver', 'registered nurse', 'senior laboratory technician'). 'Not stated' if absent.")
    claimant_weekly_income: str = Field(description=(
        "A SINGLE NOMINAL NUMBER representing the claimant's total weekly "
        "employment income (e.g. '1722.08'). NO words, NO commentary, NO "
        "qualifiers, NO $ or commas — just digits and one decimal point.\n"
        "Conversion rules:\n"
        "  - If only a weekly figure is stated, use it.\n"
        "  - If components are stated separately (e.g. salary plus commissions, "
        "or hourly plus hours), normalise to a TOTAL WEEKLY number: monthly / "
        "(52/12), annual / 52, hourly * weekly_hours, etc. Sum all components.\n"
        "  - If both pre-injury (PIAWE) and current/post-accident figures are "
        "stated, prefer the PRE-INJURY figure (it represents the claimant's "
        "earning capacity before the injury).\n"
        "  - If net and gross are both stated, prefer GROSS.\n"
        "Use 'Not stated' (literally) only if NO weekly-or-convertible income "
        "figure of any kind appears in the decision."
    ))
    claimant_weekly_income_basis: str = Field(description=(
        "Short label describing WHAT claimant_weekly_income represents and how it "
        "was derived, so the figure is auditable. State: the source measure "
        "(PIAWE / pre-injury weekly / post-injury weekly / current earnings), "
        "gross vs net, and the original period if you converted it (e.g. 'PIAWE, "
        "gross, weekly as stated'; 'gross annual salary 89500 / 52'; 'hourly 32.50 "
        "x 38h'). 'Not stated' if no income figure was found."
    ))
    employer_name: str = Field(description="Employer's legal name (workers compensation only). 'Not applicable' for CTP / non-employment cases. 'Not stated' if WC case but employer not named.")
    location_of_accident_or_injury: str = Field(description="Where the injury occurred — for CTP: road/intersection/town; for WC: workplace address/town. 'Not stated' if absent.")
    location_locality: str = Field(description="The town/suburb/locality of the accident or injury, normalised (e.g. 'Parramatta', 'Wagga Wagga'). 'Not stated' if no locality is identifiable.")
    location_state: str = Field(description="The state/territory of the accident or injury as an abbreviation (NSW/VIC/QLD/WA/SA/TAS/ACT/NT). Default 'NSW' for NSWPIC matters unless the decision clearly places the injury in another state. 'Not stated' only if genuinely indeterminable.")
    impairment_percentage: str = Field(description=(
        "Whole Person Impairment percentage when a binding assessment is made "
        "IN THIS PROCEEDING (e.g. '15'). LEAVE EMPTY if the decision merely "
        "accepts a prior Medical Assessor's certificate, or remits the matter "
        "to a medical assessor, or allows reassessment. (This is the strict "
        "'made-here' value — the lenient value goes in impairment_percentage_accepted.)"
    ))
    impairment_percentage_accepted: str = Field(description=(
        "Whole Person Impairment percentage RELIED ON BY THE TRIBUNAL for the "
        "award, regardless of whether the assessment was made in this proceeding "
        "or in a prior MAS certificate that the Member accepts. For CTP "
        "settlement approvals under MAI Act 2017 s 6.23 and damages assessments "
        "under s 7.36, this is almost always present and is the WPI underpinning "
        "the lump sum. Use the COMBINED / TOTAL value the Member relied on; if "
        "multiple component WPIs are stated for different body parts and no "
        "combined value, use the highest single value. EMPTY only if no numeric "
        "WPI appears anywhere in the decision. "
        "IMPORTANT: do NOT emit '0' as a WPI value — a true 0% finding is "
        "extremely rare and would defeat the lump sum entitlement. Where the "
        "source mentions '0%' it is almost always reciting the statutory "
        "definition of a 'minor injury' or the threshold (s 1.6 / s 4.11). "
        "Leave EMPTY in those cases."
    ))
    lump_sum_amount: str = Field(description="Total lump sum awarded for COMPENSATION (s 66 WPI or settlement principal). Regulated costs orders do NOT belong here. Empty if none.")
    weekly_benefit_amount: str = Field(description="Weekly benefit amount as a NOMINAL NUMBER (e.g. '540.50'). If multiple periods, use the LATEST. NET amount if deductions quantified.")
    non_economic_loss: str = Field(description="Damages for non-economic loss (pain and suffering) as a nominal number (digits only, no $ or commas). Leave EMPTY when not addressed or denied — the disposition goes in non_economic_loss_status.")
    non_economic_loss_status: QuantumStatusEnum = Field(description="Disposition of non-economic loss / general damages: 'Awarded' if a positive amount was allowed (put it in non_economic_loss); 'Nil' if a claim was made but explicitly refused/assessed at zero; 'Not addressed' if this head was not dealt with in the decision.")
    future_economic_loss: str = Field(description="Damages for future economic loss (incl. loss of future earnings/super and buffers) as a nominal number (digits only, no $ or commas). Leave EMPTY when not addressed or denied — the disposition goes in future_economic_loss_status.")
    future_economic_loss_status: QuantumStatusEnum = Field(description="Disposition of future economic loss: 'Awarded' if a positive amount/buffer was allowed (put it in future_economic_loss); 'Nil' if claimed but explicitly refused/assessed at zero; 'Not addressed' if not dealt with.")
    statutory_benefits: str = Field(description="Statutory benefits status (e.g. 'Weekly $522.84 from 28 Sept 2023 ongoing'; 'Terminated after 26 weeks'; 'Not addressed'). Short descriptive string.")
    medical_costs_awarded: MedicalCostsEnum = Field(description="Were medical costs explicitly awarded/ordered? 'Yes' if ordered. 'No' if explicitly denied. 'N/A' if not discussed or silent.")
    decision_nature: str = Field(description="PRIMARY category of the dispute. Simplify to one of: 'Liability Dispute', 'Permanent Impairment', 'Medical Dispute', 'Death Benefit', 'Damages', 'Settlement Approval', 'Statutory Benefits Dispute', 'Procedural'.")
    decision_result: str = Field(description="Short legal summary (e.g., 'Award for Applicant', 'Matter Remitted', 'Settlement Approved').")
    date_of_injury: str = Field(description="YYYY-MM-DD or 'Unknown'.")
    date_of_decision: str = Field(description="YYYY-MM-DD or 'Unknown'.")
    jurisdiction: JurisdictionEnum = Field(default=JurisdictionEnum.NSW, description="The legal jurisdiction of the decision.")
    regulatory_sections: List[str] = Field(description=(
        "Every statutory section the tribunal identified as APPLYING to this case "
        "(provisions the Member relied on, not mere mentions). Compact form, e.g. "
        "'s 60 Workers Compensation Act 1987', 's 3.11 Motor Accident Injuries Act "
        "2017'. Deduplicate."
    ))

    # ---- Ordinal score features ----
    injury_burden_intensity: Literal[0, 1, 2, 3, 4] = Field(description="0=minimal/soft-tissue; 1=moderate single; 2=significant single or multiple moderate; 3=severe/multiple significant; 4=catastrophic/permanent disability.")
    psychological_injury_emphasis: Literal[0, 1, 2] = Field(description="0=none mentioned; 1=secondary/minor; 2=primary/major psychological injury.")
    liability_clarity: Literal[0, 1, 2] = Field(description="0=contested/unclear; 1=partially contested; 2=clear/uncontested.")
    causation_complexity: Literal[0, 1, 2] = Field(description="0=simple/direct; 1=some complexity (e.g. minor pre-existing); 2=complex/disputed causation.")
    treatment_burden: Literal[0, 1, 2, 3] = Field(description="0=minimal/conservative; 1=moderate outpatient; 2=significant incl. surgery; 3=extensive/ongoing/multiple surgeries.")
    work_impact_severity: Literal[0, 1, 2, 3] = Field(description="0=no/minimal impact; 1=temporary partial; 2=prolonged partial or temporary total; 3=permanent incapacity.")
    pre_existing_condition_salience: Literal[0, 1, 2] = Field(description="0=none/not relevant; 1=mentioned but minor; 2=significant factor in assessment.")
    legal_procedural_complexity: Literal[0, 1, 2, 3] = Field(description="0=straightforward settlement approval; 1=minor disputes/clarifications; 2=moderate legal issues; 3=complex proceedings/appeals.")

    # ---- Long case description (500-700 words, ASCII, no proper nouns) ----
    case_description: str = Field(description=(
        "A comprehensive 500-700 word case summary in ONE PARAGRAPH (no internal "
        "newlines) covering: how the injury occurred (mechanism, date, location); "
        "specific injuries and diagnoses; treatment received and proposed; functional "
        "and economic impact; the key issue(s) in dispute; competing medical/expert "
        "evidence and which was preferred; the legal framework (sections and "
        "authorities); the tribunal's reasoning; the final outcome with specific "
        "amounts or orders. CRITICAL: do NOT include proper-noun identifiers of the "
        "parties or people involved — NO names of the claimant/applicant, the "
        "insurer or employer (use 'the insurer' / 'the employer'), the Member, "
        "treating doctors or experts (use 'a consultant psychiatrist', 'the treating "
        "GP', 'an occupational physician'), lawyers, hospitals (use 'a metropolitan "
        "public hospital'), or accounting firms. Cited case authorities (e.g. "
        "'Imbree v McNeilly') and statutory references (e.g. 's 60 Workers "
        "Compensation Act 1987') ARE preserved as legal references. Use ASCII only: "
        "straight quotes (not smart quotes), regular hyphens (not non-breaking or "
        "en/em-dashes), three dots (not ellipsis character)."
    ))
    banded_case_description: str = Field(description=(
        "A redacted version of case_description with numeric content REPLACED by "
        "band tokens to prevent target leakage in downstream models. Same prose, "
        "same paragraph, same anonymisation rules — only numbers change. "
        "Single paragraph, no newlines, ASCII only.\n\n"
        "REPLACE these categories with these EXACT tokens (lower-bound inclusive, "
        "upper-bound inclusive; boundary values go to the LOWER band):\n\n"
        "  CALENDAR DATES (any specific date like '22 July 2020', '9 July 2021', "
        "'18-Nov-2020') -> [DATE]\n"
        "  Keep relative durations VERBATIM: '12 weeks', '4 days per week', "
        "'26 weeks', '3 years', 'about three weeks in hospital', 'nearly three "
        "years'. These do not leak.\n\n"
        "  CLAIMANT AGE (a stated age in years for the claimant):\n"
        "    [AGE_BAND:<18]  [AGE_BAND:18-24]  [AGE_BAND:25-34]  [AGE_BAND:35-44]\n"
        "    [AGE_BAND:45-54]  [AGE_BAND:55-64]  [AGE_BAND:65-74]  [AGE_BAND:75+]\n"
        "  Examples: 'aged 59' -> 'aged [AGE_BAND:55-64]'; '21-year-old' -> "
        "'[AGE_BAND:18-24]-year-old'; '64' -> '[AGE_BAND:55-64]'.\n\n"
        "  WHOLE PERSON IMPAIRMENT % (WPI):\n"
        "    [WPI_RANGE:0-5%]  [WPI_RANGE:6-10%]  [WPI_RANGE:11-15%]\n"
        "    [WPI_RANGE:16-20%]  [WPI_RANGE:21-30%]  [WPI_RANGE:31-50%]  [WPI_RANGE:>50%]\n"
        "  Examples: '11% WPI' -> '[WPI_RANGE:11-15%] WPI'; '7%' -> '[WPI_RANGE:6-10%]'.\n\n"
        "  TOTAL PAYOUT / SETTLEMENT / DAMAGES AWARD (the total dollar amount):\n"
        "    [PAYOUT_RANGE:<$50k]  [PAYOUT_RANGE:$50k-$150k]\n"
        "    [PAYOUT_RANGE:$150k-$300k]  [PAYOUT_RANGE:$300k-$500k]\n"
        "    [PAYOUT_RANGE:$500k-$1M]  [PAYOUT_RANGE:$1M-$2M]  [PAYOUT_RANGE:>$2M]\n"
        "  Examples: '$350,000 total settlement' -> '[PAYOUT_RANGE:$300k-$500k] total settlement'.\n\n"
        "  NON-ECONOMIC LOSS (pain and suffering damages component):\n"
        "    [NEL_RANGE:<$50k]  [NEL_RANGE:$50k-$150k]\n"
        "    [NEL_RANGE:$150k-$300k]  [NEL_RANGE:$300k-cap]\n"
        "  Examples: '$300,000 for non-economic loss' -> '[NEL_RANGE:$150k-$300k] "
        "for non-economic loss'.\n\n"
        "  FUTURE ECONOMIC LOSS / BUFFER (future earnings, superannuation, treatment time):\n"
        "    [FEL_RANGE:<$25k]  [FEL_RANGE:$25k-$75k]  [FEL_RANGE:$75k-$200k]\n"
        "    [FEL_RANGE:$200k-$500k]  [FEL_RANGE:>$500k]\n"
        "  Examples: '$50,000 future economic loss' -> '[FEL_RANGE:$25k-$75k] "
        "future economic loss'; '$45,000 buffer' -> '[FEL_RANGE:$25k-$75k] buffer'; "
        "'about $5,000 for superannuation' -> '[FEL_RANGE:<$25k] for superannuation'.\n\n"
        "  WEEKLY INCOME (claimant's pre- or post-accident weekly employment income):\n"
        "    [INCOME_WEEKLY:<$500]  [INCOME_WEEKLY:$500-$1000]\n"
        "    [INCOME_WEEKLY:$1000-$1500]  [INCOME_WEEKLY:$1500-$2500]\n"
        "    [INCOME_WEEKLY:>$2500]\n"
        "  Examples: '$800 net per week' -> '[INCOME_WEEKLY:$500-$1000]'; "
        "'PIAWE $1,134.68' -> 'PIAWE [INCOME_WEEKLY:$1000-$1500]'.\n\n"
        "  MONTHLY COMMISSION (variable monthly commission / bonus income):\n"
        "    [COMMISSION_MONTHLY:<$1k]  [COMMISSION_MONTHLY:$1k-$5k]\n"
        "    [COMMISSION_MONTHLY:$5k-$10k]  [COMMISSION_MONTHLY:>$10k]\n"
        "  Examples: '$4,000 per month commissions' -> '[COMMISSION_MONTHLY:$1k-$5k] "
        "per month commissions'.\n\n"
        "  REGULATED COSTS ORDERS (e.g. $3,762 regulated costs) -> use PAYOUT_RANGE.\n\n"
        "PRESERVE VERBATIM (do NOT band): statutory section numbers ('s 60', "
        "'section 3.11(1)', 's 6.25'); neutral case citations ('[2022] NSWPIC "
        "137', '[2008] NSWCA 246'); paragraph numbers ('para 64'); relative "
        "durations ('12 weeks', '4 days per week', '26 weeks'); counts ('83 "
        "questions', 'four days per week'); year-only references inside Act names "
        "('Workers Compensation Act 1987'); footnote markers ('[1]').\n\n"
        "Any other numeric value (e.g. number of doctors, hospital admission "
        "length in weeks, advisory speed signs) may stay verbatim if it doesn't "
        "fit a banding category."
    ))

    # ---- Narrative sub-fields ----
    claimant_profile: str = Field(description="Age, gender, occupation, employer, pre-injury health, prior claims. 60-150 words. 'Not stated' for unstated details.")
    accident_or_injury_mechanism: str = Field(description="When/where/how the injury occurred, witnesses, immediate aftermath. 60-150 words.")
    injuries_and_diagnoses: str = Field(description="Every injury, body part, diagnosis, severity, imaging findings. 60-150 words.")
    treatment_history: str = Field(description="Chronological treatment: ED, surgeries, therapy, medications, proposed future. 60-150 words.")
    functional_impact_and_work_capacity: str = Field(description="ADL impact, work capacity, restrictions, earning loss, RTW. 60-150 words.")
    medical_evidence_summary: str = Field(description="Named experts (specialty), opinions on causation/diagnosis/impairment, which preferred and why. 60-150 words.")
    previous_insurer_actions_and_offers: str = Field(description=(
        "Chronological history of the insurer's prior actions and offers BEFORE the "
        "current proceeding: liability admissions/denials, s 78 / s 287A notices, "
        "internal reviews, weekly-payment decisions and adjustments, settlement "
        "offers (amounts and dates), refusals to pay specific expenses, IME "
        "directions. Name specific dates, amounts, and notices. 60-150 words. "
        "'Not stated' if the decision does not detail any prior actions."
    ))
    claimant_submissions: str = Field(description="The claimant's substantive submissions — legal arguments, factual positions, authorities and statutory provisions relied upon. 60-150 words.")
    insurer_submissions: str = Field(description="The insurer's substantive submissions — legal arguments, factual positions, authorities and statutory provisions relied upon. 60-150 words.")
    legal_issues_and_reasoning: str = Field(description="Legal questions, the Member's reasoning, statutory sections and case authorities cited. 60-150 words.")

    # ---- Short verbatim slices ----
    catchwords: SliceLocator
    determinations_or_orders: SliceLocator
    introduction: SliceLocator

    # ---- Event history ----
    event_history: List[HistoryEvent] = Field(description=(
        "Chronological list of dated events: date of injury, treatment milestones "
        "(GP visits, surgeries, imaging, IMEs, WPI assessments), insurer actions "
        "(liability notices, internal reviews, settlement offers, denials), "
        "procedural milestones (claim lodged, application filed, submissions, "
        "teleconference, hearing, decision). Include EVERY dated event referenced. "
        "Sort chronologically."
    ))

    # ---- Key paragraph numbers (text cut from source in code) ----
    key_paragraphs: List[KeyParagraph] = Field(description=(
        "Identify 4-8 of the most analytically important numbered paragraphs in the "
        "source — typically from the Member's reasoning/findings section. Choose "
        "paragraphs containing statutory pin-cites, neutral case citations, the "
        "framing of the legal issue, and the conclusion. Return ONLY paragraph "
        "numbers and short rationales. The actual paragraph text will be cut from "
        "the source — do NOT return it."
    ))

    # ---- Per-field provenance for the eight high-value target fields ----
    provenance: QuantumProvenance = Field(description=(
        "For each of the eight high-value fields, the VERBATIM source snippet "
        "supporting your value (or empty string if the field is genuinely 'Not "
        "stated'). Copy text exactly from the source. A non-empty quote with an "
        "empty/Not-stated value is treated as a mistake, so only leave a quote "
        "empty when the fact truly does not appear in the decision."
    ))

# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------

_ASCII_REPLACEMENTS = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "--",
    "‑": "-", "‐": "-",
    "…": "...",
    "\xa0": " ",
    "′": "'", "″": '"',
    "·": "*", "•": "*",
}


def sanitise_case_description(text):
    """Single-paragraph ASCII-only normalisation for case_description."""
    if not text:
        return ""
    for k, v in _ASCII_REPLACEMENTS.items():
        text = text.replace(k, v)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Match a non-negative number. We do NOT allow a leading "-" because none of
# the LLM fields this coercer targets (age, money amounts, percentages) can
# validly be negative. A leading "-" inside the input is always part of a
# different token (e.g. the dash in "mid-80's" or a date like "12-Jan-2024"),
# never a real sign on the value we want.
_LEADING_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_NON_NUMERIC_SENTINELS = {
    "", "not stated", "not applicable", "n/a", "unknown",
    "nil", "none", "not addressed",
}


def coerce_leading_number(val):
    """Best-effort extraction of a non-negative numeric value from an
    LLM-emitted string.

    Returns "" for sentinels ('Not stated', etc.) and unparseable strings.
    Otherwise returns the first non-negative number found, as a string
    (e.g. '47', '47.5'). Examples:
        '47'                            -> '47'
        '21 at time of accident'        -> '21'
        '55 at time of accident; 58 ...' -> '55' (at-accident value)
        "mid-80's at time of approval"  -> '80'  (NOT '-80')
        'Not stated'                    -> ''
        'Late 20s'                      -> '20'
        'Mid-80s'                       -> '80'
    """
    s = str(val or "").strip().replace("$", "").replace(",", "")
    if s.lower() in _NON_NUMERIC_SENTINELS:
        return ""
    try:
        f = float(s)
        if f < 0:
            return ""  # negative LLM output shouldn't happen; reject
        return s
    except ValueError:
        pass
    m = _LEADING_NUM_RE.search(s)
    if not m:
        return ""
    return m.group(0)


def cleanup_text(text):
    """
    Normalise whitespace from BS4 HTML extraction. Treat blank lines as
    paragraph breaks; within a paragraph collapse all whitespace to single
    spaces.
    """
    if not text:
        return ""
    text = (text
            .replace("\xa0", " ")
            .replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"'))
    paragraphs = re.split(r"\n\s*\n+", text)
    cleaned = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            cleaned.append(p)
    return "\n\n".join(cleaned)


PARA_NUM_RE = re.compile(r"(?m)^(\d{1,3})\.\s+(?=\S)")

# Whole Person Impairment extraction from decision text. Used to backfill the
# Impairment % (Accepted) column when the main LLM left it empty because the
# decision merely *accepted* a prior MAS certificate (common pattern for CTP
# settlement approvals and damages assessments under MAI Act 2017).
#
# Strict-precision rule: only fill the field when the source contains exactly
# ONE distinct WPI number. Cases with multiple component WPIs (e.g. "head 5%,
# back 7%, total 12%") are deferred to focused LLM extraction — regex picking
# the wrong component happens ~20% of the time, which is unacceptable.
_WPI_FWD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*"
    r"(?:WPI\b|whole\s+person\s+impairment|permanent\s+impairment)",
    re.IGNORECASE,
)
_WPI_REV_RE = re.compile(
    r"(?:WPI\b|whole\s+person\s+impairment|permanent\s+impairment)"
    r"\s*(?:of|at|assessed\s+at|certif(?:ied|icate)?\s*(?:as|of|at)?)?\s*"
    r"(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

# Statutory-threshold framing. Under the MAI Act 2017 the >10% WPI bar gates
# non-economic loss, so "the statutory threshold of 10% whole person impairment"
# appears in countless settlement approvals where the claimant's ACTUAL WPI is 0%.
# Treating that 10% as a finding is a systematic false positive (e.g. Quick
# [2024] NSWPIC 93). A WPI token is a threshold mention — not a finding — when
# the ~60 chars before it contain threshold language. We then drop it from the
# candidate set so the case bails to focused-LLM extraction instead.
#
# A WPI token is threshold framing when any of these hold:
#   * "threshold"/"exceed(s/ed)" within ~60 chars BEFORE it
#     ("does not exceed the statutory threshold of 10% WPI");
#   * "threshold" within ~25 chars AFTER it
#     ("the 10% whole person impairment threshold");
#   * a floor phrase (greater/more than, at least, in excess of, not less than)
#     IMMEDIATELY before it — within ~18 chars ("greater than 10% WPI" -> the
#     value is a floor, not a finding).
# The floor phrases are scoped tight on purpose: loose/distant matches wrongly
# dropped genuine findings ("compression greater than 50% ... 20% WPI",
# "exceed ... more than 10%. ... a 3% whole person impairment").
_WPI_THRESHOLD_BEFORE_RE = re.compile(r"(?:threshold|exceed(?:s|ed|ing)?)\b", re.IGNORECASE)
_WPI_THRESHOLD_AFTER_RE = re.compile(r"^.{0,25}?\bthreshold\b", re.IGNORECASE | re.DOTALL)
_WPI_FLOOR_ADJ_RE = re.compile(
    r"(?:greater\s+than|more\s+than|at\s+least|in\s+excess\s+of|no(?:t)?\s+less\s+than)"
    r"\s*$",
    re.IGNORECASE,
)
_WPI_THRESHOLD_WINDOW = 60
_WPI_FLOOR_WINDOW = 18


def _is_threshold_mention(decision_text, match_start, match_end=None):
    """True if the WPI number at match_start sits in statutory-threshold framing."""
    before = decision_text[max(0, match_start - _WPI_THRESHOLD_WINDOW):match_start]
    if _WPI_THRESHOLD_BEFORE_RE.search(before):
        return True
    floor = decision_text[max(0, match_start - _WPI_FLOOR_WINDOW):match_start]
    if _WPI_FLOOR_ADJ_RE.search(floor):
        return True
    if match_end is not None:
        after = decision_text[match_end:match_end + 30]
        if _WPI_THRESHOLD_AFTER_RE.search(after):
            return True
    return False


def find_wpi_candidates(decision_text):
    """Return the set of distinct WPI numbers (in [0,100]) found in the text.

    Numbers appearing in statutory-threshold framing (e.g. "does not exceed the
    threshold of 10% whole person impairment") are excluded — they describe the
    legislative bar, not this claimant's impairment.
    """
    if not decision_text:
        return set()
    values = set()
    for rgx in (_WPI_FWD_RE, _WPI_REV_RE):
        for m in rgx.finditer(decision_text):
            v = float(m.group(1))
            if 0 <= v <= 100 and not _is_threshold_mention(decision_text, m.start(), m.end()):
                values.add(v)
    return values


def find_wpi_tokens(decision_text):
    """Every WPI-shaped percentage in the text, WITHOUT the threshold filter.

    `find_wpi_candidates` suppresses anything sitting near threshold wording,
    which is right for a regex that cannot read context but wrong as a GATE for
    the resolution pass: a decision reading "...does not exceed the statutory
    threshold of 10% whole person impairment. Dr X assessed 8% WPI" has every
    token inside the suppression window, so the candidate set comes back empty
    and the row is never examined at all — losing a real 8% finding, and every
    genuine 0% recorded next to a minor-injury recital.

    Use this to decide WHETHER to look; use find_wpi_candidates to decide what a
    regex may safely conclude on its own.
    """
    if not decision_text:
        return set()
    values = set()
    for rgx in (_WPI_FWD_RE, _WPI_REV_RE):
        for m in rgx.finditer(decision_text):
            v = float(m.group(1))
            if 0 <= v <= 100:
                values.add(v)
    return values


def extract_wpi_confident(decision_text):
    """High-precision WPI from a decision's cleaned text.

    Returns a single float when exactly one distinct NON-ZERO WPI number
    appears in the source (~97% accuracy vs LLM on validation). Returns
    None for zero or multiple distinct values, or when the only candidate
    is 0 — '0% WPI' almost always comes from generic statutory framing
    (definition of minor injury, the s 4.11 / s 1.6 threshold language),
    not a finding about this claimant. A genuine 0% WPI claimant would
    not be entitled to non-economic loss and rarely receives a lump sum.
    """
    vals = find_wpi_candidates(decision_text)
    vals = {v for v in vals if v > 0}
    if len(vals) == 1:
        return next(iter(vals))
    return None

# AustLII NSWPIC decisions have a consistent "CATCHWORDS:" header followed by
# a body that ends at the next section heading. Parse verbatim — no LLM. Used
# as a ground-truth validation handle for the LLM-derived case description.
#
# Edge cases observed in the corpus:
#  - Standard:  "CATCHWORDS:\n\n<body>\n\nDETERMINATIONS MADE:\n..."
#  - Merged:    "CATCHWORDS: DETERMINATIONS MADE:\n\n<body>\n\n<next heading>"
#               (AustLII rendering glitch — body still recoverable from below)
#  - Lower-case next heading: "determinations made:" / "Reasons for Decision"
#
# Start: any line beginning with the literal "CATCHWORDS" (consume the rest
# of that line so the merged-header glitch still works).
# End: the next *known* section heading, case-insensitive. Using a known list
# avoids false matches on catchwords-body fragments (e.g. lines ending in
# "applied:" or "considered:").
# Start: any line beginning with "CATCHWORD" or "CATCHWORDS" (singular form
# observed in some 2022 cases).
CATCHWORDS_START_RE = re.compile(r"(?im)^[ \t]*CATCHWORDS?\b.*$")
CATCHWORDS_END_RE = re.compile(
    r"(?im)^\s*(?:"
    r"DETERMINATIONS?\s+MADE"
    r"|ORDERS?\s+MADE"
    r"|LEGISLATION\s+CITED"
    r"|CASES?\s+CITED"
    r"|REASONS?\s+FOR\s+(?:DECISION|JUDGMENT)"
    r"|REASONS?\b"
    r"|INTRODUCTION"
    r"|BACKGROUND"
    r"|ORDERS?"
    r"|DECISION"
    r"|DETERMINATION"
    r"|FINDINGS"
    r"|HEARING\s+DATES?"
    r"|DATE\s+OF\s+(?:DECISION|HEARING)"
    r"|CERTIFICATE\s+OF\s+DETERMINATION"
    r"|SUMMARY\s+OF\s+(?:DECISION|ORDERS?)"
    r"|INTERIM\s+PAYMENT\s+DIRECTION"
    r")\s*:?\s*$"
)
# Defensive cap — observed catchwords body up to ~3100 chars in genuine cases.
# >4000 means the end-anchor missed and we ran into the substantive body.
CATCHWORDS_MAX_CHARS = 4000


def extract_catchwords(decision_text):
    """Pull the verbatim CATCHWORDS body from a NSWPIC decision text.

    Returns the body string with leading/trailing whitespace trimmed and
    internal paragraph breaks preserved. Returns "" if no CATCHWORDS header
    is found (rare) or if the parsed body exceeds CATCHWORDS_MAX_CHARS (rarer
    — indicates a malformed source where the end anchor is missing).
    """
    if not decision_text:
        return ""
    m = CATCHWORDS_START_RE.search(decision_text)
    if not m:
        return ""
    start = m.end()
    end_m = CATCHWORDS_END_RE.search(decision_text, start)
    end = end_m.start() if end_m else len(decision_text)
    body = decision_text[start:end].strip()
    if len(body) > CATCHWORDS_MAX_CHARS:
        logging.warning(
            f"extract_catchwords: body length {len(body)} exceeds cap "
            f"({CATCHWORDS_MAX_CHARS}); end anchor likely missed - returning empty"
        )
        return ""
    return body


def extract_numbered_paragraphs(cleaned_text):
    """Return {paragraph_number: paragraph_text} for top-level numbered paragraphs."""
    matches = list(PARA_NUM_RE.finditer(cleaned_text))
    paragraphs = {}
    last_num = 0
    for i, m in enumerate(matches):
        num = int(m.group(1))
        if num != last_num + 1 and not (num == 1 and last_num == 0):
            if num <= last_num:
                continue
        last_num = num
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned_text)
        paragraphs[num] = cleaned_text[start:end].strip()
    return paragraphs


def _flex_pattern(marker):
    parts = re.split(r"\s+", marker.strip())
    if not parts:
        return None
    return re.compile(r"\s+".join(re.escape(p) for p in parts), re.DOTALL)


def find_slice(source, start_marker, end_marker):
    """Locate verbatim slice in `source` between markers (whitespace-tolerant)."""
    if not start_marker or not end_marker:
        return None, None, "empty marker"
    start_pat = _flex_pattern(start_marker)
    end_pat = _flex_pattern(end_marker)
    if not start_pat or not end_pat:
        return None, None, "marker collapsed to empty"
    sm = start_pat.search(source)
    if not sm:
        return None, None, "start_marker not found"
    em = end_pat.search(source, sm.start())
    if not em:
        return None, None, "end_marker not found after start_marker"
    return source[sm.start():em.end()], (sm.start(), em.end()), None


# ----------------------------------------------------------------------
# Banding validation
# ----------------------------------------------------------------------
#
# Band definitions mirror the schema description in CombinedSchema.banded_case_description.
# Convention: lower-bound inclusive, upper-bound inclusive. Boundary values
# go to the LOWER band (e.g. exactly 11% WPI -> "11-15%", not "6-10%"; exactly
# 300000 NEL -> "$150k-$300k", not "$300k-cap").

AGE_BANDS = [
    (0, 17, "<18"),
    (18, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 74, "65-74"),
    (75, 200, "75+"),
]
WPI_BANDS = [
    (0, 5, "0-5%"),
    (6, 10, "6-10%"),
    (11, 15, "11-15%"),
    (16, 20, "16-20%"),
    (21, 30, "21-30%"),
    (31, 50, "31-50%"),
    (51, 200, ">50%"),
]
PAYOUT_BANDS = [
    (0, 50_000, "<$50k"),
    (50_000, 150_000, "$50k-$150k"),
    (150_000, 300_000, "$150k-$300k"),
    (300_000, 500_000, "$300k-$500k"),
    (500_000, 1_000_000, "$500k-$1M"),
    (1_000_000, 2_000_000, "$1M-$2M"),
    (2_000_000, 10**15, ">$2M"),
]
NEL_BANDS = [
    (0, 50_000, "<$50k"),
    (50_000, 150_000, "$50k-$150k"),
    (150_000, 300_000, "$150k-$300k"),
    (300_000, 10**15, "$300k-cap"),
]
FEL_BANDS = [
    (0, 25_000, "<$25k"),
    (25_000, 75_000, "$25k-$75k"),
    (75_000, 200_000, "$75k-$200k"),
    (200_000, 500_000, "$200k-$500k"),
    (500_000, 10**15, ">$500k"),
]
INCOME_WEEKLY_BANDS = [
    (0, 500, "<$500"),
    (500, 1000, "$500-$1000"),
    (1000, 1500, "$1000-$1500"),
    (1500, 2500, "$1500-$2500"),
    (2500, 10**15, ">$2500"),
]
COMMISSION_MONTHLY_BANDS = [
    (0, 1000, "<$1k"),
    (1000, 5000, "$1k-$5k"),
    (5000, 10_000, "$5k-$10k"),
    (10_000, 10**15, ">$10k"),
]


def _band_for(value, bands):
    for low, high, label in bands:
        if low <= value <= high:
            return label
    return None


def _parse_money(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.lower() in ("not stated", "not applicable", "not addressed", "nil", "n/a", "unknown"):
        return None
    # Strip $ and commas; take first numeric token
    s = s.replace("$", "").replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_pct(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip().rstrip("%").strip()
    if not s or s.lower() in ("not stated", "not applicable", "n/a", "unknown"):
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _parse_age(s):
    if not s or not isinstance(s, str):
        return None
    m = re.search(r"\b(\d{1,3})\b", s)
    if not m:
        return None
    age = int(m.group(1))
    return age if 0 < age < 120 else None


_BAND_TOKEN_KINDS = (
    "AGE_BAND", "WPI_RANGE", "PAYOUT_RANGE", "NEL_RANGE",
    "FEL_RANGE", "INCOME_WEEKLY", "COMMISSION_MONTHLY", "DATE",
)
_BAND_TOKEN_RE = re.compile(
    r"\[(?:" + "|".join(_BAND_TOKEN_KINDS) + r")(?::[^\]]*)?\]"
)


def validate_banding(banded_text, record=None):
    """
    Validate that `banded_text` (the value of banded_case_description) does
    not leak target numerics and that any band tokens it does use are
    consistent with the structured numeric fields on `record` (when supplied).

    Returns: {
        "ok": bool,
        "tokens": {token_kind: count, ...},
        "issues": [{"type": str, "severity": "high"|"medium"|"low", "match": str|None, "detail": str|None}, ...],
    }
    """
    if not banded_text:
        return {"ok": False, "tokens": {}, "issues": [{"type": "empty", "severity": "high"}]}

    issues = []

    # Token counts
    tokens = {kind: 0 for kind in _BAND_TOKEN_KINDS}
    for kind in _BAND_TOKEN_KINDS:
        tokens[kind] = len(re.findall(r"\[" + kind + r"(?::|\])", banded_text))

    # Strip all band tokens so residual checks don't match inside them
    stripped = _BAND_TOKEN_RE.sub("", banded_text)

    # --- Residual leakage checks ---
    for m in re.finditer(r"\$\s?[\d,]+(?:\.\d+)?(?:\s*(?:million|m|k|K))?", stripped):
        issues.append({"type": "residual_currency", "severity": "high",
                       "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)\s+\d{4}\b",
        stripped,
    ):
        issues.append({"type": "residual_date", "severity": "high",
                       "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(
        r"\b\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\b",
        stripped,
    ):
        issues.append({"type": "residual_date", "severity": "high",
                       "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", stripped):
        issues.append({"type": "residual_date", "severity": "high",
                       "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(r"\baged?\s+(\d{1,3})\b", stripped, re.IGNORECASE):
        age = int(m.group(1))
        if 0 < age < 120:
            issues.append({"type": "residual_age", "severity": "high",
                           "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(r"\b(\d{1,3})[-\s]year[-\s]old\b", stripped, re.IGNORECASE):
        age = int(m.group(1))
        if 0 < age < 120:
            issues.append({"type": "residual_age", "severity": "high",
                           "match": m.group(0).strip(), "detail": None})
    for m in re.finditer(
        r"\b(\d{1,2})\s*%\s*(?:WPI|whole\s*person\s*impairment|permanent\s*impairment)\b",
        stripped, re.IGNORECASE,
    ):
        issues.append({"type": "residual_wpi", "severity": "high",
                       "match": m.group(0).strip(), "detail": None})

    # --- Cross-check band tokens against structured numeric fields ---
    if record:
        checks = [
            ("WPI_RANGE", "Impairment %",          _parse_pct,   WPI_BANDS,             "wpi"),
            ("PAYOUT_RANGE", "Lump Sum",           _parse_money, PAYOUT_BANDS,          "payout"),
            ("NEL_RANGE", "Non-Economic Loss",     _parse_money, NEL_BANDS,             "nel"),
            ("FEL_RANGE", "Future Economic Loss",  _parse_money, FEL_BANDS,             "fel"),
            ("INCOME_WEEKLY", "Claimant Weekly Income", _parse_money, INCOME_WEEKLY_BANDS, "income"),
            ("COMMISSION_MONTHLY", None,           None,         COMMISSION_MONTHLY_BANDS, "commission"),
            ("AGE_BAND", "Claimant Age",           _parse_age,   AGE_BANDS,             "age"),
        ]
        for kind, field_name, parser, bands, label in checks:
            if field_name is None or parser is None:
                continue
            structured_val = parser(record.get(field_name, ""))
            used_tokens = re.findall(r"\[" + kind + r":([^\]]+)\]", banded_text)
            if structured_val is None or not used_tokens:
                continue
            # Skip when components are split across multiple band tokens of
            # the same kind — the structured field is the SUM, which won't
            # equal any single component band (e.g. FEL $45k buffer + $5k
            # super -> two FEL_RANGE tokens; structured FEL = $50k total).
            if len(used_tokens) > 1:
                continue
            # Skip INCOME_WEEKLY cross-check when commissions are banded
            # separately — structured Weekly Income includes commission, but
            # INCOME_WEEKLY token alone represents just the base salary.
            if kind == "INCOME_WEEKLY" and tokens.get("COMMISSION_MONTHLY", 0) > 0:
                continue
            expected = _band_for(structured_val, bands)
            if expected is None:
                continue
            if not any(expected == t for t in used_tokens):
                issues.append({
                    "type": f"{label}_band_mismatch",
                    "severity": "medium",
                    "match": None,
                    "detail": f"{field_name}={structured_val}; expected {kind} band '{expected}'; "
                              f"banded text used {used_tokens}",
                })

    return {
        "ok": not issues,
        "tokens": tokens,
        "issues": issues,
    }


# ----------------------------------------------------------------------
# Field coercion, WPI reconciliation, age cross-check, loss detection
# ----------------------------------------------------------------------
#
# These run on every extraction (in _build_record_from_parsed) so the eight
# high-value fields are coerced to clean values, the WPI safeguards that used
# to live only in the backfill scripts are applied in the LIVE path, and any
# field that looks lost (empty value but a clear signal in the source) is
# flagged for a focused second pass and, if still missing, for human review.

# Plausible weekly-income band in AUD. Outside this range almost always means a
# unit error (an annual or monthly figure not converted to weekly).
INCOME_WEEKLY_MIN = float(os.getenv("NSW_INCOME_WEEKLY_MIN", "50"))
INCOME_WEEKLY_MAX = float(os.getenv("NSW_INCOME_WEEKLY_MAX", "15000"))

_VALUE_SENTINELS = {
    "", "not stated", "not applicable", "n/a", "unknown",
    "none", "nil", "not addressed",
}


def _value_present(v):
    """True if `v` is a real captured value (not empty / not a sentinel)."""
    return str(v or "").strip().lower() not in _VALUE_SENTINELS


def coerce_money(val):
    """Coerce an LLM money string to a clean numeric string (digits + one dot),
    or "" for sentinels / unparseable input. Thin wrapper over
    coerce_leading_number (which already strips $ and commas and rejects the
    'Nil'/'Not addressed' sentinels)."""
    return coerce_leading_number(val)


def _clean_wpi_value(raw):
    """Return a plausible WPI value as a tidy string, or "" to drop it.

    Drops sentinels, values outside (0, 100], and a lone 0 — '0% WPI' is almost
    always statutory-threshold/minor-injury framing rather than a finding (see
    extract_wpi_confident)."""
    p = _parse_pct(raw)
    if p is None or p <= 0 or p > 100:
        return ""
    return str(int(p)) if p == int(p) else str(p)


def reconcile_wpi(strict_raw, accepted_raw, decision_text):
    """Reconcile the two WPI columns in the live extraction path.

    - Cleans both values (plausibility + lone-zero suppression).
    - Seeds the lenient 'accepted' value from the strict 'made-here' value when
      the LLM left it blank (a value made here is, by definition, relied on).
    - Backfills 'accepted' from the high-precision single-token regex when the
      LLM left it blank and the source contains exactly one non-zero WPI.
    - Cross-checks a populated 'accepted' against that regex value and flags a
      mismatch for review (could be a legitimate combined-vs-component case).

    Returns (strict, accepted, issues).
    """
    issues = []
    strict = _clean_wpi_value(strict_raw)
    accepted = _clean_wpi_value(accepted_raw)

    if not accepted and strict:
        accepted = strict

    conf = extract_wpi_confident(decision_text)
    if not accepted:
        if conf is not None:
            accepted = str(int(conf)) if conf == int(conf) else str(conf)
            issues.append({
                "field": "Impairment % (Accepted)", "type": "wpi_regex_backfill",
                "severity": "low",
                "detail": f"LLM left WPI empty; recovered {accepted} from a lone source token",
            })
    else:
        ap = _parse_pct(accepted)
        if conf is not None and ap is not None and abs(conf - ap) > 0.01:
            issues.append({
                "field": "Impairment % (Accepted)", "type": "wpi_mismatch",
                "severity": "medium",
                "detail": f"LLM WPI={accepted} but the only source WPI token is {conf}",
            })
    return strict, accepted, issues


_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def _year_of(date_str):
    if not date_str:
        return None
    m = _YEAR_RE.search(str(date_str))
    return int(m.group(1)) if m else None


def derive_age_from_dob(dob, ref_date):
    """Age in whole years from a DOB (YYYY-MM-DD or a bare year) relative to a
    reference date (YYYY-MM-DD or a bare year). None if undeterminable or
    implausible. Year-difference precision is adequate here — we only use it as
    a cross-check, not as the primary value."""
    by, ry = _year_of(dob), _year_of(ref_date)
    if by is None or ry is None:
        return None
    age = ry - by
    return age if 0 <= age <= 120 else None


def check_age_consistency(stated_age, dob, ref_date, *, label="age"):
    """Return a review issue dict if a stated age and a DOB-derived age both
    exist and disagree by more than a year; otherwise None."""
    stated = _parse_age(stated_age)
    derived = derive_age_from_dob(dob, ref_date)
    if stated is None or derived is None:
        return None
    if abs(stated - derived) > 1:
        return {
            "field": "Claimant Age", "type": f"{label}_dob_mismatch", "severity": "medium",
            "detail": f"stated {label}={stated} but DOB-derived={derived}",
        }
    return None


def reconcile_ages(age_injury, age_decision, injury_date, decision_date):
    """Fill age-at-injury and age-at-decision from EACH OTHER using the elapsed
    years between injury and decision. No date of birth needed:

        age_at_injury  = age_at_decision - (decision_year - injury_year)
        age_at_decision = age_at_injury  + (decision_year - injury_year)

    A decision that says only 'the claimant is now 31' (age at decision) in 2021
    for a 2018 injury still fixes age-at-injury at 31 - 3 = 28. Approximate to
    +/-1 year (birthday timing within the year), well inside the tolerance used
    elsewhere. Only fills a BLANK value; never overwrites one already present.
    Returns (age_injury, age_decision) as strings."""
    ai = coerce_leading_number(age_injury)
    ad = coerce_leading_number(age_decision)
    iy, dy = _year_of(injury_date), _year_of(decision_date)
    if iy is not None and dy is not None and dy >= iy:
        gap = dy - iy
        if not ai and ad:
            v = int(float(ad)) - gap
            if 0 < v < 120:
                ai = str(v)
        elif not ad and ai:
            v = int(float(ai)) + gap
            if 0 < v < 120:
                ad = str(v)
    return ai, ad


# Source-signal detectors used by the loss gate. A money head (NEL/FEL/income)
# only counts when a dollar amount sits near the head phrase, which keeps mere
# recitations of submissions from registering as findings.
_SIG_NEL_RE = re.compile(r"non[-\s]?economic loss|general damages|pain and suffering", re.IGNORECASE)
_SIG_FEL_RE = re.compile(
    r"future economic loss|future loss of earning|loss of (?:future )?earning capacity|\bbuffer\b",
    re.IGNORECASE,
)
_SIG_INCOME_RE = re.compile(
    r"\bPIAWE\b|pre[-\s]?injury average weekly|per week|weekly (?:wage|earnings|income)|gross weekly",
    re.IGNORECASE,
)
_SIG_AGE_RE = re.compile(
    r"\baged?\s+\d{1,3}\b|\b\d{1,3}[-\s]year[-\s]old\b|date of birth|\bborn\s+(?:on|in)\b",
    re.IGNORECASE,
)
_SIG_OCC_RE = re.compile(
    r"employed as\b|worked as\b|by occupation\b|by trade\b|occupation (?:was|is|of)",
    re.IGNORECASE,
)
_SIG_MONEY_RE = re.compile(r"\$\s?\d")


def _head_with_money(text, head_re, window=160):
    """True if `head_re` matches `text` with a dollar amount within `window`
    characters of the match."""
    for m in head_re.finditer(text):
        s = max(0, m.start() - window)
        e = min(len(text), m.end() + window)
        if _SIG_MONEY_RE.search(text[s:e]):
            return True
    return False


# Maps each high-value column to (provenance-quote key, focused-field key) so
# the loss gate and the focused second pass agree on field identity.
FIELD_LOSS_SPEC = {
    "Impairment % (Accepted)": ("wpi_quote", "wpi_percent", "high"),
    "Non-Economic Loss":       ("non_economic_loss_quote", "non_economic_loss", "high"),
    "Claimant Weekly Income":  ("weekly_income_quote", "claimant_weekly_income", "high"),
    "Future Economic Loss":    ("future_economic_loss_quote", "future_economic_loss", "high"),
    "Claimant Age":            ("age_quote", "claimant_age", "high"),
    "Claimant Gender":         ("gender_quote", "claimant_gender", "medium"),
    "Claimant Occupation":     ("occupation_quote", "claimant_occupation", "medium"),
    "Accident/Injury Location": ("location_quote", "location", "medium"),
}


def _quote_present(prov, key):
    val = str((prov or {}).get(key, "") or "").strip()
    return len(val) >= 4 and val.lower() not in ("not stated", "n/a", "none")


# Idea 3: a provenance quote that actually PROVES the field is absent (a claim
# not made / not pressed, the claimant unemployed, etc.) must not count as a
# captured-but-dropped value. These phrases mark such quotes.
_ABSENCE_RE = re.compile(
    r"\b(?:not\s+claim(?:ed)?|did\s+not\s+claim|no\s+claim(?:\s+(?:for|was|is))?|"
    r"confined\s+to|not\s+(?:being\s+)?pressed|no\s+longer\s+pressed|not\s+sought|"
    r"un\-?employed|not\s+employed|not\s+working|no\s+evidence\s+of|"
    r"not\s+in\s+receipt|abandoned|withdr(?:ew|awn)|not\s+entitled|"
    r"declined\s+to\s+claim|no\s+award|not\s+awarded|did\s+not\s+(?:press|pursue|seek))\b",
    re.IGNORECASE,
)


def _quote_indicates_absence(quote):
    return bool(quote) and bool(_ABSENCE_RE.search(str(quote)))


def _quote_signal(prov, key):
    """A provenance quote that genuinely supports a value: present, and not a
    quote that proves the field's absence."""
    if not _quote_present(prov, key):
        return False
    return not _quote_indicates_absence((prov or {}).get(key, ""))


def _quote_has_clean_wpi(quote):
    """True if the WPI quote contains exactly one clean, non-zero, non-threshold
    WPI value — i.e. a value that could actually be recovered. 'greater than
    10%' / threshold framing yields no candidate (find_wpi_candidates drops it),
    so those don't qualify."""
    cand = {v for v in find_wpi_candidates(str(quote or "")) if v > 0}
    return len(cand) == 1


def detect_field_losses(record, decision_text, provenance=None):
    """Flag high-value fields that are empty in `record` but whose value looks
    present in the source (a strong textual signal, or a provenance quote the
    model itself supplied). Returns a list of issue dicts. This is the A1 gate;
    its output drives the focused second pass and the Needs Review flag."""
    prov = provenance or {}
    text = decision_text or ""
    issues = []

    def flag(field, severity, detail):
        issues.append({"field": field, "type": "possible_loss",
                       "severity": severity, "detail": detail})

    # WPI — DETECTION stays broad (any WPI token in the text, or a quote) so the
    # cheap second pass can recover values the first pass missed (e.g. a combined
    # total when only components were captured). ESCALATION to Needs Review is
    # the precise part: confirmed_high_losses requires the quote to carry a clean
    # non-threshold value, so threshold/remittal cases recover-or-ignore without
    # inflating review (Idea 2 applies there, not here).
    if not _value_present(record.get("Impairment % (Accepted)")):
        if find_wpi_candidates(text) or _quote_signal(prov, "wpi_quote"):
            flag("Impairment % (Accepted)", "high", "WPI token/quote in source but field empty")

    # Non-economic loss (Idea 1) — trust the model's disposition: a loss only
    # when AWARDED but the amount is missing (a true internal contradiction).
    # Nil / Not addressed are deliberate determinations, not losses.
    if (str(record.get("Non-Economic Loss Status", "")).strip() == "Awarded"
            and not _value_present(record.get("Non-Economic Loss"))):
        flag("Non-Economic Loss", "high", "NEL marked Awarded but amount missing")

    # Future economic loss (Idea 1)
    if (str(record.get("Future Economic Loss Status", "")).strip() == "Awarded"
            and not _value_present(record.get("Future Economic Loss"))):
        flag("Future Economic Loss", "high", "FEL marked Awarded but amount missing")

    # Weekly income
    if not _value_present(record.get("Claimant Weekly Income")):
        if _head_with_money(text, _SIG_INCOME_RE) or _quote_signal(prov, "weekly_income_quote"):
            flag("Claimant Weekly Income", "high", "Income figure in source but field empty")

    # Age — don't flag when age-at-decision IS captured: the claimant's age is
    # recorded (in the other column) and age-at-injury simply may not be
    # derivable (no injury year to subtract). reconcile_ages already fills
    # age-at-injury whenever both years are known, so a still-empty Claimant Age
    # next to a present Claimant Age At Decision is a derivation limit, not a
    # lost value.
    if (not _value_present(record.get("Claimant Age"))
            and not _value_present(record.get("Claimant Age At Decision"))):
        if _SIG_AGE_RE.search(text) or _quote_signal(prov, "age_quote"):
            flag("Claimant Age", "high", "Age/DOB in source but field empty")

    # Gender — provenance-driven only (pronoun scanning is too noisy).
    if not _value_present(record.get("Claimant Gender")) and _quote_signal(prov, "gender_quote"):
        flag("Claimant Gender", "medium", "Gender quote present but field Not stated")

    # Occupation
    if not _value_present(record.get("Claimant Occupation")):
        if _SIG_OCC_RE.search(text) or _quote_signal(prov, "occupation_quote"):
            flag("Claimant Occupation", "medium", "Occupation cue in source but field empty")

    # Location — provenance-driven only.
    if not _value_present(record.get("Accident/Injury Location")) and _quote_signal(prov, "location_quote"):
        flag("Accident/Injury Location", "medium", "Location quote present but field empty")

    return issues


def merge_focused_into_record(result, focused, target_cols):
    """Overlay non-empty values from a FocusedFields result onto `result`, but
    only for the flagged `target_cols` and only where the column is still empty
    (never override a value the first pass already captured). Returns the list
    of columns actually recovered."""
    recovered = []

    def set_if(col, value):
        if _value_present(value) and not _value_present(result.get(col)):
            result[col] = value
            recovered.append(col)
            return True
        return False

    def _status_value(attr):
        s = getattr(focused, attr, None)
        return s.value if s is not None else ""

    for col in target_cols:
        if col == "Impairment % (Accepted)":
            set_if(col, _clean_wpi_value(getattr(focused, "wpi_percent", "")))
        elif col == "Non-Economic Loss":
            amt = coerce_money(getattr(focused, "non_economic_loss", ""))
            status = _status_value("non_economic_loss_status")
            if status == "Nil" and not amt:
                amt = "0"
            if status and status != "Not addressed":
                result["Non-Economic Loss Status"] = status
            set_if(col, amt)
        elif col == "Future Economic Loss":
            amt = coerce_money(getattr(focused, "future_economic_loss", ""))
            status = _status_value("future_economic_loss_status")
            if status == "Nil" and not amt:
                amt = "0"
            if status and status != "Not addressed":
                result["Future Economic Loss Status"] = status
            set_if(col, amt)
        elif col == "Claimant Weekly Income":
            if set_if(col, coerce_money(getattr(focused, "claimant_weekly_income", ""))):
                basis = getattr(focused, "claimant_weekly_income_basis", "")
                if _value_present(basis) and not _value_present(result.get("Claimant Weekly Income Basis")):
                    result["Claimant Weekly Income Basis"] = basis
        elif col == "Claimant Age":
            set_if(col, coerce_leading_number(getattr(focused, "claimant_age", "")))
        elif col == "Claimant Gender":
            g = getattr(focused, "claimant_gender", None)
            set_if(col, g.value if g is not None else "")
        elif col == "Claimant Occupation":
            set_if(col, getattr(focused, "claimant_occupation", ""))
        elif col == "Accident/Injury Location":
            set_if(col, getattr(focused, "location", ""))
    return recovered


def damages_pass_applies(record):
    """Whether the damages breakdown pass should run for this row. Gated on
    case type because the workbook the spec targets is CTP-only; widen with
    NSW_DAMAGES_CASE_TYPES."""
    if not DAMAGES_PASS_ENABLED:
        return False
    return str(record.get("Case Type", "") or "").strip() in DAMAGES_CASE_TYPES


def merge_damages_into_record(record, parsed):
    """Merge a parsed DamagesSchema into a flat record IN PLACE.

    The consumer's load-bearing columns (`Lump Sum`, `Non-Economic Loss`,
    `Future Economic Loss` and their statuses, `Description`) are never
    overwritten — the pass's independent readings land in the `(Recheck)`
    columns so extractor accuracy stays measurable. The previously-empty
    `Weekly Benefit` is filled only when it is currently blank.

    Returns the sidecar dict for `_damages`.
    """
    flat, sidecar, _issues = damages_row_from_parsed(
        parsed, existing=record, description=record.get("Description", ""),
    )
    record.update(flat)

    if not str(record.get("Weekly Benefit", "") or "").strip():
        record["Weekly Benefit"] = flat.get("Weekly Statutory Benefit", "")

    record["_damages_version"] = DAMAGES_VERSION
    record["_damages"] = sidecar
    return sidecar


def wpi_is_legally_impossible(record):
    """True when the row's own award contradicts its WPI.

    s 4.11 of the Motor Accident Injuries Act 2017 permits damages for
    non-economic loss ONLY where impairment exceeds 10%. A row with NEL
    `Awarded` and a WPI at or below 10 is therefore wrong on its face —
    usually because the captured figure is one component or one body system
    rather than the governing total. These are the only populated rows the
    resolution pass is allowed to correct.
    """
    if str(record.get("Non-Economic Loss Status") or "").strip() != "Awarded":
        return False
    current = to_float_pct(record.get("Impairment % (Accepted)"))
    return current is not None and current <= 10


# Record fields that carry the Member's own words, cheapest first. Scanned for
# the ex gratia carve-out when the full decision text is not to hand (the cache
# backfill has no HTML for every row, and re-reading 6,600 files to answer one
# yes/no question is not worth it when the catchwords already say so).
WPI_REASONING_FIELDS = (
    "Catchwords",
    "Narrative: Legal Issues and Reasoning",
    "Result",
    "Award Breakdown",
    "WPI Resolution Notes",
)


def wpi_award_is_ex_gratia(record, text=""):
    """True if the decision says non-economic loss was paid without entitlement.

    See `nel_paid_without_entitlement`. Checks the full text when the caller has
    it, otherwise the stored reasoning fields.
    """
    return nel_paid_without_entitlement(
        text, *(record.get(f, "") for f in WPI_REASONING_FIELDS))


def apply_money_absence_semantics(record):
    """Round 3 §11: give every empty money cell a reason. Mutates in place.

    Thin wrapper — the rules live in `damages_extraction` so they stay
    testable without importing the scraper. Skipped where the damages pass
    never ran, because on those rows the money columns are defaults rather
    than findings and classifying a default would assert something we did not
    look for.
    """
    if str(record.get("Damages Extraction Status") or "").strip() != "ok":
        return record
    residual, trustworthy = damages_residual(record)
    return refine_money_absence(record, residual=residual,
                                residual_trustworthy=trustworthy)


def apply_round2_semantics(record):
    """Applicability semantics for the WPI columns. Mutates in place.

    Round 2 of the downstream spec asked for one thing: make "the answer is no"
    distinguishable from "we do not know". Everything here is decided from
    columns the row already carries plus the classified mentions in
    `_wpi_resolution`, so it replays offline over the cache — no model call, and
    no judgement that cannot be re-derived and checked.

    Four changes, in dependency order:

    1. The psychiatric gate is made self-consistent. A separately-stated
       psychiatric percentage on a row flagged `Has Psychiatric Injury = No` is
       a contradiction; `Injury Categories` arbitrates, because it is the
       multi-label field built to record exactly this and it disagrees with the
       body-system label often enough to matter. Mason [2024] NSWPIC 348 is the
       case in point: the "psychiatric" 6% is emotional and behavioural
       functioning inside a traumatic BRAIN INJURY assessment, which is why
       Professor Cameron combined it with a 7% shoulder to certify 13%.
       Psychiatric impairment would never have been combined that way.
    2. `WPI Governing System` promotes what `resolve_wpi` already works out
       internally to a column.
    3. Each empty split-WPI cell gets a reason rather than a bare `absent`.
    4. `NEL Threshold Consistent` reports the s 4.11 disagreement.
    """
    mentions = (record.get("_wpi_resolution") or {}).get("mentions") or []
    notes = []

    phys = str(record.get("WPI Physical %") or "").strip()
    psych = str(record.get("WPI Psychiatric %") or "").strip()
    total = str(record.get("Impairment % (Accepted)") or "").strip()
    has_psych = str(record.get("Has Psychiatric Injury") or "").strip() == "Yes"
    categories = [c.strip().lower()
                  for c in str(record.get("Injury Categories") or "").split("|")]

    # --- 1. gate consistency -------------------------------------------------
    if psych and not has_psych:
        if "psychiatric" in categories:
            # Two independent signals say psychiatric; the flag is the outlier.
            record["Has Psychiatric Injury"] = "Yes"
            has_psych = True
            notes.append("Has Psychiatric Injury corrected to Yes: a psychiatric "
                         "impairment percentage was separately assessed")
        else:
            # No psychiatric injury on any other signal, so the percentage is
            # misattributed - it belongs to another body system's assessment.
            notes.append(f"WPI Psychiatric % {psych} withdrawn: no psychiatric "
                         f"injury is established on this row, so the figure "
                         f"belongs to another body system's assessment")
            record["WPI Psychiatric %"] = ""
            psych = ""

    # --- 1b. recover a component the total already determines (round 5 §13.1) -
    # Physical and psychiatric are assessed separately and the GREATER governs,
    # so the accepted total IS one of the two components. Where one component is
    # stated and the total EXCEEDS it, the other component must be the greater
    # one — and therefore equals the total. This is exact, not an estimate.
    #
    # Verified against source on all 11 rows it fires on. Quigley [2026] NSWPIC
    # 280 is the instructive one: MAS Curtin certified 4% (scarring, nerve) and
    # a Review Panel 8% (brain injury, shoulder) — DIFFERENT injuries, so they
    # combine to 12 rather than competing, and MAS Lahz's combined certificate
    # independently certifies "greater than 10%". The resolution ladder had
    # read them as rival assessments and taken the median, 6; the total of 12
    # was right and the ladder's per-system figure was not, which is why this
    # recovers from the TOTAL rather than from the ladder's arithmetic.
    # ONE DIRECTION ONLY — psychiatric known, physical recovered. The mirror
    # case does not hold: a total above the stated PHYSICAL figure is far more
    # often further physical components combining than a larger psychiatric
    # assessment. Mason [2024] NSWPIC 348 is the counter-example — 7% shoulder
    # and 6% emotional/behavioural combine to 13% inside ONE brain-injury
    # assessment, and running this rule backwards there invents a 13%
    # psychiatric impairment on a claimant with no psychiatric injury at all.
    if psych and not phys:
        known_value, total_value = to_pct(psych), to_pct(total)
        if known_value is not None and total_value is not None \
                and total_value > known_value:
            record["WPI Physical %"] = total
            # The recovered component is exactly as good as the total it came from.
            record["WPI Physical % Provenance"] = (
                str(record.get("WPI Provenance") or "").strip()
                or WpiProvenanceEnum.STATED.value)
            notes.append(f"WPI Physical % {total} recovered: the accepted total "
                         f"exceeds psychiatric {psych}, and the greater body "
                         f"system governs, so the total is the physical figure")
            phys = total

    # --- 2. governing system -------------------------------------------------
    # Round 5 §13: the resolution pass computes this from the classified
    # mentions and is authoritative. Deriving it here from which cells are
    # populated was circular — with one component captured it could only name
    # the component we held, which put 145 rows in a self-fulfilling label and
    # contradicted the resolution's own notes on the rows that mattered.
    # The resolution's label stands only where it actually COMPARED two body
    # systems. Round 7: keying on the stored value instead left 78 rows holding
    # a label written before round 5, when the derivation was still circular —
    # row 500 named `physical` on a row whose resolution has no mentions at
    # all. A stale answer is indistinguishable from a computed one, so the test
    # has to be about the evidence, not about what the cell currently says.
    mention_systems = {m.get("body_system") for m in mentions
                       if m.get("about_claimant", True) and not m.get("superseded")}
    if len({s for s in mention_systems
            if s in (BodySystemEnum.PHYSICAL.value,
                     BodySystemEnum.PSYCHIATRIC.value)}) > 1:
        pass                      # computed by resolve_wpi; authoritative
    else:
        record["WPI Governing System"] = governing_system(phys, psych, total)

    # Both systems stated and no total is a gap the greater-governs rule can
    # close without judgement. Cowper [2025] NSWPIC 596: Assessors Fitzsimons
    # and Jeyasingam each found 0%, so the accepted WPI is 0 — an assessment,
    # not a null. FILL ONLY: an existing total is never overwritten, so a row
    # whose total took the lesser system is reported by `WPI Governing System`
    # rather than silently rewritten.
    if not total and phys and psych:
        governing = max(to_float_pct(phys) or 0.0, to_float_pct(psych) or 0.0)
        total = f"{governing:g}"
        record["Impairment % (Accepted)"] = total
        record["WPI Provenance"] = WpiProvenanceEnum.DERIVED.value
        record["WPI Basis"] = "; ".join(x for x in [
            record.get("WPI Basis", ""),
            f"greater of physical {phys} and psychiatric {psych} "
            f"(assessed separately; the greater governs)"] if x)[:300]

    # --- 3. why each split cell is empty ------------------------------------
    psychiatric_only = categories == ["psychiatric"]
    for system, value, column in (
            (BodySystemEnum.PHYSICAL.value, phys, "WPI Physical %"),
            (BodySystemEnum.PSYCHIATRIC.value, psych, "WPI Psychiatric %")):
        prov_col = f"{column} Provenance"
        if value:
            # A figure that IS present keeps whatever positive provenance it had.
            if record.get(prov_col) in (None, "", *PROVENANCE_ABSENCE):
                record[prov_col] = WpiProvenanceEnum.STATED.value
            continue
        record[prov_col] = classify_split_wpi_absence(
            system=system, has_psychiatric=has_psych, total_present=bool(total),
            mentions=mentions, psychiatric_only=psychiatric_only)

    # The accepted WPI itself: an empty cell with no evidence of any assessment
    # is `not_assessed`, not a defect. A withheld one stays `absent` — we know
    # the decision quantified impairment and we could not produce the total.
    if total:
        if not str(record.get("WPI Provenance") or "").strip():
            record["WPI Provenance"] = WpiProvenanceEnum.STATED.value
    elif record.get("_wpi_quarantined"):
        # Migration for rows quarantined before round 7, which stored `absent`.
        # The quarantine cannot re-fire on them — there is no value left to
        # withhold — so the correction has to happen here.
        if record.get("WPI Provenance") == WpiProvenanceEnum.ABSENT.value:
            record["WPI Provenance"] = WpiProvenanceEnum.NOT_STATED.value
    elif record.get("_wpi_ex_gratia"):
        # Set by the quarantine, and deliberately not reclassified: the
        # threshold question does not arise, which is a stronger and truer
        # statement than any of the "we looked and found nothing" values.
        record["WPI Provenance"] = WpiProvenanceEnum.NOT_APPLICABLE.value
    elif not record.get("_wpi_quarantined"):
        if record.get("WPI Provenance") in (None, "", *PROVENANCE_ABSENCE):
            # A stated component IS an assessment, so a blank total on such a
            # row is `not_stated`, not `not_assessed`. Row 500 carries a
            # physical 18% and read `not_assessed`, which denied a figure the
            # row itself holds.
            record["WPI Provenance"] = (
                WpiProvenanceEnum.NOT_STATED.value if (phys or psych)
                else classify_split_wpi_absence(
                    system=BodySystemEnum.UNCLEAR.value,
                    has_psychiatric=has_psych, total_present=False,
                    mentions=mentions))

    # --- 4. s 4.11 consistency ----------------------------------------------
    # Computed from the finding AS THE DECISION MADE IT, before step 5 fills
    # any gap by deduction. Deducing "above 10%" from the award and then
    # checking the award against it would make every row agree with itself.
    stated_finding = str(record.get("WPI Threshold Finding") or "").strip()
    # A finding THIS PASS deduced on an earlier run is not a judicial finding,
    # however it looks in the column now. Without this, a second run promotes
    # every deduction to `decision` and the consistency check starts reading a
    # deduction from the award as independent evidence about the award.
    if str(record.get("WPI Threshold Finding Basis") or "").strip() in (
            ThresholdBasisEnum.FROM_NEL_AWARD.value,
            ThresholdBasisEnum.FROM_WPI.value):
        stated_finding = ""

    if record.get("_wpi_ex_gratia"):
        # The award was not predicated on the impairment finding at all, so the
        # rule has nothing to say about it. Reporting `no` here would describe
        # a lawful payment as a violation on every downstream ingest.
        record["NEL Threshold Consistent"] = NelConsistencyEnum.UNKNOWN.value
    else:
        record["NEL Threshold Consistent"] = nel_threshold_consistency(
            nel_status=record.get("Non-Economic Loss Status"),
            threshold_finding=stated_finding,
            wpi=total)

    # --- 5. threshold coverage (§10.3) --------------------------------------
    if stated_finding and stated_finding != ThresholdFindingEnum.NONE.value:
        record["WPI Threshold Finding Basis"] = ThresholdBasisEnum.DECISION.value
    else:
        finding, basis = derive_threshold_finding(
            nel_status=record.get("Non-Economic Loss Status"),
            wpi=total,
            ex_gratia=bool(record.get("_wpi_ex_gratia")))
        # An explicit `not determined` from the decision is a real fact — the
        # court declined to decide — and outranks a deduction only when the
        # deduction found nothing either. Otherwise the deduction adds signal
        # the empty cell did not have.
        if finding != ThresholdFindingEnum.NONE.value:
            record["WPI Threshold Finding"] = finding
            record["WPI Threshold Finding Basis"] = basis
        else:
            record["WPI Threshold Finding"] = ThresholdFindingEnum.NONE.value
            record["WPI Threshold Finding Basis"] = ThresholdBasisEnum.NONE.value

    if notes:
        record["WPI Resolution Notes"] = "; ".join(
            x for x in [record.get("WPI Resolution Notes", ""), *notes] if x)[:400]
    return record


# s 4.11 is a Motor Accident Injuries Act provision. Workers compensation runs
# on an entirely different scheme (permanent impairment under s 66 of the
# Workers Compensation Act 1987), so a WC row awarding non-economic loss at 10%
# WPI is not contradictory at all. Deliberately NOT the env-widenable
# DAMAGES_CASE_TYPES: widening the damages pass to WC must not silently start
# applying motor-accident law to WC rows.
MAI_ACT_CASE_TYPES = ("CTP",)


def quarantine_impossible_wpi(record, text=""):
    """Withhold a WPI that the row's own award proves wrong. Mutates in place.

    s 4.11 permits non-economic loss only above 10%, so `NEL Awarded` on a WPI
    at or below 10 cannot both be right. The failure is nearly always the WPI -
    a component figure ("the shoulders were equally impaired ... at 8% each")
    or a superseded one read as the governing total. We cannot always recover
    the true figure: where the decision never states the combined total, there
    is nothing to promote it to. So rather than publish a number known to be
    wrong, the value moves to `WPI Candidates` for audit and the field is
    blanked, which keeps it out of WPI-conditional analysis instead of
    distorting it.

    Three rows are left alone:
      * anything that is not a motor accident claim, since s 4.11 is not the
        governing provision (see MAI_ACT_CASE_TYPES);
      * the ex gratia case, where the low WPI is CORRECT and the insurer simply
        paid anyway (`wpi_award_is_ex_gratia`); and
      * rows the resolution pass already corrected to a figure above 10.

    Returns True if the row was quarantined.
    """
    if str(record.get("Case Type", "") or "").strip() not in MAI_ACT_CASE_TYPES:
        return False
    if str(record.get("Non-Economic Loss Status") or "").strip() != "Awarded":
        return False
    value = to_float_pct(record.get("Impairment % (Accepted)"))
    if value is None or value > 10:
        return False

    shown = str(record.get("Impairment % (Accepted)") or "").strip()

    if wpi_award_is_ex_gratia(record, text):
        # The figure is CORRECT — this is the one lawful way non-economic loss
        # is paid below the threshold — but publishing it makes every
        # downstream s 4.11 check read the row as an impossible combination,
        # because a checker comparing WPI to 10 cannot see that the payment was
        # never made under s 4.11 at all. So the value is withheld here too,
        # with a provenance that says the threshold question does not arise
        # rather than one that claims a defect. The figure stays in
        # `WPI % Candidates`.
        record["_wpi_ex_gratia"] = True
        candidates = str(record.get("WPI Candidates") or "").strip()
        if shown and shown not in [c.strip() for c in candidates.split("|")]:
            record["WPI Candidates"] = f"{candidates} | {shown}".strip(" |")
        record["Impairment % (Accepted)"] = ""
        record["WPI Provenance"] = WpiProvenanceEnum.NOT_APPLICABLE.value
        record["WPI Basis"] = (
            f"withheld: {shown} is correct but no s 4.11 threshold applies — "
            f"the insurer paid non-economic loss without any legal obligation")
        note = (f"WPI {shown} withheld as not applicable: the decision states "
                f"the insurer paid non-economic loss without any legal "
                f"obligation to do so, so the s 4.11 threshold does not arise")
        record["WPI Resolution Notes"] = "; ".join(
            x for x in [record.get("WPI Resolution Notes", ""), note] if x)[:400]
        return False

    # Keep the withheld figure visible rather than destroying it.
    candidates = str(record.get("WPI Candidates") or "").strip()
    if shown and shown not in [c.strip() for c in candidates.split("|")]:
        record["WPI Candidates"] = f"{candidates} | {shown}".strip(" |")

    record["Impairment % (Accepted)"] = ""
    # `not_stated`, not `absent`. Round 7 §15.1: this is a DELIBERATE
    # withholding, so recording it as a capture failure states the opposite of
    # what happened. Impairment was assessed — the components are right there
    # in `WPI % Candidates` — but the governing total the award implies is
    # never stated in the decision, which is exactly `not_stated`.
    record["WPI Provenance"] = WpiProvenanceEnum.NOT_STATED.value
    record["WPI Basis"] = (f"withheld: {shown} contradicts the award of "
                           f"non-economic loss, which s 4.11 permits only above 10%")
    record["_wpi_quarantined"] = shown
    note = (f"WPI {shown} withheld: non-economic loss was awarded, which s 4.11 "
            f"permits only above 10% impairment")
    record["Review Notes"] = "; ".join(
        x for x in [record.get("Review Notes", ""), note] if x)[:1000]
    # Deliberately NOT `Needs Review = Yes`. That flag feeds the analysis-ready
    # gate, which would evict the whole row from the workbook over one bad
    # field - and the rest of the row is sound (Washbourne is a complete
    # $1,451,619 award with a full damages breakdown). A blank WPI is already
    # the exclusion: 215 workbook rows have none, and WPI-conditional analysis
    # filters on `WPI % Provenance = 'stated'`, which this row now fails.
    # The trail is in `Review Notes`, `WPI % Basis` and `WPI % Candidates`;
    # `backfill_wpi_nel_quarantine.py --dry-run` lists these rows on demand.
    logging.warning(f"{note} for {record.get('Case Name', '')[:50]}")
    return True


def merge_wpi_resolution_into_record(record, parsed):
    """Merge a parsed WpiResolution into a flat record IN PLACE.

    Only fills `Impairment % (Accepted)` when the ladder produced a value; a
    resolution that finds nothing must not blank a WPI the main pass already
    captured. Returns the sidecar dict for `_wpi_resolution`.
    """
    existing = str(record.get("Impairment % (Accepted)") or "").strip()
    # Remember what the main extraction had BEFORE this pass ever touched the
    # row, so a re-run under a changed ladder can start from the same place
    # instead of treating its own previous output as pre-existing truth.
    if "_wpi_pre_value" not in record:
        record["_wpi_pre_value"] = existing
    resolved = resolve_wpi(parsed, existing=existing,
                           nel_status=record.get("Non-Economic Loss Status", ""))
    value = resolved.pop("Impairment % (Accepted)", "")
    systems = resolved.pop("_wpi_systems", None) or {}
    provenance = resolved.get("WPI Provenance", "absent")

    if not existing:
        # Nothing to lose: take whatever the ladder produced, at its own
        # provenance. This is the "better than nothing" case.
        record["Impairment % (Accepted)"] = value
    elif wpi_is_legally_impossible(record) and value and to_float_pct(value) is not None             and to_float_pct(value) > 10:
        # Narrow exception to fill-only: the existing figure is contradicted by
        # the decision's own award of non-economic loss, and the ladder found a
        # figure that is not.
        logging.info(f"WPI {existing} -> {value} ({resolved.get('WPI Basis')}): "
                     f"NEL awarded requires impairment above 10%")
        record["Impairment % (Accepted)"] = value
    else:
        # FILL ONLY. This pass sees classified mentions; the main extraction
        # read the whole decision including the Member's reasoning, and an
        # audit of the corrections this pass proposed found a third of them
        # wrong (an insurer's CONCESSION of 25% classified as a rejected claim;
        # one doctor's 0% preferred over another's assessment). A resolution
        # built for empty fields must not relitigate populated ones.
        # NEVER displace a captured value with a derived or inferred estimate.
        # A median of rival medico-legal reports is a fallback for an empty
        # field, not an improvement on a figure already extracted.
        if value and value != existing:
            resolved["WPI Resolution Notes"] = (
                f"ladder produced {value} ({provenance}); kept extracted {existing}. "
                + (resolved.get("WPI Resolution Notes") or ""))[:300]
        resolved["WPI Provenance"] = "stated"
        resolved["WPI Basis"] = "retained from main extraction"
    record.update(resolved)

    # Round 6 §14.1: a governing-system label asserts that the two components
    # were COMPARED, so they cannot simultaneously be `absent`. Where the
    # resolution reduced both systems, carry them into the split columns.
    # The two columns have DIFFERENT contracts, and round 7 §15.3 turned on the
    # difference. `WPI Physical %` is "only if the decision states physical and
    # psychiatric SEPARATELY" — so it needs both. `WPI Psychiatric %` is "only
    # if separately stated", which a psychiatric assessment satisfies on its
    # own. Requiring both for each left 4 rows marked `absent` while their
    # psychiatric figure sat resolved and unused.
    if systems:
        fillable = [(BodySystemEnum.PSYCHIATRIC.value, "WPI Psychiatric %")]
        if len(systems) > 1:
            fillable.append((BodySystemEnum.PHYSICAL.value, "WPI Physical %"))
        for system, column in fillable:
            if str(record.get(column) or "").strip() or system not in systems:
                continue
            resolved_value, resolved_prov, resolved_basis = systems[system]
            if not resolved_value:
                continue
            record[column] = resolved_value
            record[f"{column} Provenance"] = resolved_prov
            note = (f"{column} {resolved_value} carried from the resolution "
                    f"({resolved_basis})")
            record["WPI Resolution Notes"] = "; ".join(
                x for x in [record.get("WPI Resolution Notes", ""), note] if x)[:400]

    # Row-level consistency: the threshold finding and the value must agree.
    # s 4.11 makes non-economic loss available only ABOVE 10%, so a row saying
    # "above 10%" while carrying 9.5% is self-contradictory on its face,
    # whatever produced it. Flag rather than silently publish.
    final = to_float_pct(record.get("Impairment % (Accepted)"))
    threshold = record.get("WPI Threshold Finding", "")
    if final is not None and (
            (threshold == "above 10%" and final <= 10)
            or (threshold == "not above 10%" and final > 10)):
        note = (f"WPI {record['Impairment % (Accepted)']} contradicts threshold "
                f"finding '{threshold}'")
        logging.warning(f"{note} for {record.get('Case Name', '')[:50]}")
        record["Review Notes"] = "; ".join(
            x for x in [record.get("Review Notes", ""), note] if x)[:1000]

    # The note above records the contradiction; this withholds the value. It
    # runs last so it sees whatever the ladder and the narrow correction above
    # settled on, and it catches the case the threshold check cannot: a row
    # whose threshold finding is `not determined` but which pays non-economic
    # loss anyway, where the award itself is the evidence.
    quarantine_impossible_wpi(record)

    record["_wpi_version"] = WPI_VERSION
    record["_wpi_resolution"] = {
        "mentions": [
            {
                "value": m.value,
                "kind": getattr(m.kind, "value", m.kind),
                "body_system": getattr(m.body_system, "value", m.body_system),
                "assessor": m.assessor,
                "superseded": bool(m.superseded),
                "about_claimant": bool(m.about_claimant),
                "quote": (m.quote or "")[:160],
            }
            for m in (getattr(parsed, "mentions", None) or [])
        ],
        "tribunal_selected_value": getattr(parsed, "tribunal_selected_value", ""),
        "tribunal_selected_quote": getattr(parsed, "tribunal_selected_quote", ""),
        "settlement_approval_without_wpi": bool(
            getattr(parsed, "settlement_approval_without_wpi", False)),
        # Needed to replay the ladder offline without re-classifying.
        "threshold_finding": getattr(
            getattr(parsed, "threshold_finding", None), "value",
            getattr(parsed, "threshold_finding", "")) or "not determined",
        "totals_are_rival_assessments": bool(
            getattr(parsed, "totals_are_rival_assessments", True)),
        "components_share_one_assessment": bool(
            getattr(parsed, "components_share_one_assessment", False)),
        "basis": record.get("WPI Basis", ""),
    }
    return record["_wpi_resolution"]


def wpi_resolution_applies(record, decision_text):
    """Whether the WPI-resolution pass is worth a call: the field is empty (or
    the source holds several figures the strict regex refused to choose
    between) and there is at least one WPI-shaped token to reason about."""
    if not WPI_RESOLUTION_ENABLED:
        return False
    # Gate on the RAW tokens, not the threshold-filtered candidates: a genuine
    # figure that merely sits near threshold wording would otherwise never be
    # looked at (see find_wpi_tokens).
    tokens = find_wpi_tokens(decision_text)
    if not tokens:
        return False
    current = str(record.get("Impairment % (Accepted)") or "").strip()
    return (not current) or len(tokens) > 1 or wpi_is_legally_impossible(record)


def normalise_medical_costs(value):
    """`N/A` round-trips through pandas as NaN, which is why the consumer sees
    `Medical Costs` as 0% populated (spec 4.1). Emit a sentinel that survives
    CSV -> DataFrame -> Excel -> DataFrame instead."""
    v = str(value or "").strip()
    return "Not addressed" if v in ("", "N/A", "NA", "n/a") else v


def worth_second_pass(losses, provenance):
    """Whether the focused second pass is worth its cost: fire for any
    high-severity loss (the five numeric/core fields) or a medium loss the
    model itself backed with a provenance quote. Skips weak medium-only,
    regex-driven hits (e.g. a generic 'employed' mention) to avoid spending an
    LLM call where recovery is unlikely."""
    for iss in losses:
        if iss["severity"] == "high":
            return True
        spec = FIELD_LOSS_SPEC.get(iss["field"])
        if spec and _quote_signal(provenance, spec[0]):
            return True
    return False


def confirmed_high_losses(remaining_losses, provenance):
    """High-severity losses the model itself corroborated with a provenance
    quote — i.e. it quoted the fact but still left the field empty, a genuine
    self-contradiction. ONLY these escalate a row to Needs Review.

    A bare regex/text signal without a corroborating quote is NOT escalated:
    sample validation showed those are mostly false positives — WPI tokens in
    remittal or statutory-threshold context, or income/FEL amounts recited in
    submissions rather than awarded. Such cases still trigger the (cheap)
    focused second pass via worth_second_pass; they're recorded in the field
    review for triage but do not exclude the row from analysis."""
    out = []
    for iss in remaining_losses:
        if iss.get("severity") != "high":
            continue
        field = iss.get("field")
        spec = FIELD_LOSS_SPEC.get(field)
        if not spec:
            continue
        if field == "Impairment % (Accepted)":
            # Idea 2: WPI escalates only when the quote holds a single clean,
            # non-threshold value (threshold 'greater than 10%' / a claimed
            # figure on a remitted matter do not).
            if _quote_has_clean_wpi((provenance or {}).get("wpi_quote", "")):
                out.append(iss)
        elif _quote_signal(provenance, spec[0]):
            out.append(iss)
    return out


def extract_html_with_paragraph_numbers(html_bytes):
    """
    NSWPIC HTML uses <ol><li value="N"> for numbered paragraphs; BS4's
    get_text() drops the `value` attribute. Inject "N. " before each numbered
    <li> before extracting text.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    main = (soup.find("article")
            or soup.find(class_="the-document")
            or soup.find(class_="austlii-doc")
            or soup.body)
    if main is None:
        return ""
    for garbage in main.find_all(["div"], class_=["austlii-header", "breadcrumb", "page-footer", "nav"]):
        garbage.decompose()
    for li in main.find_all("li"):
        val = li.get("value")
        if val and str(val).strip().isdigit():
            li.insert(0, NavigableString(f"{val}. "))
    return main.get_text(separator="\n").strip()


# ----------------------------------------------------------------------
# Cost tracker
# ----------------------------------------------------------------------

QUOTA_BREAKER_THRESHOLD = int(os.getenv("QUOTA_BREAKER_THRESHOLD", "10"))


def _is_quota_error(error_text):
    """Return True for OpenAI billing/quota errors that won't clear by retrying."""
    if not error_text:
        return False
    s = str(error_text).lower()
    return "insufficient_quota" in s or "exceeded your current quota" in s


def _is_transient_api_error(error):
    """Return True for timeout/connection-style OpenAI errors that are worth
    retrying and should be logged distinctly from schema/parse failures
    (ISSUE-016)."""
    name = type(error).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True
    s = str(error).lower()
    return ("timed out" in s or "timeout" in s or "connection error" in s
            or "temporarily unavailable" in s)


class QuotaCircuitBreaker:
    """Abort the run only on SUSTAINED insufficient_quota failure.

    OpenAI auto-recharges balance and during the recharge window (~10-30s)
    new calls 429 with insufficient_quota. With 25 concurrent workers all
    those misses pile up in <1s, so a naive "10 consecutive failures" trips
    immediately on a transient.

    The per-worker retry-with-backoff in extract_combined already absorbs
    short recharge windows. This breaker only catches the case where retries
    are exhausted AND quota failures continue across a wider time window —
    i.e. a real billing cap.

    Trip when: at least `threshold` exhausted-retry quota errors AND the most
    recent success was more than `cold_window_seconds` ago.
    """

    def __init__(self, threshold=QUOTA_BREAKER_THRESHOLD, cold_window_seconds=120):
        self._lock = Lock()
        self.threshold = threshold
        self.cold_window_seconds = cold_window_seconds
        self.consecutive = 0
        self.total = 0
        self.last_success_ts = time.monotonic()
        self.aborted = False

    def record_quota_error(self):
        with self._lock:
            self.consecutive += 1
            self.total += 1
            cold = (time.monotonic() - self.last_success_ts) >= self.cold_window_seconds
            if not self.aborted and self.consecutive >= self.threshold and cold:
                self.aborted = True
                logging.error(
                    f"QuotaCircuitBreaker tripped: {self.consecutive} exhausted-retry "
                    f"quota errors with no LLM success for "
                    f"{int(time.monotonic() - self.last_success_ts)}s. "
                    f"This looks like a real billing cap. Aborting further LLM calls."
                )
            return self.aborted

    def record_success(self):
        with self._lock:
            self.consecutive = 0
            self.last_success_ts = time.monotonic()

    def record_non_quota_error(self):
        with self._lock:
            self.consecutive = 0

    def is_aborted(self):
        with self._lock:
            return self.aborted


# ----------------------------------------------------------------------
# AustLII data error log
# ----------------------------------------------------------------------
#
# Some AustLII pages exist in the index but the actual viewer page is
# broken on AustLII's end (e.g. <article class="the-document"><h1>No title</h1>
# Can't extract contents</article>). These are not scraper bugs; they need
# manual review and potentially manual upload of the source from elsewhere.

AUSTLII_ERROR_LOG = os.path.join(OUTPUT_ROOT, "austlii_data_errors.json")
_austlii_error_lock = Lock()


def _load_austlii_error_log():
    if not os.path.exists(AUSTLII_ERROR_LOG):
        return {}
    try:
        with open(AUSTLII_ERROR_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Failed to load {AUSTLII_ERROR_LOG}: {e}")
        return {}


def record_austlii_data_error(url, *, case_name, error_type, local_file="",
                              extracted_chars=0, notes="", live_verified=False):
    """Append (or update) an AustLII data-error entry for later manual review.

    Thread-safe. Idempotent on (url, error_type): subsequent detections bump
    `detection_count` and `last_detected_at` but don't duplicate the entry.
    """
    with _austlii_error_lock:
        log = _load_austlii_error_log()
        now = datetime.datetime.now().isoformat(timespec="seconds")
        entry = log.get(url, {})
        if entry and entry.get("error_type") == error_type:
            entry["detection_count"] = entry.get("detection_count", 1) + 1
            entry["last_detected_at"] = now
        else:
            entry = {
                "url": url,
                "case_name": case_name,
                "error_type": error_type,
                "first_detected_at": now,
                "last_detected_at": now,
                "detection_count": 1,
                "local_file": local_file,
                "extracted_chars": extracted_chars,
                "notes": notes,
                "live_verified": live_verified,
                "resolved": False,
            }
        # Optional fields can be updated each call
        if local_file:
            entry["local_file"] = local_file
        if extracted_chars:
            entry["extracted_chars"] = extracted_chars
        if notes:
            entry["notes"] = notes
        if live_verified:
            entry["live_verified"] = True
        log[url] = entry

        tmp = AUSTLII_ERROR_LOG + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, AUSTLII_ERROR_LOG)
        except OSError as e:
            logging.error(f"Failed to write {AUSTLII_ERROR_LOG}: {e}")


class CostTracker:
    def __init__(self):
        self._lock = Lock()
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0

    def record(self, usage):
        if usage is None:
            return {}
        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        reasoning = 0
        cdetails = getattr(usage, "completion_tokens_details", None)
        if cdetails is not None:
            reasoning = getattr(cdetails, "reasoning_tokens", 0) or 0
        non_cached = prompt - cached
        cost = (non_cached * PRICE_INPUT_PER_M / 1_000_000
                + cached * PRICE_CACHED_INPUT_PER_M / 1_000_000
                + completion * PRICE_OUTPUT_PER_M / 1_000_000)
        with self._lock:
            self.prompt_tokens += prompt
            self.cached_tokens += cached
            self.completion_tokens += completion
            self.reasoning_tokens += reasoning
            self.calls += 1
        return {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "cost_usd": cost,
        }

    def total_cost(self):
        non_cached = self.prompt_tokens - self.cached_tokens
        return (non_cached * PRICE_INPUT_PER_M / 1_000_000
                + self.cached_tokens * PRICE_CACHED_INPUT_PER_M / 1_000_000
                + self.completion_tokens * PRICE_OUTPUT_PER_M / 1_000_000)


# ----------------------------------------------------------------------
# LLM extractor (single combined call)
# ----------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a senior legal analyst preparing a complete dossier from a NSW
Personal Injury Commission decision. Produce one structured response with:

(1) STRUCTURED FACTS — named parties, dates, amounts, classifications.
    Amounts as nominal numbers (e.g. 150000.00); no $ or commas. If a money
    figure is a regulated costs order rather than compensation/damages, do
    NOT put it in lump_sum_amount.

    HIGH-VALUE TARGET FIELDS — these eight are the most important outputs and
    are frequently buried deep in the decision. Actively SEARCH the whole
    document (including the quantum/assessment and orders sections, which are
    usually near the end) before deciding any of them is absent. Use the
    'Not stated' / empty / 'Not addressed' sentinel ONLY after a genuine
    search, never as a shortcut:
      - WPI % (impairment_percentage / impairment_percentage_accepted)
      - non_economic_loss (+ non_economic_loss_status)
      - claimant_weekly_income (+ claimant_weekly_income_basis)
      - future_economic_loss (+ future_economic_loss_status)
      - claimant_age (at injury), claimant_age_at_decision, claimant_date_of_birth
      - claimant_gender
      - claimant_occupation
      - location_of_accident_or_injury (+ location_locality, location_state)

    WPI fields — TWO fields, different semantics:
      * impairment_percentage = WPI MADE in this proceeding (the Member made
        the binding finding). EMPTY for CTP settlement approvals and damages
        assessments — those merely accept a prior MAS certificate, they
        don't make the assessment.
      * impairment_percentage_accepted = WPI USED for the award, regardless
        of who assessed it. For settlement approvals and damages this is
        almost always present (and equals the prior MAS figure). Use the
        COMBINED/TOTAL value the Member relied on; if only component WPIs
        are stated for different body parts, use the HIGHEST component.
      Do NOT emit 0 — a '0% WPI' mention is almost always statutory threshold
      / minor-injury framing, not a finding about this claimant.

    DAMAGES HEADS use a status + amount pair so 'refused' is never confused
    with 'not dealt with':
      * non_economic_loss / future_economic_loss = the dollar amount (number
        only) when AWARDED; leave EMPTY otherwise.
      * *_status = Awarded (amount allowed) / Nil (claimed but refused or
        assessed at zero) / Not addressed (head not dealt with in this case).

    INCOME: emit a single normalised weekly number AND record in
    claimant_weekly_income_basis what it represents (PIAWE/pre/post, gross/net)
    and any conversion you performed (annual/52, monthly, hourly x hours).

    AGE: claimant_age = age AT INJURY; claimant_age_at_decision = age at the
    decision; claimant_date_of_birth = DOB (full date or year) for cross-check.
    DERIVE age by arithmetic when not stated directly: from year of birth, or
    from a stated current age minus the injury->decision gap (a claimant 'now
    31' in a 2021 decision for a 2018 injury was 28 at injury). Do not return
    'Not stated' for age when a current age and the injury year are both known.

(1b) PROVENANCE — in the `provenance` object, for EACH of the eight target
    fields, copy the VERBATIM source snippet that ESTABLISHES your value (<=200
    chars). The quote must support a CAPTURED value, not describe its absence:
    if the field is 'Not stated' / Nil / Not addressed (e.g. the claim was not
    pressed, the claimant was unemployed, the matter was remitted for later
    assessment), leave that quote EMPTY. A non-empty quote beside an empty value
    is treated as an error and will be re-checked, so keep them consistent.

(2) CASE_DESCRIPTION — a comprehensive 500-700 word executive summary in ONE
    PARAGRAPH (no newlines), covering mechanism, injuries, treatment,
    function, evidence, legal framework, reasoning, and outcome with specific
    amounts/orders. CRITICAL: do NOT include proper-noun names of parties,
    the Member, doctors, experts, lawyers, hospitals, or firms. Use generic
    references: 'the claimant', 'the insurer', 'the employer', 'the Member',
    'a consultant psychiatrist', 'the treating GP', 'a metropolitan public
    hospital'. Cited case authorities and statutory references ARE preserved.
    Use ASCII only (straight quotes, regular hyphens, three dots — no smart
    quotes, em-dashes, non-breaking hyphens, ellipsis character).

    Also produce BANDED_CASE_DESCRIPTION: the same paragraph with numeric
    target-leaking content replaced by band tokens per the rules in the
    schema field description (dates -> [DATE]; ages -> [AGE_BAND:...];
    WPI -> [WPI_RANGE:...]; payouts/NEL/FEL/income/commissions -> their
    respective range tokens). Same prose otherwise. Same anonymisation
    rules. Relative durations and section numbers stay verbatim.

(3) NARRATIVE SUB-FIELDS — 60-150 words each. If a fact is not stated in the
    source, write 'Not stated' for that detail rather than omitting it.
    Includes three submission-related fields: previous_insurer_actions_and_offers
    (the insurer's pre-proceeding conduct), claimant_submissions, and
    insurer_submissions (the arguments advanced at the hearing).

(4) SHORT SLICES — for each of catchwords, determinations/orders, and
    introduction, return start_marker and end_marker substrings copied
    VERBATIM from the source (30-80 chars each). Do NOT paraphrase markers —
    another program will use them to cut the slice from the source.

(5) KEY_PARAGRAPHS — pick 4-8 of the most analytically important numbered
    paragraphs from the Member's reasoning/findings. Return paragraph NUMBERS
    only (e.g. 64). The verbatim text will be cut from the source in code —
    do NOT return paragraph text. Choose paragraphs with statutory pin-cites,
    neutral case citations, the framing of the issue, and the conclusion.

(6) ORDINAL SCORES — assign integer scores per each field's rubric. Score
    against the rubric in the schema, not against your subjective impression.
    For procedural / fault-only decisions, score the dimensions on the facts
    discussed; missing dimensions score 0.

(7) EVENT_HISTORY — chronological dated events with actor and tag.
"""


# Quantum / claimant-profile keywords — where the eight high-value target
# fields live. These are filled FIRST so they win the truncation budget even in
# a very long decision whose generic structural headings come earlier (ISSUE-017).
_TRUNCATE_TARGET_KEYWORDS = (
    "non-economic loss", "non economic loss", "general damages", "pain and suffering",
    "future economic loss", "loss of earning", "earning capacity", "buffer", "quantum",
    "damages assessment", "whole person impairment", "impairment", "wpi",
    "piawe", "per week", "weekly", "pre-injury", "salary", "wage",
    "date of birth", "born", "aged", "occupation", "employed",
)
# Generic structural headings — lower priority; dropped first when the cap binds.
_TRUNCATE_STRUCTURE_KEYWORDS = (
    "background", "facts", "history", "particulars", "mechanism", "medical evidence",
    "pre-existing", "treatment", "diagnosis", "surgery", "expert evidence", "submissions",
    "reasoning", "findings", "reasons", "discussion", "issues", "orders", "conclusion", "decision",
)
_TRUNCATE_BEFORE, _TRUNCATE_AFTER = 3000, 10000


def _merge_ranges(ranges):
    """Merge a list of [start, end) into sorted, non-overlapping ranges."""
    if not ranges:
        return []
    rs = sorted([list(r) for r in ranges])
    merged = [rs[0]]
    for s, e in rs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _ranges_len(ranges):
    return sum(e - s for s, e in ranges)


def _narrative_truncate(text):
    """If the source overflows the single-pass char limit, keep the start chunk
    plus windows around key sections — but allocate the budget to the high-value
    QUANTUM/CLAIMANT-PROFILE keywords BEFORE generic structural headings, and
    deduplicate overlapping windows by RANGE rather than exact text (ISSUE-017).

    Without this, a long decision with many early Background/Facts/Reasons hits
    could fill the cap before the late non-economic-loss / WPI / PIAWE / date-of-
    birth sections survive, producing false empty fields. Windows are anchored on
    both the first and last occurrence of each keyword (the operative quantum
    assessment is usually the later mention)."""
    cap = SINGLE_PASS_LIMIT_CHARS
    if len(text) <= cap:
        return text
    lowered = text.lower()
    n = len(text)

    def windows_for(kw):
        out = []
        first = lowered.find(kw)
        if first != -1:
            out.append((max(0, first - _TRUNCATE_BEFORE), min(n, first + _TRUNCATE_AFTER)))
            last = lowered.rfind(kw)
            if last != -1 and abs(last - first) > 8000:
                out.append((max(0, last - _TRUNCATE_BEFORE), min(n, last + _TRUNCATE_AFTER)))
        return out

    # Priority order: start chunk, then TARGET keyword windows, then STRUCTURE.
    candidates = [(0, min(30000, cap))]
    for kw in _TRUNCATE_TARGET_KEYWORDS:
        candidates.extend(windows_for(kw))
    for kw in _TRUNCATE_STRUCTURE_KEYWORDS:
        candidates.extend(windows_for(kw))

    # Greedily accept windows in priority order, merging by range, never letting
    # the kept content exceed the cap. Earlier (higher-priority) windows win the
    # budget. When a window doesn't fully fit, TRIM it to the remaining budget
    # (keeping its front, which holds the keyword at offset _TRUNCATE_BEFORE)
    # rather than dropping it wholesale.
    accepted = []
    total = 0
    for s, e in candidates:
        remaining = cap - total
        if remaining <= 0:
            break
        e = min(e, s + remaining)
        if e <= s:
            continue
        trial = _merge_ranges(accepted + [[s, e]])
        tlen = _ranges_len(trial)
        if tlen <= cap:
            accepted, total = trial, tlen
    accepted.sort()
    return "\n\n...[SECTION BREAK]...\n\n".join(text[s:e].strip() for s, e in accepted)


# ----------------------------------------------------------------------
# Focused second pass (A6)
# ----------------------------------------------------------------------
#
# Fires only when the loss gate (detect_field_losses) believes a high-value
# field is present in the source but the first pass left it empty. It re-asks
# ONLY for the suspect fields, optionally at a higher reasoning effort. Cheap,
# because it runs on the handful of decisions that need it and returns a small
# schema.

class FocusedFields(BaseModel):
    wpi_percent: str = Field(description="Whole Person Impairment % relied on for the award (combined/total, or highest component). Number only. EMPTY if no WPI is stated; never emit 0 for threshold/minor-injury framing.")
    non_economic_loss: str = Field(description="Non-economic loss / general damages amount as a number (no $/commas). EMPTY if not awarded.")
    non_economic_loss_status: QuantumStatusEnum = Field(description="Awarded / Nil (refused) / Not addressed.")
    claimant_weekly_income: str = Field(description="Total weekly employment income as a single number (convert annual/monthly/hourly to weekly; prefer pre-injury PIAWE, gross). EMPTY if none.")
    claimant_weekly_income_basis: str = Field(description="What the income figure represents and any conversion (e.g. 'PIAWE gross weekly'; 'annual 89500 / 52'). 'Not stated' if none.")
    future_economic_loss: str = Field(description="Future economic loss / buffer amount as a number (no $/commas). EMPTY if not awarded.")
    future_economic_loss_status: QuantumStatusEnum = Field(description="Awarded / Nil (refused) / Not addressed.")
    claimant_age: str = Field(description="Claimant's age at injury as a number (derive from year of birth if needed). 'Not stated' if neither age nor birth year appears.")
    claimant_gender: GenderEnum = Field(description="Male / Female / Other / Not stated (from explicit statement or unambiguous gendered references).")
    claimant_occupation: str = Field(description="Claimant's occupation at time of injury. 'Not stated' if absent.")
    location: str = Field(description="Where the injury/accident occurred. 'Not stated' if absent.")


_FOCUSED_SYSTEM_INSTRUCTION = """\
You are re-examining a NSW Personal Injury Commission decision because a first
extraction pass may have MISSED one or more specific high-value fields. Read
the WHOLE source carefully — especially the quantum/assessment and orders
sections, which are usually near the end — and extract the requested fields.

Semantics:
  - Money amounts are plain numbers (no $ or commas).
  - WPI: the percentage relied on for the award; combined/total, or the highest
    component if only components are stated. Never emit 0 for statutory
    threshold / minor-injury language.
  - Income: a single normalised weekly number; convert annual/monthly/hourly;
    prefer pre-injury PIAWE and gross.
  - Damages heads use status (Awarded/Nil/Not addressed) + amount.
  - Age: prefer age at injury; derive from year of birth if needed.

Only use a 'Not stated' / empty sentinel after a genuine search confirms the
fact is absent. Fill every field in the schema, but the caller will only use
the fields it flagged as missing.
"""


class LLMExtractor:
    def __init__(self, api_key, timeout=OPENAI_TIMEOUT):
        # Explicit per-request timeout so a stalled call fails on a bounded
        # deadline instead of pinning a worker thread (ISSUE-016). max_retries=0
        # because _parse_with_retry owns the retry/backoff policy.
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def extract_text_from_html(self, html_content):
        return extract_html_with_paragraph_numbers(html_content)

    def extract_text_from_pdf(self, pdf_content):
        try:
            reader = PdfReader(io.BytesIO(pdf_content))
        except Exception as e:
            logging.error(f"Unable to open PDF for extraction: {e}")
            return ""
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as e:
                logging.warning(f"Failed to extract text from PDF page {page_number}: {e}")
                page_text = ""
            if page_text.strip():
                pages.append(page_text.strip())
        return "\n\n".join(pages).strip()

    def _parse_with_retry(self, system_instruction, user_content, response_format,
                          context=None, reasoning_effort=REASONING_EFFORT):
        """Shared structured-parse call with insufficient_quota backoff.

        Returns (parsed, usage, error). OpenAI auto-recharges when the balance
        drops below a threshold, and during the ~10-30s recharge window the API
        returns 429/insufficient_quota even though the account is in good
        standing — treat it like a transient rate limit. Backoff schedule:
        2, 5, 10, 20, 40, 80 seconds (covers ~2.5 minutes of recharge / brief
        outage).
        """
        backoff_schedule = [2, 5, 10, 20, 40, 80]
        last_error = None
        for attempt in range(len(backoff_schedule) + 1):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                )
                return completion.choices[0].message.parsed, completion.usage, None
            except Exception as e:
                last_error = e
                ctx = f" ({context})" if context else ""
                retryable = _is_quota_error(str(e)) or _is_transient_api_error(e)
                if retryable and attempt < len(backoff_schedule):
                    delay = backoff_schedule[attempt]
                    kind = "insufficient_quota" if _is_quota_error(str(e)) else "timeout/connection"
                    logging.warning(
                        f"{kind}{ctx} - retry {attempt+1}/{len(backoff_schedule)} in {delay}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                # Distinguish timeout/connection failures from schema/parse errors.
                if _is_transient_api_error(e):
                    logging.error(f"LLM timeout/connection error{ctx} after retries: {e}")
                else:
                    logging.error(f"LLM parse error{ctx}: {e}")
                return None, None, str(e)
        return None, None, str(last_error)

    def extract_combined(self, source_text, context=None):
        """
        Single combined gpt-5 call. Returns (parsed, usage, error). The caller
        is responsible for further post-processing (sanitising case_description,
        resolving slices and key paragraphs against the full source).
        """
        if not source_text:
            return None, None, "empty source"

        processed = _narrative_truncate(source_text)
        user_content = (
            "Source text of the decision follows. Produce the combined extraction.\n\n"
            "---\n"
            f"{processed}\n"
            "---\n"
        )
        return self._parse_with_retry(
            _SYSTEM_INSTRUCTION, user_content, CombinedSchema,
            context=context, reasoning_effort=REASONING_EFFORT,
        )

    def extract_focused(self, source_text, fields_needed, context=None):
        """Focused second pass for fields the first pass appears to have missed.

        `fields_needed` is an iterable of focused-field keys (the second element
        of FIELD_LOSS_SPEC values, e.g. 'wpi_percent', 'claimant_age'). Returns
        (parsed_FocusedFields, usage, error).
        """
        if not source_text:
            return None, None, "empty source"
        needed = sorted(set(fields_needed or []))
        if not needed:
            return None, None, "no fields requested"

        processed = _narrative_truncate(source_text)
        user_content = (
            "A first extraction pass may have MISSED these fields:\n"
            f"  {', '.join(needed)}\n\n"
            "Re-read the decision below and extract them (fill the whole schema; "
            "only the listed fields will be used). Search the entire document, "
            "including the quantum/assessment and orders sections.\n\n"
            "---\n"
            f"{processed}\n"
            "---\n"
        )
        return self._parse_with_retry(
            _FOCUSED_SYSTEM_INSTRUCTION, user_content, FocusedFields,
            context=context, reasoning_effort=FOCUSED_REASONING_EFFORT,
        )

    def extract_wpi_resolution(self, source_text, context=None):
        """Classify every WPI figure in a decision so the caller can resolve
        them (see wpi_resolution.py). Used when the source holds more than one
        figure, where a regex cannot tell a component from a rival assessment
        from a statutory-threshold recital."""
        if not source_text:
            return None, None, "empty source"
        processed = _narrative_truncate(source_text)
        user_content = (
            "Classify every whole-person-impairment percentage in the decision "
            "below. Do NOT do arithmetic; report the figures as you find them.\n\n"
            "---\n"
            f"{processed}\n"
            "---\n"
        )
        return self._parse_with_retry(
            WPI_SYSTEM_INSTRUCTION, user_content, WpiResolution,
            context=context, reasoning_effort=WPI_RESOLUTION_REASONING_EFFORT,
        )

    def extract_damages(self, source_text, context=None):
        """Damages-breakdown pass (downstream spec 2026-07-27).

        A separate call rather than extra fields on CombinedSchema: the eight
        high-value fields the consumer already relies on must not regress, and
        the claimed-vs-allowed distinction needs a prompt that talks about
        nothing else. Returns (parsed_DamagesSchema, usage, error).
        """
        if not source_text:
            return None, None, "empty source"
        processed = build_damages_context(source_text)
        user_content = (
            "Reconstruct the award breakdown from the decision below. Report "
            "what was ALLOWED, never what was claimed or refused, and leave a "
            "figure EMPTY rather than guessing it.\n\n"
            "---\n"
            f"{processed}\n"
            "---\n"
        )
        return self._parse_with_retry(
            DAMAGES_SYSTEM_INSTRUCTION, user_content, DamagesSchema,
            context=context, reasoning_effort=DAMAGES_REASONING_EFFORT,
        )


class DecisionScraper:
    def __init__(self, base_url, output_folder="nsw_decisions", api_key=None):
        self.base_url = base_url
        self.output_folder = output_folder
        self.extractor = LLMExtractor(api_key) if api_key else None
        self.cache_file = CACHE_FILE
        self.sidecar_file = SIDECAR_FILE
        self.cache_lock = Lock()
        self.cache = self._load_cache()
        self.cost_tracker = CostTracker()
        self.quota_breaker = QuotaCircuitBreaker()
        self.rate_limit_lock = Lock()
        self.rate_limit_triggered = False
        self.next_request_time = 0.0
        self.rate_limit_delay = float(os.getenv("AUSTLII_RATE_LIMIT_DELAY", "5"))
        self.rate_limit_success_count = 0
        self.rate_limit_reset_threshold = int(os.getenv("AUSTLII_RATE_LIMIT_RESET_THRESHOLD", "3"))

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        }

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _is_current_schema(self, row):
        return isinstance(row, dict) and row.get("_schema_version") == SCHEMA_VERSION

    def _backup_corrupt_cache(self, reason):
        """Move an unusable cache aside to a TIMESTAMPED path so prior corrupt
        input is preserved for diagnosis and repeated events don't clobber each
        other. Tolerant of OSError so recovery still proceeds (ISSUE-008)."""
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{self.cache_file}.corrupted.{stamp}"
        try:
            shutil.move(self.cache_file, backup)
            logging.error(f"Cache {self.cache_file} unusable ({reason}); moved to {backup}; starting empty.")
        except OSError as e:
            logging.error(f"Cache {self.cache_file} unusable ({reason}) and backup failed ({e}); "
                          f"starting empty without moving the file.")

    def _load_cache(self):
        """Loads cache safely. If corrupted or not a dict, backs up (timestamped)
        and starts empty. Never raises on a bad cache file (ISSUE-008)."""
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                loaded_cache = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            self._backup_corrupt_cache(f"read/parse error: {e}")
            return {}
        if not isinstance(loaded_cache, dict):
            self._backup_corrupt_cache("top-level JSON is not an object")
            return {}
        return {
            url: annotate_analysis_fields(data) if isinstance(data, dict) else data
            for url, data in loaded_cache.items()
        }

    def _save_cache(self, max_retries=3):
        """Atomic cache write via a per-process temp file (ISSUE-003). Callers
        that also write the sidecar/reports should hold `dataset_lock()` around
        the whole batch; this method does NOT self-lock (the lock is not
        reentrant)."""
        with self.cache_lock:
            cache_copy = self.cache.copy()
        for attempt in range(1, max_retries + 1):
            try:
                atomic_write_json(self.cache_file, cache_copy)
                return True
            except Exception as e:
                logging.error(f"Failed to save cache (attempt {attempt}/{max_retries}): {e}")
                time.sleep(min(2 ** attempt, 5))
        return False

    def update_cache(self, url, data):
        """Thread-safe cache update helper"""
        with self.cache_lock:
            self.cache[url] = annotate_analysis_fields(data)

    def _save_sidecar(self):
        """
        Write a sidecar JSON file containing only the long/nested fields
        (narrative sub-fields, slices, key paragraphs, event history,
        regulatory sections, token usage) keyed by URL. The flat CSV columns
        already live in the main cache; this is the rich data CSV cells
        cannot comfortably hold.
        """
        sidecar = {}
        with self.cache_lock:
            for url, row in self.cache.items():
                if not isinstance(row, dict):
                    continue
                entry = {
                    "Case Name": row.get("Case Name", ""),
                    "File Saved": row.get("File Saved", ""),
                    "Status": row.get("Status", ""),
                    "Needs Review": row.get("Needs Review", ""),
                    "Review Notes": row.get("Review Notes", ""),
                    "_schema_version": row.get("_schema_version"),
                    "narrative": row.get("_narrative", {}),
                    "slices": row.get("_slices", {}),
                    "key_paragraphs": row.get("_key_paragraphs", []),
                    "event_history": row.get("_event_history", []),
                    "regulatory_sections": (row.get("Regulatory Sections") or "").split(" | "),
                    "token_usage": row.get("_token_usage", {}),
                    "banding_validation": row.get("_banding_validation", {}),
                    "provenance": row.get("_provenance", {}),
                    "field_review": row.get("_field_review", {}),
                    "damages": row.get("_damages", {}),
                    "wpi_resolution": row.get("_wpi_resolution", {}),
                    "wpi_version": row.get("_wpi_version", 0),
                    "damages_version": row.get("_damages_version", 0),
                }
                # Privacy transforms apply to the OUTPUT sidecar only; the cache
                # keeps full data (ISSUE-011). No-op unless a privacy knob is set.
                sidecar[url] = apply_privacy_to_sidecar(entry)

        try:
            atomic_write_json(self.sidecar_file, sidecar, ensure_ascii=False)
            logging.info(f"Sidecar JSON saved to {self.sidecar_file} ({len(sidecar)} entries)")
            return True
        except Exception as e:
            logging.error(f"Failed to save sidecar: {e}")
            return False

    def _throttle_if_rate_limited(self):
        delay = max(self.rate_limit_delay, 0)
        if delay == 0:
            return
        with self.rate_limit_lock:
            if not self.rate_limit_triggered:
                return
            now = time.monotonic()
            if now < self.next_request_time:
                time.sleep(self.next_request_time - now)
                now = time.monotonic()
            self.next_request_time = now + delay

    def _make_request_with_retry(self, url, max_retries=5):
        # Use curl_cffi with Chrome TLS impersonation so we pass Cloudflare's
        # bot management check (without it AustLII returns 403 on every request).
        for attempt in range(max_retries):
            self._throttle_if_rate_limited()
            try:
                response = cf_requests.get(url, impersonate="chrome", timeout=30)

                if response.status_code == 200:
                    with self.rate_limit_lock:
                        if self.rate_limit_triggered:
                            self.rate_limit_success_count += 1
                            if self.rate_limit_success_count >= self.rate_limit_reset_threshold:
                                self.rate_limit_triggered = False
                                self.rate_limit_success_count = 0
                                self.next_request_time = 0.0
                    return response

                if response.status_code in [403, 429, 500, 502, 503, 504]:
                    if response.status_code in [403, 429]:
                        with self.rate_limit_lock:
                            self.rate_limit_triggered = True
                            self.rate_limit_success_count = 0
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Request failed ({response.status_code}) for {url}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Request failed ({response.status_code}) for {url}. No retry.")
                    return None

            except Exception as e:
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                logging.warning(f"Connection error ({e}) for {url}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)

        logging.error(f"Max retries exceeded for {url}")
        return None

    def get_decision_links(self, index_url):
        logging.info(f"Fetching index: {index_url}")
        
        response = self._make_request_with_retry(index_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        decision_url_pattern = re.compile(r"\/NSWPIC\/\d{4}\/[^\/]+\.(?:html|pdf)", re.IGNORECASE)
        nswpic_pattern = re.compile(r"\/NSWPIC\/\d{4}\/", re.IGNORECASE)

        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(self.base_url, href)
            
            if decision_url_pattern.search(full_url):
                title = a.get_text(" ", strip=True)
                if 5 < len(title) < 250:
                    links.append((title, full_url))
            elif nswpic_pattern.search(full_url):
                logging.warning(f"Potential decision link did not match expected pattern: {full_url}")
        
        unique_links = {}
        for title, url in links:
            if url not in unique_links:
                unique_links[url] = title
        
        return [(title, url) for url, title in unique_links.items()]

    def process_decision(self, title, url):
        if not self.extractor:
            raise RuntimeError("LLM extractor is not configured. Ensure OPENAI_API_KEY is set.")
        safe_title_base = re.sub(r'[^\w\s\-\.]', '', title)
        safe_title_base = re.sub(r'[\s]+', '_', safe_title_base)
        case_id_match = re.search(r"/NSWPIC/(\d{4})/(\d+)\.(?:html|pdf)", url, re.IGNORECASE)
        case_id = None
        if case_id_match:
            case_id = f"{case_id_match.group(1)}_{case_id_match.group(2)}"
        if case_id:
            suffix = f"_{case_id}"
        else:
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
            suffix = f"_unknown_case_{url_hash}"
        safe_title_base = f"{safe_title_base[:100]}{suffix}"

        log_title = (title[:75] + '...') if len(title) > 75 else title

        # Reuse cache only if the entry was produced by the current schema.
        with self.cache_lock:
            cached = self.cache.get(url)
            if self._is_current_schema(cached):
                return cached

        # If the quota breaker has already tripped, skip BEFORE any network/file
        # work — no further LLM extraction can succeed, so don't burn I/O on it
        # (ISSUE-013).
        if self.quota_breaker.is_aborted():
            return None

        # Predict the local filename from URL+title so we can skip the network
        # entirely when we already have the file on disk.
        url_ext = ".pdf" if url.lower().endswith(".pdf") else ".html"
        expected_safe_title = f"{safe_title_base}{url_ext}"
        expected_full_path = os.path.join(self.output_folder, expected_safe_title)

        is_pdf = (url_ext == ".pdf")
        raw_text = None
        safe_title = expected_safe_title
        full_path = expected_full_path

        if os.path.exists(expected_full_path):
            logging.info(f"Using local file for: {log_title}")
            try:
                with open(expected_full_path, "rb") as f:
                    file_bytes = f.read()
                if is_pdf:
                    raw_text = self.extractor.extract_text_from_pdf(file_bytes)
                else:
                    raw_text = self.extractor.extract_text_from_html(file_bytes)
            except Exception as e:
                logging.warning(f"Failed to read local file {expected_full_path}: {e} - refetching")
                raw_text = None

        if raw_text is None:
            logging.info(f"Fetching: {log_title}")
            response = self._make_request_with_retry(url)
            if not response:
                logging.error(f"Failed to fetch content for {log_title} after retries.")
                return None

            content_type = response.headers.get("Content-Type", "").lower()
            is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type
            is_html = "html" in content_type or not content_type

            if is_pdf:
                file_extension = ".pdf"
                raw_text = self.extractor.extract_text_from_pdf(response.content)
            elif is_html:
                file_extension = ".html"
                raw_text = self.extractor.extract_text_from_html(response.content)
            else:
                logging.warning(f"Unsupported decision content skipped: {url} ({content_type})")
                return None

            safe_title = f"{safe_title_base}{file_extension}"
            full_path = os.path.join(self.output_folder, safe_title)
            if os.path.exists(full_path):
                url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
                root_name, extension = os.path.splitext(safe_title)
                safe_title = f"{root_name}_{url_hash}{extension}"
                full_path = os.path.join(self.output_folder, safe_title)
            with open(full_path, "wb") as f:
                f.write(response.content)

        decision_text = cleanup_text(raw_text)
        if len(decision_text) < 500:
            logging.warning(f"Decision text too short; skipping {log_title}")
            record_austlii_data_error(
                url,
                case_name=title,
                error_type="html_no_content",
                local_file=safe_title,
                extracted_chars=len(decision_text),
                notes=(
                    "AustLII viewer rendered an empty <article class='the-document'> "
                    "with 'No title / Can\\'t extract contents'. Decision listed in index "
                    "but body is not served; needs manual review."
                ),
            )
            return None

        if self.quota_breaker.is_aborted():
            logging.warning(f"Skipping {log_title}: quota breaker tripped")
            return None

        parsed, usage, llm_error = self.extractor.extract_combined(
            decision_text, context=f"title={log_title}, url={url}",
        )
        token_usage = self.cost_tracker.record(usage) if usage else {}

        # Cache write rule: ONLY on a valid parsed extraction. Every failure
        # path leaves the cache untouched so the URL is reprocessed naturally
        # on the next run.
        if llm_error or parsed is None:
            if _is_quota_error(llm_error):
                self.quota_breaker.record_quota_error()
            else:
                self.quota_breaker.record_non_quota_error()
                logging.error(f"LLM error for {log_title}: {llm_error or 'parse failed'}")
            return None

        self.quota_breaker.record_success()
        result_data = self._build_record_from_parsed(
            title=title, url=url, file_saved=safe_title,
            parsed=parsed, decision_text=decision_text,
            token_usage=token_usage,
        )
        self.update_cache(url, result_data)
        return result_data

    def _build_record_from_parsed(self, *, title, url, file_saved, parsed, decision_text, token_usage):
        sanitised_case_description = sanitise_case_description(parsed.case_description)
        sanitised_banded_description = sanitise_case_description(parsed.banded_case_description)

        # Resolve slices against the cleaned source.
        slices = {}
        for attr in ("catchwords", "determinations_or_orders", "introduction"):
            loc = getattr(parsed, attr)
            entry = {
                "present": bool(loc.present),
                "start_marker": loc.start_marker or "",
                "end_marker": loc.end_marker or "",
                "text": "",
                "resolution_error": "",
            }
            if loc.present:
                slice_text, _, err = find_slice(decision_text, loc.start_marker, loc.end_marker)
                if slice_text is None:
                    entry["resolution_error"] = err or "unresolved"
                else:
                    entry["text"] = slice_text
            slices[attr] = entry

        # Resolve key paragraphs against the cleaned source.
        para_lookup = extract_numbered_paragraphs(decision_text)
        key_paragraphs = []
        for kp in (parsed.key_paragraphs or []):
            text = para_lookup.get(kp.paragraph_number, "")
            key_paragraphs.append({
                "paragraph_number": kp.paragraph_number,
                "rationale": kp.rationale,
                "text": text,
                "resolved": bool(text),
            })

        event_history = [
            {"date": ev.date, "actor": ev.actor, "tag": ev.tag}
            for ev in (parsed.event_history or [])
        ]

        narrative = {
            "claimant_profile": parsed.claimant_profile,
            "accident_or_injury_mechanism": parsed.accident_or_injury_mechanism,
            "injuries_and_diagnoses": parsed.injuries_and_diagnoses,
            "treatment_history": parsed.treatment_history,
            "functional_impact_and_work_capacity": parsed.functional_impact_and_work_capacity,
            "medical_evidence_summary": parsed.medical_evidence_summary,
            "previous_insurer_actions_and_offers": parsed.previous_insurer_actions_and_offers,
            "claimant_submissions": parsed.claimant_submissions,
            "insurer_submissions": parsed.insurer_submissions,
            "legal_issues_and_reasoning": parsed.legal_issues_and_reasoning,
        }

        # ---- Provenance (A2): per-field verbatim quotes from the model ----
        prov = getattr(parsed, "provenance", None)
        provenance = {
            "wpi_quote": getattr(prov, "wpi_quote", "") or "",
            "non_economic_loss_quote": getattr(prov, "non_economic_loss_quote", "") or "",
            "weekly_income_quote": getattr(prov, "weekly_income_quote", "") or "",
            "future_economic_loss_quote": getattr(prov, "future_economic_loss_quote", "") or "",
            "age_quote": getattr(prov, "age_quote", "") or "",
            "gender_quote": getattr(prov, "gender_quote", "") or "",
            "occupation_quote": getattr(prov, "occupation_quote", "") or "",
            "location_quote": getattr(prov, "location_quote", "") or "",
        }

        review_issues = []

        # ---- WPI reconciliation (B1): live regex/threshold safeguards ----
        wpi_strict, wpi_accepted, wpi_issues = reconcile_wpi(
            parsed.impairment_percentage,
            getattr(parsed, "impairment_percentage_accepted", ""),
            decision_text,
        )
        review_issues.extend(wpi_issues)

        # ---- Damages heads (B2): status + coerced numeric amount ----
        nel_status = parsed.non_economic_loss_status.value
        nel_amount = coerce_money(parsed.non_economic_loss)
        if nel_status == "Nil" and not nel_amount:
            nel_amount = "0"
        fel_status = parsed.future_economic_loss_status.value
        fel_amount = coerce_money(parsed.future_economic_loss)
        if fel_status == "Nil" and not fel_amount:
            fel_amount = "0"

        # ---- Weekly income (B3): coercion + range plausibility ----
        income_amount = coerce_money(parsed.claimant_weekly_income)
        if income_amount:
            try:
                iv = float(income_amount)
                if iv < INCOME_WEEKLY_MIN or iv > INCOME_WEEKLY_MAX:
                    review_issues.append({
                        "field": "Claimant Weekly Income", "type": "income_out_of_range",
                        "severity": "medium",
                        "detail": f"weekly income {income_amount} outside "
                                  f"[{INCOME_WEEKLY_MIN:g}, {INCOME_WEEKLY_MAX:g}] - possible unit error",
                    })
            except ValueError:
                pass

        # ---- Age (B4): at-injury / at-decision + DOB cross-check ----
        age_injury = coerce_leading_number(parsed.claimant_age)
        age_decision = coerce_leading_number(getattr(parsed, "claimant_age_at_decision", ""))
        # Derive one age from the other via the injury->decision gap (no DOB
        # needed): e.g. 'now 31' at a 2021 decision for a 2018 injury => 28 at
        # injury. Fills blanks only.
        age_injury, age_decision = reconcile_ages(
            age_injury, age_decision, parsed.date_of_injury, parsed.date_of_decision)
        dob = (getattr(parsed, "claimant_date_of_birth", "") or "").strip()
        age_issue = check_age_consistency(age_injury, dob, parsed.date_of_injury, label="age")
        if age_issue:
            review_issues.append(age_issue)

        flat_overrides = {
            "Jurisdiction": parsed.jurisdiction.value,
            "Case Type": parsed.case_type.value,
            "Decision Date": parsed.date_of_decision,
            "Injury Date": parsed.date_of_injury,
            "Applicant": parsed.applicant_name,
            "Respondent": parsed.respondent_name,
            # claimant_age may carry context like "21 at time of accident";
            # extract the leading number so the CSV/Excel has clean numerics.
            # The full contextual narrative remains in _narrative.claimant_profile.
            "Claimant Age": age_injury,
            "Claimant Age At Decision": age_decision,
            "Claimant Gender": parsed.claimant_gender.value,
            "Claimant Occupation": parsed.claimant_occupation,
            "Claimant Weekly Income": income_amount,
            "Claimant Weekly Income Basis": getattr(parsed, "claimant_weekly_income_basis", ""),
            "Employer Name": parsed.employer_name,
            "Accident/Injury Location": parsed.location_of_accident_or_injury,
            "Location Locality": getattr(parsed, "location_locality", ""),
            "Location State": getattr(parsed, "location_state", ""),
            "Claimant Outcome": parsed.claimant_outcome.value,
            "Impairment %": wpi_strict,
            "Impairment % (Accepted)": wpi_accepted,
            # Coerce to a clean numeric string like the other money fields, so
            # text such as "$300,000 inclusive of costs" / "Weekly $522.84
            # ongoing" doesn't survive as non-numeric and get dropped by numeric
            # filters downstream (ISSUE-006).
            "Lump Sum": coerce_money(parsed.lump_sum_amount),
            "Weekly Benefit": coerce_money(parsed.weekly_benefit_amount),
            "Non-Economic Loss": nel_amount,
            "Non-Economic Loss Status": nel_status,
            "Future Economic Loss": fel_amount,
            "Future Economic Loss Status": fel_status,
            "Statutory Benefits": parsed.statutory_benefits,
            "Medical Costs": normalise_medical_costs(parsed.medical_costs_awarded.value),
            "Nature": parsed.decision_nature,
            "Result": parsed.decision_result,
            "Description": sanitised_case_description,
            "Banded Description": sanitised_banded_description,
            "Catchwords": extract_catchwords(decision_text),
            "Injury Burden Intensity": parsed.injury_burden_intensity,
            "Psychological Injury Emphasis": parsed.psychological_injury_emphasis,
            "Liability Clarity": parsed.liability_clarity,
            "Causation Complexity": parsed.causation_complexity,
            "Treatment Burden": parsed.treatment_burden,
            "Work Impact Severity": parsed.work_impact_severity,
            "Pre-existing Condition Salience": parsed.pre_existing_condition_salience,
            "Legal Procedural Complexity": parsed.legal_procedural_complexity,
            "Regulatory Sections": " | ".join(parsed.regulatory_sections or []),
        }

        result = build_result_record(
            title, url, file_saved=file_saved, status="ok",
            **flat_overrides,
        )

        token_usage = dict(token_usage or {})

        # ---- Field-loss gate (A1) + focused second pass (A6) ----
        initial_losses = detect_field_losses(result, decision_text, provenance)
        second_pass = {"requested": [], "recovered": [], "error": ""}
        if initial_losses and FOCUSED_SECOND_PASS_ENABLED and self.extractor \
                and not self.quota_breaker.is_aborted() \
                and worth_second_pass(initial_losses, provenance):
            target_cols = sorted({iss["field"] for iss in initial_losses})
            fields_needed = [FIELD_LOSS_SPEC[c][1] for c in target_cols if c in FIELD_LOSS_SPEC]
            second_pass["requested"] = target_cols
            fparsed, fusage, ferr = self.extractor.extract_focused(
                decision_text, fields_needed,
                context=f"focused second pass for {url}",
            )
            if fusage is not None:
                token_usage["focused_pass"] = self.cost_tracker.record(fusage)
            if ferr or fparsed is None:
                second_pass["error"] = ferr or "focused parse failed"
            else:
                recovered = merge_focused_into_record(result, fparsed, target_cols)
                second_pass["recovered"] = recovered
                if recovered:
                    logging.info(
                        f"Focused second pass recovered {recovered} for {url}"
                    )

        # ---- WPI resolution pass ----
        # Fires when the source holds a WPI the strict regex refused to choose
        # between (several figures) or none was captured at all. The model
        # classifies the mentions; resolve_wpi does the arithmetic.
        if wpi_resolution_applies(result, decision_text) and self.extractor                 and not self.quota_breaker.is_aborted():
            wparsed, wusage, werr = self.extractor.extract_wpi_resolution(
                decision_text, context=f"WPI resolution for {url}",
            )
            if wusage is not None:
                token_usage["wpi_resolution"] = self.cost_tracker.record(wusage)
            if werr or wparsed is None:
                logging.warning(f"WPI resolution failed for {url}: {werr or 'parse failed'}")
            else:
                merge_wpi_resolution_into_record(result, wparsed)

        # ---- Damages breakdown pass (downstream spec 2026-07-27) ----
        # Runs after the loss gate so it sees the final Lump Sum / NEL / FEL
        # values it has to reconcile against.
        damages_error = ""
        if damages_pass_applies(result) and self.extractor \
                and not self.quota_breaker.is_aborted():
            dparsed, dusage, derr = self.extractor.extract_damages(
                decision_text, context=f"damages pass for {url}",
            )
            if dusage is not None:
                token_usage["damages_pass"] = self.cost_tracker.record(dusage)
            if derr or dparsed is None:
                damages_error = derr or "damages parse failed"
                result["Damages Extraction Status"] = "error"
                result["Damages Notes"] = damages_error[:1000]
                logging.warning(f"Damages pass failed for {url}: {damages_error}")
            else:
                merge_damages_into_record(result, dparsed)
        elif not damages_pass_applies(result):
            result["Damages Extraction Status"] = "not applicable"

        remaining_losses = detect_field_losses(result, decision_text, provenance)

        # ---- Needs Review (C4): only provenance-confirmed losses exclude ----
        confirmed = confirmed_high_losses(remaining_losses, provenance)
        if confirmed:
            result["Needs Review"] = "Yes"
        note_bits = [f"{iss['field']}: {iss['detail']}" for iss in remaining_losses]
        note_bits += [f"{iss['field']}: {iss['detail']}" for iss in review_issues]
        result["Review Notes"] = "; ".join(note_bits)[:1000]

        result["_narrative"] = narrative
        result["_slices"] = slices
        result["_key_paragraphs"] = key_paragraphs
        result["_event_history"] = event_history
        result["_token_usage"] = token_usage
        result["_provenance"] = provenance
        result["_field_review"] = {
            "date_of_birth": dob,
            "age_at_decision": age_decision,
            "initial_losses": initial_losses,
            "remaining_losses": remaining_losses,
            "confirmed_losses": confirmed,
            "other_issues": review_issues,
            "second_pass": second_pass,
        }
        result["_banding_validation"] = validate_banding(sanitised_banded_description, record=result)

        # Catches rows the WPI resolution pass never saw (disabled, failed, or
        # skipped): the s 4.11 contradiction is visible from the record alone.
        # Idempotent - a quarantined row has no value left to withhold.
        quarantine_impossible_wpi(result, decision_text)

        # Runs after the quarantine so the applicability columns describe the
        # WPI the row actually publishes, not the one it was about to withhold.
        apply_round2_semantics(result)
        apply_money_absence_semantics(result)

        # Re-annotate so the Needs Review flag feeds the analysis-ready gate.
        return annotate_analysis_fields(result)

def damages_manifest_fields(rows):
    """Damages-pass coverage for the run manifest, so a consumer can tell at a
    glance which damages version produced the file and how much of it carries a
    breakdown."""
    ok = sum(1 for r in rows if r.get("Damages Extraction Status") == "ok")
    return {
        "damages_version": DAMAGES_VERSION,
        "damages_rows_extracted": ok,
        "damages_rows_not_run": len(rows) - ok,
    }


def _report_sort_key(row):
    decision_date = (row.get("Decision Date") or "").strip()
    return decision_date if has_valid_iso_date(decision_date) else "0000-00-00"


def generate_reports(scraper, detailed_path, analysis_ready_path, *, script,
                     current_run_urls=None, manifest_extra=None):
    """Regenerate the CSV reports + run manifest from the cache as ONE step
    (ISSUE-001/002/015/022).

    - Only current-schema rows flow into reports; stale-schema rows are excluded
      and counted.
    - CSVs are ALWAYS written (header-only when empty) via atomic per-process
      temp files; never leave a previous run's CSV in place.
    - Privacy transforms (ISSUE-011) apply to OUTPUT rows only.
    - A manifest records schema version, timestamp, and counts so consumers can
      detect stale/empty output.
    Returns (all_data, analysis_ready_data) of the CURRENT-schema rows written.
    """
    with scraper.cache_lock:
        snapshot = [(u, ensure_damages_defaults(annotate_analysis_fields(r)))
                    for u, r in scraper.cache.items() if isinstance(r, dict)]
    total_cache = len(snapshot)
    current = [(u, r) for u, r in snapshot if r.get("_schema_version") == SCHEMA_VERSION]
    stale = total_cache - len(current)
    unseen = 0
    if current_run_urls is not None:
        crs = set(current_run_urls)
        unseen = sum(1 for u, _ in current if u not in crs)

    all_data = [r for _, r in current]
    all_data.sort(key=_report_sort_key, reverse=True)
    analysis_ready = [r for r in all_data if r.get("Analysis Ready") == "Yes"]

    out_all = [apply_privacy_to_row(r) for r in all_data]
    out_ready = [apply_privacy_to_row(r) for r in analysis_ready]
    atomic_write_csv(detailed_path, RESULT_FIELDS, out_all)
    atomic_write_csv(analysis_ready_path, RESULT_FIELDS, out_ready)

    if not all_data:
        logging.warning(f"{detailed_path}: 0 current-schema rows — wrote header-only CSV.")
    else:
        logging.info(f"Summary report saved to {detailed_path} ({len(all_data)} rows).")
    if not analysis_ready:
        logging.warning(f"{analysis_ready_path}: 0 analysis-ready rows — wrote header-only CSV.")
    else:
        logging.info(f"Analysis-ready report saved to {analysis_ready_path} ({len(analysis_ready)} rows).")
    if stale:
        logging.warning(f"Excluded {stale} non-current-schema (v!={SCHEMA_VERSION}) cache rows from reports.")
    if current_run_urls is not None and unseen:
        logging.info(f"{unseen} current-schema cache rows were not seen in this run's index scan "
                     f"(retained in cache and reports).")

    needs_review = sum(1 for r in all_data if str(r.get("Needs Review", "")).strip() == "Yes")
    extra = {"stale_rows_excluded": stale, "cache_rows_not_in_current_run": unseen,
             "total_cache_rows": total_cache}
    extra.update(damages_manifest_fields(all_data))
    if manifest_extra:
        extra.update(manifest_extra)
    write_run_manifest(script=script, total_rows=len(all_data),
                       analysis_ready_rows=len(analysis_ready),
                       needs_review_rows=needs_review, **extra)
    return all_data, analysis_ready


def regenerate_reports_from_cache(cache, detailed_path, analysis_ready_path, *, script,
                                  **manifest_extra):
    """Report regeneration for the maintenance scripts, which hold a raw cache
    dict rather than a DecisionScraper. Same guarantees as generate_reports:
    current-schema rows only, always-write (header-only when empty), atomic,
    privacy-applied outputs, run manifest (ISSUE-001/002/015/022). Callers
    should wrap this in `dataset_lock()`."""
    rows = [ensure_damages_defaults(annotate_analysis_fields(r))
            for r in cache.values() if isinstance(r, dict)]
    total_cache = len(rows)
    current = [r for r in rows if r.get("_schema_version") == SCHEMA_VERSION]
    stale = total_cache - len(current)
    current.sort(key=_report_sort_key, reverse=True)
    analysis_ready = [r for r in current if r.get("Analysis Ready") == "Yes"]

    atomic_write_csv(detailed_path, RESULT_FIELDS, [apply_privacy_to_row(r) for r in current])
    atomic_write_csv(analysis_ready_path, RESULT_FIELDS, [apply_privacy_to_row(r) for r in analysis_ready])
    logging.info(f"Rewrote {detailed_path} ({len(current)} rows) and "
                 f"{analysis_ready_path} ({len(analysis_ready)} rows).")
    if not current:
        logging.warning(f"{detailed_path}: 0 current-schema rows — wrote header-only CSV.")
    if stale:
        logging.warning(f"Excluded {stale} non-current-schema (v!={SCHEMA_VERSION}) cache rows from reports.")
    needs_review = sum(1 for r in current if str(r.get("Needs Review", "")).strip() == "Yes")
    manifest_extra = dict(manifest_extra)
    manifest_extra.update(damages_manifest_fields(current))
    write_run_manifest(script=script, total_rows=len(current),
                       analysis_ready_rows=len(analysis_ready),
                       needs_review_rows=needs_review,
                       stale_rows_excluded=stale, total_cache_rows=total_cache, **manifest_extra)
    return current, analysis_ready


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("⚠️ OPENAI_API_KEY not found in .env file.")
        return

    BASE_DOMAIN = "https://www.austlii.edu.au"
    OUTPUT_DIR = DECISIONS_DIR
    if privacy_active():
        logging.info(f"Privacy mode active for OUTPUTS: {privacy_summary()} "
                     f"(cache retains full data). Decision text is sent to the OpenAI API.")
    scraper = DecisionScraper(BASE_DOMAIN, OUTPUT_DIR, api_key)
    index_delay_seconds = float(os.getenv("AUSTLII_INDEX_DELAY", "2"))

    years = list(range(2021, datetime.datetime.now().year + 1))
    all_links = []

    for year in years:
        index_url = f"https://www.austlii.edu.au/cgi-bin/viewdb/au/cases/nsw/NSWPIC/{year}/"
        logging.info(f"Scanning Year: {year} ...")
        links = scraper.get_decision_links(index_url)
        all_links.extend(links)
        time.sleep(max(index_delay_seconds, 0))

    logging.info(f"Total decisions found (2021-Present): {len(all_links)}")

    target_links = all_links
    current_run_urls = {url for _, url in target_links}
    max_workers = get_worker_count()  # validated/clamped (ISSUE-018)

    logging.info(f"Starting parallel processing of {len(target_links)} decisions ({max_workers} threads)...")
    results = []
    wall_t0 = time.monotonic()

    # Bounded submission: keep ~window tasks in flight rather than enqueueing the
    # whole corpus, and cancel queued work the moment the quota breaker trips
    # (ISSUE-013).
    window = max(max_workers * 4, max_workers)
    total = len(target_links)
    done = 0
    aborted_logged = False
    it = iter(target_links)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight = set()
        for _ in range(window):
            nxt = next(it, None)
            if nxt is None:
                break
            in_flight.add(executor.submit(scraper.process_decision, *nxt))

        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in finished:
                done += 1
                try:
                    data = fut.result()
                    if data:
                        results.append(data)
                except Exception as e:
                    logging.error(f"Unhandled exception while processing a decision: {e}")
                if done % 25 == 0:
                    with dataset_lock():
                        scraper._save_cache()
                    logging.info(
                        f"Progress: {done}/{total} processed, "
                        f"running cost ${scraper.cost_tracker.total_cost():.2f} "
                        f"({scraper.cost_tracker.calls} LLM calls)"
                    )

            if scraper.quota_breaker.is_aborted():
                if not aborted_logged:
                    aborted_logged = True
                    logging.error(
                        f"QUOTA BREAKER TRIPPED at {done}/{total} — cancelling queued work and "
                        f"halting submission. Top up the OpenAI account and re-run; cached rows "
                        f"are preserved.")
                for f in in_flight:
                    f.cancel()
                in_flight = set()
                break

            # Refill the window (only if not aborted).
            for _ in range(len(finished)):
                nxt = next(it, None)
                if nxt is None:
                    break
                in_flight.add(executor.submit(scraper.process_decision, *nxt))

    with dataset_lock():
        scraper._save_cache()
        scraper._save_sidecar()
        wall_elapsed = time.monotonic() - wall_t0

        # Cost summary
        ct = scraper.cost_tracker
        logging.info("=" * 70)
        logging.info("LLM USAGE / COST")
        logging.info("=" * 70)
        logging.info(f"  Wall-clock:        {wall_elapsed:.1f}s")
        logging.info(f"  LLM calls:         {ct.calls}")
        logging.info(f"  Prompt tokens:     {ct.prompt_tokens:,}  (cached {ct.cached_tokens:,})")
        logging.info(f"  Completion tokens: {ct.completion_tokens:,}  (reasoning {ct.reasoning_tokens:,})")
        logging.info(f"  Total cost:        ${ct.total_cost():.2f}")
        if ct.calls:
            logging.info(f"  Mean per call:     ${ct.total_cost() / ct.calls:.4f}")

        all_data, analysis_ready_data = generate_reports(
            scraper, CSV_REPORT, ANALYSIS_READY_REPORT,
            script="nsw_court_scraper.main", current_run_urls=current_run_urls,
            manifest_extra={"quota_aborted": bool(aborted_logged),
                            "index_links": total, "llm_calls": ct.calls},
        )

    print_summary(all_data, analysis_ready_data)

def print_summary(all_data, analysis_ready_data):
    """Print data summary to console for analysis-ready rows, split by Case Type."""
    if not all_data:
        print("\nNo data to summarise.")
        return

    exclusion_counts = defaultdict(int)
    for row in all_data:
        if row.get("Analysis Ready") == "Yes":
            continue
        reasons = row.get("Analysis Exclusion Reason", "").split("; ")
        for reason in filter(None, reasons):
            exclusion_counts[reason] += 1

    print("\n" + "=" * 70)
    print("DATA QUALITY")
    print("=" * 70)
    print(f"Total rows collected:               {len(all_data)}")
    print(f"Analysis-ready rows:               {len(analysis_ready_data)}")
    print(f"Excluded from downstream analysis: {len(all_data) - len(analysis_ready_data)}")
    if exclusion_counts:
        for reason in sorted(exclusion_counts):
            print(f"  - {reason}: {exclusion_counts[reason]}")

    needs_review = sum(1 for r in all_data if str(r.get("Needs Review", "")).strip() == "Yes")
    print(f"Rows flagged Needs Review:          {needs_review}")

    # C3: per-field coverage for the eight high-value target fields. A sudden
    # coverage drop after a prompt/schema change shows up here immediately.
    print("\n" + "-" * 70)
    print("KEY FIELD COVERAGE (all rows)")
    print("-" * 70)
    total = len(all_data) or 1
    for field in KEY_TARGET_FIELDS:
        populated = sum(1 for r in all_data if _value_present(r.get(field, "")))
        pct = 100.0 * populated / total
        print(f"  {field:<28} {populated:>5}/{len(all_data):<5} ({pct:5.1f}%)")

    if not analysis_ready_data:
        print("\nNo analysis-ready data to summarise.")
        return

    lump_sum_counts = defaultdict(int)
    impairment_counts = defaultdict(int)       # strict "made-here" WPI
    accepted_counts = defaultdict(int)         # accepted/relied-on WPI (analysis-useful)
    both_counts = defaultdict(int)             # lump sum AND accepted WPI
    injury_dates = defaultdict(list)
    decision_dates = defaultdict(list)

    def _has_numeric_value(val):
        """Check if a value is a non-empty numeric string."""
        if not val or not isinstance(val, str):
            return False
        val = val.strip()
        if not val or val.lower() in ("n/a", "unknown", "none", "nan"):
            return False
        try:
            float(val)
            return True
        except ValueError:
            return False

    for row in analysis_ready_data:
        case_type = row.get("Case Type") or "Unknown"
        has_lump = _has_numeric_value(row.get("Lump Sum", ""))
        has_impairment = _has_numeric_value(row.get("Impairment %", ""))
        # Accepted WPI is the analysis-useful field for CTP payout-vs-WPI work;
        # after the v3 split many valid CTP rows have blank strict Impairment %
        # but a populated accepted WPI, so count both (ISSUE-009).
        has_accepted = _has_numeric_value(row.get("Impairment % (Accepted)", ""))

        if has_lump:
            lump_sum_counts[case_type] += 1
        if has_impairment:
            impairment_counts[case_type] += 1
        if has_accepted:
            accepted_counts[case_type] += 1
        if has_lump and has_accepted:
            both_counts[case_type] += 1

        inj = row.get("Injury Date", "").strip()
        if inj and inj != "Unknown":
            injury_dates[case_type].append(inj)

        dec = row.get("Decision Date", "").strip()
        if dec and dec != "Unknown":
            decision_dates[case_type].append(dec)

    case_types = sorted(set(
        list(lump_sum_counts) + list(impairment_counts) + list(accepted_counts)
        + list(both_counts) + list(injury_dates) + list(decision_dates)
    ))

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    for ct in case_types:
        print(f"\n--- {ct} ---")
        print(f"  Rows with Lump Sum:              {lump_sum_counts.get(ct, 0)}")
        print(f"  Rows with Impairment % (strict): {impairment_counts.get(ct, 0)}")
        print(f"  Rows with WPI (Accepted):        {accepted_counts.get(ct, 0)}")
        print(f"  Rows with Lump Sum + Accepted:   {both_counts.get(ct, 0)}")

        inj = injury_dates.get(ct, [])
        if inj:
            print(f"  Injury dates:                    {min(inj)} to {max(inj)}")
        else:
            print(f"  Injury dates:                    N/A")

        dec = decision_dates.get(ct, [])
        if dec:
            print(f"  Decision dates:                  {min(dec)} to {max(dec)}")
        else:
            print(f"  Decision dates:                  N/A")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
