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
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import random
from threading import Lock
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Bump SCHEMA_VERSION when the cached extraction shape changes. Rows with a
# mismatched _schema_version are re-processed.
SCHEMA_VERSION = 2

RESULT_FIELDS = [
    # Identity
    "Case Name", "URL", "File Saved",
    # Decision metadata
    "Jurisdiction", "Case Type", "Decision Date", "Injury Date",
    "Applicant", "Respondent",
    # Claimant info
    "Claimant Age", "Claimant Gender", "Claimant Occupation",
    "Claimant Weekly Income", "Employer Name", "Accident/Injury Location",
    # Outcome
    "Claimant Outcome",
    "Impairment %", "Lump Sum", "Weekly Benefit",
    "Non-Economic Loss", "Future Economic Loss", "Statutory Benefits",
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
    "Analysis Ready", "Analysis Exclusion Reason",
]

# Keys stored on cached rows but excluded from the flat CSV output.
SIDECAR_KEYS = (
    "_narrative",            # dict of narrative sub-fields (incl. submissions)
    "_slices",               # dict of catchwords / determinations / introduction
    "_key_paragraphs",       # list of {paragraph_number, rationale, text}
    "_event_history",        # list of {date, actor, tag}
    "_schema_version",
    "_token_usage",          # last extraction's token usage
    "_banding_validation",   # banded_case_description validation result
)

MODEL = "gpt-5"
REASONING_EFFORT = "low"
DEFAULT_WORKERS = 25
SINGLE_PASS_LIMIT_CHARS = 100_000

# Pricing (USD per 1M tokens) — override via env if needed
PRICE_INPUT_PER_M = float(os.getenv("GPT5_PRICE_INPUT_PER_M", "1.25"))
PRICE_CACHED_INPUT_PER_M = float(os.getenv("GPT5_PRICE_CACHED_INPUT_PER_M", "0.125"))
PRICE_OUTPUT_PER_M = float(os.getenv("GPT5_PRICE_OUTPUT_PER_M", "10.00"))


def has_valid_iso_date(value):
    if not isinstance(value, str):
        return False
    return bool(ISO_DATE_PATTERN.fullmatch(value.strip()))


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

    return reasons


def annotate_analysis_fields(row):
    annotated = dict(row)
    reasons = get_analysis_exclusion_reasons(annotated)
    annotated["Analysis Ready"] = "Yes" if not reasons else "No"
    annotated["Analysis Exclusion Reason"] = "; ".join(reasons)
    return annotated


def build_result_record(title, url, file_saved="", status="", llm_error="", **overrides):
    row = {field: "" for field in RESULT_FIELDS}
    row.update({
        "Case Name": title,
        "URL": url,
        "File Saved": file_saved,
        "Status": status,
        "LLM Error": llm_error,
    })
    # Sidecar defaults
    row["_schema_version"] = SCHEMA_VERSION
    row["_narrative"] = {}
    row["_slices"] = {}
    row["_key_paragraphs"] = []
    row["_event_history"] = []
    row["_token_usage"] = {}
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
        "Claimant's age. Prefer age at time of injury/accident. "
        "Examples: '47', '21 at time of accident'. "
        "If only year-of-birth is stated (e.g. 'born in 1995' or 'date of birth: "
        "12 March 1995'), DERIVE the age = injury_year - birth_year and emit "
        "the number (e.g. '28'). 'Not stated' only if neither an age nor a "
        "year-of-birth can be found anywhere in the decision."
    ))
    claimant_gender: str = Field(description="Claimant's gender if stated (Male / Female / Other / Not stated).")
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
    employer_name: str = Field(description="Employer's legal name (workers compensation only). 'Not applicable' for CTP / non-employment cases. 'Not stated' if WC case but employer not named.")
    location_of_accident_or_injury: str = Field(description="Where the injury occurred — for CTP: road/intersection/town; for WC: workplace address/town. 'Not stated' if absent.")
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
    non_economic_loss: str = Field(description="Damages for non-economic loss (pain and suffering) as a nominal number. 'Not stated' if not addressed; 'Nil' if explicitly denied.")
    future_economic_loss: str = Field(description="Damages for future economic loss (incl. loss of future earnings/super) as a nominal number. 'Not stated' if not addressed; 'Nil' if explicitly denied.")
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

AUSTLII_ERROR_LOG = "austlii_data_errors.json"
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


def _narrative_truncate(text):
    """If the source overflows the single-pass char limit, keep the start +
    a chunk near each likely-key section."""
    if len(text) <= SINGLE_PASS_LIMIT_CHARS:
        return text
    keywords = (
        "Background", "Facts", "History",
        "Particulars", "Mechanism", "Medical evidence",
        "Pre-existing", "Treatment", "Diagnosis", "Surgery",
        "Expert evidence", "Submissions", "Reasoning", "Findings",
        "Reasons", "Discussion", "Issues", "Orders", "Conclusion", "Decision",
    )
    lowered = text.lower()
    segments = [text[:30000]]
    seen = {text[:30000]}
    for kw in keywords:
        idx = lowered.find(kw.lower())
        if idx == -1:
            continue
        start = max(0, idx - 3000)
        end = min(len(text), idx + 10000)
        seg = text[start:end].strip()
        if seg and seg not in seen:
            seen.add(seg)
            segments.append(seg)
    combined = "\n\n...[SECTION BREAK]...\n\n".join(segments)
    if len(combined) > SINGLE_PASS_LIMIT_CHARS:
        combined = combined[:SINGLE_PASS_LIMIT_CHARS]
    return combined


class LLMExtractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

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
        # Per-worker retry on insufficient_quota — OpenAI auto-recharges when
        # the balance drops below a threshold, and during the ~10-30s recharge
        # window the API returns 429/insufficient_quota even though the
        # account is in good standing. Treat it like a transient rate limit.
        # Backoff schedule: 2, 5, 10, 20, 40, 80 seconds (covers ~2.5 minutes
        # of recharge / brief outage).
        backoff_schedule = [2, 5, 10, 20, 40, 80]
        last_error = None
        for attempt in range(len(backoff_schedule) + 1):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_INSTRUCTION},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=CombinedSchema,
                    reasoning_effort=REASONING_EFFORT,
                )
                return completion.choices[0].message.parsed, completion.usage, None
            except Exception as e:
                last_error = e
                if _is_quota_error(str(e)) and attempt < len(backoff_schedule):
                    delay = backoff_schedule[attempt]
                    ctx = f" ({context})" if context else ""
                    logging.warning(
                        f"insufficient_quota{ctx} - retry {attempt+1}/{len(backoff_schedule)} in {delay}s "
                        f"(treating as transient auto-recharge window)"
                    )
                    time.sleep(delay)
                    continue
                ctx = f" ({context})" if context else ""
                logging.error(f"Combined LLM error{ctx}: {e}")
                return None, None, str(e)
        return None, None, str(last_error)

class DecisionScraper:
    def __init__(self, base_url, output_folder="nsw_decisions", api_key=None):
        self.base_url = base_url
        self.output_folder = output_folder
        self.extractor = LLMExtractor(api_key) if api_key else None
        self.cache_file = "processed_cache.json"
        self.sidecar_file = "processed_sidecar.json"
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

    def _load_cache(self):
        """Loads cache safely. If corrupted, backs up and starts empty."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    loaded_cache = json.load(f)
                if isinstance(loaded_cache, dict):
                    return {
                        url: annotate_analysis_fields(data) if isinstance(data, dict) else data
                        for url, data in loaded_cache.items()
                    }
                logging.error(f"Cache file {self.cache_file} does not contain a dictionary. Starting with empty cache.")
                return {}
            except json.JSONDecodeError:
                logging.error(f"Cache file {self.cache_file} is corrupted. Starting with empty cache.")
                shutil.move(self.cache_file, self.cache_file + ".corrupted")
                return {}
        return {}

    def _save_cache(self, max_retries=3):
        """
        Thread-safe atomic write.
        Copies cache under lock, dumps to temp file, then renames.
        """
        temp_file = self.cache_file + ".tmp"
        
        # Create a thread-safe snapshot of the data
        with self.cache_lock:
            cache_copy = self.cache.copy()

        for attempt in range(1, max_retries + 1):
            try:
                with open(temp_file, 'w') as f:
                    json.dump(cache_copy, f, indent=2, default=str)
                os.replace(temp_file, self.cache_file)
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
                    "_schema_version": row.get("_schema_version"),
                    "narrative": row.get("_narrative", {}),
                    "slices": row.get("_slices", {}),
                    "key_paragraphs": row.get("_key_paragraphs", []),
                    "event_history": row.get("_event_history", []),
                    "regulatory_sections": (row.get("Regulatory Sections") or "").split(" | "),
                    "token_usage": row.get("_token_usage", {}),
                    "banding_validation": row.get("_banding_validation", {}),
                }
                sidecar[url] = entry

        temp_file = self.sidecar_file + ".tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2, default=str, ensure_ascii=False)
            os.replace(temp_file, self.sidecar_file)
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

        flat_overrides = {
            "Jurisdiction": parsed.jurisdiction.value,
            "Case Type": parsed.case_type.value,
            "Decision Date": parsed.date_of_decision,
            "Injury Date": parsed.date_of_injury,
            "Applicant": parsed.applicant_name,
            "Respondent": parsed.respondent_name,
            # claimant_age schema allows context like "21 at time of accident";
            # extract the leading number so the CSV/Excel has clean numerics.
            # The full contextual narrative remains in _narrative.claimant_profile.
            "Claimant Age": coerce_leading_number(parsed.claimant_age),
            "Claimant Gender": parsed.claimant_gender,
            "Claimant Occupation": parsed.claimant_occupation,
            "Claimant Weekly Income": parsed.claimant_weekly_income,
            "Employer Name": parsed.employer_name,
            "Accident/Injury Location": parsed.location_of_accident_or_injury,
            "Claimant Outcome": parsed.claimant_outcome.value,
            "Impairment %": (parsed.impairment_percentage or "").replace("%", "").strip(),
            "Impairment % (Accepted)": (getattr(parsed, "impairment_percentage_accepted", "") or "").replace("%", "").strip(),
            "Lump Sum": (parsed.lump_sum_amount or "").replace("$", "").replace(",", "").strip(),
            "Weekly Benefit": parsed.weekly_benefit_amount,
            "Non-Economic Loss": parsed.non_economic_loss,
            "Future Economic Loss": parsed.future_economic_loss,
            "Statutory Benefits": parsed.statutory_benefits,
            "Medical Costs": parsed.medical_costs_awarded.value,
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
        result["_narrative"] = narrative
        result["_slices"] = slices
        result["_key_paragraphs"] = key_paragraphs
        result["_event_history"] = event_history
        result["_token_usage"] = token_usage
        result["_banding_validation"] = validate_banding(sanitised_banded_description, record=result)
        return result

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("⚠️ OPENAI_API_KEY not found in .env file.")
        return

    BASE_DOMAIN = "https://www.austlii.edu.au"
    OUTPUT_DIR = "nsw_pic_decisions"
    CSV_REPORT = "detailed_payout_summary.csv"
    ANALYSIS_READY_REPORT = "analysis_ready_payout_summary.csv"
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
    max_workers = int(os.getenv("EXTRACTION_WORKERS", str(DEFAULT_WORKERS)))

    logging.info(f"Starting parallel processing of {len(target_links)} decisions ({max_workers} threads)...")
    results = []
    wall_t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(scraper.process_decision, title, url): url
            for title, url in target_links
        }

        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logging.error(f"Unhandled exception while processing {url}: {e}")

            # Periodic save every 25 completions.
            if i > 0 and i % 25 == 0:
                scraper._save_cache()
                logging.info(
                    f"Progress: {i+1}/{len(target_links)} processed, "
                    f"running cost ${scraper.cost_tracker.total_cost():.2f} "
                    f"({scraper.cost_tracker.calls} LLM calls)"
                )

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

    # Use thread-safe snapshot for final report generation
    with scraper.cache_lock:
        all_data = [annotate_analysis_fields(row) for row in scraper.cache.values()]

    analysis_ready_data = [
        row for row in all_data
        if row.get("Analysis Ready") == "Yes"
    ]

    def _decision_date_sort_key(row):
        decision_date = (row.get("Decision Date") or "").strip()
        if has_valid_iso_date(decision_date):
            return decision_date
        return "0000-00-00"

    all_data.sort(key=_decision_date_sort_key, reverse=True)
    analysis_ready_data.sort(key=_decision_date_sort_key, reverse=True)
    
    if all_data:
        with open(CSV_REPORT, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS, extrasaction='ignore')
            dict_writer.writeheader()
            dict_writer.writerows(all_data)
        logging.info(f"Summary report saved to {CSV_REPORT}")

    if analysis_ready_data:
        with open(ANALYSIS_READY_REPORT, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS, extrasaction='ignore')
            dict_writer.writeheader()
            dict_writer.writerows(analysis_ready_data)
        logging.info(f"Analysis-ready report saved to {ANALYSIS_READY_REPORT}")
    else:
        logging.warning("No analysis-ready rows were produced.")

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

    if not analysis_ready_data:
        print("\nNo analysis-ready data to summarise.")
        return

    lump_sum_counts = defaultdict(int)
    impairment_counts = defaultdict(int)
    both_counts = defaultdict(int)
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

        if has_lump:
            lump_sum_counts[case_type] += 1
        if has_impairment:
            impairment_counts[case_type] += 1
        if has_lump and has_impairment:
            both_counts[case_type] += 1

        inj = row.get("Injury Date", "").strip()
        if inj and inj != "Unknown":
            injury_dates[case_type].append(inj)

        dec = row.get("Decision Date", "").strip()
        if dec and dec != "Unknown":
            decision_dates[case_type].append(dec)

    case_types = sorted(set(
        list(lump_sum_counts) + list(impairment_counts) + list(both_counts)
        + list(injury_dates) + list(decision_dates)
    ))

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    for ct in case_types:
        print(f"\n--- {ct} ---")
        print(f"  Rows with Lump Sum:              {lump_sum_counts.get(ct, 0)}")
        print(f"  Rows with Impairment %:          {impairment_counts.get(ct, 0)}")
        print(f"  Rows with both:                  {both_counts.get(ct, 0)}")

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
