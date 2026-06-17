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
]

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
)

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


def _narrative_truncate(text):
    """If the source overflows the single-pass char limit, keep the start +
    a chunk near each likely-key section.

    The keyword list deliberately includes the QUANTUM / CLAIMANT-PROFILE
    section terms (non-economic loss, future economic loss, PIAWE, weekly,
    impairment/WPI, damages, buffer, date of birth, aged, occupation). These
    are where the eight high-value target fields live, and in long decisions
    they sit near the END — so they must survive truncation. We also anchor on
    BOTH the first and last occurrence of each keyword because the operative
    quantum assessment is usually the later mention."""
    if len(text) <= SINGLE_PASS_LIMIT_CHARS:
        return text
    keywords = (
        # Structure
        "Background", "Facts", "History",
        "Particulars", "Mechanism", "Medical evidence",
        "Pre-existing", "Treatment", "Diagnosis", "Surgery",
        "Expert evidence", "Submissions", "Reasoning", "Findings",
        "Reasons", "Discussion", "Issues", "Orders", "Conclusion", "Decision",
        # Quantum / damages heads (where the 8 target fields live)
        "non-economic loss", "non economic loss", "general damages",
        "pain and suffering", "future economic loss", "loss of earning",
        "earning capacity", "buffer", "quantum", "damages assessment",
        "whole person impairment", "impairment", "WPI",
        # Claimant profile / income
        "PIAWE", "weekly", "per week", "pre-injury", "salary", "wage",
        "date of birth", "born", "aged", "occupation", "employed",
    )
    lowered = text.lower()
    segments = [text[:30000]]
    seen = {text[:30000]}

    def add_window(idx):
        start = max(0, idx - 3000)
        end = min(len(text), idx + 10000)
        seg = text[start:end].strip()
        if seg and seg not in seen:
            seen.add(seg)
            segments.append(seg)

    for kw in keywords:
        kwl = kw.lower()
        first = lowered.find(kwl)
        if first == -1:
            continue
        add_window(first)
        last = lowered.rfind(kwl)
        if last != -1 and abs(last - first) > 8000:
            add_window(last)

    combined = "\n\n...[SECTION BREAK]...\n\n".join(segments)
    if len(combined) > SINGLE_PASS_LIMIT_CHARS:
        combined = combined[:SINGLE_PASS_LIMIT_CHARS]
    return combined


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
            "Lump Sum": (parsed.lump_sum_amount or "").replace("$", "").replace(",", "").strip(),
            "Weekly Benefit": parsed.weekly_benefit_amount,
            "Non-Economic Loss": nel_amount,
            "Non-Economic Loss Status": nel_status,
            "Future Economic Loss": fel_amount,
            "Future Economic Loss Status": fel_status,
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

        # Re-annotate so the Needs Review flag feeds the analysis-ready gate.
        return annotate_analysis_fields(result)

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
