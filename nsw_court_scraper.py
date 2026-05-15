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
    "Nature", "Result", "Description",
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
    "_narrative",          # dict of narrative sub-fields (incl. submissions)
    "_slices",             # dict of catchwords / determinations / introduction
    "_key_paragraphs",     # list of {paragraph_number, rationale, text}
    "_event_history",      # list of {date, actor, tag}
    "_schema_version",
    "_token_usage",        # last extraction's token usage
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
    claimant_age: str = Field(description="Claimant's age at time of injury or hearing as stated (e.g. '47' or '21 at time of accident'). 'Not stated' if absent.")
    claimant_gender: str = Field(description="Claimant's gender if stated (Male / Female / Other / Not stated).")
    claimant_occupation: str = Field(description="Claimant's occupation at time of injury (e.g. 'bus driver', 'registered nurse', 'senior laboratory technician'). 'Not stated' if absent.")
    claimant_weekly_income: str = Field(description=(
        "Claimant's weekly employment income as stated in the decision. Capture any "
        "specific weekly figure mentioned — pre-injury (PIAWE), current/post-accident, "
        "or at hearing. If multiple components are stated (e.g. base salary plus "
        "commissions, or net plus super), include each. Add a short qualifier "
        "indicating which period the figure applies to. Use nominal numbers with no "
        "$ or commas. Examples: '1230.00 PIAWE', '800.00 net per week plus approx "
        "4000.00 per month commissions (current, post-accident)', '461.16 current "
        "weekly earnings; PIAWE 1134.68'. 'Not stated' only if NO weekly income "
        "figure of any kind appears."
    ))
    employer_name: str = Field(description="Employer's legal name (workers compensation only). 'Not applicable' for CTP / non-employment cases. 'Not stated' if WC case but employer not named.")
    location_of_accident_or_injury: str = Field(description="Where the injury occurred — for CTP: road/intersection/town; for WC: workplace address/town. 'Not stated' if absent.")
    impairment_percentage: str = Field(description="Whole Person Impairment percentage ONLY if a final assessment is made (e.g. '15'). LEAVE EMPTY if the decision allows reassessment or remits to a medical assessor.")
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
            ctx = f" ({context})" if context else ""
            logging.error(f"Combined LLM error{ctx}: {e}")
            return None, None, str(e)

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
        for attempt in range(max_retries):
            self._throttle_if_rate_limited()
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                
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
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
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

        logging.info(f"Processing: {log_title}")

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
            result_data = build_result_record(title, url, status="skipped_unsupported_content")
            self.update_cache(url, result_data)
            return result_data

        decision_text = cleanup_text(raw_text)
        safe_title = f"{safe_title_base}{file_extension}"

        if len(decision_text) < 500:
            logging.warning(f"Decision text too short; skipping {log_title}")
            result_data = build_result_record(title, url, file_saved=safe_title, status="skipped_short_text")
            self.update_cache(url, result_data)
            return result_data

        full_path = os.path.join(self.output_folder, safe_title)
        if os.path.exists(full_path):
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
            root_name, extension = os.path.splitext(safe_title)
            safe_title = f"{root_name}_{url_hash}{extension}"
            full_path = os.path.join(self.output_folder, safe_title)
        with open(full_path, 'wb') as f:
            f.write(response.content)

        parsed, usage, llm_error = self.extractor.extract_combined(
            decision_text, context=f"title={log_title}, url={url}",
        )
        token_usage = self.cost_tracker.record(usage) if usage else {}

        if llm_error or parsed is None:
            result_data = build_result_record(
                title, url, file_saved=safe_title,
                status="llm_error", llm_error=llm_error or "parse failed",
            )
            result_data["_token_usage"] = token_usage
            self.update_cache(url, result_data)
            return result_data

        result_data = self._build_record_from_parsed(
            title=title, url=url, file_saved=safe_title,
            parsed=parsed, decision_text=decision_text,
            token_usage=token_usage,
        )
        self.update_cache(url, result_data)
        return result_data

    def _build_record_from_parsed(self, *, title, url, file_saved, parsed, decision_text, token_usage):
        sanitised_case_description = sanitise_case_description(parsed.case_description)

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
            "Claimant Age": parsed.claimant_age,
            "Claimant Gender": parsed.claimant_gender,
            "Claimant Occupation": parsed.claimant_occupation,
            "Claimant Weekly Income": parsed.claimant_weekly_income,
            "Employer Name": parsed.employer_name,
            "Accident/Injury Location": parsed.location_of_accident_or_injury,
            "Claimant Outcome": parsed.claimant_outcome.value,
            "Impairment %": (parsed.impairment_percentage or "").replace("%", "").strip(),
            "Lump Sum": (parsed.lump_sum_amount or "").replace("$", "").replace(",", "").strip(),
            "Weekly Benefit": parsed.weekly_benefit_amount,
            "Non-Economic Loss": parsed.non_economic_loss,
            "Future Economic Loss": parsed.future_economic_loss,
            "Statutory Benefits": parsed.statutory_benefits,
            "Medical Costs": parsed.medical_costs_awarded.value,
            "Nature": parsed.decision_nature,
            "Result": parsed.decision_result,
            "Description": sanitised_case_description,
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
