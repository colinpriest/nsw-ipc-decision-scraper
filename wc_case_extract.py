"""
Workers-compensation case extract: one row per unique WC decision.

Builds output/wc_case_extract.xlsx from the flat CSV (structured fields the LLM
pipeline already produced) plus rules applied to the decision text for the
fields the pipeline never populated on WC rows.

Why text rules at all: damages_extraction.py only ever ran on CTP rows, so on
WC rows `Damages Extraction Status` is "not run" and the whole damages /
provenance / injury-taxonomy layer is empty (Primary Injury Category,
Accident Mechanism and Injury Categories are 0% populated). Insurer name and
legal-costs quantum were never in the schema at all. Those six fields are
derived here and each carries a companion `*_source` / `*_context` column so a
reader can audit the rule that fired.

De-duplication: nsw_pic_decisions/ holds ~6,639 files for ~3,503 decisions
because a filename-convention change (case-id suffix) made the scraper's
on-disk cache check miss pre-existing files and re-save them. This module keys
everything on the AustLII URL, which is one-per-decision, and reports the
redundant file count in `duplicate_html_files`.

All executable logic lives in main() (ISSUE-004): importing this module has no
side effects, so the helpers can be tested without reading production files or
overwriting the workbook.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from nsw_court_scraper import (
    CostTracker,
    CSV_REPORT,
    MODEL,
    OUTPUT_ROOT,
    LLMExtractor,
    _is_quota_error,
    _is_transient_api_error,
)

DECISIONS_FOLDER = "nsw_pic_decisions"
OUTPUT_XLSX = os.path.join(OUTPUT_ROOT, "wc_case_extract.xlsx")
TEXT_CACHE = os.path.join(OUTPUT_ROOT, "wc_text_cache")

# Excel hard-limits cell text at 32,767 chars. 67% of WC decisions are longer
# than that, which is why the workbook carries the HTML filename rather than
# the decision text.
CELL_CHAR_CAP = 32_000

MONEY = r"\$\s?[\d,]+(?:\.\d{1,2})?"
PERCENT = r"\d{1,3}(?:\.\d+)?\s?(?:%|per cent|percent)"


WC_REASONING_EFFORT = os.getenv("NSW_WC_REASONING_EFFORT", "low")

# Bump when WCCaseSchema or the system instruction changes meaningfully:
# cache entries written under an older version are discarded on load.
WC_SCHEMA_VERSION = 8

# The second pass is versioned INDEPENDENTLY of WC_SCHEMA_VERSION, and that is
# the whole point of it existing. Bumping WC_SCHEMA_VERSION discards the cache
# and re-extracts all 191 fields with a model measured at 91% run-to-run
# stability, which would move every figure already written into the data
# dictionary and the worked examples -- 494/129/77, the 47.8% consequential
# rate, the 92.7% overlap, the 23.3-vs-4.7 definitional spread, the frozen
# 259-case set. Documentation quoting numbers the workbook no longer contains
# is the Diab failure mode with the docs as the victim. An additive pass adds
# fields without disturbing any of it, at roughly a third of the cost.
WC_CONDUCT_VERSION = 1

# The relief-degree pilot is versioned apart from the conduct pass for the same
# reason the conduct pass is versioned apart from the main schema, applied one
# level down: conduct_finding and denial_scope work and cost $46 to produce, so
# a field still being piloted must not be able to invalidate them. Bumping this
# re-runs one ordinal and nothing else.
WC_RELIEF_VERSION = 1
LLM_CACHE_FILE = os.path.join(OUTPUT_ROOT, "wc_llm_cache.json")
DICTIONARY_XLSX = os.path.join("docs", "wc_data_dictionary.xlsx")
# Worksheets carry case names and catchwords, so they are written under
# output/ (git-ignored) rather than into the tracked dictionary.
WORKSHEETS_XLSX = os.path.join(OUTPUT_ROOT, "wc_reference_sets.xlsx")


# ----------------------------------------------------------------------
# LLM schema for the fields a regex cannot honestly resolve
# ----------------------------------------------------------------------
#
# Split of responsibility. Deterministic fields (citation, filename, dates,
# duration, the pipeline's existing ordinals) are copied, never inferred.
# Fields whose answer depends on reading the decision — who the insurer was,
# which side each rival WPI figure belongs to, whether liability was actually
# in issue, whether the worker was represented — go to the LLM, because they
# turn on discourse structure that proximity matching gets wrong. The regex
# rules below are retained as a cross-check, and every such column carries a
# `*_source` telling you which produced the value plus a `*_agreement` flag
# where both ran.

class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"
    unknown = "Unknown"


class WCOutcomeEnum(str, Enum):
    claimant = "claimant"
    insurer = "insurer"
    mixed = "mixed"
    not_determined = "not_determined"


class LiabilityPostureEnum(str, Enum):
    liability_denied = "liability_denied"
    quantum_or_entitlement_only = "quantum_or_entitlement_only"
    # Both sides must be able to express the same categories. Without this,
    # the rule's procedural label could never be agreed with, manufacturing a
    # disagreement on every procedural row -- the NO_RULE_BASELINE trap again.
    not_applicable_procedural = "not_applicable_procedural"
    unclear = "unclear"


class CostsDirectionEnum(str, Enum):
    respondent_pays_applicant = "respondent_pays_applicant"
    applicant_pays_respondent = "applicant_pays_respondent"
    each_party_bears_own = "each_party_bears_own"
    no_order_as_to_costs = "no_order_as_to_costs"
    costs_reserved = "costs_reserved"
    costs_assessment_application = "costs_assessment_application"
    not_addressed = "not_addressed"


class ProceedingPostureEnum(str, Enum):
    first_instance = "first_instance"
    reconsideration = "reconsideration"
    related_earlier_proceedings = "related_earlier_proceedings"
    not_stated = "not_stated"


class LumpSumTypeEnum(str, Enum):
    s66_permanent_impairment = "s66_permanent_impairment"
    s67_pain_and_suffering = "s67_pain_and_suffering"
    death_benefit = "death_benefit"
    other = "other"
    none = "none"


class WCInjuryEnum(str, Enum):
    psychological = "psychological"
    spinal = "spinal"
    upper_limb = "upper_limb"
    lower_limb = "lower_limb"
    head_brain = "head_brain"
    hearing = "hearing"
    vision = "vision"
    respiratory_dust = "respiratory_dust"
    cancer_disease = "cancer_disease"
    cardiac = "cardiac"
    skin_scarring = "skin_scarring"
    internal_other = "internal_other"
    multiple = "multiple"
    not_stated = "not_stated"
    unclassified = "unclassified"


class WCMechanismEnum(str, Enum):
    workplace_stress_bullying = "workplace_stress_bullying"
    exposure_to_trauma = "exposure_to_trauma"
    assault_violence = "assault_violence"
    manual_handling = "manual_handling"
    slip_trip_fall = "slip_trip_fall"
    struck_by_object = "struck_by_object"
    equipment_machinery = "equipment_machinery"
    motor_vehicle = "motor_vehicle"
    repetitive_strain = "repetitive_strain"
    occupational_exposure = "occupational_exposure"
    disease_infection = "disease_infection"
    not_stated = "not_stated"
    unclassified = "unclassified"


class WCCaseSchema(BaseModel):
    insurer_name: str = Field(description=(
        "Name of the workers compensation INSURER or scheme agent (e.g. EML/Employers Mutual, "
        "icare, Allianz, GIO, QBE, StateCover, Gallagher Bassett, Workers Compensation Nominal "
        "Insurer). NOT the employer, and NOT an insurer that appears only inside a cited case "
        "name. If the decision says the employer was SELF-INSURED, or names the Treasury Managed "
        "Fund, answer 'self-insured'. Most workers compensation decisions name the employer as "
        "respondent and never name the insurer at all — leave this EMPTY in that case rather than "
        "inferring one from the employer's identity."))
    insurer_evidence: str = Field(description="Short verbatim quote identifying the insurer. EMPTY if none.")

    outcome: WCOutcomeEnum = Field(description=(
        "Who won, from the INJURED WORKER's perspective, regardless of who was the applicant "
        "(employers and insurers are often the applicant). 'claimant' = the worker got "
        "substantially what was sought; 'insurer' = the worker got nothing; 'mixed' = the worker "
        "succeeded on some issues/heads and failed on others. A remittal to a Medical Assessor "
        "after the worker won the disputed issue is 'claimant', not 'mixed'."))
    outcome_reason: str = Field(description="One sentence explaining the outcome classification.")

    liability_posture: LiabilityPostureEnum = Field(description=(
        "'liability_denied' = the respondent put injury/liability itself in issue (denied injury, "
        "s 4/s 9A causation, s 11A reasonable action defence). "
        "'quantum_or_entitlement_only' = liability/injury was admitted or not in issue, and the "
        "dispute was about entitlement, treatment, or amount. 'unclear' only if genuinely absent. "
        "Boundary rules, applied STRICTLY IN THIS ORDER - stop at the first that matches: "
        "(1) if the respondent disputes whether a claimed CONSEQUENTIAL or SECONDARY condition "
        "was caused by the accepted injury, that is 'liability_denied' - liability for that "
        "condition is the live question, and this takes precedence even where the primary "
        "injury is admitted and its effects are said to have resolved; "
        "(2) otherwise, if the respondent ADMITS the injury but says its effects have RESOLVED "
        "or CEASED, that is 'quantum_or_entitlement_only' - the argument is about continuing "
        "entitlement, not about whether the injury happened; "
        "(3) a dispute confined to whether proposed treatment is reasonably necessary under s 60, "
        "where injury and causation are not in issue, is 'quantum_or_entitlement_only'; "
        "(4) a procedural, interlocutory or reconsideration application that puts no liability "
        "question in issue at all is 'not_applicable_procedural', NOT "
        "'quantum_or_entitlement_only'."))
    liability_posture_evidence: str = Field(description="Short verbatim quote supporting the posture. EMPTY if none.")

    wpi_contended_by_claimant: str = Field(description=(
        "Whole person impairment % the APPLICANT/worker contended for (their expert's or their "
        "submission's figure; the highest if several). Number only, no % sign. EMPTY if the "
        "worker advanced no figure. Never report statutory threshold language (e.g. 'more than "
        "10%', 'at least 15%') as a contended figure."))
    wpi_contended_by_insurer: str = Field(description=(
        "Whole person impairment % the RESPONDENT/insurer contended for. Number only. EMPTY if "
        "none. Same threshold exclusion as above."))
    wpi_contended_evidence: str = Field(description="Short quote(s) showing who advanced which figure. EMPTY if none.")

    claimant_legal_representation: YesNoEnum = Field(description=(
        "Was the WORKER legally represented (counsel and/or solicitor acting for them)? 'No' only "
        "where the decision indicates they were self-represented/unrepresented/appeared in person."))
    claimant_representation_evidence: str = Field(description="Short quote. EMPTY if none.")

    claimant_age: str = Field(description=(
        "The WORKER's age at the date of injury, as a plain number. Resolve in this order and "
        "stop at the first that works: (1) an age the decision states for the worker; (2) their "
        "date of birth, from which you compute age at injury; (3) their year of birth, giving "
        "injury year minus birth year. EMPTY if none of the three appears. Never use the age of "
        "a doctor, dependant, witness or other person, and never infer age from years of service "
        "or from the length of a working life."))
    claimant_date_of_birth: str = Field(description=(
        "The WORKER's date of birth as YYYY-MM-DD, or just YYYY if only a year of birth is "
        "given. EMPTY if the decision does not state it."))
    claimant_age_basis: str = Field(description=(
        "How claimant_age was obtained: 'stated' / 'from_date_of_birth' / 'from_year_of_birth' / "
        "'not_stated'."))

    claimant_interpreter_used: YesNoEnum = Field(description=(
        "Did the WORKER use an interpreter at any point (hearing, medical examination, or "
        "statement)? 'No' where none is mentioned or one was expressly not required."))
    interpreter_evidence: str = Field(description="Short quote. EMPTY if none.")

    medical_assessor_involved: YesNoEnum = Field(description=(
        "Is a Medical Assessor / Approved Medical Specialist involved in this matter at ANY "
        "point, past or future? Answer Yes if a Medical Assessment Certificate was issued, "
        "relied on or challenged, AND ALSO Yes where THIS decision remits or refers the matter "
        "to the President for referral to a Medical Assessor. A referral yet to happen still "
        "counts as involvement."))
    remitted_to_medical_assessor: YesNoEnum = Field(description=(
        "Did THIS decision remit or refer the matter to the President for referral to a Medical "
        "Assessor (typically for WPI assessment)?"))

    consequential_condition_claimed: YesNoEnum = Field(description=(
        "Did the worker claim a CONSEQUENTIAL condition — a further condition said to result from "
        "the accepted primary injury? 'No' if the phrase appears only in cited authority."))
    fatality: YesNoEnum = Field(description=(
        "Did THE WORKER whose claim this is die? Yes only for a death-benefit or dependency "
        "claim brought in respect of that worker. Answer No when someone else is deceased - a "
        "treating doctor, a relative, a witness, a party in a cited authority - and No where "
        "the words 'deceased' or 'dependant' appear only in quoted legislation or case law."))

    lump_sum_amount: str = Field(description=(
        "Total lump sum ORDERED to be paid, as a plain number (no $ or commas). Exclude weekly "
        "payments, medical expenses and legal costs. EMPTY if no lump sum was ordered — including "
        "where the matter was remitted for assessment instead."))
    lump_sum_type: LumpSumTypeEnum = Field(description="Head of compensation the lump sum was paid under.")

    weekly_benefit_amount: str = Field(description=(
        "Weekly compensation rate ORDERED, as a plain number. Where a schedule of stepped rates "
        "is ordered, give the HIGHEST rate. EMPTY if no weekly benefit was ordered."))

    legal_costs_amount: str = Field(description=(
        "LEGAL costs figure disclosed for EITHER party (professional costs, party/party costs, "
        "disbursements, an assessed or ordered costs sum), as a plain number. This is rare: an "
        "order that costs be paid 'as agreed or assessed' has NO figure and must be EMPTY. Never "
        "report treatment, surgery, travel or funds-management costs here."))
    legal_costs_evidence: str = Field(description="Short quote showing the costs figure. EMPTY if none.")

    costs_order_direction: CostsDirectionEnum = Field(description=(
        "WHO bears costs, independent of any amount. NSW workers compensation costs are regulated "
        "and do NOT follow the event: a worker who loses is not generally exposed to the insurer's "
        "costs, and much of a worker's legal work is funded through IRO/ILARS rather than by a "
        "costs order. Classify what THIS decision orders. "
        "'respondent_pays_applicant' where the respondent/insurer is ordered to pay the worker's "
        "costs (including 'costs as agreed or assessed' in the worker's favour); "
        "'applicant_pays_respondent' for the rare reverse order; "
        "'each_party_bears_own' where each side bears its own costs, including the s 341 default; "
        "'no_order_as_to_costs' where the decision expressly makes no costs order; "
        "'costs_reserved' where costs are deferred; "
        "'costs_assessment_application' where the PROCEEDING ITSELF is an application to assess "
        "costs rather than a merits dispute; "
        "'not_addressed' where the decision says nothing about costs."))
    costs_order_evidence: str = Field(description="Short quote showing the costs direction. EMPTY if none.")
    costs_complexity_uplift_percent: str = Field(description=(
        "Percentage uplift for complexity ordered on the costs, as a plain number (e.g. '20' for "
        "a 20% uplift). EMPTY if no uplift was ordered."))
    exempt_worker: YesNoEnum = Field(description=(
        "Is the worker an EXEMPT worker — a police officer, paramedic/ambulance officer, "
        "firefighter, or coal miner — whose costs and entitlements are treated differently? "
        "Answer Yes only where the decision indicates the worker falls in one of those classes, "
        "or expressly calls them an exempt worker."))
    iro_ilars_funding_mentioned: YesNoEnum = Field(description=(
        "Does the decision mention IRO, ILARS or Independent Review Office funding of the "
        "worker's legal costs?"))

    proceeding_posture: ProceedingPostureEnum = Field(description=(
        "Is this a fresh first-instance dispute, or connected to an earlier determination between "
        "these parties on this claim? Use 'reconsideration' where the Commission is asked to "
        "reconsider, revoke or set aside its own earlier Certificate of Determination. Use "
        "'related_earlier_proceedings' where the decision refers to an EARLIER determination, "
        "certificate or proceeding between the same parties on the same claim. Use "
        "'first_instance' otherwise. Do NOT treat citation of appellate AUTHORITY, or a general "
        "discussion of appeal rights, as an earlier proceeding — only an actual prior proceeding "
        "in THIS claim counts."))
    proceeding_posture_evidence: str = Field(description="Short quote. EMPTY if none.")

    dispute_notice_date: str = Field(description=(
        "Date of the insurer's dispute notice under s 74 or s 78 of the 1998 Act (the notice "
        "declining or disputing liability), as YYYY-MM-DD. Where several are mentioned, give the "
        "one that gave rise to THIS dispute. EMPTY if no such notice date is stated."))

    wpi_determined: str = Field(description=(
        "Whole person impairment percentage the DECISION ITSELF finds, accepts, or records as "
        "assessed — a Medical Assessor's certified figure, an agreed figure, or the Member's own "
        "finding. Plain number, no % sign. This is distinct from the figures each side contended "
        "for. EMPTY where the decision makes no impairment finding, which is the norm when the "
        "matter is remitted for assessment. Never report statutory threshold language as a "
        "determined figure."))

    primary_injury: WCInjuryEnum = Field(description=(
        "Principal injury by body system. Apply this test IN ORDER and stop at the first that "
        "resolves: (1) the body system the ORDERS/DETERMINATIONS address - what compensation, "
        "treatment or impairment was actually decided; (2) if the orders address more than one "
        "equally, the one named FIRST in the catchwords; (3) if still tied, 'multiple'. "
        "Use 'not_stated' where the decision never identifies an injury (for example a purely "
        "procedural or costs determination). Do not choose a body part merely because it is "
        "mentioned in the medical history."))
    mechanism: WCMechanismEnum = Field(description=(
        "How the injury came about, ONLY where the decision actually says. For psychological "
        "claims arising from management action, bullying or workload use "
        "'workplace_stress_bullying'; for repeated exposure to traumatic incidents (police, "
        "ambulance, fire) use 'exposure_to_trauma'. "
        "Use 'not_stated' when the decision does not describe how the injury happened — common "
        "where liability is admitted and only treatment or entitlement is in issue, or where the "
        "injury is a disease or gradual-onset condition attributed to the nature and conditions "
        "of employment rather than an incident. Do NOT default to 'slip_trip_fall' or any other "
        "incident type merely because an injury occurred; a fall must actually be described. "
        "A gradual condition from repeated duties is 'repetitive_strain', not an incident."))

    legal_complexity: int = Field(ge=0, le=2, description=(
        "Legal and procedural complexity, 0-2, judged RELATIVE TO OTHER CONTESTED workers "
        "compensation decisions. Every case in this corpus is disputed, so being disputed is not "
        "complexity, and neither is length of reasons nor volume of medical evidence. "
        "Apply this test IN ORDER and stop at the first that matches — do not average, and do not "
        "default to 1. "
        "SCORE 2 if ANY ONE of these is present: a s 11A(1) defence decided on its merits; a "
        "contest about worker or deemed-worker status, or whether the injury arose out of or in "
        "the course of employment; three or more parties, or apportionment, contribution or "
        "s 151Z recovery between insurers or employers; a jurisdictional, limitation, estoppel or "
        "res judicata question; reconsideration or setting aside of an earlier determination; "
        "construction of a provision on which the Member finds no settled authority; oral evidence "
        "or cross-examination leading to express findings on credit. "
        "SCORE 0 if ALL of these hold: a single issue; no contested question of statutory "
        "construction; no oral evidence; and the outcome follows settled principle on largely "
        "uncontested facts. Typical 0s: whether proposed surgery is reasonably necessary under "
        "s 60; consent or largely agreed orders; distributing a death benefit among agreed "
        "dependants; an arithmetic weekly-benefits calculation. "
        "SCORE 1 otherwise: the ordinary contested matter — a statutory test such as s 4(b)(ii) "
        "or main contributing factor applied to conflicting expert opinion, a consequential "
        "condition, a disputed PIAWE, or several substantive issues none of which meets the "
        "2 checklist."))
    legal_complexity_reason: str = Field(description=(
        "One short sentence naming the specific legal or procedural work that set the score."))


WC_CASE_SYSTEM_INSTRUCTION = """\
You are extracting structured research fields from a NSW Personal Injury
Commission WORKERS COMPENSATION decision. Read the whole document: the
catchwords and the orders/determinations block carry the holding, and the
reasons carry the disputed figures.

Rules that matter here:
  - Perspective. Outcome is always from the INJURED WORKER's point of view.
    Employers, insurers and the Nominal Insurer are frequently the applicant,
    so never equate "applicant" with "worker".
  - Attribution. A WPI percentage belongs to the side whose expert or
    submission advanced it. Figures from a Medical Assessor, or the Member's
    own finding, belong to NEITHER side — leave those out of the contended
    fields.
  - Statutory thresholds are not findings. "More than 10%", "at least 15%",
    "20% or more" describe a legal test, not an assessment. Never report them
    as a contended or awarded WPI.
  - Ordered, not sought. Money fields report what was ORDERED. A sum merely
    claimed, offered or refused is not an award.
  - Costs. Legal costs are almost never quantified in this jurisdiction; an
    order to pay costs "as agreed or assessed" carries no figure. Treatment,
    surgery, travel and NSW Trustee funds-management costs are NOT legal costs.
  - Insurer vs employer. The respondent is usually the employer. Identify the
    insurer only from text that actually names it as the insurer or scheme
    agent; ignore insurer names that appear solely inside cited case names.
  - Mechanism. Report how the injury happened only if the decision says.
    Where liability is admitted and only treatment or entitlement is in issue,
    the mechanism is often never described — that is 'not_stated', not a
    guess. Never fall back on 'slip_trip_fall' as a default.
  - Complexity is graded WITHIN this corpus. Every decision here is a
    contested matter, so treat the ordinary contested case as the middle of
    the range, not the top of it. Long reasons and thick medical evidence are
    not complexity; contested legal tests, multi-party apportionment,
    jurisdictional questions and credit findings are.

Leave a field EMPTY (or 'Unknown'/'none') rather than guessing. Every evidence
field must be a short verbatim quote from the decision, or EMPTY.
"""


# Deterministic decoding. gpt-5 accepts `seed` but rejects `temperature`
# (capability-tested: "does not support 0 with this model, only the default"),
# so seed is the whole of the pinning we can do at the API layer.
WC_SEED = int(os.getenv("NSW_WC_SEED", "20260815"))

# Reuses the scraper's tracker and its gpt-5 prices, so cost is reported in
# dollars rather than as a multiplier.
COST = CostTracker()

# Context budget, in characters. Fixed sizes keep the prompt byte-identical
# across runs for a given decision.
HEAD_BUDGET = 12_000     # citation, parties, member, catchwords
ORDERS_BUDGET = 12_000   # determinations / orders block: what was actually decided
REASONS_BUDGET = 36_000  # start of the reasons, where the facts and issues sit

ORDERS_MARKERS = (
    "DETERMINATIONS MADE", "THE COMMISSION DETERMINES", "ORDERS", "DETERMINATION",
    "CERTIFICATE OF DETERMINATION", "FINDINGS AND ORDERS",
)
REASONS_MARKERS = ("STATEMENT OF REASONS", "REASONS", "BACKGROUND")


def build_wc_context(source_text):
    """Assemble a DETERMINISTIC slice of the decision for the prompt.

    A truncation that varies run to run makes the prompt vary, which is a
    variance source before the model even reads anything. This takes three
    fixed-size windows anchored on structural markers — header/catchwords, the
    orders block, and the opening of the reasons — so the same decision always
    produces the same prompt. Everything is sliced by character offset, never
    sampled.
    """
    text = source_text or ""
    if len(text) <= HEAD_BUDGET + ORDERS_BUDGET + REASONS_BUDGET:
        return text

    upper = text.upper()
    head = text[:HEAD_BUDGET]

    def find_first(markers, start=0):
        positions = [upper.find(marker, start) for marker in markers]
        positions = [p for p in positions if p >= 0]
        return min(positions) if positions else -1

    orders_start = find_first(ORDERS_MARKERS)
    orders = text[orders_start:orders_start + ORDERS_BUDGET] if orders_start >= 0 else ""

    reasons_start = find_first(REASONS_MARKERS, max(orders_start, 0))
    if reasons_start < 0:
        reasons_start = min(len(text), HEAD_BUDGET)
    reasons = text[reasons_start:reasons_start + REASONS_BUDGET]

    parts = [head]
    if orders:
        parts.append("\n\n[ORDERS / DETERMINATIONS]\n" + orders)
    parts.append("\n\n[REASONS]\n" + reasons)
    return "".join(parts)


def _parse_pinned(extractor, system_instruction, user_content, response_format,
                  context=None, reasoning_effort=WC_REASONING_EFFORT, seed=None):
    """Structured parse with a pinned seed, mirroring the scraper's retry policy.

    LLMExtractor._parse_with_retry does not expose `seed`, so the call is made
    here against the same client. The backoff schedule matches the scraper's so
    this pass behaves identically under quota/transient failure.
    """
    backoff_schedule = [2, 5, 10, 20, 40, 80]
    last_error = None
    for attempt in range(len(backoff_schedule) + 1):
        try:
            completion = extractor.client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                seed=WC_SEED if seed is None else seed,
            )
            COST.record(completion.usage)
            return completion.choices[0].message.parsed, completion.usage, None
        except Exception as error:
            last_error = error
            suffix = f" ({context})" if context else ""
            retryable = _is_quota_error(str(error)) or _is_transient_api_error(error)
            if retryable and attempt < len(backoff_schedule):
                time.sleep(backoff_schedule[attempt])
                continue
            logging.error(f"WC LLM error{suffix}: {error}")
            return None, None, str(error)
    return None, None, str(last_error)


class DenialScopeEnum(str, Enum):
    """WHAT was denied, which is what makes partial denial countable.

    Under the liability_posture binary, an insurer that accepts the injury and
    denies every consequential condition, treatment and impairment head reads as
    'quantum only' - the cooperative category. This enumeration turns "deny in
    part, never in whole" from an inference into a measurement.
    """
    primary_injury = "primary_injury"
    consequential_condition = "consequential_condition"
    specific_treatment = "specific_treatment"
    impairment_head = "impairment_head"
    weekly_entitlement = "weekly_entitlement"
    work_capacity = "work_capacity"
    nothing_denied = "nothing_denied"
    not_stated = "not_stated"


class ConductFindingEnum(str, Enum):
    """Three-way ON PURPOSE.

    'The Member said nothing about the insurer's conduct' and 'the Member
    expressly considered it and did not criticise it' are opposite evidence for
    convergent validity and identical under a boolean. The distinction costs
    nothing at extraction time and cannot be recovered afterwards.
    """
    criticism_made = "criticism_made"
    considered_not_criticised = "considered_not_criticised"
    not_addressed = "not_addressed"


class ConductScopeEnum(str, Enum):
    """WHAT was criticised, mapped to SIRA's Standards of Practice.

    Deliberately the regulator's vocabulary rather than a generic
    disapproval scale. It lets the metric be validated against the actual
    standard - 'does predicted loss correlate with Standards-relevant conduct
    findings' - instead of against whether a Member was unimpressed, and it
    absorbs surveillance and defective notices as scope values rather than
    spawning separate booleans.
    """
    reasons_adequacy = "reasons_adequacy"
    timeliness = "timeliness"
    investigation_proportionality = "investigation_proportionality"
    surveillance = "surveillance"
    failure_to_consider_evidence = "failure_to_consider_evidence"
    defective_notice = "defective_notice"
    other_conduct = "other_conduct"


class WCConductSchema(BaseModel):
    """Second-pass fields: denial scope, success granularity, conduct findings.

    Kept out of WCCaseSchema so it can be versioned and re-run on its own.
    """
    denial_scope: list[DenialScopeEnum] = Field(description=(
        "Every head liability was denied for. ['nothing_denied'] where liability was wholly "
        "admitted. A dispute confined to quantum is 'nothing_denied', NOT 'not_stated'."))
    denial_scope_evidence: str = Field(description=(
        "Short verbatim quote showing what was denied. EMPTY if nothing was denied."))
    heads_claimed: int = Field(description=(
        "How many distinct heads of relief the worker actually pressed for decision: weekly "
        "compensation, medical expenses, lump sum, interest, costs, and so on. 0 if undeterminable."))
    heads_succeeded: int = Field(description=(
        "How many of those heads the worker actually won. Never greater than heads_claimed. "
        "This is what turns 'mixed' from one bucket into a degree: winning four heads of five "
        "and winning one of five are currently the same value."))
    conduct_finding: ConductFindingEnum = Field(description=(
        "Did the Member remark on the INSURER's conduct of the claim? 'criticism_made' only "
        "where the Member is critical. 'considered_not_criticised' where conduct was squarely "
        "raised or addressed and found acceptable. 'not_addressed' where the decision simply "
        "does not deal with it - the common case, and NOT the same as approval."))
    conduct_scope: list[ConductScopeEnum] = Field(description=(
        "What was criticised, where conduct_finding is 'criticism_made'. Empty otherwise."))
    conduct_evidence: str = Field(description=(
        "Short verbatim quote of the Member's remark on conduct. EMPTY where not_addressed."))


WC_CONDUCT_SYSTEM_INSTRUCTION = (
    "You extract a small set of research fields from NSW Personal Injury Commission workers "
    "compensation decisions. Report only what the decision states.\n\n"
    "Rules that matter:\n"
    "- Report what the MEMBER found, never what a party asserted.\n"
    "- Absence is a finding. If the decision does not address the insurer's conduct, say "
    "'not_addressed'. Do NOT infer approval from silence, and do NOT infer criticism from the "
    "insurer losing: an insurer can lose with impeccable conduct on fresh medical evidence, and "
    "can win having handled the matter badly.\n"
    "- Criticism means the Member is critical of how the insurer handled the claim - inadequate "
    "reasons, delay, disproportionate investigation or surveillance, failure to consider material "
    "evidence, defective notices. Disagreeing with the insurer's case is NOT criticism of "
    "conduct.\n"
    "- heads_succeeded can never exceed heads_claimed. Count heads actually pressed for decision, "
    "not every remedy mentioned in the application.\n"
    "- Quote verbatim in the evidence fields, and leave them empty rather than paraphrasing."
)


class ReliefGrantedEnum(str, Enum):
    """How much of what the worker pressed was actually granted.

    Replaces the head-count ratio, which measured the wrong granularity: 64.4%
    of this corpus presses a SINGLE head, so a heads_succeeded/heads_claimed
    ratio can only ever be 0 or 1, and 154 of the 255 'mixed' decisions came
    back at 1.0. Mixed in workers compensation usually means partial success
    WITHIN one head - a reduced period of weekly payments, some treatment
    approved and some refused - which head-counting cannot see.

    This is a degree judgment rather than a fact, which is why it is piloted
    before it is bought: soft ordinals are where run-to-run variance
    concentrates, and an unstable ordinal yields a sensitivity curve made of
    noise.
    """
    fully_granted = "fully_granted"
    substantially_granted = "substantially_granted"
    partly_granted = "partly_granted"
    minimally_granted = "minimally_granted"
    refused = "refused"
    not_determinable = "not_determinable"


class WCReliefSchema(BaseModel):
    """Pilot schema: one ordinal and its evidence."""
    relief_granted_degree: ReliefGrantedEnum = Field(description=(
        "How much of what the worker actually pressed did the Member grant, counting partial "
        "success WITHIN a head and not only whole heads won or lost. 'fully_granted' = "
        "substantially everything sought. 'substantially_granted' = the great majority, with a "
        "minor reduction or carve-out. 'partly_granted' = a real split, roughly half. "
        "'minimally_granted' = a token or small part only. 'refused' = nothing of substance. "
        "'not_determinable' where the orders do not permit the comparison - use it rather than "
        "guessing."))
    relief_granted_evidence: str = Field(description=(
        "Short verbatim quote from the ORDERS supporting the degree. EMPTY if not determinable."))


WC_RELIEF_SYSTEM_INSTRUCTION = (
    "You judge how much of the relief a worker pressed for was granted by the NSW Personal Injury "
    "Commission, from the decision text.\n\n"
    "Rules that matter:\n"
    "- Compare what was ORDERED against what was actually PRESSED for decision, not against "
    "everything ever mentioned in the application.\n"
    "- Partial success within a single head counts. A claim for weekly payments over three years "
    "allowed for eight months is 'partly_granted' or 'minimally_granted', not 'fully_granted'.\n"
    "- Concessions the worker made before the hearing are not losses: judge against what remained "
    "in issue.\n"
    "- 'not_determinable' is a real answer and is preferred to a guess. Use it where the orders "
    "are procedural, remit the matter, or do not state what was sought.\n"
    "- Quote verbatim from the orders in the evidence field."
)


def extract_wc_relief_llm(extractor, source_text, context=None, seed=None):
    """Run the relief-degree pilot on one decision. Returns (parsed, usage, error)."""
    if not source_text:
        return None, None, "empty source"
    user_content = (
        "Judge how much of the relief pressed for was granted in the decision below.\n\n"
        "---\n"
        f"{build_wc_context(source_text)}\n"
        "---\n"
    )
    return _parse_pinned(
        extractor, WC_RELIEF_SYSTEM_INSTRUCTION, user_content, WCReliefSchema,
        context=context, reasoning_effort=WC_REASONING_EFFORT, seed=seed,
    )


def extract_wc_conduct_llm(extractor, source_text, context=None, seed=None):
    """Run the additive second pass on one decision. Returns (parsed, usage, error)."""
    if not source_text:
        return None, None, "empty source"
    user_content = (
        "Extract the denial scope, success granularity and insurer-conduct fields from the "
        "decision below. Report what the Member FOUND.\n\n"
        "---\n"
        f"{build_wc_context(source_text)}\n"
        "---\n"
    )
    return _parse_pinned(
        extractor, WC_CONDUCT_SYSTEM_INSTRUCTION, user_content, WCConductSchema,
        context=context, reasoning_effort=WC_REASONING_EFFORT, seed=seed,
    )


def extract_wc_case_llm(extractor, source_text, context=None, seed=None):
    """Run the WC structured pass on one decision.

    Uses the scraper's OpenAI client (same failure policy) but a pinned seed
    and a deterministic context slice. Returns (parsed, usage, error).
    """
    if not source_text:
        return None, None, "empty source"
    user_content = (
        "Extract the workers compensation research fields from the decision "
        "below. Report what was ORDERED and who actually advanced each figure.\n\n"
        "---\n"
        f"{build_wc_context(source_text)}\n"
        "---\n"
    )
    return _parse_pinned(
        extractor, WC_CASE_SYSTEM_INSTRUCTION, user_content, WCCaseSchema,
        context=context, reasoning_effort=WC_REASONING_EFFORT, seed=seed,
    )


# ----------------------------------------------------------------------
# Text extraction
# ----------------------------------------------------------------------

def html_to_text(html_bytes):
    """Same extraction the scraper uses, so text rules see what the LLM saw.

    NSWPIC HTML numbers paragraphs with <ol><li value="N">; BS4 drops the
    `value` attribute, so inject "N. " before each numbered <li>.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    main = (soup.find("article")
            or soup.find(class_="the-document")
            or soup.find(class_="austlii-doc")
            or soup.body)
    if main is None:
        return ""
    for garbage in main.find_all(["div"], class_=["austlii-header", "breadcrumb",
                                                 "page-footer", "nav"]):
        garbage.decompose()
    for li in main.find_all("li"):
        val = li.get("value")
        if val and str(val).strip().isdigit():
            li.insert(0, NavigableString(f"{val}. "))
    return main.get_text(separator="\n").strip()


def flatten(text):
    """Collapse whitespace so proximity windows in the rules mean characters,
    not newline soup."""
    return re.sub(r"\s+", " ", text or "")


# ----------------------------------------------------------------------
# Case identity
# ----------------------------------------------------------------------

def case_id_from_url(url):
    """'.../NSWPIC/2024/721.html' -> ('[2024] NSWPIC 721', 2024, 721)."""
    match = re.search(r"/NSWPIC/(\d{4})/(\d+)\.(?:html|pdf)", str(url), re.IGNORECASE)
    if not match:
        return "", None, None
    year, number = int(match.group(1)), int(match.group(2))
    return f"[{year}] NSWPIC {number}", year, number


def index_decision_files(folder):
    """Map (year, number) -> {'canonical': name, 'all': [names]}.

    Prefers the case-id-suffixed filename: it is the convention the scraper
    writes today, and it is the copy guaranteed to carry the citation.
    """
    index = {}
    if not os.path.isdir(folder):
        return index
    for name in sorted(os.listdir(folder)):
        match = re.search(r"_(\d{4})_(\d+)\.html$", name)
        if match:
            key = (int(match.group(1)), int(match.group(2)))
            index.setdefault(key, {"canonical": None, "all": []})
            index[key]["all"].append(name)
            if index[key]["canonical"] is None:
                index[key]["canonical"] = name
    # Legacy un-suffixed copies belong to the case whose suffixed name shares
    # the title prefix. They are duplicates, never the canonical file.
    prefixes = {re.sub(r"_\d{4}_\d+\.html$", "", entry["canonical"]): key
                for key, entry in index.items() if entry["canonical"]}
    for name in sorted(os.listdir(folder)):
        if re.search(r"_\d{4}_\d+\.html$", name) or not name.endswith(".html"):
            continue
        key = prefixes.get(name[:-len(".html")])
        if key:
            index[key]["all"].append(name)
    return index


# ----------------------------------------------------------------------
# Insurer
# ----------------------------------------------------------------------

# Canonical name -> alias patterns. NSW workers-comp respondents are usually
# the employer, so the insurer has to be recovered from the body of the text.
INSURERS = [
    ("Employers Mutual (EML)", r"Employers Mutual(?: NSW)?(?: Limited| Ltd)?|\bEML\b"),
    ("icare / Insurance and Care NSW", r"\bicare\b|Insurance and Care NSW|Insurance & Care NSW"),
    ("Workers Compensation Nominal Insurer", r"Workers Compensation Nominal Insurer|\bNominal Insurer\b"),
    ("Allianz Australia Insurance", r"Allianz(?: Australia)?(?: Insurance)?(?: Limited| Ltd)?"),
    ("GIO", r"\bGIO\b|AAI Limited t/as GIO|AAI Limited"),
    ("QBE", r"QBE(?: Insurance)?(?: \(Australia\))?(?: Limited| Ltd)?"),
    ("StateCover Mutual", r"StateCover(?: Mutual)?(?: Limited| Ltd)?"),
    ("Gallagher Bassett", r"Gallagher Bassett"),
    ("Hospitality Employers Mutual", r"Hospitality Employers Mutual|\bHEM\b"),
    ("Coal Mines Insurance", r"Coal Mines Insurance"),
    ("Guild Insurance", r"Guild Insurance"),
    ("CGU", r"\bCGU\b"),
    ("Zurich", r"Zurich(?: Australian Insurance)?(?: Limited| Ltd)?"),
    ("Catholic Church Insurance", r"Catholic Church Insurance"),
    ("Insurance Australia (IAG/NRMA)", r"Insurance Australia (?:Group|Limited|Ltd)|\bIAG\b|\bNRMA\b"),
    ("DXC / Xchanging", r"\bDXC\b|Xchanging"),
    # Not an insurer as such, but the answer to "who carried the risk" for
    # large employers and the NSW government, which is what the field is for.
    ("Self-insured / Treasury Managed Fund", r"self[- ]insur\w*|Treasury Managed Fund|\bTMF\b"),
]

# "the insurer, X" and friends — the highest-confidence textual signal.
INSURER_APPOSITION = re.compile(
    r"(?:the (?:respondent'?s? )?insurer[,:]?\s+|insurer\s*\(\s*|on behalf of the (?:respondent|employer)[,]?\s+by\s+)"
    r"([A-Z][A-Za-z&.'/\- ]{2,60}?)(?=[,.;)]|\s+(?:issued|declined|disputed|denied|accepted|submitted|paid))",
    re.I)


# A mention inside a case citation is about some other litigation, not this
# claim. Frequency is a blunt proxy for "is this mention about our case": it
# admits an insurer cited twice as authority and rejects a legitimate single
# mention in the procedural history. Testing the CONTEXT of each mention is
# both cheaper and more accurate.
NEUTRAL_CITATION = re.compile(
    r"\[\d{4}\]\s*[A-Za-z]{2,12}\s*\d+"      # [2020] NSWCA 123
    r"|\(\d{4}\)\s*\d+\s*[A-Z]"              # (2020) 99 NSWLR 1
    r"|\d+\s+NSWLR\s+\d+", re.I)
CASE_NAME_V = re.compile(r"\s+v\s+", re.I)


def is_citation_context(text, start, end):
    """True when this mention sits inside a cited case name or citation.

    Two signals: a neutral or report citation within ~60 characters, or a
    ' v ' adjacent enough to make the insurer a party to some OTHER case.
    Insurers that are parties to THIS matter are already caught by the
    party-name tier, which runs first.
    """
    window = text[max(0, start - 60): end + 60]
    if NEUTRAL_CITATION.search(window):
        return True
    adjacent = text[max(0, start - 45): end + 45]
    return bool(CASE_NAME_V.search(adjacent))


def substantive_mentions(text, pattern):
    """Occurrences of an insurer alias that are NOT inside a citation."""
    return [match for match in re.finditer(pattern, text, re.I)
            if not is_citation_context(text, match.start(), match.end())]


def detect_insurer(party_text, body_text):
    """Return (insurer, source).

    Tiers, strongest first: a party to this matter; an explicit 'the insurer,
    X' apposition; then any mention that survives citation-stripping. A wrong
    insurer is worse than a missing one — insurer identity is joinable from
    icare's own records, but a wrong name silently corrupts per-insurer
    analysis — so every tier prefers returning nothing to guessing.
    """
    for canonical, pattern in INSURERS:
        if re.search(pattern, party_text or "", re.I):
            return canonical, "party_name"

    body = body_text or ""
    apposition = INSURER_APPOSITION.search(body)
    if apposition:
        named = apposition.group(1).strip()
        for canonical, pattern in INSURERS:
            if re.search(pattern, named, re.I):
                return canonical, "named_as_insurer"

    counts = Counter()
    for canonical, pattern in INSURERS:
        hits = len(substantive_mentions(body, pattern))
        if hits:
            counts[canonical] = hits
    if counts:
        top, hits = counts.most_common(1)[0]
        return top, f"text_mention(n={hits})"
    return "", "not_stated"


# ----------------------------------------------------------------------
# Outcome
# ----------------------------------------------------------------------

PARTIAL = re.compile(r"\bin part\b|\bpartly\b|\bpartial(?:ly)?\b", re.I)
AWARD_APPLICANT = re.compile(r"award (?:for|in favour of) (?:the )?(?:applicant|first applicant)", re.I)
AWARD_RESPONDENT = re.compile(r"award (?:for|in favour of) (?:the )?(?:first |second |third )?respondent", re.I)


def classify_outcome(claimant_outcome, result_text, body_text):
    """Collapse to claimant / insurer / mixed.

    `Claimant Outcome` is already normalised to the injured worker (so it
    survives insurer-as-applicant matters), but it is binary. "Mixed" is
    recovered from explicit partial-success language, or from an orders block
    that awards for the applicant on one issue and the respondent on another.
    """
    result = str(result_text or "")
    body = body_text or ""
    if PARTIAL.search(result):
        return "mixed", "partial_language_in_result"
    if AWARD_APPLICANT.search(result) and AWARD_RESPONDENT.search(result):
        return "mixed", "result_awards_both_ways"
    orders = body[:40_000]
    if AWARD_APPLICANT.search(orders) and AWARD_RESPONDENT.search(orders):
        return "mixed", "orders_award_both_ways"
    outcome = str(claimant_outcome or "").strip()
    if outcome == "For Claimant":
        return "claimant", "claimant_outcome_field"
    if outcome == "Against Claimant":
        return "insurer", "claimant_outcome_field"
    return "", "not_determined"


# ----------------------------------------------------------------------
# Lump sum
# ----------------------------------------------------------------------

# An order to pay a lump sum, with a figure. Split by head of compensation so
# the workbook can say which kind of lump sum it was.
LUMP_PATTERNS = [
    ("s66_permanent_impairment", re.compile(
        r"s(?:ection)?\.?\s?66\b[^.\n]{0,160}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,160}?s(?:ection)?\.?\s?66\b"
        + r"|permanent impairment compensation[^.\n]{0,120}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,140}?(?:permanent impairment|whole person impairment)", re.I)),
    ("s67_pain_and_suffering", re.compile(
        r"s(?:ection)?\.?\s?67\b[^.\n]{0,160}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,160}?s(?:ection)?\.?\s?67\b"
        + r"|pain and suffering[^.\n]{0,120}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,120}?pain and suffering", re.I)),
    ("death_benefit", re.compile(
        r"s(?:ection)?\.?\s?(?:25|85A)\b[^.\n]{0,200}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,200}?s(?:ection)?\.?\s?(?:25|85A)\b"
        + r"|death benefit[^.\n]{0,150}?" + MONEY
        + r"|" + MONEY + r"[^.\n]{0,150}?death benefit", re.I)),
]

# The figure must sit in an operative sentence, not in a party's submission.
ORDERED = re.compile(
    r"\b(?:order|orders|ordered|award|awards|awarded|determines?|directs?|"
    r"is to pay|are to pay|will pay|to pay the applicant|to pay to)\b", re.I)


def extract_lump_sum(flat_text):
    """Return (amount, kind, context) for the largest ordered lump sum."""
    best = (None, "", "")
    for kind, pattern in LUMP_PATTERNS:
        for match in pattern.finditer(flat_text):
            window = flat_text[max(0, match.start() - 250): match.end() + 120]
            if not ORDERED.search(window):
                continue
            amounts = [to_money(a) for a in re.findall(MONEY, match.group(0))]
            amounts = [a for a in amounts if a is not None]
            if not amounts:
                continue
            top = max(amounts)
            if best[0] is None or top > best[0]:
                best = (top, kind, match.group(0)[:200])
    return best


def to_money(raw):
    try:
        return float(str(raw).replace("$", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# Legal costs
# ----------------------------------------------------------------------

# Terms that mean LEGAL costs. Deliberately narrow: an unrestricted "costs"
# near a dollar figure is overwhelmingly treatment cost in this corpus
# (133 naive hits, ~6 of them genuine).
LEGAL_COSTS = re.compile(
    r"professional costs"
    r"|legal (?:costs|fees)"
    r"|party[/ \-]?party costs|party and party costs"
    r"|solicitor[/ \-](?:and )?client costs"
    r"|costs? assess(?:or|ment|ed)"
    r"|costs (?:in the sum of|in the amount of|fixed at|assessed at)"
    r"|Schedule 6[^.\n]{0,60}costs|costs[^.\n]{0,60}Schedule 6"
    r"|item WK\d+", re.I)

# Nearby words that mark the dollar figure as something other than legal costs.
NOT_LEGAL_COSTS = re.compile(
    r"costs? of and incidental to|reasonably necessary|surgery|treatment|"
    r"physiotherapy|medication|prescription|per week|weekly|permanent impairment|"
    r"pain and suffering|funeral|death benefit|PIAWE|hospital|MRI|scan|injection|"
    r"orthodontic|dental|running costs|court costs|management fee|"
    r"s(?:ection)?\.?\s?25\(1A\)|(?:cl|clause)\.?\s?177", re.I)


def extract_legal_costs(flat_text):
    """Return (amount, context) for a disclosed legal-costs figure, else
    (None, ''). Takes the largest surviving figure: where a decision quantifies
    both a claim and an award, the claim is the fuller disclosure."""
    best_amount, best_context = None, ""
    for match in re.finditer(MONEY, flat_text):
        window = flat_text[max(0, match.start() - 90): match.end() + 90]
        if not LEGAL_COSTS.search(window) or NOT_LEGAL_COSTS.search(window):
            continue
        amount = to_money(match.group(0))
        if amount is None:
            continue
        if best_amount is None or amount > best_amount:
            best_amount, best_context = amount, window.strip()[:250]
    return best_amount, best_context


# ----------------------------------------------------------------------
# WPI mentioned in text
# ----------------------------------------------------------------------

WPI_FIGURE = re.compile(
    PERCENT + r"[^.\n]{0,60}?(?:whole person impairment|WPI|permanent impairment|impairment)"
    + r"|(?:whole person impairment|WPI|permanent impairment)[^.\n]{0,60}?" + PERCENT, re.I)

# Statutory-threshold boilerplate reads like a finding but is not one. This is
# the corpus's single biggest source of false Accepted-WPI values.
WPI_THRESHOLD_TALK = re.compile(
    r"(?:greater than|more than|at least|exceeds?|not (?:more|greater) than|less than|"
    r"equal to or greater than)\s*(?:10|11|15|20|21|30)\s?(?:%|per cent|percent)"
    r"|(?:10|11|15|20|21|30)\s?(?:%|per cent|percent)\s*(?:or more|or greater|threshold)"
    r"|threshold", re.I)


def wpi_mentioned(flat_text):
    """True when the text states an impairment percentage that is not merely
    statutory-threshold language."""
    for match in WPI_FIGURE.finditer(flat_text):
        window = flat_text[max(0, match.start() - 120): match.end() + 120]
        if not WPI_THRESHOLD_TALK.search(window):
            return True
    return False


# ----------------------------------------------------------------------
# Injury and mechanism taxonomies
# ----------------------------------------------------------------------

# Ordered most- to least-specific: the first category to win on score wins,
# and ties break toward the earlier entry.
INJURY_CATEGORIES = [
    ("psychological", r"psychological|psychiatric|PTSD|post[- ]traumatic|depress|anxiet|"
                      r"adjustment disorder|mental (?:health|injury)|nervous shock"),
    ("hearing", r"hearing loss|deafness|tinnitus|binaural|industrial deafness|noise[- ]induced"),
    ("spinal", r"lumbar|cervical|thoracic|spine|spinal|back injury|neck injury|disc "
               r"(?:bulge|protrusion|herniation)|sciatica|radiculopathy"),
    ("upper_limb", r"shoulder|rotator cuff|elbow|wrist|hand|finger|thumb|carpal tunnel|"
                   r"upper extremity|epicondylitis|forearm"),
    ("lower_limb", r"knee|ankle|foot|feet|toe|hip|leg|lower extremity|meniscus|"
                   r"achilles|plantar|patell"),
    ("head_brain", r"traumatic brain injury|\bTBI\b|head injury|concussion|skull|"
                   r"intracranial|cerebral"),
    ("respiratory_dust", r"silicosis|asbestos|mesothelioma|asbestosis|pneumoconiosis|"
                         r"dust disease|respiratory|lung|occupational asthma|COPD"),
    ("cancer_disease", r"\bcancer\b|carcinoma|melanoma|leukaemia|lymphoma|"
                       r"Q fever|hepatitis|COVID|infectious disease"),
    ("cardiac", r"myocardial|heart attack|cardiac|coronary|cardiovascular"),
    ("vision", r"vision|eye injury|ocular|blindness|retina|cornea"),
    ("skin_scarring", r"scarring|burns?\b|dermatitis|skin condition|laceration"),
    ("internal_other", r"hernia|abdominal|bowel|bladder|kidney|reproductive"),
]

MECHANISM_CATEGORIES = [
    ("workplace_stress_bullying", r"bullying|harassment|workload|performance (?:management|appraisal)|"
                                  r"disciplinary|investigation into|restructure|redundanc|"
                                  r"interpersonal conflict|s(?:ection)?\.?\s?11A|reasonable action"),
    ("exposure_to_trauma", r"exposure to trauma|traumatic (?:incident|event)s?|critical incident|"
                           r"attended (?:the )?scene|body recovery|fatal(?:ity|ities)? attended|"
                           r"vicarious trauma|cumulative trauma exposure"),
    ("assault_violence", r"assault|attacked|violence|punched|threatened with|armed robbery|"
                         r"aggressive (?:patient|client|customer|student)"),
    ("manual_handling", r"lifting|manual handling|carrying|pushing|pulling|"
                        r"moving (?:a |the )?(?:patient|box|load)|twisting while"),
    ("slip_trip_fall", r"slip(?:ped)?|trip(?:ped)?|fell|fall(?:ing)? (?:from|down|off)|"
                       r"lost (?:his|her|their) footing|uneven surface|wet floor|ladder"),
    ("struck_by_object", r"struck by|hit by|falling object|crush(?:ed|ing)|caught between|"
                         r"impact from|collided with"),
    ("equipment_machinery", r"machine(?:ry)?|equipment (?:failure|malfunction)|forklift|conveyor|"
                            r"power tool|saw\b|grinder|defective (?:plant|equipment)|guard(?:ing)? "
                            r"(?:was )?(?:missing|removed)"),
    ("motor_vehicle", r"motor vehicle accident|car accident|truck (?:accident|rollover)|"
                      r"road (?:traffic )?accident|while driving|vehicle collision"),
    ("repetitive_strain", r"repetitive|overuse|cumulative (?:strain|effect)|"
                          r"nature and conditions of (?:his|her|their|the) employment|"
                          r"gradual onset|repeated (?:movement|task)"),
    ("occupational_exposure", r"exposure to (?:dust|noise|silica|asbestos|chemical|fume|solvent)|"
                              r"inhal(?:ed|ation)|contaminat|toxic"),
    ("disease_infection", r"contracted|infection|virus|zoonotic|Q fever|COVID-19"),
]


def classify(categories, tiers):
    """Score keyword hits tier by tier and return (label, tier, evidence).

    `tiers` is [(name, text), ...] most authoritative first. The first tier
    that matches anything decides. Weighting all tiers into one score instead
    lets a long judgment out-vote its own headnote — a body that mentions a
    body part forty times in passing beats catchwords that name the injury
    once — so reliability is expressed as precedence, not weight.

    Ties break toward the earlier category, which is why the taxonomies are
    ordered most- to least-specific.
    """
    for tier_name, tier_text in tiers:
        scores = Counter()
        for label, pattern in categories:
            hits = len(re.findall(pattern, tier_text or "", re.I))
            if hits:
                scores[label] = hits
        if not scores:
            continue
        ordered = [label for label, _ in categories if label in scores]
        top = max(ordered, key=lambda label: (scores[label], -ordered.index(label)))
        evidence = f"{tier_name}: " + "; ".join(f"{label}={scores[label]}" for label in ordered)
        return top, tier_name, evidence[:250]
    return "unknown", "none", ""


# ----------------------------------------------------------------------
# Representation, interpreter, Medical Assessor, liability posture
# ----------------------------------------------------------------------

# Applicant-specific representation. PIC decisions rarely carry a formal
# REPRESENTATION block (6.4%), so most of the signal is narrative.
APPLICANT_REPRESENTED = re.compile(
    r"counsel for the applicant|applicant'?s counsel"
    r"|solicitors? for the applicant|applicant'?s solicitors?"
    r"|appear(?:ed|ing) for the applicant"
    r"|on behalf of the applicant[,]? (?:Mr|Ms|Mrs|Dr)"
    r"|(?:Mr|Ms|Mrs)\s+\w+[^.\n]{0,40}of counsel[^.\n]{0,60}applicant", re.I)
APPLICANT_UNREPRESENTED = re.compile(
    r"applicant (?:is|was|appeared) (?:self[- ]represented|unrepresented|in person)"
    r"|self[- ]represented applicant|unrepresented applicant"
    r"|applicant[^.\n]{0,30}appeared in person", re.I)
# Generic representation: someone ran an advocacy role, side unspecified.
ANY_REPRESENTATION = re.compile(
    r"\bof counsel\b|instructed by|\bsolicitor\b|\bcounsel\b", re.I)

INTERPRETER_USED = re.compile(
    r"(?:assisted|aided) by (?:an|the|a) (?:\w+ )?interpreter"
    r"|(?:an|the) interpreter (?:was|is) (?:used|present|engaged|provided|required|assisting)"
    r"|with the (?:assistance|aid) of (?:an|the) interpreter"
    r"|through (?:an|the) interpreter"
    r"|gave evidence (?:through|with) (?:an|the) interpreter"
    r"|(?:\w+) interpreter (?:was|is) (?:present|used|provided)", re.I)
INTERPRETER_NOT_USED = re.compile(
    r"no interpreter (?:was |is )?(?:required|needed|necessary|used)"
    r"|did not (?:require|need|use) an interpreter"
    r"|without (?:the (?:assistance|aid) of )?an interpreter", re.I)

MEDICAL_ASSESSOR = re.compile(
    r"Medical Assessor|Medical Assessment Certificate|\bMAC\b"
    r"|Approved Medical Specialist|\bAMS\b", re.I)

LIABILITY_DENIED = re.compile(
    r"denied liability|liability (?:is |was |has been )?(?:wholly |partly )?denied"
    r"|disput(?:ed|es|ing) liability|liability (?:is |was )?(?:in )?(?:dispute|issue)"
    r"|declined liability|dispute[sd]? that the applicant (?:sustained|suffered)"
    r"|whether the applicant (?:sustained|suffered) (?:an? )?injury"
    r"|s(?:ection)?\.?\s?11A|s(?:ection)?\.?\s?9A"
    r"|no injury (?:was )?(?:sustained|suffered)"
    r"|denies? (?:that )?(?:the applicant|she|he|they) (?:sustained|suffered)", re.I)
LIABILITY_NOT_IN_ISSUE = re.compile(
    r"liability (?:is|was) not (?:in )?(?:issue|dispute|disputed)"
    r"|(?:injury|liability) (?:is|was|has been) (?:admitted|accepted|conceded)"
    r"|there is no (?:issue|dispute) that[^.\n]{0,120}(?:injury|injured|sustained)"
    r"|accepted (?:work(?:place)? )?injury|accepted liability"
    r"|no issue (?:is taken |arises )?(?:as to|with|regarding) liability", re.I)


def detect_representation(flat_text):
    """Return (Yes/No/Unknown, basis) for the claimant specifically."""
    if APPLICANT_UNREPRESENTED.search(flat_text):
        return "No", "explicit_unrepresented"
    if APPLICANT_REPRESENTED.search(flat_text):
        return "Yes", "applicant_specific"
    if ANY_REPRESENTATION.search(flat_text):
        return "Yes", "generic_representation_language"
    return "Unknown", "no_signal"


def detect_interpreter(flat_text):
    if INTERPRETER_USED.search(flat_text):
        return "Yes", "interpreter_used"
    if INTERPRETER_NOT_USED.search(flat_text):
        return "No", "explicitly_not_required"
    if re.search(r"\binterpreter\b", flat_text, re.I):
        return "Unknown", "mentioned_without_clear_use"
    return "No", "no_mention"


# A death-benefit claim is structural: it is brought under s 25 or s 85A, by or
# for dependants, in respect of a deceased worker. The old rule fired on a bare
# "deceased" or "dependan", which also matches a deceased treating doctor, a
# dependent relative, and quoted legislation — a 15% false-positive rate on a
# question that should be unambiguous.
FATALITY_STRUCTURAL = re.compile(
    r"s(?:ection)?\.?\s?(?:25|25\(1\)|85A)\b[^.\n]{0,120}(?:death|deceased|dependan)"
    r"|death benefit|lump sum death"
    r"|(?:the )?deceased worker|deceased'?s? (?:dependan|estate|widow|spouse)"
    r"|claim (?:for|in respect of) the death of"
    r"|died (?:as a result of|in the course of|from) ", re.I)
# Someone else died: exclude before concluding the worker did.
FATALITY_OTHER_PARTY = re.compile(
    r"deceased (?:doctor|specialist|practitioner|witness|father|mother|brother|sister)"
    r"|(?:his|her|their) (?:late |deceased )(?:father|mother|husband|wife|brother|sister|son|daughter)",
    re.I)


def detect_fatality(flat_text, nature, catchwords):
    """Structural test for a death/dependency claim.

    Nature and catchwords are the authoritative signals — the Commission
    classifies these matters explicitly — so they decide first, and the body
    text is consulted only as a fallback.
    """
    if str(nature or "").strip() == "Death Benefit":
        return True
    head = str(catchwords or "")
    if re.search(r"death benefit|dependan|deceased worker|compensation to relatives", head, re.I):
        return True
    body = flat_text or ""
    if FATALITY_STRUCTURAL.search(body):
        window_hits = len(FATALITY_STRUCTURAL.findall(body))
        # A single structural mention alongside an explicit other-party death
        # is more likely to be that other person.
        if window_hits == 1 and FATALITY_OTHER_PARTY.search(body):
            return False
        return True
    return False


# Costs DIRECTION is stated far more often than any amount, so it is the
# recoverable half of the costs question. Ordered most- to least-specific.
COSTS_DIRECTION_RULES = [
    ("costs_assessment_application", re.compile(
        r"application (?:for|to have) (?:the )?costs (?:be )?assess|costs assessment application"
        r"|assessment of (?:the )?(?:applicant'?s? |respondent'?s? )?costs under", re.I)),
    ("applicant_pays_respondent", re.compile(
        r"applicant[^.\n]{0,60}?(?:to |is to |will |must )pay[^.\n]{0,50}?respondent[^.\n]{0,20}?costs", re.I)),
    ("respondent_pays_applicant", re.compile(
        r"respondent[^.\n]{0,60}?(?:to |is to |will |must )pay[^.\n]{0,50}?(?:applicant|worker)[^.\n]{0,20}?costs"
        r"|pay the applicant'?s? costs", re.I)),
    ("no_order_as_to_costs", re.compile(r"no order (?:as to|for|in respect of) costs", re.I)),
    ("each_party_bears_own", re.compile(
        r"each party[^.\n]{0,50}(?:bear|pay)[^.\n]{0,20}own costs|bear (?:its|their|his|her) own costs"
        r"|s(?:ection)?\.?\s?341[^.\n]{0,80}own costs", re.I)),
    ("costs_reserved", re.compile(r"costs?\s+(?:are |is |be )?reserved", re.I)),
]
# A bare "costs as agreed or assessed" attaches to whichever party is named.
COSTS_AGREED_ASSESSED = re.compile(
    r"(applicant|respondent|worker)'?s? costs[^.\n]{0,40}(?:as )?agreed or assessed"
    r"|costs[^.\n]{0,40}(?:as )?agreed or assessed", re.I)

UPLIFT = re.compile(r"uplift[^.\n]{0,40}?(\d{1,2})\s?%|(\d{1,2})\s?%[^.\n]{0,40}?uplift", re.I)

# Exempt status attaches to the WORKER's role, not the employer. NSW Police
# employs civilian radio operators; a cleaner at a fire station is not a
# firefighter. Matching employer names alone produced a 15% false-positive
# rate, so occupation decides and the employer is only corroborating.
EXEMPT_OCCUPATION = re.compile(
    r"police officer|constable|detective|sergeant|police prosecutor"
    r"|paramedic|ambulance officer"
    r"|fire ?fighter|fire officer|station officer"
    r"|coal miner|mine worker|underground miner", re.I)
EXEMPT_DECLARED = re.compile(r"exempt worker", re.I)
# Employers whose workforce is predominantly, but not exclusively, exempt.
EXEMPT_EMPLOYER = re.compile(
    r"NSW Police Force|Police Force of New South Wales"
    r"|Ambulance Service|NSW Ambulance"
    r"|Fire (?:and|&) Rescue|Rural Fire Service"
    r"|Coal Mines Insurance", re.I)
# Civilian roles inside an otherwise-exempt employer.
NON_OPERATIONAL_ROLE = re.compile(
    r"radio operation|communications officer|administrat|clerical|cleaner|"
    r"civilian|call ?taker|control room|analyst|technician|caterer", re.I)


def detect_exempt_worker(occupation, employer, flat_text):
    """Exempt worker status, decided on the worker's ROLE.

    Returns Yes/No. An express 'exempt worker' finding wins; then occupation;
    then an exempt employer, but only where the stated occupation does not
    look like a civilian role within it.
    """
    if EXEMPT_DECLARED.search(flat_text or ""):
        return "Yes"
    role = str(occupation or "")
    if EXEMPT_OCCUPATION.search(role):
        return "Yes"
    if NON_OPERATIONAL_ROLE.search(role):
        return "No"
    if EXEMPT_EMPLOYER.search(str(employer or "")):
        # Employer is exempt-heavy and the role is unstated: treat as unknown
        # rather than asserting exemption.
        return "Unknown" if not role or role.strip().lower() == "not stated" else "Yes"
    return "No"
IRO_ILARS = re.compile(r"\bILARS\b|Independent Review Office|\bIRO\b", re.I)

# Deliberately narrow: the WORD "appeal" is useless here (judicial-review and
# Court of Appeal language appears in 31% of decisions purely as cited
# authority). Only an actual earlier proceeding in this claim counts.
RECONSIDERATION = re.compile(
    r"reconsider(?:ation)? of (?:the |its )?(?:earlier |previous )?(?:Certificate of Determination|determination|decision)"
    r"|application (?:for|to) reconsider|revoke the Certificate of Determination"
    r"|s(?:ection)?\.?\s?57\b[^.\n]{0,60}reconsider", re.I)
EARLIER_PROCEEDING = re.compile(
    r"Certificate of Determination dated|previously determined|earlier proceedings"
    r"|previous proceedings|earlier Certificate of Determination"
    r"|determined by (?:Member|Arbitrator) [A-Z]\w+ on \d", re.I)

DISPUTE_NOTICE = re.compile(
    r"s(?:ection)?\.?\s?7[48]\b[^.\n]{0,160}?(\d{1,2}\s+\w+\s+\d{4})"
    r"|(\d{1,2}\s+\w+\s+\d{4})[^.\n]{0,160}?s(?:ection)?\.?\s?7[48]\b", re.I)


def detect_costs_direction(flat_text):
    """Return (direction, evidence). Regulated-costs semantics, not loser-pays."""
    for label, pattern in COSTS_DIRECTION_RULES:
        match = pattern.search(flat_text)
        if match:
            return label, match.group(0)[:200]
    match = COSTS_AGREED_ASSESSED.search(flat_text)
    if match:
        party = (match.group(1) or "").lower()
        if party == "respondent":
            return "applicant_pays_respondent", match.group(0)[:200]
        # Unattributed "costs as agreed or assessed" in a worker's matter is
        # conventionally the respondent paying the worker's costs.
        return "respondent_pays_applicant", match.group(0)[:200]
    return "not_addressed", ""


def detect_proceeding_posture(flat_text):
    if RECONSIDERATION.search(flat_text):
        return "reconsideration"
    if EARLIER_PROCEEDING.search(flat_text):
        return "related_earlier_proceedings"
    return "first_instance"


def extract_dispute_notice_date(flat_text):
    """Date of the s 74 / s 78 dispute notice, as YYYY-MM-DD. ~44% of decisions
    put a date within reach of the section reference."""
    match = DISPUTE_NOTICE.search(flat_text)
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    parsed = _naive(raw, dayfirst=True)
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def extract_uplift(flat_text):
    match = UPLIFT.search(flat_text)
    if not match:
        return None
    value = to_money(match.group(1) or match.group(2))
    return value if value is not None and 0 < value <= 100 else None


def detect_medical_assessor(flat_text):
    return ("Yes", "referenced") if MEDICAL_ASSESSOR.search(flat_text) else ("No", "no_mention")


# A consequential-condition dispute has an ACCEPTED primary injury but a
# DENIED consequential one. The concession language ("accepted injury") and the
# denial then both appear, and reading the concession first gets it backwards —
# the live question is liability for the consequential condition.
CONSEQUENTIAL_DENIED = re.compile(
    r"(?:disputes?|denies?|denied|in issue|placed in issue|does not accept)[^.\n]{0,80}"
    r"consequential (?:condition|injury)"
    r"|consequential (?:condition|injury)[^.\n]{0,80}"
    r"(?:is |was |are |were )?(?:disputed|denied|in issue|placed in issue|not accepted)"
    r"|liability is disputed for the consequential", re.I)


# Liability for a consequential condition is often ACCEPTED, and the fight is
# then about incapacity or treatment. Firing on the mere presence of a
# consequential condition alongside dispute language mislabels those.
CONSEQUENTIAL_ACCEPTED = re.compile(
    r"liability (?:for|in respect of)[^.\n]{0,60}(?:injuries|injury|condition)[^.\n]{0,30}accepted"
    r"|liability (?:is |was |has been )?(?:accepted|admitted)"
    r"|(?:accepts?|admits?|admitted|accepted) liability", re.I)


def classify_liability_posture(flat_text, nature):
    """liability_denied / quantum_or_entitlement_only / unclear.

    An explicit concession outranks denial language, because a decision that
    records "liability is not in dispute" often still recites the insurer's
    earlier s 78 denial — EXCEPT where what is denied is a consequential
    condition, which is itself a liability question.
    """
    consequential = CONSEQUENTIAL_DENIED.search(flat_text)
    if consequential and not CONSEQUENTIAL_ACCEPTED.search(flat_text):
        return "liability_denied", "consequential_condition_disputed"
    concede = LIABILITY_NOT_IN_ISSUE.search(flat_text)
    deny = LIABILITY_DENIED.search(flat_text)
    # Saade: liability for the injuries was expressly ACCEPTED and the only
    # live issue was incapacity, but the recited s 74/s 78 denial still tripped
    # the generic denial path. An express acceptance outranks a recited denial
    # wherever no consequential condition is in issue.
    if deny and CONSEQUENTIAL_ACCEPTED.search(flat_text) and not consequential:
        return "quantum_or_entitlement_only", "liability_expressly_accepted"
    if concede and not deny:
        return "quantum_or_entitlement_only", "explicit_concession"
    if deny and not concede:
        return "liability_denied", "explicit_denial"
    if deny and concede:
        # Both present: the earlier-positioned signal usually frames the issue.
        return (("liability_denied", "both_signals_denial_first")
                if deny.start() < concede.start()
                else ("quantum_or_entitlement_only", "both_signals_concession_first"))
    nature_text = str(nature or "")
    if nature_text.startswith("Procedural"):
        # Reconsiderations and interlocutory matters do not put liability in
        # issue at all; "unclear" implied a failed read rather than a category
        # that does not apply.
        return "not_applicable_procedural", "procedural_matter"
    if nature_text == "Liability Dispute":
        return "liability_denied", "nature_field"
    if nature_text in ("Medical Dispute", "Statutory Benefits Dispute",
                       "Permanent Impairment", "Death Benefit"):
        return "quantum_or_entitlement_only", "nature_field"
    return "unclear", "no_signal"


# ----------------------------------------------------------------------
# Contended WPI (claimant vs insurer)
# ----------------------------------------------------------------------

# Who a percentage belongs to is signalled by the party that qualified the
# examiner, or by the party making the submission.
CLAIMANT_SIDE = re.compile(
    r"applicant'?s?(?: (?:expert|doctor|specialist|IME|qualified))?"
    r"|qualified by the applicant|on behalf of the applicant"
    r"|applicant (?:submits|submitted|contends|contended|asserts|argued)"
    r"|for the applicant", re.I)
INSURER_SIDE = re.compile(
    r"respondent'?s?(?: (?:expert|doctor|specialist|IME|qualified))?"
    r"|qualified by the respondent|on behalf of the respondent"
    r"|respondent (?:submits|submitted|contends|contended|asserts|argued)"
    r"|for the respondent|insurer'?s?", re.I)

WPI_VALUE = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s?(?:%|per cent|percent)[^.\n]{0,60}?"
    r"(?:whole person impairment|WPI|permanent impairment)"
    r"|(?:whole person impairment|WPI|permanent impairment)[^.\n]{0,60}?"
    r"(\d{1,3}(?:\.\d+)?)\s?(?:%|per cent|percent)", re.I)


def contended_wpi(flat_text):
    """Attribute impairment percentages to each side by proximity.

    Returns (claimant_pct, insurer_pct, context). Heuristic and noisy — a
    percentage is credited to whichever party marker sits closest before it —
    so the matched window is exported for auditing. Threshold boilerplate is
    excluded on the same rule as wpi_mentioned().
    """
    claimant, insurer, notes = [], [], []
    for match in WPI_VALUE.finditer(flat_text):
        raw = match.group(1) or match.group(2)
        value = to_money(raw)
        if value is None or value > 100:
            continue
        window = flat_text[max(0, match.start() - 220): match.end() + 60]
        if WPI_THRESHOLD_TALK.search(window):
            continue
        before = flat_text[max(0, match.start() - 220): match.start()]
        claimant_hit = list(CLAIMANT_SIDE.finditer(before))
        insurer_hit = list(INSURER_SIDE.finditer(before))
        last_claimant = claimant_hit[-1].start() if claimant_hit else -1
        last_insurer = insurer_hit[-1].start() if insurer_hit else -1
        if last_claimant < 0 and last_insurer < 0:
            continue
        if last_claimant > last_insurer:
            claimant.append(value)
            notes.append(f"claimant={value}")
        else:
            insurer.append(value)
            notes.append(f"insurer={value}")
    return (max(claimant) if claimant else None,
            max(insurer) if insurer else None,
            "; ".join(notes[:12])[:250])


# ----------------------------------------------------------------------
# Weekly benefit
# ----------------------------------------------------------------------

WEEKLY_RATE = re.compile(MONEY + r"\s*(?:\([^)]{0,40}\)\s*)?per week|at the rate of\s*" + MONEY, re.I)


def extract_weekly_benefit(flat_text):
    """Highest ordered weekly rate, plus how many distinct rates appear.

    WC awards routinely step the rate across many periods, so a single figure
    understates the award; the count flags that a schedule exists.
    """
    rates = []
    for match in WEEKLY_RATE.finditer(flat_text):
        window = flat_text[max(0, match.start() - 250): match.end() + 80]
        if not ORDERED.search(window):
            continue
        for raw in re.findall(MONEY, match.group(0)):
            value = to_money(raw)
            if value is not None and 0 < value < 10_000:
                rates.append(value)
    if not rates:
        return None, 0
    return max(rates), len(set(rates))


# ----------------------------------------------------------------------
# Misc field helpers
# ----------------------------------------------------------------------

# The label must own its line and carry a colon. Without both, "CERTIFICATE OF
# DETERMINATION OF MEMBER" matches and the parser returns the next header
# ("CITATION:") as the member's name.
MEMBER_LABEL = re.compile(
    r"^[ \t]*(SENIOR MEMBER|PRINCIPAL MEMBER|ACTING PRESIDENT|DEPUTY PRESIDENT|PRESIDENT"
    r"|MEMBER|ARBITRATOR)[ \t]*:[ \t]*$", re.I | re.M)
DATEISH = re.compile(r"\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}")
OTHER_LABEL = re.compile(r"^[A-Z][A-Z \t/&'-]{2,40}:$")


def extract_member(raw_text):
    """Read the AustLII header's label/value table.

    The two columns occasionally transpose (the value lands under the wrong
    label), so reject anything that is really a date or another header label.
    """
    head = raw_text[:8000]
    for match in MEMBER_LABEL.finditer(head):
        role = match.group(1).strip().title()
        for line in head[match.end():].split("\n")[1:8]:
            name = line.strip()
            if not name:
                continue
            if DATEISH.search(name) or OTHER_LABEL.match(name):
                break
            if len(name) < 3 or len(name.split()) > 6:
                break
            return name, role
    return "", ""


def canonical_insurer(name):
    """Map a free-text insurer name onto the controlled vocabulary.

    The model returns whatever the decision calls it ("EML", "Employers Mutual
    (NSW) Limited"), which would read as a disagreement with the rule pass on
    every row. Returns (canonical_or_original, matched_bool).
    """
    text = str(name or "").strip()
    if not text:
        return "", False
    for canonical, pattern in INSURERS:
        if re.search(pattern, text, re.I):
            return canonical, True
    return text, False


def clip_ordinal(value, ceiling=2):
    """The pipeline scored Legal Procedural Complexity and Work Impact Severity
    on 0-3; the requested schema is 0-2. Clip, and keep the raw value beside
    it so nothing is silently lost."""
    try:
        return min(int(float(value)), ceiling)
    except (TypeError, ValueError):
        return None


def as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else result


def blank_to_none(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


# ----------------------------------------------------------------------
# Row builder
# ----------------------------------------------------------------------

def build_row(csv_row, raw_text, canonical_file, all_files):
    """One workbook row. `csv_row` is the flat-CSV record; `raw_text` is the
    decision text (empty string when the HTML is missing)."""
    flat = flatten(raw_text)
    case_id, year, number = case_id_from_url(csv_row.get("URL"))

    party_text = " ".join(str(csv_row.get(k) or "") for k in ("Applicant", "Respondent", "Case Name"))
    insurer, insurer_source = detect_insurer(party_text, flat)

    outcome, outcome_basis = classify_outcome(
        csv_row.get("Claimant Outcome"), csv_row.get("Result"), flat)

    csv_lump = as_float(csv_row.get("Lump Sum"))
    text_lump, lump_kind, lump_context = extract_lump_sum(flat)
    if csv_lump is not None and text_lump is not None:
        lump_source = "csv+text"
    elif csv_lump is not None:
        lump_source = "csv_only"
    elif text_lump is not None:
        lump_source = "text_only"
    else:
        lump_source = "none"
    lump_amount = csv_lump if csv_lump is not None else text_lump

    legal_costs, legal_costs_context = extract_legal_costs(flat)

    catchwords = str(csv_row.get("Catchwords") or "")
    description = str(csv_row.get("Description") or "")
    # Catchwords name what the Member actually decided; the description is a
    # faithful summary; the body mentions everything argued. Fall through only
    # when the tier above is silent.
    # The body tier is DELIBERATELY EXCLUDED. A keyword count over full text
    # matches body parts in medical history, prior injuries, and cited
    # authority, so it is structurally biased toward over-detection — which is
    # what drove `multiple` and the 39% disagreement rate. Catchwords and the
    # description state what the decision is ABOUT; if neither resolves it, the
    # honest answer is "unknown", not a full-text guess. See FALLBACK_PRINCIPLE.
    tiers = [("catchwords", catchwords), ("description", description)]
    injury, injury_tier, injury_evidence = classify(INJURY_CATEGORIES, tiers)
    mechanism, mechanism_tier, mechanism_evidence = classify(MECHANISM_CATEGORIES, tiers)

    member, member_role = extract_member(raw_text)

    wpi = as_float(csv_row.get("Impairment % (Accepted)"))
    wpi_claimant, wpi_insurer, wpi_contended_context = contended_wpi(flat)

    represented, represented_basis = detect_representation(flat)
    interpreter, interpreter_basis = detect_interpreter(flat)
    assessor, assessor_basis = detect_medical_assessor(flat)
    liability_posture, liability_basis = classify_liability_posture(flat, csv_row.get("Nature"))
    weekly_amount, weekly_rate_count = extract_weekly_benefit(flat)
    duration_days, duration_years = injury_to_decision(
        csv_row.get("Injury Date"), csv_row.get("Decision Date"))
    costs_direction, costs_direction_evidence = detect_costs_direction(flat)
    notice_date = extract_dispute_notice_date(flat)
    notice_to_decision = date_gap(notice_date, csv_row.get("Decision Date"))
    age_rule, age_basis_rule = rule_claimant_age(csv_row.get("Claimant Age"))

    return {
        # --- identity -------------------------------------------------
        "case_id": case_id,
        "case_year": year,
        "case_number": number,
        "case_name": blank_to_none(csv_row.get("Case Name")),
        "source_html_file": canonical_file or "",
        "source_url": csv_row.get("URL"),
        "duplicate_html_files": max(0, len(all_files) - 1),
        "decision_text_chars": len(raw_text or ""),

        # --- 3. insurer ----------------------------------------------
        "insurer_name": insurer,
        "insurer_source": insurer_source,
        "applicant": blank_to_none(csv_row.get("Applicant")),
        "respondent": blank_to_none(csv_row.get("Respondent")),

        # --- 4/5. dates ----------------------------------------------
        "accident_date": blank_to_none(csv_row.get("Injury Date")),
        "decision_date": blank_to_none(csv_row.get("Decision Date")),
        "injury_to_decision_days": duration_days,
        "injury_to_decision_years": duration_years,

        # --- 6. nature ------------------------------------------------
        "nature_of_case": blank_to_none(csv_row.get("Nature")),

        # --- 7. outcome -----------------------------------------------
        "outcome": outcome,
        "outcome_basis": outcome_basis,
        "outcome_analysable": str(csv_row.get("Nature") or "").strip().startswith("Procedural") is False,
        "result_text": blank_to_none(csv_row.get("Result")),

        # --- 8. lump sum ----------------------------------------------
        "lump_sum_amount": lump_amount,
        "lump_sum_present": lump_source != "none",
        "lump_sum_type": lump_kind,
        "lump_sum_source": lump_source,
        "lump_sum_context": lump_context,

        # --- 9. WPI ---------------------------------------------------
        "wpi_percent": wpi,
        "wpi_provenance": blank_to_none(csv_row.get("WPI Provenance")),
        "wpi_percent_in_text": wpi_mentioned(flat),
        "wpi_contended_by_claimant": wpi_claimant,
        "wpi_contended_by_insurer": wpi_insurer,
        "wpi_contended_context": wpi_contended_context,
        "wpi_determined": None,

        # --- 10-13. claimant ------------------------------------------
        "claimant_age": age_rule,
        "claimant_age_basis": age_basis_rule,
        "claimant_date_of_birth": None,
        "claimant_age_at_decision": as_int(csv_row.get("Claimant Age At Decision")),
        "claimant_gender": blank_to_none(csv_row.get("Claimant Gender")),
        "employer_name": blank_to_none(csv_row.get("Employer Name")),
        "claimant_occupation": blank_to_none(csv_row.get("Claimant Occupation")),

        # --- 14-17, 20. ordinals --------------------------------------
        "psych_injury_emphasis": as_int(csv_row.get("Psychological Injury Emphasis")),
        "legal_complexity": clip_ordinal(csv_row.get("Legal Procedural Complexity")),
        "legal_complexity_raw": as_int(csv_row.get("Legal Procedural Complexity")),
        "liability_clarity": as_int(csv_row.get("Liability Clarity")),
        "pre_existing_conditions": as_int(csv_row.get("Pre-existing Condition Salience")),
        "work_impact_severity": clip_ordinal(csv_row.get("Work Impact Severity")),
        "work_impact_severity_raw": as_int(csv_row.get("Work Impact Severity")),
        "ability_to_work": invert_ordinal(clip_ordinal(csv_row.get("Work Impact Severity"))),

        # --- 18. income -----------------------------------------------
        "pre_injury_weekly_income": as_float(csv_row.get("Claimant Weekly Income")),
        "pre_injury_income_basis": blank_to_none(csv_row.get("Claimant Weekly Income Basis")),

        # --- 19. legal costs ------------------------------------------
        "legal_costs_amount": legal_costs,
        "legal_costs_disclosed": legal_costs is not None,
        "legal_costs_context": legal_costs_context,
        "costs_order_direction": costs_direction,
        "costs_order_direction_evidence_rule": costs_direction_evidence,
        "costs_complexity_uplift_percent": extract_uplift(flat),
        "exempt_worker": detect_exempt_worker(
            csv_row.get("Claimant Occupation"), csv_row.get("Employer Name"), flat),
        "iro_ilars_funding_mentioned": "Yes" if IRO_ILARS.search(flat) else "No",

        # --- 21/22. taxonomies ----------------------------------------
        "primary_injury": injury,
        "primary_injury_tier": injury_tier,
        "primary_injury_evidence": injury_evidence,
        "mechanism": mechanism,
        "mechanism_tier": mechanism_tier,
        "mechanism_evidence": mechanism_evidence,

        # --- extras worth having --------------------------------------
        # --- process / conduct of the case ----------------------------
        "claimant_legal_representation": represented,
        "claimant_legal_representation_basis": represented_basis,
        "claimant_interpreter_used": interpreter,
        "claimant_interpreter_basis": interpreter_basis,
        "medical_assessor_involved": assessor,
        "medical_assessor_basis": assessor_basis,
        "remitted_to_medical_assessor": bool(
            re.search(r"remit(?:ted)?[^.\n]{0,80}Medical Assessor|refer(?:ral|red)?[^.\n]{0,60}"
                      r"Medical Assessor", flat, re.I)),
        "liability_posture": liability_posture,
        "liability_posture_basis": liability_basis,
        "consequential_condition_claimed": bool(
            re.search(r"consequential (?:condition|injury)", flat, re.I)),
        "fatality": detect_fatality(flat, csv_row.get("Nature"), csv_row.get("Catchwords")),
        "s11A_defence_raised": bool(re.search(r"s(?:ection)?\.?\s?11A", flat, re.I)),
        "proceeding_posture": detect_proceeding_posture(flat),
        "dispute_notice_date": notice_date,
        "notice_to_decision_days": notice_to_decision,

        # --- weekly benefit -------------------------------------------
        "weekly_benefit_amount": weekly_amount,
        "weekly_benefit_rate_count": weekly_rate_count,
        "weekly_benefit_text": blank_to_none(csv_row.get("Weekly Benefit")),

        # --- extras worth having --------------------------------------
        "member_name": member,
        "member_role": member_role,
        "regulatory_sections": blank_to_none(csv_row.get("Regulatory Sections")),
        "catchwords": (catchwords or "")[:CELL_CHAR_CAP],
        "medical_costs_addressed": blank_to_none(csv_row.get("Medical Costs")),
        "injury_burden_intensity": as_int(csv_row.get("Injury Burden Intensity")),
        "causation_complexity": as_int(csv_row.get("Causation Complexity")),
        "treatment_burden": as_int(csv_row.get("Treatment Burden")),
        "needs_review": blank_to_none(csv_row.get("Needs Review")),
    }


def _yes_no(value):
    text = str(getattr(value, "value", value) or "").strip()
    return text if text in ("Yes", "No", "Unknown") else "Unknown"


def _enum(value):
    return str(getattr(value, "value", value) or "").strip()


def _num(value):
    return to_money(str(value).replace("%", "")) if str(value or "").strip() else None


def _int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# LLM field -> (row column, converter). The rule-based value is preserved
# alongside as `<column>_rule` so disagreements stay visible instead of being
# silently overwritten.
LLM_OVERLAY = [
    ("insurer_name", "insurer_name", lambda v: canonical_insurer(v)[0]),
    ("outcome", "outcome", _enum),
    ("liability_posture", "liability_posture", _enum),
    ("wpi_contended_by_claimant", "wpi_contended_by_claimant", _num),
    ("wpi_contended_by_insurer", "wpi_contended_by_insurer", _num),
    ("claimant_legal_representation", "claimant_legal_representation", _yes_no),
    ("claimant_interpreter_used", "claimant_interpreter_used", _yes_no),
    ("medical_assessor_involved", "medical_assessor_involved", _yes_no),
    ("remitted_to_medical_assessor", "remitted_to_medical_assessor", _yes_no),
    ("consequential_condition_claimed", "consequential_condition_claimed", _yes_no),
    ("fatality", "fatality", _yes_no),
    ("lump_sum_amount", "lump_sum_amount", _num),
    ("lump_sum_type", "lump_sum_type", _enum),
    ("weekly_benefit_amount", "weekly_benefit_amount", _num),
    ("legal_costs_amount", "legal_costs_amount", _num),
    ("primary_injury", "primary_injury", _enum),
    ("mechanism", "mechanism", _enum),
    ("legal_complexity", "legal_complexity", _int_or_none),
    ("claimant_age", "claimant_age", _int_or_none),
    ("claimant_date_of_birth", "claimant_date_of_birth", lambda v: str(v or "").strip()),
    ("wpi_determined", "wpi_determined", _num),
    ("costs_order_direction", "costs_order_direction", _enum),
    ("costs_complexity_uplift_percent", "costs_complexity_uplift_percent", _num),
    ("exempt_worker", "exempt_worker", _yes_no),
    ("iro_ilars_funding_mentioned", "iro_ilars_funding_mentioned", _yes_no),
    ("proceeding_posture", "proceeding_posture", _enum),
    ("dispute_notice_date", "dispute_notice_date", lambda v: str(v or "").strip()),
]

# Evidence quotes the model must supply for its harder calls.
LLM_EVIDENCE = [
    ("insurer_evidence", "insurer_evidence"),
    ("outcome_reason", "outcome_reason"),
    ("liability_posture_evidence", "liability_posture_evidence"),
    ("wpi_contended_evidence", "wpi_contended_evidence"),
    ("claimant_representation_evidence", "claimant_representation_evidence"),
    ("interpreter_evidence", "interpreter_evidence"),
    ("legal_costs_evidence", "legal_costs_evidence"),
    ("legal_complexity_reason", "legal_complexity_reason"),
    ("costs_order_evidence", "costs_order_evidence"),
    ("proceeding_posture_evidence", "proceeding_posture_evidence"),
    ("claimant_age_basis", "claimant_age_basis_llm"),
]


def merge_llm_into_row(row, parsed, error=None):
    """Overlay the LLM pass onto a rule-based row, IN PLACE.

    The LLM wins on every field it covers — those are the fields the rules
    cannot resolve honestly — but the rule value is kept as `<field>_rule` and
    an `<field>_agreement` flag records whether the two independent methods
    said the same thing. Disagreement is the audit signal: it marks the rows
    worth reading by hand.
    """
    row["llm_status"] = "ok" if parsed is not None else (error or "not run")
    if parsed is None:
        for _, column, _conv in LLM_OVERLAY:
            row[f"{column}_rule"] = row.get(column)
            row[f"{column}_source"] = "rule"
        return row

    for attribute, column, convert in LLM_OVERLAY:
        rule_value = row.get(column)
        llm_value = convert(getattr(parsed, attribute, None))
        row[f"{column}_rule"] = rule_value
        empty = llm_value in (None, "", "none", "unclassified", "not_determined")
        row[column] = rule_value if empty else llm_value
        row[f"{column}_source"] = "rule" if empty else "llm"
        row[f"{column}_agreement"] = ("no_rule_baseline" if column in NO_RULE_BASELINE
                                      else _agreement(rule_value, llm_value))

    for attribute, column in LLM_EVIDENCE:
        row[column] = str(getattr(parsed, attribute, "") or "")[:600]

    row["lump_sum_present"] = row.get("lump_sum_amount") is not None
    row["legal_costs_disclosed"] = row.get("legal_costs_amount") is not None
    return row


# Second-pass columns. No rule counterpart exists for any of them, so unlike
# LLM_OVERLAY there is nothing to cross-check against and no _rule/_agreement
# pair is emitted. conduct_finding in particular is single-source by nature:
# no regex can tell criticism from disagreement.
CONDUCT_OVERLAY = [
    ("denial_scope", "denial_scope"),
    ("denial_scope_evidence", "denial_scope_evidence"),
    ("heads_claimed", "heads_claimed"),
    ("heads_succeeded", "heads_succeeded"),
    ("conduct_finding", "conduct_finding"),
    ("conduct_scope", "conduct_scope"),
    ("conduct_evidence", "conduct_evidence"),
]


def apply_conduct(row, parsed, error=None):
    """Write the second-pass fields onto a row.

    `conduct_status` distinguishes 'the pass has not been run' from 'the pass
    ran and found nothing', for the same reason conduct_finding is three-way:
    a blank that could mean either is not evidence of anything.
    """
    row["conduct_status"] = "ok" if parsed is not None else (error or "not run")
    if parsed is None:
        for _attribute, column in CONDUCT_OVERLAY:
            row.setdefault(column, None)
        row["claimant_success_degree"] = None
        return row

    for attribute, column in CONDUCT_OVERLAY:
        value = getattr(parsed, attribute, None)
        if isinstance(value, list):
            value = ";".join(item.value if isinstance(item, Enum) else str(item)
                             for item in value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, str):
            value = value[:600]
        row[column] = value

    claimed = row.get("heads_claimed") or 0
    succeeded = row.get("heads_succeeded") or 0
    # Guard the model's arithmetic rather than trusting it: heads_succeeded is
    # told never to exceed heads_claimed, and a violation is a signal about the
    # extraction, not a value to publish.
    if claimed and succeeded > claimed:
        row["conduct_status"] = "heads_inconsistent"
        row["claimant_success_degree"] = None
    elif claimed:
        # The sensitivity curve's x-axis: the share of pressed heads actually
        # won, so 'mixed' stops being one bucket covering four-of-five and
        # one-of-five alike.
        row["claimant_success_degree"] = round(succeeded / claimed, 3)
    else:
        row["claimant_success_degree"] = None
    return row


# Fields with NO usable rule counterpart. legal_complexity's "rule" is the
# pipeline's own score, which is 2 on 93% of the corpus — comparing against a
# constant manufactures a disagreement on every row and would be read as a
# quality signal when it is an artefact.
#
# primary_injury and mechanism joined it after restricting the keyword rule to
# catchwords+description failed to move the disagreement rate (injury 39/100
# unchanged, mechanism 33->46/100). A keyword count and a reading of the orders
# are measuring different things, so their disagreement is not a second
# opinion and must not be reported as a quality signal. Validate these two
# against a HAND-LABELLED sample instead: see --validation-sample.
NO_RULE_BASELINE = {"legal_complexity", "primary_injury", "mechanism",
                    "proceeding_posture"}

# Facts the CORPUS genuinely almost never records. An all-empty column here is
# a property of the source material, not a failed extractor, and must not be
# read as a bug. Corpus-wide measurements are in the data dictionary.
STRUCTURALLY_SPARSE = {
    "legal_costs_amount",            # 6 decisions in 2,385 quantify legal costs
    "claimant_date_of_birth",        # year of birth appears in ~3%
    "iro_ilars_funding_mentioned",   # ILARS funding is invisible in the text
    "costs_complexity_uplift_percent",
}

# Fields where a rule/LLM disagreement is a genuine quality signal rather than
# a wording difference. Taxonomy fields are excluded: they disagree often by
# design, because the rule is a keyword count and the LLM reads the orders.
DISAGREEMENT_WATCHLIST = (
    "fatality", "outcome", "lump_sum_amount", "wpi_contended_by_claimant",
    "remitted_to_medical_assessor", "claimant_age",
)


def derive_review_flags(row):
    """Recalibrate the review flag against failure modes we actually observe.

    The pipeline's own `needs_review` never fires on this subset — it was tuned
    for the CTP damages pass — so it cannot see an empty WPI in a permanent-
    impairment matter, or a binary field where two independent extractors
    disagree. This replaces it with specific, named flags.
    """
    flags = []

    if not row.get("decision_text_chars"):
        flags.append("no_decision_text")
    elif row.get("decision_text_chars", 0) < 3000:
        flags.append("suspiciously_short_text")

    if row.get("llm_status") not in ("ok", None):
        flags.append("llm_failed")

    # A binary factual question that two methods answer differently is wrong
    # somewhere, and wrong the same way on rows nobody is looking at.
    for field in DISAGREEMENT_WATCHLIST:
        if row.get(f"{field}_agreement") == "differs":
            flags.append(f"disagreement:{field}")

    nature = str(row.get("nature_of_case") or "")
    if nature == "Permanent Impairment" and row.get("wpi_percent") is None:
        # Expected where the matter was remitted; a problem where it was not.
        if str(row.get("remitted_to_medical_assessor")) not in ("Yes", "True", "True "):
            flags.append("permanent_impairment_without_wpi")

    if row.get("lump_sum_present") and row.get("lump_sum_amount") is None:
        flags.append("lump_sum_flagged_without_amount")

    # A single mention is acceptable evidence now that citation context is
    # stripped, so n=1 is no longer flagged. What IS worth flagging is a
    # text-derived age, which sampling showed to be wrong more often than right.
    if row.get("claimant_age_source") == "rule" \
            and str(row.get("claimant_age_basis", "")).startswith("text_"):
        flags.append("age_from_text_rule_unverified")

    age = row.get("claimant_age")
    if age is not None and not (14 <= age <= 100):
        flags.append("implausible_age")

    if row.get("outcome") == "mixed" and not str(row.get("outcome_reason") or "").strip():
        flags.append("mixed_outcome_without_reason")

    wpi = row.get("wpi_percent")
    if wpi is not None and not (0 <= wpi <= 100):
        flags.append("implausible_wpi")

    if row.get("wpi_percent") is None and row.get("wpi_determined") is not None:
        flags.append("wpi_recovered_by_llm_only")

    row["review_flags"] = "; ".join(flags)
    row["needs_review_derived"] = "Yes" if flags else "No"
    return row


def normalise_row(row):
    """Deterministic post-processing, so harmless wording variation does not
    read as a change between runs (variance lever 5).

    Collapses whitespace in free text, trims evidence quotes to a fixed length,
    and rounds floats to a fixed precision. Applied after the LLM overlay so it
    catches model output as well as rule output.
    """
    for key, value in list(row.items()):
        if isinstance(value, str):
            cleaned = re.sub(r"\s+", " ", value).strip()
            if key.endswith(("_evidence", "_reason", "_context")):
                cleaned = cleaned[:400]
            row[key] = cleaned
        elif isinstance(value, float) and value is not None and not pd.isna(value):
            row[key] = round(value, 2)
    return row


def _agreement(rule_value, llm_value):
    """same / differs / rule_only / llm_only / both_empty.

    A rule that answers False is asserting "No", not declining to answer, so
    booleans are normalised before the emptiness test — otherwise every
    correctly-agreed negative reads as 'llm_only'.
    """
    if isinstance(rule_value, bool):
        rule_value = "Yes" if rule_value else "No"
    empties = (None, "", "none", "unclassified", "not_determined", "Unknown")
    rule_empty = rule_value in empties
    llm_empty = llm_value in empties
    if rule_empty and llm_empty:
        return "both_empty"
    if rule_empty:
        return "llm_only"
    if llm_empty:
        return "rule_only"
    if isinstance(rule_value, float) and isinstance(llm_value, float):
        return "same" if abs(rule_value - llm_value) < 0.01 else "differs"
    return "same" if str(rule_value).strip().lower() == str(llm_value).strip().lower() else "differs"


# Age is stated in only ~33% of decisions and a year of birth in ~3%, so the
# realistic ceiling is ~38% however it is extracted. Ordered most- to
# least-reliable; each pattern must name the claimant-ish subject nearby.
AGE_PATTERNS = [
    re.compile(r"\b(\d{2})\s+years?\s+(?:of\s+age|old)\b", re.I),
    re.compile(r"\bage[d]?\s+(\d{2})\b", re.I),
    re.compile(r"\b(\d{2})[- ]year[- ]old\b", re.I),
    re.compile(r"\bage of\s+(\d{2})\b", re.I),
]
YOB_PATTERNS = [
    re.compile(r"born\s+(?:on\s+)?\d{1,2}\s+\w+\s+(\d{4})\b", re.I),
    re.compile(r"date\s+of\s+birth[:\s]+[^.\n]{0,40}?(\d{4})\b", re.I),
    re.compile(r"d\.?o\.?b\.?[:\s]+[^.\n]{0,40}?(\d{4})\b", re.I),
    re.compile(r"\bborn\s+(?:in\s+)?(?:early\s+|late\s+|mid[- ])?(\d{4})\b", re.I),
]


# The age phrase must be about the CLAIMANT. Without this, the first
# age-shaped match in a decision is as likely to be the deceased's son, the
# worker's age when hired decades earlier, or a doctor writing "a 49 year old
# who has worked in heavy work" — all three observed in a 20-case sample.
AGE_SUBJECT = re.compile(r"applicant|worker|claimant|plaintiff", re.I)
AGE_WRONG_SUBJECT = re.compile(
    r"\bson\b|\bdaughter\b|\bchild\b|\bchildren\b|deceased'?s?\b|widow|spouse|"
    r"\bDr\b|doctor|assessor|witness|\bhis wife\b|\bher husband\b|"
    r"commenced (?:employment|work)|started (?:work|employment)|began (?:work|employment)|"
    r"at the time (?:he|she|they) (?:commenced|started|began)", re.I)


def rule_claimant_age(csv_age, flat_text=None, injury_date=None):
    """Age at injury from the pipeline CSV only. Returns (age, basis).

    The text fallback that used to sit here is DELIBERATELY REMOVED. Sampling
    showed it wrong more often than right — it returned a worker's age when
    hired decades earlier, the deceased's son's age, and a doctor's generic
    "a 49 year old" — and the overlay consumes the rule value whenever the
    model declines, so every false positive would have been delivered as fact.

    Age feeds the fairness cohorts, where a wrong value silently misassigns a
    cohort while a missing value merely drops the row. Coverage falls to ~30%,
    which funnel stage 9 documents as non-randomly missing. A documented bias
    beats an undocumented corruption. See FALLBACK_PRINCIPLE.
    """
    existing = as_int(csv_age)
    if existing is not None and 14 <= existing <= 100:
        return existing, "csv"
    return None, "not_stated"


def _naive(value, dayfirst=False):
    """Parse to a tz-naive Timestamp. Some decision dates parse tz-aware, and
    subtracting those from a naive one raises."""
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=dayfirst)
    if pd.isna(parsed):
        return parsed
    try:
        if parsed.tzinfo is not None:
            parsed = parsed.tz_localize(None)
    except (AttributeError, TypeError):
        return pd.NaT
    return parsed


def date_gap(start, end):
    """Whole days between two dates, or None if either is unusable."""
    first, second = _naive(start), _naive(end)
    if pd.isna(first) or pd.isna(second):
        return None
    return int((second - first).days)


def injury_to_decision(injury_date, decision_date):
    """(days, years) between accident and decision. Note the accident date may
    be a deemed date for disease/gradual-onset claims, so this is 'time from
    the operative injury date', not always a real elapsed-since-accident."""
    start, end = _naive(injury_date), _naive(decision_date)
    if pd.isna(start) or pd.isna(end):
        return None, None
    days = int((end - start).days)
    return days, round(days / 365.25, 2)


def invert_ordinal(value, ceiling=2):
    """Work Impact Severity runs high = worse. 'Ability to work' is its
    complement, so 2 = little or no impact on capacity."""
    return None if value is None else ceiling - value


# ----------------------------------------------------------------------
# Data dictionary
# ----------------------------------------------------------------------
#
# One entry per PRIMARY column. The cross-check triples that accompany every
# LLM-owned field (`_rule`, `_source`, `_agreement`) and the evidence quotes
# are generated from LLM_OVERLAY / LLM_EVIDENCE rather than listed by hand, so
# they cannot drift out of step with the schema. build_dictionary() asserts
# that every emitted column is documented.
#
# `provenance` values:
#   csv        copied from the pipeline's flat CSV, not re-derived
#   derived    computed deterministically here (dates, counts, filenames)
#   llm        decided by the WCCaseSchema pass, with a rule cross-check
#   rule       decided by a text rule in this module
#   audit      provenance/agreement metadata about another column

FIELD_DOCS = [
    # field, group, type, provenance, allowed values, definition
    ("case_id", "identity", "text", "derived", "[YYYY] NSWPIC N",
     "Medium-neutral citation. The unique key: one row per case_id."),
    ("case_year", "identity", "int", "derived", "",
     "Year component of the citation."),
    ("case_number", "identity", "int", "derived", "",
     "Sequence number component of the citation."),
    ("case_name", "identity", "text", "csv", "",
     "Full case name as published by AustLII."),
    ("source_html_file", "identity", "text", "derived", "",
     "Canonical downloaded HTML in nsw_pic_decisions/, holding the full decision text. The text "
     "is not embedded in this workbook: 67% of decisions exceed Excel's 32,767-character cell "
     "limit."),
    ("source_url", "identity", "text", "csv", "",
     "AustLII URL. Rows are keyed on this, so the duplicate files on disk cannot double count."),
    ("duplicate_html_files", "identity", "int", "derived", "",
     "Extra copies of this decision on disk under other filenames. Informational: ~6,639 files "
     "cover ~3,503 decisions because a filename-convention change made the scraper's on-disk "
     "cache check miss pre-existing files and re-save them."),
    ("decision_text_chars", "identity", "int", "derived", "",
     "Length of the extracted decision text. 0 means the HTML was missing or unreadable, in "
     "which case every text-derived field on the row is blank."),
    ("llm_status", "identity", "text", "audit", "ok / not run / error text",
     "Whether the LLM pass succeeded for this row."),
    ("llm_cached", "identity", "bool", "audit", "",
     "True when the LLM values were served from wc_llm_cache.json rather than a fresh call."),

    ("insurer_name", "3 insurer", "text", "llm", "controlled list, see 'enumerations'",
     "Workers compensation insurer or scheme agent, normalised to the controlled list. Blank far "
     "more often than for CTP: only 50.1% of WC decisions name an insurer, because the RESPONDENT "
     "is the employer and the scheme agent acts behind it. A further 31.2% say 'the insurer' "
     "without naming it and 18.7% never mention one. Blank means not stated, never 'no insurer'."),
    ("insurer_source", "3 insurer", "text", "audit",
     "party_name / named_as_insurer / text_mention(n=..) / not_stated",
     "Which rule found the insurer, strongest first. Mentions inside a cited case name or "
     "neutral citation are stripped before counting, so n=1 is acceptable evidence: the count is "
     "of mentions about THIS matter. A wrong insurer is worse than a missing one - identity is "
     "joinable from icare's own records, but a wrong name silently corrupts per-insurer work - "
     "so every tier prefers returning nothing to guessing."),
    ("applicant", "3 insurer", "text", "csv", "",
     "Applicant party. NOT necessarily the worker: employers, insurers and the Nominal Insurer "
     "are frequently the applicant."),
    ("respondent", "3 insurer", "text", "csv", "",
     "Respondent party, usually the employer."),

    ("accident_date", "4-5 dates", "date", "csv", "ISO date",
     "Date of injury. May be a DEEMED date for disease or gradual-onset claims, not the date of "
     "an incident."),
    ("decision_date", "4-5 dates", "date", "csv", "ISO date", "Date the decision issued."),
    ("injury_to_decision_days", "4-5 dates", "int", "derived", "",
     "Days from accident_date to decision_date. Inherits the deemed-date caveat above."),
    ("injury_to_decision_years", "4-5 dates", "float", "derived", "",
     "The same interval in years, to 2dp."),

    ("nature_of_case", "6 nature", "text", "csv",
     "Liability Dispute / Medical Dispute / Statutory Benefits Dispute / Permanent Impairment / "
     "Death Benefit / Procedural / other",
     "Dispute type as classified by the pipeline."),
    ("result_text", "6 nature", "text", "csv", "",
     "Free-text result as recorded by the pipeline, e.g. 'Award for Applicant'."),

    ("outcome", "7 outcome", "text", "llm", "claimant / insurer / mixed",
     "BASE RATE (full corpus, rule-side): claimant 75.2%, insurer 19.5%, mixed 5.2%; excluding "
     "Procedural, 76.9 / 17.6 / 5.4. 'Insurer loses' is the MAJORITY class - this is not a "
     "rare-event problem, and the informative minority is where the insurer wins. "
     "Who won, ALWAYS from the injured worker's perspective regardless of who was the applicant. "
     "'mixed' means the worker succeeded on some issues and failed on others. A remittal to a "
     "Medical Assessor after the worker won the disputed issue counts as 'claimant'."),
    ("outcome_basis", "7 outcome", "text", "audit", "",
     "Which rule produced the rule-side outcome."),

    ("lump_sum_amount", "8 lump sum", "number", "llm", "",
     "Gross lump sum ORDERED. Blank where none was ordered, which includes the common case of a "
     "matter remitted for assessment instead. Sparse by nature: weekly payments, not lump sums, "
     "are the dominant WC remedy."),
    ("lump_sum_present", "8 lump sum", "bool", "derived", "",
     "True when lump_sum_amount is populated."),
    ("lump_sum_type", "8 lump sum", "text", "llm",
     "s66_permanent_impairment / s67_pain_and_suffering / death_benefit / other / none",
     "Head of compensation the lump sum was paid under."),
    ("lump_sum_source", "8 lump sum", "text", "audit", "csv+text / csv_only / text_only / none",
     "Which rule-side method found a lump sum, before the LLM overlay."),
    ("lump_sum_context", "8 lump sum", "text", "audit", "",
     "Matched text for the rule-side lump sum, for auditing."),

    ("wpi_percent", "9 WPI", "number", "csv", "0-100",
     "Whole person impairment the pipeline ACCEPTED as the case's finding. Populated on 21.2% of "
     "WC rows. Blank does not mean no percentage appears - see wpi_percent_in_text."),
    ("wpi_provenance", "9 WPI", "text", "csv", "stated / not_assessed",
     "Why wpi_percent is or is not populated."),
    ("wpi_percent_in_text", "9 WPI", "bool", "rule", "",
     "True when the decision states an impairment percentage that is not statutory-threshold "
     "boilerplate. True on 34.3% of WC decisions versus 21.2% for wpi_percent: the gap is "
     "competing expert assessments, insurer offers and apportionment figures the Member never "
     "adopted, usually because the matter was remitted to a Medical Assessor."),
    ("wpi_contended_by_claimant", "9 WPI", "number", "llm", "0-100",
     "Impairment percentage the WORKER contended for. Excludes Medical Assessor figures, the "
     "Member's own finding, and statutory threshold language."),
    ("wpi_contended_by_insurer", "9 WPI", "number", "llm", "0-100",
     "Impairment percentage the INSURER/respondent contended for. Same exclusions. 0 is a "
     "meaningful value here - insurers do contend for 0%."),
    ("wpi_contended_context", "9 WPI", "text", "audit", "",
     "Rule-side attribution trace for the contended figures."),

    ("claimant_age", "10-13 claimant", "int", "llm", "",
     "Age at injury. ~30% populated, and NON-RANDOMLY missing - age is stated more often where "
     "it matters to the reasoning. The text fallback that once filled this was REMOVED after "
     "sampling showed it wrong more often than right (age at hiring, the deceased's son, a "
     "doctor's generic 'a 49 year old'). Age feeds the fairness cohorts, where a wrong value "
     "silently misassigns a cohort while a missing value merely drops the row. See the FALLBACK "
     "PRINCIPLE note."),
    ("claimant_age_at_decision", "10-13 claimant", "int", "csv", "", "Age when the decision issued."),
    ("claimant_gender", "10-13 claimant", "text", "csv", "Male / Female / Not stated",
     "Corpus split is 1,436 Male / 896 Female / 53 Not stated."),
    ("employer_name", "10-13 claimant", "text", "csv", "", "Employer at the time of injury."),
    ("claimant_occupation", "10-13 claimant", "text", "csv", "",
     "Occupation at the time of injury. 'Not stated' where absent."),

    ("psych_injury_emphasis", "14-17,20 ordinals", "int", "csv", "0-2",
     "How prominent the psychological component is. Higher = more prominent."),
    ("legal_complexity", "14-17,20 ordinals", "int", "llm", "0-2",
     "NO RULE BASELINE: legal_complexity_agreement is reported as 'no_rule_baseline' because the "
     "pipeline score it would be compared against is 2 on 93% of the corpus, so any comparison "
     "manufactures a disagreement. Do not read legal_complexity_raw as a second opinion. "
     "Legal and procedural complexity RELATIVE TO OTHER CONTESTED WC matters. Scored by an "
     "ordered decision rule: 2 if any of a s 11A defence decided on merits, a worker/deemed-worker "
     "or course-of-employment contest, three or more parties or s 151Z recovery, a jurisdictional/"
     "limitation/estoppel question, reconsideration, unsettled construction, or credit findings; "
     "0 if all of single issue, no construction question, no oral evidence, settled principle; "
     "1 otherwise. REPLACES the pipeline's score, which was calibrated against litigation "
     "generally and therefore rated 2,212 of 2,385 disputed cases as maximally complex."),
    ("legal_complexity_raw", "14-17,20 ordinals", "int", "csv", "0-3",
     "The pipeline's original Legal Procedural Complexity, unclipped. Retained for comparison; "
     "see the caveat on legal_complexity."),
    ("liability_clarity", "14-17,20 ordinals", "int", "csv", "0-2",
     "How clearly liability was established. Higher = clearer / less disputed."),
    ("pre_existing_conditions", "14-17,20 ordinals", "int", "csv", "0-2",
     "Salience of pre-existing conditions. Higher = more salient."),
    ("work_impact_severity", "14-17,20 ordinals", "int", "csv", "0-2",
     "Impact of the injury on capacity for work, clipped from the pipeline's 0-3. HIGHER = WORSE."),
    ("work_impact_severity_raw", "14-17,20 ordinals", "int", "csv", "0-3",
     "The pipeline's unclipped Work Impact Severity."),
    ("ability_to_work", "14-17,20 ordinals", "int", "derived", "0-2",
     "Complement of work_impact_severity. HIGHER = MORE able to work. Provided because the "
     "requested schema asked for ability rather than severity; the two are mirror images, so use "
     "one or the other, never both as independent features."),

    ("pre_injury_weekly_income", "18 income", "number", "csv", "",
     "Pre-injury weekly income, gross, usually PIAWE. ~32% populated."),
    ("pre_injury_income_basis", "18 income", "text", "csv", "",
     "What the income figure represents and any conversion applied."),

    ("legal_costs_amount", "19 legal costs", "number", "llm", "",
     "A disclosed LEGAL costs figure for either party. Almost always blank: costs in this "
     "jurisdiction are ordered 'as agreed or assessed' without a figure. Only 6 decisions in the "
     "whole corpus quantify legal costs. Treatment, travel and NSW Trustee funds-management costs "
     "are excluded."),
    ("legal_costs_disclosed", "19 legal costs", "bool", "derived", "",
     "True when legal_costs_amount is populated."),
    ("legal_costs_context", "19 legal costs", "text", "audit", "",
     "Matched text for the rule-side costs figure, for auditing."),

    ("primary_injury", "21-22 taxonomy", "text", "llm", "see 'enumerations'",
     "NO RULE BASELINE - reported as 'no_rule_baseline'. The keyword rule counts body-part words "
     "and so over-detects (medical history, prior injuries, cited authority); restricting it to "
     "catchwords+description did not move the 39/100 disagreement. A keyword count and a reading "
     "of the orders measure different things, so their disagreement is not a quality signal. "
     "Validate against the hand-labelling sheet 'validation_worksheet'. "
     "Principal injury by body system. 'multiple' only where two or more are equally central."),
    ("primary_injury_tier", "21-22 taxonomy", "text", "audit", "catchwords / description / body / none",
     "Which text tier decided the rule-side injury. Tiers are tried in order of reliability; the "
     "first that matches decides, so a long judgment cannot out-vote its own headnote."),
    ("primary_injury_evidence", "21-22 taxonomy", "text", "audit", "",
     "Rule-side keyword score vector."),
    ("mechanism", "21-22 taxonomy", "text", "llm", "see 'enumerations'",
     "NO RULE BASELINE - see primary_injury. Restricting the rule to catchwords+description made "
     "the disagreement worse (33->46/100), confirming the two methods measure different things. "
     "Validate against 'validation_worksheet'. "
     "How the injury came about. 'not_stated' is a real and common answer: where liability is "
     "admitted and only treatment or entitlement is in issue, the decision often never describes "
     "the mechanism."),
    ("mechanism_tier", "21-22 taxonomy", "text", "audit", "catchwords / description / body / none",
     "As primary_injury_tier."),
    ("mechanism_evidence", "21-22 taxonomy", "text", "audit", "", "Rule-side keyword score vector."),

    ("claimant_legal_representation", "process", "text", "llm", "Yes / No / Unknown",
     "Whether the WORKER was legally represented. 'No' means the decision indicates they were "
     "self-represented or appeared in person."),
    ("claimant_legal_representation_basis", "process", "text", "audit", "",
     "Which rule produced the rule-side value."),
    ("claimant_interpreter_used", "process", "text", "llm", "Yes / No / Unknown",
     "Whether the worker used an interpreter at any point. Interpreters are mentioned in 6.5% of "
     "WC decisions."),
    ("claimant_interpreter_basis", "process", "text", "audit", "",
     "Which rule produced the rule-side value."),
    ("medical_assessor_involved", "process", "text", "llm", "Yes / No / Unknown",
     "Whether a Medical Assessor or AMS is involved at ANY point, including a referral this "
     "decision makes. A referral yet to happen still counts."),
    ("medical_assessor_basis", "process", "text", "audit", "",
     "Which rule produced the rule-side value."),
    ("remitted_to_medical_assessor", "process", "text", "llm", "Yes / No / Unknown",
     "Whether THIS decision remits or refers the matter for medical assessment. The usual reason "
     "a permanent-impairment case carries no lump sum figure."),
    ("liability_posture", "process", "text", "llm",
     "liability_denied / quantum_or_entitlement_only / not_applicable_procedural / unclear",
     "Whether the respondent put injury or liability itself in issue, or admitted it and disputed "
     "only entitlement, treatment or amount. The field where the rule pass is least reliable and "
     "the LLM most clearly better."),
    ("liability_posture_basis", "process", "text", "audit", "",
     "Which rule produced the rule-side value."),
    # --- Second pass (--conduct-pass), versioned independently -------------
    ("denial_scope", "conduct", "text", "llm_second_pass",
     "semicolon-separated: primary_injury / consequential_condition / specific_treatment / "
     "impairment_head / weekly_entitlement / work_capacity / nothing_denied / not_stated",
     "Every head liability was denied for. This is what makes partial denial COUNTABLE: under the "
     "liability_posture binary, denying every consequential condition while accepting the injury "
     "reads as the cooperative 'quantum only' category."),
    ("denial_scope_evidence", "conduct", "text", "llm_second_pass", "",
     "Verbatim quote showing what was denied. Empty where nothing was denied."),
    ("heads_claimed", "conduct", "integer", "llm_second_pass", "0 = undeterminable",
     "How many distinct heads of relief the worker pressed for decision."),
    ("heads_succeeded", "conduct", "integer", "llm_second_pass", "<= heads_claimed",
     "How many of those heads the worker won. Cross-checked against heads_claimed; a violation "
     "sets conduct_status to 'heads_inconsistent' rather than publishing the value."),
    ("claimant_success_degree", "conduct", "float", "derived", "0.0-1.0, null if heads_claimed = 0",
     "heads_succeeded / heads_claimed. Turns the definitional question from a two-point "
     "comparison into a sensitivity curve: 'mixed' currently covers winning four heads of five "
     "and one of five alike."),
    ("conduct_finding", "conduct", "text", "llm_second_pass",
     "criticism_made / considered_not_criticised / not_addressed",
     "Whether the Member remarked on the INSURER's conduct of the claim. Three-way on purpose: "
     "silence and express approval are opposite evidence for convergent validity and identical "
     "under a boolean. Losing is not criticism -- an insurer can lose with impeccable conduct on "
     "fresh medical evidence, or win having handled the matter badly. MEASURED: criticism_made "
     "5.7%, considered_not_criticised 2.1%, not_addressed 92.1%. Claimant success runs 87.6% / "
     "62.7% / 71.6% across those three, so the arms fall on BOTH sides of baseline. "
     "ENDOGENEITY CAVEAT, which anyone using this as a validation target must state: the conduct "
     "remark and the outcome are recorded in the same document by the same person, so criticism "
     "is partly downstream of the finding rather than independent of it. The "
     "considered_not_criticised arm is a partial answer -- pure endogeneity would not place it "
     "BELOW the not_addressed baseline -- but it is a mitigation, not a refutation."),
    ("conduct_scope", "conduct", "text", "llm_second_pass",
     "semicolon-separated: reasons_adequacy / timeliness / investigation_proportionality / "
     "surveillance / failure_to_consider_evidence / defective_notice / other_conduct",
     "What was criticised, in SIRA Standards of Practice terms. Deliberately the regulator's "
     "vocabulary: it lets the metric be validated against the actual standard rather than against "
     "generalised judicial disapproval, and absorbs surveillance and defective notices as scope "
     "values rather than separate booleans."),
    ("conduct_evidence", "conduct", "text", "llm_second_pass", "",
     "Verbatim quote of the Member's remark on conduct."),
    ("relief_granted_degree", "conduct", "text", "llm_second_pass",
     "fully_granted / substantially_granted / partly_granted / minimally_granted / refused / "
     "not_determinable",
     "How much of what the worker PRESSED was granted, counting partial success within a head. "
     "Replaces the heads_succeeded/heads_claimed ratio, which measured the wrong granularity: "
     "64.4% of the corpus presses a single head, so the ratio could only be 0 or 1 and scored 60% "
     "of 'mixed' decisions at a perfect 1.0. This is the field that makes the definitional "
     "question a sensitivity curve rather than a two-point comparison."),
    ("relief_granted_evidence", "conduct", "text", "llm_second_pass", "",
     "Verbatim quote from the orders supporting the degree."),
    ("relief_status", "conduct", "text", "audit", "ok / not run / error text",
     "Whether the relief pass ran."),
    ("relief_agreement", "conduct", "text", "audit",
     "single_vote / unanimous / majority / no_majority / not_run",
     "How the delivered degree was arrived at. Re-voting is TARGETED at the middle three "
     "categories, where the sensitivity threshold slides, so an extreme carrying 'single_vote' is "
     "not less trustworthy than a re-voted middle case -- it was never in doubt."),
    ("relief_votes", "conduct", "text", "audit", "comma-separated",
     "Every vote cast, keyed internally by seed so the production vote and the pilot's first vote "
     "(which share a seed) cannot be counted twice."),
    ("relief_stability", "conduct", "text", "audit", "",
     "Records that this field is SINGLE-VOTE. Piloted at 89% unanimous over 3 votes on 100 cases, "
     "against 91% on the gated fields. Gating was declined deliberately: the field feeds a "
     "sensitivity curve, where a label moving one adjacent step blurs the line, rather than a "
     "cohort assignment, where it would relocate a case between analysis populations. Treat "
     "roughly 11% of values as soft."),
    ("conduct_status", "conduct", "text", "audit",
     "ok / not run / heads_inconsistent / error text",
     "Whether the second pass ran. Distinguishes 'not run' from 'ran and found nothing', for the "
     "same reason conduct_finding is three-way."),
    ("consequential_condition_claimed", "process", "text", "llm", "Yes / No / Unknown",
     "Whether the worker claimed a condition said to flow from the accepted primary injury."),
    ("fatality", "process", "text", "llm", "Yes / No / Unknown",
     "Whether the worker died, making this a death-benefit or dependency claim."),
    ("s11A_defence_raised", "process", "bool", "rule", "",
     "True when s 11A is mentioned. Mention only - it does not mean the defence was determined, "
     "or succeeded."),

    ("weekly_benefit_amount", "weekly benefit", "number", "llm", "",
     "Weekly compensation rate ORDERED; the highest rate where a stepped schedule is ordered."),
    ("weekly_benefit_rate_count", "weekly benefit", "int", "rule", "",
     "Distinct weekly rates found. >1 means a stepped schedule, so a single figure understates "
     "the award."),
    ("weekly_benefit_text", "weekly benefit", "text", "csv", "",
     "The pipeline's free-text weekly benefit description. ~22% populated."),

    ("member_name", "extras", "text", "rule", "",
     "Deciding Member or Arbitrator, parsed from the AustLII header label/value table."),
    ("member_role", "extras", "text", "rule",
     "Member / Principal Member / Senior Member / President / Deputy President / Arbitrator",
     "The decision-maker's role."),
    ("regulatory_sections", "extras", "text", "csv", "",
     "Statutory provisions the decision turns on. ~98% populated."),
    ("catchwords", "extras", "text", "csv", "",
     "AustLII headnote. The most reliable single summary of what was decided."),
    ("medical_costs_addressed", "extras", "text", "csv", "Yes / No / Not addressed",
     "Whether medical costs were dealt with."),
    ("injury_burden_intensity", "extras", "int", "csv", "0-4", "Pipeline ordinal."),
    ("causation_complexity", "extras", "int", "csv", "0-2", "Pipeline ordinal."),
    ("treatment_burden", "extras", "int", "csv", "0-3", "Pipeline ordinal."),
    ("needs_review", "extras", "text", "csv", "Yes / No",
     "The PIPELINE's review flag. Tuned for the CTP damages pass and never fires on WC rows - use "
     "needs_review_derived instead."),
    ("outcome_analysable", "7 outcome", "bool", "derived", "",
     "False for Procedural matters, which have no substantive winner and would pollute any "
     "outcome metric or conduct signal. Filter on this before computing win rates, or treat "
     "Procedural as its own stratum - it splits 50.6/46.8 claimant/insurer against 76.9/17.6 for "
     "everything else, so pooling them shifts the base rate."),
    ("needs_review_derived", "review", "text", "derived", "Yes / No",
     "Whether this row tripped any quality flag defined here. Replaces the pipeline flag, which "
     "cannot see an empty WPI in a permanent-impairment matter or a binary field where two "
     "independent extractors disagree."),
    ("review_flags", "review", "text", "derived", "semicolon-separated",
     "Named quality problems on this row: no_decision_text, suspiciously_short_text, llm_failed, "
     "disagreement:<field> (on the binary/factual watchlist only, not the taxonomies), "
     "permanent_impairment_without_wpi, lump_sum_flagged_without_amount, "
     "implausible_age, implausible_wpi, "
     "mixed_outcome_without_reason, wpi_recovered_by_llm_only."),

    ("claimant_age_basis", "10-13 claimant", "text", "rule",
     "csv / text_stated_age / text_year_of_birth / not_stated",
     "How the rule-side age was obtained. Age appears in only ~33% of decisions and a year of "
     "birth in ~3%, so ~38% is the realistic ceiling however it is extracted."),
    ("claimant_date_of_birth", "10-13 claimant", "text", "llm", "YYYY-MM-DD or YYYY",
     "The worker's date or year of birth where stated. Rarely present."),

    ("wpi_determined", "9 WPI", "number", "llm", "0-100",
     "Impairment percentage the DECISION ITSELF finds, accepts or records as assessed - a Medical "
     "Assessor's certified figure, an agreed figure, or the Member's own finding. Distinct from "
     "the contended figures, and from wpi_percent, which is the older pipeline pass. Recovers "
     "cases the pipeline left blank."),

    ("costs_order_direction", "19 legal costs", "text", "llm",
     "respondent_pays_applicant / applicant_pays_respondent / each_party_bears_own / "
     "no_order_as_to_costs / costs_reserved / costs_assessment_application / not_addressed",
     "WHO bears costs, independent of amount - the recoverable half of the costs question, stated "
     "far more often than any figure. NSW workers compensation costs are REGULATED and do not "
     "follow the event: a worker who loses is not generally exposed to the insurer's costs, and "
     "much of a worker's legal work is funded through IRO/ILARS rather than by a costs order. Do "
     "not read this as a civil loser-pays outcome."),
    ("costs_order_direction_evidence_rule", "19 legal costs", "text", "audit", "",
     "Matched text for the rule-side costs direction."),
    ("costs_complexity_uplift_percent", "19 legal costs", "number", "llm", "0-100",
     "Percentage uplift for complexity ordered on costs, where one was ordered."),
    ("exempt_worker", "19 legal costs", "text", "llm", "Yes / No / Unknown",
     "Decided on the worker's ROLE, not the employer: NSW Police employs civilian radio "
     "operators, and matching employer names alone gave a 15% false-positive rate. 'Unknown' "
     "where an exempt-heavy employer is named but the occupation is not stated. "
     "Whether the worker is an EXEMPT worker - police, paramedic, firefighter or coal miner - "
     "whose costs are insurer-paid in most cases under the Workers Compensation Regulation 2016. "
     "A different costs regime applies, so this conditions any costs analysis."),
    ("iro_ilars_funding_mentioned", "19 legal costs", "text", "llm", "Yes / No / Unknown",
     "Whether IRO/ILARS funding of the worker's legal costs is mentioned. Relevant because ILARS "
     "funds much worker-side legal work that never appears as a costs order."),

    ("proceeding_posture", "process", "text", "llm",
     "first_instance / reconsideration / related_earlier_proceedings / not_stated",
     "Whether this decision is a fresh dispute or connected to an earlier determination on the "
     "same claim. Repeat matters on one claim are NOT independent observations - cluster or drop "
     "them before modelling. Deliberately keyed to explicit markers ('Certificate of "
     "Determination dated', 'previously determined', reconsideration applications) rather than "
     "the word 'appeal', which appears in 31% of decisions purely as cited authority."),
    ("dispute_notice_date", "process", "date", "llm", "ISO date",
     "Date of the insurer's s 74 / s 78 dispute notice. A date sits within reach of the section "
     "reference in ~44% of decisions, so treat this as available-for-a-subset, not a core field."),
    ("notice_to_decision_days", "process", "int", "derived", "",
     "Days from the dispute notice to the decision - dispute duration, as distinct from "
     "injury_to_decision_days. Only populated where dispute_notice_date is."),
]

FALLBACK_PRINCIPLE = (
    "A fallback tier earns its place only if its precision exceeds the cost of contamination, "
    "and for any field feeding attribution or fairness that bar is very high. This dataset has "
    "now failed that test three times, in three different ways:\n"
    "  * insurer_name - a frequency count over full text named NRMA as the insurer in ~266 "
    "decisions that merely cited Diab v NRMA Ltd. Fixed by stripping citation context, not by "
    "raising the count threshold.\n"
    "  * claimant_age - a text rule returned a worker's age when hired decades earlier, the "
    "deceased's son's age, and a doctor's generic 'a 49 year old'. Wrong more often than right, "
    "so the fallback was REMOVED and coverage dropped to ~30%.\n"
    "  * primary_injury / mechanism - a keyword count over full text over-detects, matching body "
    "parts in medical history, prior injuries and cited authority. Restricting it to "
    "catchwords+description did not help, so it was demoted to no-baseline.\n"
    "The asymmetry is the whole argument: a MISSING value is honest and removes a row, while a "
    "WRONG value is silently delivered as fact and corrupts everything computed from it. Prefer "
    "returning nothing."
)


DICTIONARY_NOTES = [
    ("Scope", "One row per unique workers compensation decision: 2,385 of the 3,501 extracted "
              "decisions (CTP accounts for the other 1,116)."),
    ("De-duplication", "nsw_pic_decisions/ holds ~6,639 files for ~3,503 decisions. A filename-"
                       "convention change added a case-id suffix, and the scraper's on-disk cache "
                       "check predicts the new name, so pre-existing files were invisible, "
                       "re-fetched and saved again. Differences between copies are AustLII page "
                       "chrome, not judgment text. This workbook keys on URL, so nothing is "
                       "double counted; duplicate_html_files records the redundancy."),
    ("Blank means unknown", "A blank cell means the decision did not state the fact, never zero "
                            "and never 'nil'. Do not impute. Where a field has a paired "
                            "*_source / *_provenance column, that column says WHY it is blank."),
    ("Two independent methods", "Semantically hard fields are decided by an LLM structured pass "
                                "and cross-checked by a text rule. The rule value is kept as "
                                "<field>_rule and <field>_agreement compares them. Rows where "
                                "they differ are the ones worth reading by hand."),
    ("Damages layer not run", "damages_extraction.py only ever ran on CTP rows, so on WC rows the "
                              "pipeline's damages, provenance and injury-taxonomy columns are "
                              "empty. Insurer, injury, mechanism, lump sum, weekly benefit and "
                              "legal costs are therefore derived here rather than copied."),
    ("Decision text", "Not embedded: 67% of decisions exceed Excel's 32,767-character cell limit. "
                      "Use source_html_file to open the full text."),
    ("LLM variance - MEASURED",
     "Two independent runs over the same 100 cases, identical prompt and identical pinned seed, "
     "gave these per-field stabilities: costs_order_direction, fatality and lump_sum_amount "
     "100%; proceeding_posture, medical_assessor_involved and insurer_name 99%; outcome, "
     "liability_posture, claimant_age, wpi_determined and dispute_notice_date 98%; mechanism "
     "95%; primary_injury and legal_complexity 91%. "
     "primary_injury and legal_complexity are therefore subject to a STABILITY GATE (see "
     "*_stability columns): each is extracted twice, and where the two disagree a third vote "
     "breaks the tie by majority. "
     "Note also that CHANGING THE SCHEMA shifts unrelated fields - editing one field's "
     "description alters the whole schema sent to the model. Apparent drift in outcome across "
     "earlier runs (11-12-11 mixed) was schema-change sensitivity, not sampling variance: pure "
     "run-to-run variance on outcome is 2 cases in 100, and both were partial-success cases, "
     "i.e. the definitional boundary itself. Results are cached in wc_llm_cache.json and pinned "
     "once written, so a completed run is stable."),
    ("Ordinal direction", "work_impact_severity is higher = worse; ability_to_work is its "
                          "complement, higher = better. Use one, not both."),
    ("FALLBACK PRINCIPLE - when a rule earns its place", FALLBACK_PRINCIPLE),
    ("The claimant_age pattern - the template for a distrusted rule",
     "claimant_age is the model to copy when a rule is informative but not trustworthy enough to "
     "deliver: the rule's value is RETAINED in claimant_age_rule as evidence and for audit, but "
     "it is never allowed to populate the field itself, which is LLM-only at ~29/100 - close to "
     "the honest ceiling of ~38%. Retain the rule as evidence; do not let it populate. This is "
     "strictly better than either deleting the rule (losing the audit trail) or letting it fill "
     "gaps (delivering wrong values as fact)."),
    ("Structurally absent vs extraction failure",
     "Some columns are almost entirely empty because the DECISIONS do not record the fact, not "
     "because extraction failed. Confirmed structurally absent: legal_costs_amount (only 6 of "
     "2,385 decisions quantify legal costs; costs are ordered 'as agreed or assessed'), "
     "claimant_date_of_birth (a year of birth appears in ~3%), iro_ilars_funding_mentioned "
     "(ILARS funding is essentially invisible in decision text), and "
     "costs_complexity_uplift_percent (~2%). Do not treat these as bugs, and do not expect a "
     "better extractor to fill them. By contrast an empty wpi_determined_rule IS a rule "
     "limitation - the LLM recovers figures the rule does not."),
    ("SELECTION BIAS - read first",
     "Every row here is a DISPUTED matter that was determined and published. It is not a sample "
     "of workers compensation claims. Accepted claims never appear; settled and discontinued "
     "matters never appear; and Medical Assessment Certificates are not published as a public "
     "registry, so a WPI figure surfaces only when a published decision reproduces or challenges "
     "one. Counts of WPI assessments, lump sums and PIC involvement in this dataset therefore "
     "UNDERSTATE the scheme, and rates computed here are rates among published disputes. See the "
     "'selection_funnel' sheet for the stage-by-stage filter, the direction each biases, and "
     "whether it can be bounded."),
]


def enumeration_rows():
    """Controlled vocabularies, read off the schema so they cannot drift."""
    rows = []
    for label, enum_class in (("primary_injury", WCInjuryEnum),
                              ("mechanism", WCMechanismEnum),
                              ("outcome", WCOutcomeEnum),
                              ("liability_posture", LiabilityPostureEnum),
                              ("lump_sum_type", LumpSumTypeEnum),
                              ("Yes/No fields", YesNoEnum)):
        for member in enum_class:
            rows.append((label, member.value))
    for canonical, _pattern in INSURERS:
        rows.append(("insurer_name", canonical))
    return rows


# Every filter between "a worker is injured in NSW" and "a row in this
# workbook". Counts are filled in from the data ONLY where this dataset can
# see the stage; the upstream stages are unobservable from published decisions
# and are marked as such rather than guessed at.
SELECTION_STAGES = [
    dict(
        stage="1. Injury occurs",
        population="Workers injured in NSW employment",
        filter_applied="—",
        removed="Nothing yet",
        bias_direction="—",
        boundable="Scheme-wide injury counts are published by SIRA, so the base can be "
                  "bounded externally, though not from this dataset.",
    ),
    dict(
        stage="2. Injury becomes a claim",
        population="Workers who lodge a workers compensation claim",
        filter_applied="Worker reports the injury and claims",
        removed="Unreported and unclaimed injuries",
        bias_direction="Removes minor injuries, short absences, and workers least willing or "
                       "able to claim (insecure work, language barriers, fear of reprisal). "
                       "Biases the surviving population toward more severe and better-supported "
                       "claims.",
        boundable="Not from this dataset. Partially boundable against SIRA scheme statistics.",
    ),
    dict(
        stage="3. Claim becomes disputed",
        population="Claims the insurer declines or disputes",
        filter_applied="Insurer issues a s 74 / s 78 dispute notice",
        removed="Accepted claims — the large majority",
        bias_direction="THE DOMINANT FILTER. Everything in this dataset is a contested matter, "
                       "so the corpus describes disputes, not claims. Any rate computed here "
                       "(lump sum, WPI, outcome) is a rate among disputes and will not "
                       "generalise to the scheme.",
        boundable="Direction is certain, magnitude is not. Scheme dispute rates would bound it.",
    ),
    dict(
        stage="4. Dispute reaches the Commission",
        population="Disputes lodged with PIC",
        filter_applied="Worker lodges an Application to Resolve a Dispute",
        removed="Disputes resolved by internal review, negotiation, or abandonment",
        bias_direction="Removes disputes the worker could not fund or chose not to pursue. "
                       "IRO/ILARS funding eligibility acts as a gate, so unfunded and "
                       "ineligible workers drop out disproportionately.",
        boundable="NOT BOUNDABLE from this dataset, and measured here as a negative result: "
                  "iro_ilars_funding_mentioned is uniformly 'No' because ILARS funding is "
                  "invisible in decision text. The funding-eligibility gate at this stage can "
                  "therefore be described but never measured from published decisions. IRO "
                  "publishes aggregate ILARS grant volumes, which would bound it externally.",
    ),
    dict(
        stage="5. Lodgement survives to determination",
        population="Matters determined by a Member",
        filter_applied="Not settled, discontinued, or resolved by complying agreement",
        removed="Settlements, discontinuances and consent resolutions",
        bias_direction="Removes matters both sides thought predictable — the clearer cases "
                       "settle. What remains is enriched for genuine legal or factual "
                       "difficulty, contested causation, and disputed impairment. This inflates "
                       "apparent complexity and depresses apparent claimant win rates relative "
                       "to all disputes.",
        boundable="Settlement terms are confidential, so not boundable from public sources. The "
                  "OBSERVED consequence is in the outcome base rate: on the full corpus the "
                  "worker succeeds in 75.2% of determined matters (76.9% excluding Procedural), "
                  "the insurer in 19.5%. 'Insurer loses' is the MAJORITY class, not a rare "
                  "event, and the surviving matters are enriched for cases the insurer arguably "
                  "should have conceded.",
    ),
    dict(
        stage="6. Determination is published",
        population="Published decisions",
        filter_applied="PIC publishes the Certificate of Determination and reasons",
        removed="Unpublished and de-identified determinations; Medical Assessment "
                "Certificates, which are NOT published as a public registry",
        bias_direction="MACs being unpublished is why impairment appears rare here: a WPI "
                       "figure surfaces only when reproduced or challenged in a published "
                       "decision. The dataset therefore UNDERSTATES medical assessments and "
                       "permanent-impairment claims, and is enriched for matters that were "
                       "contested or appealed.",
        boundable="Partially. Consent determinations are published inconsistently.",
    ),
    dict(
        stage="7. Published decision is captured",
        population="Decisions scraped from AustLII",
        filter_applied="AustLII coverage and this scraper's run",
        removed="Decisions outside the crawl, or not on AustLII",
        bias_direction="Believed close to neutral within the covered period, but coverage is "
                       "not verified against the PIC decisions register.",
        boundable="Yes — compare counts per year against the PIC register.",
    ),
    dict(
        stage="8. Capture survives extraction",
        population="Rows in this workbook",
        filter_applied="Text extracted and the LLM pass succeeded",
        removed="Files that failed to parse or extract",
        bias_direction="Small and close to neutral; failures are encoding faults, not case "
                       "characteristics.",
        boundable="Yes — counted directly, see the counts column.",
    ),
    dict(
        stage="9. Row has the field you need",
        population="Rows usable for a given analysis",
        filter_applied="Per-field missingness",
        removed="Rows where the decision never stated the fact",
        bias_direction="Field-specific and often NOT random. COHORT ASSIGNMENT IS ITSELF A "
                       "FILTER: primary_injury is 'multiple' or 'not_stated' on ~23% of cases, "
                       "so any injury-cohort analysis silently runs on the ~77% that can be "
                       "assigned - state that before computing cohort rates. "
                       "Age appears in ~38% of decisions "
                       "and is stated more often where it matters to the reasoning. Insurer is "
                       "nameable in ~50% because the employer, not the insurer, is respondent. "
                       "Complete-case analysis on these fields is a further, silent selection.",
        boundable="Yes — coverage is measurable per field; see the 'fields' sheet.",
    ),
]


def build_selection_funnel(corpus=None, extract=None):
    """Selection diagram from injury to row, with observed counts where the
    data can see the stage and an explicit blank where it cannot."""
    rows = []
    for stage in SELECTION_STAGES:
        record = dict(stage)
        record["observed_count"] = ""
        rows.append(record)

    def annotate(prefix, text):
        for record in rows:
            if record["stage"].startswith(prefix):
                record["observed_count"] = text

    annotate("7.", "6,639 HTML files on disk covering 3,503 unique citations "
                   "(the surplus is a filename-convention duplication, not extra cases)")
    if corpus is not None:
        workers_comp = corpus[corpus["Case Type"].astype(str).str.strip() == "Workers Compensation"]
        annotate("8.", f"{len(corpus)} decisions extracted, of which "
                       f"{len(workers_comp)} workers compensation "
                       f"(2 of 3,503 citations failed extraction)")
    if extract is not None and len(extract):
        coverage = []
        for column, label in (("wpi_percent", "accepted WPI"),
                              ("claimant_age", "age"),
                              ("insurer_name", "insurer"),
                              ("lump_sum_amount", "lump sum"),
                              ("legal_costs_amount", "legal costs")):
            if column in extract.columns:
                populated = extract[column].replace("", pd.NA).notna().sum()
                coverage.append(f"{label} {populated}/{len(extract)}")
        annotate("9.", "; ".join(coverage))
    return pd.DataFrame(rows, columns=[
        "stage", "population", "filter_applied", "removed", "bias_direction",
        "boundable", "observed_count"])



# The failure worth teaching from: not random noise, but error correlated with
# the very variable a student would analyse against.
WORKED_EXAMPLES = [
    dict(
        example="The Diab contamination",
        what_happened="insurer_name was derived by counting insurer-name mentions in the "
                      "decision text and taking the most frequent. In ~266 workers compensation "
                      "decisions this returned 'Insurance Australia (IAG/NRMA)'.",
        why_it_was_wrong="Every one of those decisions was citing Diab v NRMA Ltd [2014] "
                         "NSWWCCPD 72, the leading authority on whether treatment is reasonably "
                         "necessary under s 60. NRMA was not the insurer. It was a case name.",
        why_it_matters="The error was NOT random - it was correlated with dispute type. Diab is "
                       "cited in treatment-necessity matters, so NRMA would have appeared to "
                       "specialise in medical disputes and to win or lose them at whatever rate "
                       "medical disputes do. A per-insurer analysis would have produced a "
                       "plausible, internally consistent, entirely fabricated profile - and it "
                       "would have survived a sanity check, because nothing about it looks odd.",
        how_it_was_caught="Two independent extractors disagreed, and the disagreement was "
                          "logged rather than silently resolved. Reading the matched text showed "
                          "the citation.",
        the_fix="Strip mentions inside a cited case name or neutral citation, then accept a "
                "single surviving mention. A frequency threshold would NOT have fixed it: "
                "decisions citing Diab twice would still pass, and legitimate single mentions in "
                "the procedural history would be discarded.",
        lesson="Correlated measurement error is more dangerous than noise. Noise attenuates an "
               "estimate; correlated error invents a finding. Provenance discipline - recording "
               "WHERE a value came from, and being able to read the evidence back - is what "
               "makes this class of error findable at all.",
    ),
    dict(
        example="'The insurer loses' is a definition, not an observation",
        what_happened="Over the full corpus the rule-side outcome classifier puts the worker's "
                      "success rate at 75.2% and the LLM pass at 72.3%, with the LLM finding "
                      "roughly twice as much 'mixed' (10.7% vs 5.2%). The two disagree on 259 "
                      "cases, 10.9% of the corpus - and 'mixed' appears on one side or the other "
                      "in 92.7% of them.",
        why_it_was_wrong="Neither is wrong. They disagree about whether a partial success - the "
                         "worker wins on some heads and loses on others - counts as a win. The "
                         "disagreement set and the partial-success population are effectively the "
                         "same object.",
        why_it_matters="Counting 'mixed' as a win rather than a loss moves the headline base rate "
                       "by 10.7 points, and - see the definitional_metric_shift sheet - it does "
                       "not move it evenly: 23.3 points on Permanent Impairment against 4.7 on "
                       "Death Benefit. So the definition does not merely shift a number, it "
                       "REDISTRIBUTES penalty between providers according to their book. A "
                       "participant heavy in impairment matters is both disadvantaged by the base "
                       "rate and most exposed to the definitional choice; the two compound. "
                       "Whoever sets the definition is making a distributive decision wearing "
                       "technical clothes, and a scheme participant will contest it long before "
                       "it contests any model built on top of it.",
        how_it_was_caught="Two classifiers with different granularity were run over the same "
                          "cases and their disagreement reported rather than averaged away.",
        the_fix="There is no fix - there is a decision. Define 'loss' explicitly, state it with "
                "the metric, and report sensitivity to the alternative definition.",
        lesson="Definitional risk is quantifiable and belongs in the metric design, not in a "
               "footnote. Report the base rate under both definitions.",
        footnote="The earlier version of this example quoted 66% and a nine-point swing, taken "
                 "from the 100-case sample. Re-derived over the full corpus the figures are "
                 "72.3% and 10.7 points. The sample was stratified on gender and WPI presence "
                 "and was faithful on both, but drifted on dispute type, which nobody had "
                 "stratified on - see 'The sample that was representative of the wrong things'.",
    ),
    dict(
        example="Partial denial: when a category disagreement is a category error",
        what_happened="liability_posture was extracted twice, by a text rule and by the LLM, into "
                      "a binary: was liability itself in issue, or was it admitted and only "
                      "entitlement disputed? The two methods disagreed on 711 of 2,385 cases "
                      "(29.8%), and the disagreement was directional 4:1 - 494 cases where the "
                      "LLM read denial and the rule read quantum-only, against 129 the other way.",
        why_it_was_wrong="Neither extractor was misreading. In the dominant cell a consequential "
                         "condition is claimed in 47.8% of cases against a 25.6% base rate, and "
                         "the rule is quoting explicit concession language in 406 of the 494. "
                         "Both are reading real words on the page: liability WAS admitted for the "
                         "primary injury, and it WAS denied for the consequential condition, the "
                         "treatment, or the impairment head. The category had no value for that, "
                         "so each method rounded to a different side.",
        why_it_matters="This is a gaming channel, not just a data-quality defect, and it is now "
                       "MEASURED rather than inferred. Extracting what was denied (denial_scope) "
                       "shows that of the 748 cases the binary reads as 'quantum only' - the "
                       "cooperative category - 51.3% carry a recorded denial of a consequential "
                       "condition, a treatment or an impairment head, and 49.1% denied something "
                       "without denying the primary injury. Corpus-wide the pattern covers 828 "
                       "cases, 34.7%. Deny in part, never in whole, and a conduct metric built on "
                       "this field cannot see it: half of what it scores as cooperative contains "
                       "a denial. The field sits close to the metric's target concept, so the "
                       "error propagates rather than washing out.",
        how_it_was_caught="The cross-check was nearly suppressed: the marginals almost match "
                          "(denied 46 v 49, quantum 47 v 40, procedural 7 v 11), so the field "
                          "looked fine at the aggregate. Similar marginals with case-level "
                          "disagreement is exactly the signature worth investigating, and the "
                          "cross-tab - not the marginals - showed the direction.",
        the_fix="Do not adjudicate a winner. Draw a reference set FROM the disagreement cells "
                "(25/12/13 across the three, deliberately non-proportional), offer the labeller "
                "the third value the schema cannot return, and cross the dominant cell on the "
                "hypothesised mechanism and on rule strength so 'the category is wrong' and 'the "
                "rule is weak' can be told apart. If it holds, liability_denied_in_part becomes a "
                "third value.",
        footnote="A hypothesis died here and it is worth keeping the corpse. The prediction was "
                 "that partial denial would be CONCENTRATED in the 494-case disagreement cell, "
                 "and it is not: inside that cell primary injury is denied 59.1% of the time - "
                 "the LLM's reading backed against the rule's, so rule weakness - and the "
                 "partial-not-primary signature is 36.6% against a 32.0% baseline on rows where "
                 "the methods AGREE. Partial denial is a property of the category corpus-wide, "
                 "not of where the two methods disagree. The finding survived losing its "
                 "mechanism, and got larger: the channel does not need the disagreement to "
                 "exist.",
        lesson="When two competent methods disagree about a category and both can quote the text, "
               "suspect the category before you suspect the methods. A binary that cannot express "
               "a real state of the world does not merely lose information - it creates a channel "
               "for anyone who benefits from occupying that state.",
    ),
    dict(
        example="The sample that was representative of the wrong things",
        what_happened="A 100-case sample, drawn stratified on claimant gender and WPI presence, "
                      "put liability_denied at 49%. The full corpus put it at 63%. Every figure "
                      "quoted from that sample - including a worked example in this very sheet - "
                      "inherited the gap.",
        why_it_was_wrong="The stratification did its job: sample WPI presence 5.0% against a "
                         "corpus 4.9%, male 59% against 60%. But nothing was stratified on "
                         "dispute type, and dispute type drifted - Procedural 11% against 6.5%, "
                         "Liability Dispute 24% against 28% - across categories whose denial "
                         "rates span 8.3% to 97.8%. Re-weighting the corpus to the sample's "
                         "dispute mix recovers 3 of the 14 points; the rest is an unlucky draw at "
                         "roughly 2.2 standard errors.",
        why_it_matters="A sample stratified on two variables is UNSTRATIFIED on every other one, "
                       "and the ones that bite are whichever happen to correlate with the "
                       "outcome. Nothing about the sample looked wrong: the variables anyone had "
                       "thought to check reproduced the corpus almost exactly.",
        how_it_was_caught="The same figure was computed over the full corpus and did not match. "
                          "The sample's own cases were then re-scored in the full run and gave "
                          "49% again, which ruled out run-to-run instability and left composition "
                          "and luck.",
        the_fix="Re-derive every sample-quoted figure from the full corpus once the full run "
                "exists. Keep hand-labelled samples for what only a human can supply - a label - "
                "and never for a base rate.",
        lesson="Stratification buys representativeness on the variables you named and nothing "
               "else. Check the marginals that matter for the CONCLUSION, not the ones you "
               "happened to stratify on.",
    ),
    dict(
        example="Half the budget went to two fields",
        what_happened="The full run cost $115.37 over 5,044 calls for 2,385 decisions. Only 2,385 "
                      "of those calls were the extraction. The other 2,659 were the stability "
                      "gate: a best-of-three vote on primary_injury and legal_complexity, the two "
                      "fields measured at 91% run-to-run stability.",
        why_it_was_wrong="Nothing was wrong with the gate. What was wrong is that it was a "
                         "default rather than a decision. Each vote re-sends the ENTIRE decision "
                         "text - mean 50.7k characters - to re-answer two fields, so the gate "
                         "pays full input price for a fraction of the schema and accounts for "
                         "roughly half the run.",
        why_it_matters="Input tokens are about 69% of this workload's cost, so architecture "
                       "choices that re-send the source dominate the bill, and they are invisible "
                       "in a per-call average. 'Stabilise the two shakiest fields' sounds free and "
                       "costs $50. The same reasoning decided the conduct fields: an additive "
                       "second pass keyed separately from the main schema costs about $42, where "
                       "re-running everything to add them would have cost $115 - and would have "
                       "moved every figure already published in this workbook.",
        how_it_was_caught="Dividing the run's call count by the corpus size gave 2.11 passes' "
                          "worth of text for one pass of output, which does not reconcile until "
                          "you notice the gate is re-sending everything.",
        the_fix="Price the architecture, not just the model. Ask what each mechanism re-sends, "
                "how often, and for how much of the schema; then decide whether the quality it "
                "buys is worth that, and record the decision either way.",
        lesson="Cost lives in the architecture as much as the price list. Its companion lesson - "
               "estimate token cost from the corpus LENGTH DISTRIBUTION, never from a "
               "small-sample per-case average - is about getting the unit price right; this one "
               "is about knowing how many units your design quietly ordered.",
    ),
    dict(
        example="The field that was specified at the wrong granularity",
        what_happened="claimant_success_degree was defined as heads_succeeded / heads_claimed, to "
                      "turn the definitional question from a two-point comparison into a "
                      "sensitivity curve. Extracted over the full corpus at a cost of $46, it "
                      "came back flat: 64.4% of decisions press a SINGLE head, so the ratio can "
                      "only be 0 or 1, and 154 of the 255 'mixed' decisions scored 1.0.",
        why_it_was_wrong="The specification assumed 'mixed' means winning some heads and losing "
                         "others. In workers compensation it usually means partial success WITHIN "
                         "one head - a reduced period of weekly payments, some treatment approved "
                         "and some refused. The field measured a real quantity accurately; it was "
                         "the wrong quantity.",
        why_it_matters="Nothing in the schema, the prompt or the validation would have caught "
                       "this. The field is well defined, the model answered it correctly, and "
                       "every value is defensible. It fails only against its PURPOSE, and that "
                       "failure is invisible until you look at the distribution - where it is "
                       "obvious at a glance, and would have been obvious in 100 cases for about "
                       "$2 rather than 2,385 for $46.",
        how_it_was_caught="Cross-tabulating the new field against the outcome it was meant to "
                          "refine. A field that disagrees with the thing it is supposed to be a "
                          "finer-grained version of is either wrong or measuring something else.",
        the_fix="Pilot the DISTRIBUTION of a derived field before buying the full pass, and pilot "
                "it with repeated votes where the field is a judgment rather than a fact. The "
                "replacement here - an ordinal for within-head partiality - carries the larger "
                "risk, because degree judgments are where run-to-run variance concentrates and an "
                "unstable ordinal yields a sensitivity curve made of noise.",
        lesson="Specification errors are invisible in the schema and obvious in the histogram. "
               "Validation asks whether the field is right; only the distribution asks whether it "
               "is USEFUL, and the two come apart exactly when a field is derived rather than "
               "observed.",
    ),
]


# Fields that carry weight and now have no rule cross-check, so a human
# reference set is the only validation available. liability_posture is here
# too: it HAS a cross-check, but it feeds the metric directly, so an
# independent reference set is worth having regardless.
REFERENCE_SET_FIELDS = [
    ("primary_injury", "Body system the ORDERS address. 'multiple' only if two are equally "
                       "central; 'not_stated' if the decision never identifies an injury."),
    ("mechanism", "How the injury happened, per the decision. 'not_stated' is correct and common "
                  "where liability is admitted and only treatment is in issue."),
    ("legal_complexity", "0 = single issue, settled principle, no oral evidence. 2 = s 11A on "
                         "merits, worker-status contest, 3+ parties or s 151Z, jurisdiction/"
                         "limitation/estoppel, reconsideration, unsettled construction, or credit "
                         "findings. 1 = otherwise."),
    ("liability_posture", "Was injury/liability itself in issue, or was it admitted and only "
                          "entitlement/treatment/amount disputed? Answer liability_denied_in_part "
                          "where liability was admitted for the primary injury but denied for a "
                          "consequential condition, a particular treatment, or an impairment "
                          "head. That third value is THE HYPOTHESIS UNDER TEST: neither extractor "
                          "could return it, so do not force such a case into either binary "
                          "answer, and do not shy off it because it is not in the schema."),
]


# Per-field overrides on --reference-size.
#
# liability_posture carries 80 rather than 50, but NOT to power a localisation
# test. Whether partial denial operates differently by dispute type is already
# answered at corpus scale -- the outcome-lift table is n=711 -- and that
# evidence does not need labels. What the labels answer is the prior question:
# is liability_denied_in_part a real category that a human can apply
# consistently? 40 dominant-cell cases answer that well, and they are the
# precondition for everything else. Once in_part is in the schema and extracted
# across all 2,385, localisation re-runs at full scale for free, against the
# actual label instead of the disagreement proxy the lift table uses now.
#
# So 80 buys consistency-checking room and coverage of every dispute type,
# not statistical power. It should not grow further: a bigger sheet would be
# buying power for a question this sample is not the right instrument for.
REFERENCE_SET_SIZES = {"liability_posture": 80}

# Allocation for the liability_posture adjudication draw. Deliberately NOT
# proportional to the disagreement cells: proportional would spend 35 of 50 on
# the dominant cell and leave the reverse cell with 9, which cannot tell 'rare'
# from 'absent'.
LIABILITY_ADJUDICATION_SHARES = (
    ("llm_denied__rule_quantum", 0.50),
    ("llm_quantum__rule_denied", 0.24),
    ("procedural_either_side", 0.26),
)

# Rule bases that quote actual concession language, as against inferring the
# posture from the Nature field alone. The two carry very different weight when
# the rule contradicts the LLM.
CONCESSION_BASES = {"liability_expressly_accepted", "both_signals_concession_first",
                    "explicit_concession"}


def dispute_group(nature):
    """Collapse nature_of_case to the groups the partial-denial mechanism
    distinguishes.

    Liability, impairment and medical are kept apart because the outcome lift
    given a posture disagreement differs sharply across them (x2.26, x1.15,
    x1.11). Everything else is pooled: those types are either too small to
    stratify on or show no lift worth separating.
    """
    nature = str(nature).strip().lower()
    if "liability" in nature:
        return "liability"
    if "impairment" in nature:
        return "impairment"
    if "medical" in nature:
        return "medical"
    return "other"


def liability_disagreement_cell(llm, rule):
    """Which rule-vs-LLM disagreement cell a row sits in, or None if they agree."""
    llm, rule = str(llm).strip(), str(rule).strip()
    if llm == rule:
        return None
    if llm == "liability_denied" and rule == "quantum_or_entitlement_only":
        return "llm_denied__rule_quantum"
    if llm == "quantum_or_entitlement_only" and rule == "liability_denied":
        return "llm_quantum__rule_denied"
    if "not_applicable_procedural" in (llm, rule):
        return "procedural_either_side"
    return "other"


def build_liability_adjudication_sample(extract, size=50, seed=20260815):
    """Draw liability_posture cases from the DISAGREEMENT cells, not the label.

    Stratifying on the model's own label (what every other reference set does)
    is right when the only question is 'how often is the model wrong'. It is the
    wrong design here, because the cross-tab already answers a different
    question. The disagreement is directional 4:1 -- 494 cases where the LLM
    reads denial and the rule reads quantum-only, against 129 the other way --
    and a consequential condition is claimed in 47.8% of that dominant cell
    against a 25.6% base rate. Both readers are quoting real language.

    On the figure: the methods disagree on 711 of 2,385 cases, 29.8%. Commit
    003a899, which is otherwise the record of why this cross-check was
    reinstated, says 21% -- that was measured before the full run and is wrong.
    It matters because 21% reads like a judgement call about a marginal field,
    and 29.8% is the largest genuine method discrepancy in the workbook.

    That is the signature of PARTIAL denial: liability admitted for the primary
    injury, denied for the consequential condition, the treatment, or the
    impairment head. So this sample's job is not to pick a winner between the
    methods; it is to test whether the category needs a third value.

    The dominant cell is therefore crossed on two axes:
      - dispute type, because that is where the MECHANISM varies. Conditioning
        the outcome on the posture disagreement gives a lift of x2.26 in
        liability disputes but only x1.15 in permanent impairment: where denial
        should be explicit, a posture disagreement more than doubles the mixed
        rate, which reads as genuine partial liability producing partial
        outcome; where disagreement is commonest it barely moves the outcome,
        which reads as category-and-rule failure instead. Two mechanisms, and
        they separate on dispute type.
      - whether the rule had explicit concession language or only the Nature
        field (88 of the 494 are nature_field alone, which is a weak-rule
        failure rather than an under-specified category).

    consequential_condition_claimed was the first axis tried and is deliberately
    NOT stratified on now. It sits at 47.8% inside this cell, so a random draw
    represents it comfortably; it is recorded per case and can be tested after
    the fact. Liability Dispute is 25% of the cell and is the diagnostic
    stratum, and an unstratified draw gave it THREE cases out of 25 -- enough to
    load the sample with the dispute types whose answer is likely 'rule failure'
    and then read that back as a finding about the mechanism.

    The general rule, worth keeping: stratify the scarce diagnostic variable and
    let the near-balanced one fall out. A variable near 50/50 self-balances in
    any reasonable draw; a 25% variable that the conclusion turns on does not.

    What the dispute-type stratum buys here is COVERAGE, not power. Whether
    partial denial varies by dispute type is already evidenced at n=711; these
    labels answer whether the category is real and consistently applicable, and
    for that every dispute type needs to be present rather than numerous.
    """
    if extract is None or not len(extract):
        return pd.DataFrame()
    needed = {"liability_posture", "liability_posture_rule"}
    if not needed.issubset(extract.columns):
        return pd.DataFrame()

    frame = extract.copy()
    frame["_cell"] = [liability_disagreement_cell(a, b)
                      for a, b in zip(frame["liability_posture"],
                                      frame["liability_posture_rule"])]
    disagreements = frame[frame["_cell"].notna() & (frame["_cell"] != "other")]
    if not len(disagreements):
        return pd.DataFrame()

    # Largest-remainder allocation over the fixed shares, capped by what each
    # cell actually holds; anything that cannot be placed falls through to the
    # cells that still have rows.
    available = {name: int((disagreements["_cell"] == name).sum())
                 for name, _share in LIABILITY_ADJUDICATION_SHARES}
    size = min(size, len(disagreements))
    exact = {name: size * share for name, share in LIABILITY_ADJUDICATION_SHARES}
    allocation = {name: min(int(exact[name]), available[name]) for name in exact}
    remainders = sorted(exact, key=lambda name: exact[name] - int(exact[name]), reverse=True)
    while sum(allocation.values()) < size:
        progressed = False
        for name in remainders:
            if sum(allocation.values()) >= size:
                break
            if allocation[name] < available[name]:
                allocation[name] += 1
                progressed = True
        if not progressed:
            break

    chunks = []
    for name, take in allocation.items():
        if take <= 0:
            continue
        cell = disagreements[disagreements["_cell"] == name]
        if name == "llm_denied__rule_quantum":
            nature = cell.get("nature_of_case", pd.Series("", index=cell.index)).astype(str)
            basis = cell.get("liability_posture_basis",
                             pd.Series("", index=cell.index)).astype(str)
            cell = cell.assign(_substratum=[
                f"{dispute_group(n)}|{'concession' if b in CONCESSION_BASES else 'nature_field_only'}"
                for n, b in zip(nature, basis)])
            chunk, _ = stratified_sample(cell, "_substratum", take, seed)
            chunk = chunk.drop(columns=["_substratum"])
        else:
            chunk = cell.sample(n=min(take, len(cell)), random_state=seed)
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks).sample(frac=1, random_state=seed)


def build_reference_worksheets(extract, size=50, seed=20260815, prefill=False):
    """One blank hand-labelling worksheet per field needing a human reference set.

    Returns {field: DataFrame}. Each is stratified on the model's own label so
    rare categories appear, and carries the evidence a labeller needs — the
    catchwords state what was decided — plus the source filename for the full
    text.

    `prefill` is off by default ON PURPOSE. Pre-filling the model's answer into
    the human column would anchor the labeller and inflate measured accuracy;
    the whole point of the reference set is independence.
    """
    worksheets = {}
    if extract is None or not len(extract):
        return worksheets
    for field, guidance in REFERENCE_SET_FIELDS:
        if field not in extract.columns:
            continue
        take = min(REFERENCE_SET_SIZES.get(field, size), len(extract))
        adjudication = field == "liability_posture"
        if adjudication:
            sample = build_liability_adjudication_sample(extract, size=take, seed=seed)
            if not len(sample):
                continue
        else:
            sample, _ = stratified_sample(extract, field, take, seed)
        columns = [c for c in ("case_id", "case_name", "source_html_file", "nature_of_case",
                               "catchwords")
                   if c in sample.columns]
        sheet = sample[columns].copy()
        evidence = f"{field}_evidence" if f"{field}_evidence" in sample.columns else None
        reason = f"{field}_reason" if f"{field}_reason" in sample.columns else None
        # On the adjudication sheet the evidence quote is the LLM's OWN choice of
        # supporting words, so showing it up front argues one side of the very
        # disagreement being adjudicated. It moves to the hideable tail instead.
        for extra in (evidence, reason):
            if extra and not adjudication:
                sheet[extra] = sample[extra]
        sheet["GUIDANCE"] = guidance
        if prefill:
            sheet[f"HUMAN_{field}"] = sample[field]
        else:
            sheet[f"HUMAN_{field}"] = ""
        sheet["HUMAN_confident"] = ""
        sheet["HUMAN_notes"] = ""
        # Everything from here is hideable: it reveals what the extractors said.
        # The model's answer goes LAST so a labeller can hide the column.
        if adjudication:
            # The cell name encodes BOTH answers, so it is tail material too --
            # but it has to survive into the workbook, because scoring is
            # per-cell and a whole-sheet accuracy on a disagreement-stratified
            # draw is not a corpus accuracy.
            for extra in (evidence, "liability_posture_basis",
                          "consequential_condition_claimed", "_cell"):
                if extra and extra in sample.columns:
                    sheet[extra] = sample[extra]
            sheet[f"RULE_{field}"] = sample[f"{field}_rule"]
        sheet[f"MODEL_{field}"] = sample[field]
        worksheets[field] = sheet
    return worksheets


def wilson_interval(successes, total, z=1.96):
    """95% Wilson score interval, as whole percentages.

    Reported instead of a bare rate because every group in this report is small
    enough that a point estimate is misleading on its own: 7 of 9 is 78%, and
    also anything from 45% to 94%. Wilson rather than normal-approximation
    because it stays inside [0, 1] and behaves at the extremes, which is exactly
    where these counts live.
    """
    if not total:
        return (0, 0)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(proportion * (1 - proportion) / total
                            + z * z / (4 * total * total)) / denominator)
    return (round(100 * max(0.0, centre - spread)), round(100 * min(1.0, centre + spread)))


def _rate(successes, total):
    """'7/9 = 78% [45-94]' -- count, rate and interval travelling together."""
    low, high = wilson_interval(successes, total)
    return f"{successes}/{total} = {round(100 * successes / total)}% [{low}-{high}]"


def count_human_labels(path):
    """How many HUMAN_ cells in an existing worksheet have been filled in.

    Returns 0 for a missing or unreadable file: the guard this feeds must fail
    open, since refusing to write because a workbook could not be parsed would
    block the pipeline over a file nobody has labelled.
    """
    if not path or not os.path.exists(path):
        return 0
    try:
        book = pd.ExcelFile(path)
    except Exception:
        return 0
    filled = 0
    for name in book.sheet_names:
        try:
            sheet = pd.read_excel(book, name)
        except Exception:
            continue
        for column in sheet.columns:
            if str(column).startswith("HUMAN_"):
                values = sheet[column]
                filled += int((values.notna()
                               & values.astype(str).str.strip().ne("")).sum())
    return filled


def score_reference_set(path):
    """Read back completed worksheets and report LLM accuracy per field.

    Scores only rows where a human label was actually entered, and reports the
    confusion pairs so a systematic bias (rather than scattered error) is
    visible.
    """
    book = pd.ExcelFile(path)
    report = []
    for field, _guidance in REFERENCE_SET_FIELDS:
        if field not in book.sheet_names:
            continue
        sheet = pd.read_excel(book, field)
        human_column, model_column = f"HUMAN_{field}", f"MODEL_{field}"
        if human_column not in sheet.columns or model_column not in sheet.columns:
            continue
        # A sheet carrying _cell was drawn from the disagreement cells on
        # purpose, so its accuracy is NOT an estimate of corpus accuracy -- it
        # is conditioned on the two methods already disagreeing, where the LLM
        # is wrong far more often than it is at large. Reporting the two designs
        # under one unlabelled 'accuracy' column would invite exactly that
        # misreading.
        adjudication = "_cell" in sheet.columns
        design = ("adjudication: disagreement-stratified, NOT a corpus accuracy"
                  if adjudication else "label-stratified")
        labelled = sheet[sheet[human_column].astype(str).str.strip().ne("")
                         & sheet[human_column].notna()]
        if not len(labelled):
            report.append({"field": field, "labelled": 0, "accuracy": None,
                           "design": design, "note": "no human labels entered yet"})
            continue
        human = labelled[human_column].astype(str).str.strip()
        model = labelled[model_column].astype(str).str.strip()
        correct = (human == model).sum()
        confusions = Counter(f"{m} -> {h}" for m, h in zip(model, human) if m != h)
        note = "; ".join(f"{k} ({v})" for k, v in confusions.most_common(5)) or "no errors"

        if adjudication:
            rule_column = f"RULE_{field}"
            rule = (labelled[rule_column].astype(str).str.strip() if rule_column in labelled
                    else pd.Series("", index=labelled.index))
            partial = human.str.contains("in_part", case=False, na=False)
            parts = [f"partial-denial verdicts {partial.sum()}/{len(labelled)} "
                     f"({round(100 * partial.mean(), 1)}%)"]
            if rule_column in labelled:
                parts.append(f"human sides with LLM {(human == model).sum()}, "
                             f"with rule {(human == rule).sum()}, "
                             f"with neither {((human != model) & (human != rule)).sum()}")
            for cell, rows in labelled.groupby(labelled["_cell"].astype(str)):
                cell_human = rows[human_column].astype(str).str.strip()
                cell_model = rows[model_column].astype(str).str.strip()
                cell_partial = cell_human.str.contains("in_part", case=False, na=False).sum()
                parts.append(f"{cell}: n={len(rows)}, LLM right "
                             f"{(cell_human == cell_model).sum()}, partial {cell_partial}")
                # The dominant cell holds two mechanisms, not one. Reading it as
                # a single number is what the arms exist to prevent: if the
                # human sides with the rule at a materially different rate where
                # the rule had only the Nature field than where it quoted
                # concession language, that is a weak rule and an
                # under-specified category, separately, rather than one story.
                if (cell == "llm_denied__rule_quantum"
                        and "liability_posture_basis" in rows.columns
                        and rule_column in rows.columns):
                    cell_rule = rows[rule_column].astype(str).str.strip()
                    arms = ["concession" if str(b) in CONCESSION_BASES else "nature_field_only"
                            for b in rows["liability_posture_basis"]]
                    arm_rates = {}
                    for arm in ("concession", "nature_field_only"):
                        pick = [a == arm for a in arms]
                        if not any(pick):
                            continue
                        arm_human = cell_human[pick]
                        sided_rule = (arm_human == cell_rule[pick]).sum()
                        sided_llm = (arm_human == cell_model[pick]).sum()
                        arm_partial = arm_human.str.contains("in_part", case=False,
                                                             na=False).sum()
                        arm_rates[arm] = sided_rule / len(arm_human)
                        parts.append(f"    arm {arm}: sides with rule "
                                     f"{_rate(sided_rule, len(arm_human))}, with LLM "
                                     f"{sided_llm}, partial {arm_partial}")
                    # The caveat has to travel WITH the number. A workbook
                    # outlives the conversation that hedged it, and whoever
                    # opens this in six months will quote the comparison
                    # without knowing what it can and cannot carry.
                    if len(arm_rates) == 2:
                        gap = round(100 * (arm_rates["nature_field_only"]
                                           - arm_rates["concession"]))
                        parts.append(
                            f"    arm difference: {gap:+d} points (nature_field_only minus "
                            f"concession). CAVEAT: the arms are ~31 v ~9 by construction, so "
                            f"only a LARGE divergence is readable here. A small or zero "
                            f"difference is uninformative - it is NOT evidence the two arms "
                            f"are one story. Read direction and size, not significance.")
                    # The localisation test. Pooling these would let the dispute
                    # types whose answer is 'rule failure' -- medical and
                    # impairment, half of the disagreements -- speak for the
                    # ones where partial denial looks like the real cause.
                    if "nature_of_case" in rows.columns:
                        groups = [dispute_group(n) for n in rows["nature_of_case"]]
                        group_rates = {}
                        for group in ("liability", "impairment", "medical", "other"):
                            pick = [g == group for g in groups]
                            if not any(pick):
                                continue
                            group_human = cell_human[pick]
                            group_partial = group_human.str.contains("in_part", case=False,
                                                                     na=False).sum()
                            group_rates[group] = group_partial / len(group_human)
                            parts.append(
                                f"    dispute {group}: partial "
                                f"{_rate(group_partial, len(group_human))}, sides with rule "
                                f"{(group_human == cell_rule[pick]).sum()}")
                        if len(group_rates) >= 2:
                            top = max(group_rates, key=group_rates.get)
                            bottom = min(group_rates, key=group_rates.get)
                            spread = round(100 * (group_rates[top] - group_rates[bottom]))
                            parts.append(
                                f"    dispute spread: {spread} points, {top} highest, {bottom} "
                                f"lowest. This sheet is not the instrument for localisation - "
                                f"that runs at corpus scale (n=711 today, all 2,385 once in_part "
                                f"is extracted). Read this as whether the category APPLIES "
                                f"everywhere, not as the size of the dispute-type effect.")
            note = " | ".join(parts) + f" || confusions: {note}"

        report.append({
            "field": field,
            "labelled": len(labelled),
            "accuracy": round(100 * correct / len(labelled), 1),
            "design": design,
            "note": note,
        })
    return pd.DataFrame(report)


def freeze_definitional_set(extract):
    """The outcome disagreements, frozen as a named teaching set.

    These are NOT extraction errors to be resolved. They are the
    mixed-versus-clean-win boundary, where two competent readers disagree
    because the definition is underdetermined.

    At full-corpus scale this changed in kind, not just in size. The set is 259
    cases, 10.9% of the corpus, against an LLM-side 'mixed' rate of 10.7%: the
    disagreement set and the partial-success population are effectively the same
    object. 'mixed' appears on one side or the other in 92.7% of them. The
    methods are not disagreeing about the facts of these cases -- they are
    disagreeing about where partial success falls.

    So the exercise is no longer arguing case by case about ten decisions. It is
    characterising a boundary: which dispute types, postures and injury
    categories cluster there, and how far the headline metric moves as the line
    is redrawn (see summarise_definitional_shift).

    The 19 cases that are a clean claimant/insurer flip with no 'mixed' on
    either side are marked as such rather than dropped. They are a different
    animal -- a disagreement about what happened, not about where the line sits
    -- and leaving them unmarked in a boundary-characterisation exercise would
    put 19 factual errors into a set whose whole premise is that nobody is
    wrong.
    """
    if extract is None or "outcome_agreement" not in extract.columns:
        return pd.DataFrame()
    disputed = extract[extract["outcome_agreement"] == "differs"].copy()
    columns = [c for c in ("case_id", "case_name", "nature_of_case", "source_html_file",
                           "outcome", "outcome_rule", "outcome_reason", "result_text",
                           "outcome_analysable", "catchwords",
                           # Carried so the boundary can be characterised rather
                           # than merely argued over.
                           "liability_posture", "primary_injury", "legal_complexity")
               if c in disputed.columns]
    frozen = disputed[columns].copy()
    frozen.insert(0, "set_name", "DEFINITIONAL_RISK_OUTCOME_V1")
    if {"outcome", "outcome_rule"}.issubset(frozen.columns):
        mixed_either = (frozen["outcome"].astype(str).eq("mixed")
                        | frozen["outcome_rule"].astype(str).eq("mixed"))
        frozen.insert(1, "disagreement_kind",
                      ["boundary_partial_success" if flag else "clean_flip"
                       for flag in mixed_either])
    frozen["question_for_students"] = (
        "Where does partial success fall? These are not extraction errors: in 92.7% of them one "
        "reader called the case 'mixed' and the other resolved it to a side. Do not argue them "
        "one at a time. Characterise the boundary - which dispute types, postures and injury "
        "categories cluster here - then state a rule for partial success, apply it to the whole "
        "set, and report how far the headline success rate moves under your rule versus the "
        "alternative. Rows marked clean_flip are the exception: there the two readers disagree "
        "about what happened, so treat them separately.")
    return frozen


def summarise_definitional_shift(extract):
    """How far the headline metric moves as the partial-success line is redrawn.

    The point students need is not that a boundary exists but that it is worth
    something. Reported overall and by dispute type, because the boundary is not
    evenly distributed: Permanent Impairment is 30.1% of the disagreement set
    against 15.8% of the corpus, so a definitional change hits some dispute
    types roughly twice as hard as others.
    """
    if extract is None or not len(extract):
        return pd.DataFrame()
    if not {"outcome", "outcome_rule"}.issubset(extract.columns):
        return pd.DataFrame()

    def rates(frame):
        llm, rule = frame["outcome"].astype(str), frame["outcome_rule"].astype(str)
        mixed = llm.eq("mixed")
        return {
            "n": len(frame),
            # 'mixed counts as a loss' is the strict reading; 'mixed counts as a
            # win' is the generous one. The gap is the definitional exposure.
            "worker_success_strict_%": round(100 * llm.eq("claimant").mean(), 1),
            "worker_success_generous_%": round(100 * (llm.eq("claimant") | mixed).mean(), 1),
            "definitional_swing_points": round(100 * mixed.mean(), 1),
            "llm_mixed_%": round(100 * mixed.mean(), 1),
            "rule_mixed_%": round(100 * rule.eq("mixed").mean(), 1),
            "methods_differ_%": round(100 * llm.ne(rule).mean(), 1),
        }

    rows = [dict(scope="ALL", **rates(extract))]
    if "nature_of_case" in extract.columns:
        for nature, frame in extract.groupby(extract["nature_of_case"].astype(str)):
            if len(frame) >= 25:
                rows.append(dict(scope=nature, **rates(frame)))
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def build_validation_worksheet(extract, size=40, seed=20260815):
    """Blank worksheet for HAND-LABELLING the taxonomy fields.

    primary_injury and mechanism have no usable rule baseline, so the only
    honest validation is a human label on a sample. This emits the model's
    answer, its evidence, and empty columns for the human's verdict, stratified
    across the labels so rare categories are actually represented.
    """
    if extract is None or not len(extract):
        return pd.DataFrame()
    available = min(size, len(extract))
    if "primary_injury" in extract.columns:
        sample, _ = stratified_sample(extract, "primary_injury", available, seed)
    else:
        sample = extract.head(available)
    columns = [c for c in ("case_id", "case_name", "source_html_file", "catchwords",
                           "primary_injury", "primary_injury_evidence",
                           "mechanism", "mechanism_evidence",
                           "legal_complexity", "legal_complexity_reason")
               if c in sample.columns]
    worksheet = sample[columns].copy()
    worksheet["HUMAN_primary_injury"] = ""
    worksheet["HUMAN_mechanism"] = ""
    worksheet["HUMAN_legal_complexity"] = ""
    worksheet["HUMAN_notes"] = ""
    return worksheet


def write_reference_sets(path, extract, size=50, seed=20260815, prefill=False):
    """Hand-labelling worksheets and the frozen definitional set.

    Kept OUT of the tracked dictionary: these carry case names and catchwords,
    and the repo's convention is that litigant-bearing artifacts stay under
    output/.
    """
    if extract is None or not len(extract):
        return {}
    existing = count_human_labels(path)
    if existing:
        # Every other output here is regenerable; hand labels are not. A run
        # whose point is to ADD fields must not silently destroy the human work
        # the fields were designed around.
        raise RuntimeError(
            f"{path} already holds {existing} hand-entered label(s). Refusing to overwrite. "
            f"Move it aside, or pass --reference-out with a different path.")
    worksheets = build_reference_worksheets(extract, size=size, seed=seed, prefill=prefill)
    frozen = freeze_definitional_set(extract)
    shift = summarise_definitional_shift(extract)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for field, sheet in worksheets.items():
            sheet.to_excel(writer, sheet_name=field[:31], index=False)
        if len(frozen):
            frozen.to_excel(writer, sheet_name="definitional_outcome_set", index=False)
        if len(shift):
            shift.to_excel(writer, sheet_name="definitional_metric_shift", index=False)
        pd.DataFrame([
            {"instruction": "Label HUMAN_<field> WITHOUT reading MODEL_<field> (rightmost "
                            "column) - it is placed last so you can hide it. Anchoring on the "
                            "model's answer inflates measured accuracy and defeats the purpose."},
            {"instruction": "Use the catchwords first: they state what the Member decided. Open "
                            "source_html_file only where the catchwords are inconclusive."},
            {"instruction": "Leave HUMAN_<field> blank if genuinely undecidable; blanks are "
                            "skipped in scoring rather than counted as errors."},
            {"instruction": "liability_posture is an ADJUDICATION sheet, drawn only from cases "
                            "where the rule and the LLM disagree. Its accuracy is therefore not "
                            "the field's accuracy on the corpus - it is conditioned on a known "
                            "disagreement. Do not quote it as a field-level accuracy."},
            {"instruction": "On liability_posture, liability_denied_in_part is a permitted answer "
                            "and is the point of the exercise: neither extractor could return it. "
                            "Use it where liability was admitted for the primary injury but "
                            "denied for a consequential condition, a treatment, or an impairment "
                            "head."},
            {"instruction": "Score with: python wc_case_extract.py --score-validation "
                            "output/wc_reference_sets.xlsx"},
        ]).to_excel(writer, sheet_name="how_to_label", index=False)
    return worksheets


def build_dictionary(columns=None):
    """Expand FIELD_DOCS with the generated cross-check columns.

    Passing `columns` asserts coverage: any emitted column without an entry is
    returned as an 'UNDOCUMENTED' row rather than silently omitted, so the
    dictionary cannot fall behind the schema.
    """
    rows = [dict(zip(("field", "group", "type", "provenance", "allowed_values", "definition"), doc))
            for doc in FIELD_DOCS]
    documented = {row["field"] for row in rows}

    for _attribute, column, _convert in LLM_OVERLAY:
        for suffix, dtype, definition in (
            ("_rule", "varies", f"Value the text rule produced for {column}, kept for comparison."),
            ("_source", "text", f"Which method supplied the delivered {column}: 'llm' or 'rule' "
                                "(rule is used when the LLM returned nothing)."),
            ("_agreement", "text", f"Rule vs LLM for {column}: same / differs / rule_only / "
                                   "llm_only / both_empty. 'differs' flags a row worth reading."),
        ):
            name = f"{column}{suffix}"
            if name not in documented:
                rows.append({"field": name, "group": "cross-check", "type": dtype,
                             "provenance": "audit", "allowed_values": "", "definition": definition})
                documented.add(name)

    for field in STABILITY_FIELDS:
        for suffix, definition in (
            ("_votes", f"The independent votes cast for {field} under the stability gate, "
                       "comma-separated, in seed order. One value means the gate did not run."),
            ("_stability", f"Stability-gate outcome for {field}: 'stable' (two votes agreed), "
                           "'resolved_by_majority' (they disagreed and a third vote broke the "
                           "tie), 'unresolved_three_way' (three different answers - NO majority "
                           "exists, the first vote is retained and the row should be treated as "
                           "unreliable for this field), 'single_vote_only' (a repeat call "
                           "failed), or 'not_gated' (--stability-gate was off)."),
        ):
            name = f"{field}{suffix}"
            if name not in documented:
                rows.append({"field": name, "group": "stability", "type": "text",
                             "provenance": "audit", "allowed_values": "",
                             "definition": definition})
                documented.add(name)

    for _attribute, column in LLM_EVIDENCE:
        if column not in documented:
            rows.append({"field": column, "group": "evidence", "type": "text",
                         "provenance": "llm", "allowed_values": "",
                         "definition": "Verbatim quote from the decision supporting the related "
                                       "field, or a one-sentence reason. Blank where the model "
                                       "found no support."})
            documented.add(column)

    if columns is not None:
        order = {name: position for position, name in enumerate(columns)}
        for name in columns:
            if name not in documented:
                rows.append({"field": name, "group": "UNDOCUMENTED", "type": "",
                             "provenance": "", "allowed_values": "",
                             "definition": "TODO: no dictionary entry for this column."})
        rows.sort(key=lambda row: order.get(row["field"], len(order)))
    return pd.DataFrame(rows, columns=["field", "group", "type", "provenance",
                                       "allowed_values", "definition"])


def write_data_dictionary(path, columns=None, corpus=None, extract=None):
    """Write the standalone dictionary workbook. Returns the fields frame."""
    dictionary = build_dictionary(columns)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        dictionary.to_excel(writer, sheet_name="fields", index=False)
        pd.DataFrame(DICTIONARY_NOTES, columns=["topic", "note"]).to_excel(
            writer, sheet_name="notes", index=False)
        build_selection_funnel(corpus, extract).to_excel(
            writer, sheet_name="selection_funnel", index=False)
        pd.DataFrame(WORKED_EXAMPLES).to_excel(
            writer, sheet_name="worked_examples", index=False)
        pd.DataFrame(enumeration_rows(), columns=["field", "value"]).to_excel(
            writer, sheet_name="enumerations", index=False)
    return dictionary



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def load_llm_cache(path):
    """Read the LLM cache. Each pass checks its OWN version on use.

    Keyed on URL (one per decision) rather than filename, so the duplicate
    files on disk can never produce two cache entries for one case.

    Entries are no longer filtered on WC_SCHEMA_VERSION at load time. They were,
    and that quietly coupled the two passes: a main-schema bump would have taken
    the conduct data with it, forcing both to be re-derived together and
    defeating the point of versioning them apart. The version gate now lives in
    each reader -- cached_llm_extract checks _wc_schema_version,
    cached_conduct_extract checks _conduct_version -- so a stale main record and
    a current conduct record can coexist in one entry, and only the stale one is
    re-extracted.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        print(f"WARNING: could not read LLM cache ({error}); starting empty")
        return {}
    return {url: entry for url, entry in raw.items() if isinstance(entry, dict)}


def cache_pass_counts(cache):
    """How many entries are current for each pass. Reported at startup so a
    version bump cannot silently look like an empty cache."""
    main = sum(1 for entry in cache.values()
               if entry.get("_wc_schema_version") == WC_SCHEMA_VERSION)
    conduct = sum(1 for entry in cache.values()
                  if entry.get("_conduct_version") == WC_CONDUCT_VERSION)
    return {"entries": len(cache), "main_current": main, "conduct_current": conduct}


def save_llm_cache(path, cache):
    """Write via a temp file so an interrupted run cannot truncate the cache."""
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    os.replace(temp_path, path)


# Measured at 91% run-to-run stability, against 95-100% for everything else.
# These two get a best-of-three vote instead of a single draw.
STABILITY_FIELDS = ("primary_injury", "legal_complexity")


def _vote_value(parsed, field):
    return str(getattr(getattr(parsed, field, None), "value", getattr(parsed, field, None)))


def stability_gate(extractor, cache, url, text, parsed, context=None, lock=None):
    """Best-of-three for the two least stable fields.

    Runs a second extraction with a different seed. If the gated fields agree,
    the value is accepted as stable at no further cost. If they disagree, a
    third vote breaks the tie by majority. Costs 2x on every case and 3x on the
    ~9% that disagree, so it is opt-in via --stability-gate.

    Returns (votes, statuses) keyed by field, and mutates `parsed` to carry the
    majority value.
    """
    votes = {field: [_vote_value(parsed, field)] for field in STABILITY_FIELDS}
    statuses = {}

    second = _cached_vote(extractor, cache, url, text, WC_SEED + 1, context, lock)
    if second is None:
        return votes, {field: "single_vote_only" for field in STABILITY_FIELDS}
    for field in STABILITY_FIELDS:
        votes[field].append(second.get(field, ""))

    if all(len(set(v)) == 1 for v in votes.values()):
        return votes, {field: "stable" for field in STABILITY_FIELDS}

    third = _cached_vote(extractor, cache, url, text, WC_SEED + 2, context, lock)
    for field in STABILITY_FIELDS:
        if third is not None:
            votes[field].append(third.get(field, ""))
        tally = Counter(v for v in votes[field] if v)
        if not tally:
            statuses[field] = "no_votes"
            continue
        winner, count = tally.most_common(1)[0]
        if len(set(votes[field])) == 1:
            statuses[field] = "stable"
        elif count >= 2:
            statuses[field] = "resolved_by_majority"
            _apply_vote(parsed, field, winner)
        else:
            # Three different answers: no majority exists. Keep the first vote
            # and flag it rather than pretending the tie was broken.
            statuses[field] = "unresolved_three_way"
    return votes, statuses


def _apply_vote(parsed, field, winner):
    """Write the majority value back, respecting the field's declared type."""
    current = getattr(parsed, field, None)
    try:
        if isinstance(current, int) and not isinstance(current, bool):
            setattr(parsed, field, int(winner))
        elif isinstance(current, Enum):
            setattr(parsed, field, type(current)(winner))
        else:
            setattr(parsed, field, winner)
    except (ValueError, TypeError):
        pass


def _cached_vote(extractor, cache, url, text, seed, context, lock):
    """One extra vote, memoised per (url, seed). Only the gated fields are
    stored — the rest of the repeat parse is discarded."""
    entry = cache.get(url) or {}
    stored = (entry.get("_votes") or {}).get(str(seed))
    if stored is not None:
        return stored
    if extractor is None:
        return None
    parsed, _usage, _error = extract_wc_case_llm(
        extractor, text, context=f"{context} seed={seed}", seed=seed)
    if parsed is None:
        return None
    record = {field: _vote_value(parsed, field) for field in STABILITY_FIELDS}
    if lock is not None:
        with lock:
            cache.setdefault(url, {}).setdefault("_votes", {})[str(seed)] = record
    else:
        cache.setdefault(url, {}).setdefault("_votes", {})[str(seed)] = record
    return record


def cached_llm_extract(extractor, cache, url, text, context=None, lock=None):
    """Return (parsed, error, was_cached). A cache hit costs no API call.

    `lock` guards the shared cache dict when called from worker threads. The
    API call itself is deliberately made OUTSIDE the lock — holding it across
    a multi-second request would serialise the whole pool.
    """
    def _read():
        return cache.get(url)

    entry = _read() if lock is None else _with(lock, _read)
    # The version gate moved here from load_llm_cache so the conduct pass can
    # keep its own data in an entry whose main record is stale.
    if entry is not None and entry.get("_wc_schema_version") == WC_SCHEMA_VERSION:
        payload = {k: v for k, v in entry.items() if not k.startswith("_")}
        try:
            return WCCaseSchema(**payload), None, True
        except Exception:
            pass  # Malformed entry: fall through and re-extract.
    if extractor is None:
        return None, "llm disabled", False
    parsed, _usage, error = extract_wc_case_llm(extractor, text, context=context)
    if parsed is not None:
        record = parsed.model_dump(mode="json")
        record["_wc_schema_version"] = WC_SCHEMA_VERSION

        def _write():
            # Carry the second pass across a main-schema re-extraction. Without
            # this the two passes are coupled again by the back door: bumping
            # WC_SCHEMA_VERSION would silently discard conduct data that is
            # still current, and the additive pass would cost full price every
            # time the main schema moved.
            previous = cache.get(url) or {}
            # Carry every side-pass's data, not a named list of them: a new pass
            # added later would otherwise be silently dropped by an old write,
            # which is the coupling this whole design exists to prevent.
            # _votes is the one exception -- those are values of MAIN-schema
            # fields, so a re-extraction makes them stale rather than reusable.
            for key, value in previous.items():
                if key.startswith("_") and key not in ("_votes", "_wc_schema_version"):
                    record[key] = value
            cache[url] = record
        _write() if lock is None else _with(lock, _write)
    return parsed, error, False


RELIEF_PILOT_SEEDS = (20260815, 20260816, 20260817)


def cached_relief_votes(extractor, cache, url, text, context=None, lock=None,
                        seeds=RELIEF_PILOT_SEEDS):
    """Run the relief ordinal under several seeds, memoised per (url, seed).

    Returns the list of votes. The pilot needs both answers this produces --
    what the distribution looks like and whether it holds still -- and one set
    of calls gives both, so there is no cheaper design that answers the
    question.
    """
    def _read():
        return ((cache.get(url) or {}).get("_relief_votes") or {})

    stored = _read() if lock is None else _with(lock, _read)
    votes = []
    for seed in seeds:
        key = str(seed)
        if stored.get(key) is not None:
            votes.append(stored[key])
            continue
        if extractor is None:
            continue
        parsed, _usage, _error = extract_wc_relief_llm(
            extractor, text, context=f"{context} seed={seed}", seed=seed)
        if parsed is None:
            continue
        value = parsed.relief_granted_degree.value

        def _write():
            slot = cache.setdefault(url, {})
            slot.setdefault("_relief_votes", {})[key] = value
            slot["_relief_version"] = WC_RELIEF_VERSION
        _write() if lock is None else _with(lock, _write)
        votes.append(value)
    return votes


def run_relief_revote(args, records, file_index, extractor, llm_cache, cache_lock):
    """Two extra votes on the middle categories, plus a bounding sample of extremes.

    Gating the whole corpus costs 3x to sharpen 79% of cases that are not in
    doubt. Re-voting only where the threshold slides costs about a third of
    that and resolves the noise that actually reaches the curve. The price is a
    blind spot -- an extreme misread as an extreme is never revisited -- which
    the bounding sample measures rather than assumes.
    """
    middle, extreme = relief_revote_targets(llm_cache, extremes=args.relief_revote_extremes,
                                            seed=args.seed)
    targets = set(middle) | set(extreme)
    print(f"\nRelief re-vote: {len(middle)} middle-category cases"
          f" + {len(extreme)} sampled extremes, 2 extra votes each")
    if not targets:
        print("Nothing to re-vote: run --relief-pass first.")
        return {}

    by_url = {}
    for csv_row in records:
        url = csv_row.get("URL")
        if url in targets:
            by_url[url] = csv_row

    before = {url: (llm_cache.get(url, {}).get("_relief") or {}).get("relief_granted_degree")
              for url in targets}

    def revote(url):
        csv_row = by_url.get(url)
        if csv_row is None:
            return
        _case_id, year, number = case_id_from_url(url)
        entry = file_index.get((year, number), {"canonical": None, "all": []})
        text = load_text(url, entry["canonical"], args.decisions, args.cache)
        if not text:
            return
        cached_relief_votes(extractor, llm_cache, url, text, context=url,
                            lock=cache_lock, seeds=RELIEF_PILOT_SEEDS[1:])

    workers = max(1, min(args.workers, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(revote, sorted(targets)))
    save_llm_cache(args.llm_cache, llm_cache)

    def summarise(urls, label):
        if not urls:
            return
        agreements, changed, to_middle = Counter(), 0, 0
        for url in urls:
            value, agreement, _votes = resolve_relief_votes(llm_cache.get(url))
            agreements[agreement] += 1
            if value and value != before.get(url):
                changed += 1
                if value in MIDDLE_DEGREES:
                    to_middle += 1
        print(f"\n{label} (n={len(urls)}):")
        for name, count in agreements.most_common():
            print(f"  {name:<14} {count:>4}  {100 * count / len(urls):5.1f}%")
        print(f"  value changed  {changed:>4}  {100 * changed / len(urls):5.1f}%")
        if label.startswith("Extremes"):
            print(f"  moved INTO a middle category {to_middle} "
                  f"({100 * to_middle / len(urls):.1f}%) <- the blind-spot bound")

    summarise(middle, "Middle categories")
    summarise(extreme, "Extremes (bounding sample)")
    print(f"\nLLM spend: ${COST.total_cost():.4f} over {COST.calls} calls")
    return {"middle": len(middle), "extremes": len(extreme)}


def run_relief_pilot(args, records, file_index, extractor, llm_cache, cache_lock):
    """Pilot the relief ordinal and report. Writes nothing but the cache.

    Kept off the workbook path on purpose: a field under test must not be able
    to land in the deliverable, and a pilot that overwrote wc_case_extract.xlsx
    with a 100-row sample would be a far more expensive mistake than the one it
    is guarding against.
    """
    # A seeded RANDOM draw, not records[:N]. Taking the head of the frame is
    # the failure this repo already documents: the first N rows are in citation
    # order, and any variable correlated with time or court order rides along
    # unnoticed. The first version of this pilot did exactly that and drew a
    # sample carrying 15% 'mixed' against a 10.7% corpus rate -- an
    # over-representation of precisely the cases the ordinal is meant to
    # spread, which flatters the result it was testing.
    frame = pd.DataFrame(records)
    take = min(args.relief_pilot, len(frame))
    sample = frame.sample(n=take, random_state=args.seed).to_dict("records")
    print(f"\nRelief-degree pilot: {len(sample)} cases x {len(RELIEF_PILOT_SEEDS)} votes "
          f"(random, seed {args.seed})")

    def vote(csv_row):
        url = csv_row.get("URL")
        case_id, year, number = case_id_from_url(url)
        entry = file_index.get((year, number), {"canonical": None, "all": []})
        text = load_text(url, entry["canonical"], args.decisions, args.cache)
        if not text:
            return []
        return cached_relief_votes(extractor, llm_cache, url, text,
                                   context=case_id, lock=cache_lock)

    workers = max(1, min(args.workers, len(sample)))
    all_votes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for votes in pool.map(vote, sample):
            all_votes.append(votes)

    save_llm_cache(args.llm_cache, llm_cache)
    report = summarise_relief_pilot(all_votes)

    print(f"\nCases with >=2 votes: {report['cases']}")
    print("\nDistribution (first vote):")
    for value, count in report["distribution"].items():
        share = 100 * count / report["cases"] if report["cases"] else 0
        print(f"  {value:<24} {count:>4}  {share:5.1f}%")
    print(f"\n  distinct values used   {report['distinct_values']}")
    print(f"  largest bucket         {report['largest_bucket_%']}%")
    print("\nRun-to-run stability:")
    print(f"  unanimous              {report['unanimous_%']}%")
    print(f"  majority only          {report['majority_only_%']}%")
    print(f"  no majority            {report['no_majority_%']}%")
    print("\nThe two failure modes this is testing for:")
    print("  - piles up on one value  -> cannot carry a sensitivity curve however accurate")
    print("  - moves between runs     -> the curve would be made of noise")
    print(f"\nLLM spend: ${COST.total_cost():.4f} over {COST.calls} calls")
    return report


def summarise_relief_pilot(all_votes):
    """The two questions the pilot exists to answer, and nothing else.

    Deliberately reports the DISTRIBUTION and the STABILITY rather than an
    accuracy: there is no ground truth here, and the failure modes that killed
    the head-count ratio are both visible without one. A field that piles up on
    one value cannot carry a sensitivity curve however accurate it is, and an
    ordinal that moves between runs produces a curve made of noise.
    """
    complete = [votes for votes in all_votes if len(votes) >= 2]
    distribution = Counter(votes[0] for votes in complete)
    unanimous = sum(1 for votes in complete if len(set(votes)) == 1)
    majority = sum(1 for votes in complete
                   if len(set(votes)) > 1 and Counter(votes).most_common(1)[0][1] >= 2)
    scattered = len(complete) - unanimous - majority
    top_share = (100 * max(distribution.values()) / len(complete)) if complete else 0.0
    return {
        "cases": len(complete),
        "distribution": dict(distribution.most_common()),
        "distinct_values": len(distribution),
        "largest_bucket_%": round(top_share, 1),
        "unanimous_%": round(100 * unanimous / len(complete), 1) if complete else 0.0,
        "majority_only_%": round(100 * majority / len(complete), 1) if complete else 0.0,
        "no_majority_%": round(100 * scattered / len(complete), 1) if complete else 0.0,
    }


def cached_relief_extract(extractor, cache, url, text, context=None, lock=None):
    """Single-vote production pass for the relief ordinal.

    Stores the full record under `_relief`, separate from the pilot's
    `_relief_votes`. The 100 pilot cases are therefore re-extracted rather than
    reusing vote one: the votes hold only the ordinal, and taking them would
    leave those rows with a degree and no evidence quote -- a silent hole in
    exactly the column that makes the value auditable, to save about a dollar.
    The votes stay in the cache as the stability record.
    """
    def _read():
        return cache.get(url) or {}

    entry = _read() if lock is None else _with(lock, _read)
    if entry.get("_relief_version") == WC_RELIEF_VERSION and entry.get("_relief"):
        try:
            return WCReliefSchema(**entry["_relief"]), None, True
        except Exception:
            pass  # Malformed: fall through and re-extract.
    if extractor is None:
        return None, "llm disabled", False
    parsed, _usage, error = extract_wc_relief_llm(
        extractor, text, context=context, seed=RELIEF_PILOT_SEEDS[0])
    if parsed is not None:
        def _write():
            slot = cache.setdefault(url, {})
            slot["_relief"] = parsed.model_dump(mode="json")
            slot["_relief_version"] = WC_RELIEF_VERSION
        _write() if lock is None else _with(lock, _write)
    return parsed, error, False


# The three values the sensitivity threshold actually slides across. Re-voting
# is targeted here rather than run corpus-wide: gating everything costs 3x to
# sharpen the 79% of cases sitting at fully_granted or refused, where the pilot
# found 56/58 and 18/19 agreement with the outcome field and nothing is in
# doubt.
MIDDLE_DEGREES = {"substantially_granted", "partly_granted", "minimally_granted"}


def resolve_relief_votes(entry):
    """Combine the delivered vote with any re-votes. Returns (value, agreement, votes).

    Votes are keyed by SEED, not appended to a list: the production pass and the
    pilot's first vote both use RELIEF_PILOT_SEEDS[0], so a naive concatenation
    would count one answer twice on the 100 pilot cases and manufacture
    unanimity out of a single opinion.
    """
    entry = entry or {}
    base = (entry.get("_relief") or {}).get("relief_granted_degree")
    by_seed = {str(seed): value for seed, value in (entry.get("_relief_votes") or {}).items()
               if value}
    if base:
        by_seed[str(RELIEF_PILOT_SEEDS[0])] = base
    votes = [by_seed[key] for key in sorted(by_seed)]
    if not votes:
        return None, "not_run", []
    if len(votes) == 1:
        return votes[0], "single_vote", votes
    tally = Counter(votes)
    winner, count = tally.most_common(1)[0]
    if len(set(votes)) == 1:
        return winner, "unanimous", votes
    if count >= 2:
        return winner, "majority", votes
    # Every vote different: keep the delivered value and flag it rather than
    # pretending a tie was broken.
    return base or votes[0], "no_majority", votes


def relief_revote_targets(cache, extremes=0, seed=20260815):
    """URLs needing extra votes: every middle-category case, plus a random
    sample of extremes to BOUND the blind spot.

    The targeted design cannot see an extreme→middle error, because it never
    re-votes an extreme. Sampling some anyway turns that from an unknown into a
    measured rate, which is the difference between a limitation and a hole.
    """
    middle, extreme = [], []
    for url, entry in cache.items():
        degree = (entry.get("_relief") or {}).get("relief_granted_degree")
        if not degree:
            continue
        (middle if degree in MIDDLE_DEGREES else extreme).append(url)
    if extremes and extreme:
        rng = random.Random(seed)
        extreme = rng.sample(extreme, min(extremes, len(extreme)))
    else:
        extreme = []
    return sorted(middle), sorted(extreme)


RELIEF_OVERLAY = [
    ("relief_granted_degree", "relief_granted_degree"),
    ("relief_granted_evidence", "relief_granted_evidence"),
]


def apply_relief(row, parsed, error=None, resolution=None):
    """Write the relief ordinal onto a row.

    Single-vote by choice: the pilot measured 89% unanimity, and this field
    feeds a sensitivity CURVE rather than a cohort assignment. A label that
    moves one adjacent step blurs the line; it does not relocate a case between
    analysis populations the way primary_injury would. Paying 3x to sharpen a
    sensitivity analysis is the trade the stability-gate example says to make
    explicitly, and here it was declined. relief_stability records that.
    """
    row["relief_status"] = "ok" if parsed is not None else (error or "not run")
    if parsed is None:
        for _attribute, column in RELIEF_OVERLAY:
            row.setdefault(column, None)
        row["relief_agreement"] = "not_run"
        row["relief_votes"] = ""
        row["relief_stability"] = "not_run"
        return row
    for attribute, column in RELIEF_OVERLAY:
        value = getattr(parsed, attribute, None)
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, str):
            value = value[:600]
        row[column] = value

    value, agreement, votes = (resolution or (None, "single_vote", []))
    if value:
        # The resolved value REPLACES the single vote where re-voting happened.
        row["relief_granted_degree"] = value
    row["relief_agreement"] = agreement
    row["relief_votes"] = ",".join(votes)
    row["relief_stability"] = (
        "single_vote_89pc_unanimous_in_pilot" if agreement == "single_vote"
        else f"revoted_{agreement}")
    return row


def cached_conduct_extract(extractor, cache, url, text, context=None, lock=None):
    """Second pass, memoised under its own version key. Returns (parsed, error,
    was_cached).

    Stored as a nested `_conduct` dict rather than as top-level keys so it stays
    invisible to WCCaseSchema(**payload), which reconstructs the main record
    from every non-underscore key in the entry.
    """
    def _read():
        return cache.get(url) or {}

    entry = _read() if lock is None else _with(lock, _read)
    if entry.get("_conduct_version") == WC_CONDUCT_VERSION:
        try:
            return WCConductSchema(**(entry.get("_conduct") or {})), None, True
        except Exception:
            pass  # Malformed: fall through and re-extract.
    if extractor is None:
        return None, "llm disabled", False
    parsed, _usage, error = extract_wc_conduct_llm(extractor, text, context=context)
    if parsed is not None:
        def _write():
            slot = cache.setdefault(url, {})
            slot["_conduct"] = parsed.model_dump(mode="json")
            slot["_conduct_version"] = WC_CONDUCT_VERSION
        _write() if lock is None else _with(lock, _write)
    return parsed, error, False


def _with(lock, function):
    with lock:
        return function()


def load_text(url, canonical_file, folder, cache_dir):
    """Read decision text, memoising the parse so re-runs are fast."""
    if not canonical_file:
        return ""
    cache_path = None
    if cache_dir:
        case_id, year, number = case_id_from_url(url)
        if year is not None:
            cache_path = os.path.join(cache_dir, f"{year}_{number}.txt")
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as handle:
                return handle.read()
    path = os.path.join(folder, canonical_file)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as handle:
        text = html_to_text(handle.read())
    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text


# Printed for review, in the order the spec asked for them.
REVIEW_LAYOUT = [
    ("identity", ["case_id", "case_name", "source_html_file", "duplicate_html_files",
                  "decision_text_chars", "llm_status"]),
    ("3 insurer", ["insurer_name", "insurer_source", "insurer_name_rule",
                   "insurer_name_agreement", "insurer_evidence"]),
    ("4-5 dates", ["accident_date", "decision_date", "injury_to_decision_days",
                   "injury_to_decision_years"]),
    ("6 nature", ["nature_of_case", "result_text"]),
    ("7 outcome", ["outcome", "outcome_source", "outcome_rule", "outcome_agreement",
                   "outcome_reason"]),
    ("8 lump sum", ["lump_sum_amount", "lump_sum_source", "lump_sum_amount_rule",
                    "lump_sum_amount_agreement", "lump_sum_type"]),
    ("9 WPI", ["wpi_percent", "wpi_provenance", "wpi_percent_in_text",
               "wpi_contended_by_claimant", "wpi_contended_by_claimant_rule",
               "wpi_contended_by_insurer", "wpi_contended_by_insurer_rule",
               "wpi_contended_evidence"]),
    ("10-13 claimant", ["claimant_age", "claimant_gender", "claimant_occupation",
                        "employer_name"]),
    ("14-17,20 ordinals", ["psych_injury_emphasis", "legal_complexity", "liability_clarity",
                           "pre_existing_conditions", "work_impact_severity", "ability_to_work",
                           "legal_complexity_rule", "legal_complexity_source", "legal_complexity_reason"]),
    ("18 income", ["pre_injury_weekly_income", "pre_injury_income_basis"]),
    ("19 legal costs", ["legal_costs_amount", "legal_costs_amount_source",
                        "legal_costs_amount_rule", "legal_costs_evidence"]),
    ("21-22 taxonomy", ["primary_injury", "primary_injury_source", "primary_injury_rule",
                        "mechanism", "mechanism_source", "mechanism_rule"]),
    ("new fields", ["remitted_to_medical_assessor", "medical_assessor_involved",
                    "consequential_condition_claimed", "fatality",
                    "weekly_benefit_amount", "weekly_benefit_amount_rule",
                    "claimant_legal_representation", "claimant_legal_representation_rule",
                    "claimant_interpreter_used", "claimant_interpreter_used_rule",
                    "liability_posture", "liability_posture_rule", "liability_posture_evidence"]),
    ("extras", ["member_name", "member_role", "s11A_defence_raised", "regulatory_sections"]),
]


def stratified_sample(frame, column, size, seed):
    """Proportional allocation with a guaranteed floor of one per stratum.

    A plain random draw of 10 from a 60/38/2 gender split can easily return no
    women and never returns a 'Not stated' row, which is exactly the stratum
    where extraction is most likely to be wrong. Largest-remainder allocation
    keeps the sample proportional; the floor keeps every stratum visible.
    """
    strata = frame[column].fillna("Not stated").astype(str)
    counts = strata.value_counts()
    present = [name for name in counts.index if counts[name] > 0]
    size = min(size, len(frame))
    if size <= 0 or not present:
        return frame.head(0), {}

    # Floor of one per stratum, as far as the sample size allows.
    allocation = {name: 0 for name in present}
    for name in present[:size]:
        allocation[name] = 1
    remaining = size - sum(allocation.values())

    if remaining > 0:
        total = counts[present].sum()
        exact = {name: remaining * counts[name] / total for name in present}
        for name in present:
            allocation[name] += int(exact[name])
        leftover = size - sum(allocation.values())
        for name in sorted(present, key=lambda n: exact[n] - int(exact[n]), reverse=True)[:leftover]:
            allocation[name] += 1

    parts = []
    for name, wanted in allocation.items():
        pool = frame[strata == name]
        take = min(wanted, len(pool))
        if take:
            parts.append(pool.sample(n=take, random_state=seed))
    sample = pd.concat(parts) if parts else frame.head(0)
    return sample.sample(frac=1, random_state=seed), allocation


def print_review(extract):
    """Vertical per-case dump: 80-odd columns are unreadable as a wide table."""
    for _, row in extract.iterrows():
        print("\n" + "=" * 78)
        print(f"{row.get('case_id', '')}  {str(row.get('case_name', ''))[:60]}")
        print("=" * 78)
        for group, columns in REVIEW_LAYOUT:
            print(f"\n-- {group}")
            for column in columns:
                if column not in extract.columns:
                    continue
                value = row[column]
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    value = ""
                text = str(value).replace("\n", " ")
                if len(text) > 150:
                    text = text[:150] + "..."
                print(f"   {column:<36} {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=CSV_REPORT)
    parser.add_argument("--decisions", default=DECISIONS_FOLDER)
    parser.add_argument("--out", default=OUTPUT_XLSX)
    parser.add_argument("--cache", default=TEXT_CACHE,
                        help="Directory for memoised decision text ('' to disable)")
    parser.add_argument("--sample", type=int, default=10,
                        help="Random sample size while the spec is being settled. "
                             "Use --all for the full corpus.")
    parser.add_argument("--seed", type=int, default=20260815,
                        help="Sampling seed, so a review run is reproducible")
    parser.add_argument("--all", action="store_true", help="Process every WC case")
    parser.add_argument("--stratify", default="Claimant Gender,has_wpi",
                        help="Comma-separated columns to stratify the sample on. 'has_wpi' is a "
                             "synthetic stratum for whether an accepted WPI exists. "
                             "'' for a plain random draw.")
    parser.add_argument("--no-llm", action="store_true",
                        help="Rule-based fields only; skip the structured LLM pass")
    parser.add_argument("--llm-cache", default=LLM_CACHE_FILE,
                        help="JSON cache of LLM results, keyed on URL ('' to disable)")
    parser.add_argument("--stability-gate", action="store_true",
                        help="Best-of-three vote on primary_injury and legal_complexity, the two "
                             "fields measured at 91%% run-to-run stability. Costs 2x per case, "
                             "3x on the ~9%% that disagree.")
    parser.add_argument("--refresh-llm", action="store_true",
                        help="Ignore cached LLM results and re-extract")
    parser.add_argument("--relief-pass", action="store_true",
                        help="Run the relief_granted_degree pass, single vote per case. "
                             "Versioned apart from the conduct pass. ~$28 over the full corpus.")
    parser.add_argument("--relief-revote", action="store_true",
                        help="Two extra votes on cases sitting in the middle three degrees, "
                             "where the sensitivity threshold slides, then majority-resolve. "
                             "Targets ~19%% of the corpus instead of gating all of it.")
    parser.add_argument("--relief-revote-extremes", type=int, default=100, metavar="N",
                        help="Also re-vote N random cases at the extremes, to BOUND the "
                             "extreme-to-middle errors the targeted design cannot see. 0 to skip.")
    parser.add_argument("--relief-pilot", type=int, metavar="N",
                        help="Pilot the relief_granted_degree ordinal on N cases with 3 votes "
                             "each, report the distribution and run-to-run stability, and exit. "
                             "Writes nothing but the cache. ~$0.06/case.")
    parser.add_argument("--conduct-pass", action="store_true",
                        help="Run the additive second pass: denial scope, success granularity "
                             "and insurer-conduct findings. Versioned separately from the main "
                             "schema, so it adds fields without re-deriving the other 191. "
                             "One pass over the full corpus, no stability gate: ~$42.")
    parser.add_argument("--workers", type=int, default=int(os.getenv("NSW_WC_WORKERS", "20")),
                        help="Concurrent LLM calls (default 20)")
    parser.add_argument("--dictionary-out", default=DICTIONARY_XLSX,
                        help="Path for the standalone data dictionary workbook")
    parser.add_argument("--reference-out", default=WORKSHEETS_XLSX,
                        help="Path for hand-labelling worksheets and the frozen definitional set")
    parser.add_argument("--reference-size", type=int, default=50,
                        help="Cases per field in the hand-labelling reference set")
    parser.add_argument("--score-validation", metavar="XLSX",
                        help="Score completed worksheets and exit")
    parser.add_argument("--quiet", action="store_true", help="Skip the per-case review dump")
    args = parser.parse_args()

    if args.score_validation:
        report = score_reference_set(args.score_validation)
        if not len(report):
            print("No scorable sheets found.")
            return
        print("\nLLM accuracy against the human reference set:")
        for _, row in report.iterrows():
            accuracy = "n/a" if pd.isna(row["accuracy"]) else f"{row['accuracy']}%"
            print(f"  {row['field']:<22} {accuracy:>7}  (n={row['labelled']})  [{row['design']}]")
            for part in str(row["note"]).split(" | "):
                print(f"  {'':<22} {part}")
        return

    frame = pd.read_csv(args.csv, low_memory=False)
    workers_comp = frame[frame["Case Type"].astype(str).str.strip() == "Workers Compensation"]
    workers_comp = workers_comp.drop_duplicates(subset=["URL"])
    total = len(workers_comp)

    if args.all:
        selected, label = workers_comp, f"all {total}"
    elif args.stratify:
        columns = [c.strip() for c in args.stratify.split(",") if c.strip()]
        pool = workers_comp.copy()
        # A composite key lets the sample cover, say, gender AND whether a WPI
        # figure exists - the default seed happened to draw 0 WPI rows out of
        # 20, which is a 0.8% event and made the sample useless for that field.
        pool["_stratum"] = ""
        for column in columns:
            if column == "has_wpi":
                part = pool["Impairment % (Accepted)"].notna().map({True: "wpi", False: "nowpi"})
            else:
                part = pool[column].fillna("Not stated").astype(str)
            pool["_stratum"] = pool["_stratum"] + "|" + part
        selected, allocation = stratified_sample(pool, "_stratum", args.sample, args.seed)
        selected = selected.drop(columns=["_stratum"])
        label = (f"stratified {len(selected)} of {total} by {'+'.join(columns)} "
                 f"{allocation} (seed {args.seed})")
    else:
        size = min(args.sample, total)
        selected = workers_comp.sample(n=size, random_state=args.seed)
        label = f"random {size} of {total} (seed {args.seed})"
    print(f"Workers compensation cases: {label}")

    file_index = index_decision_files(args.decisions)
    print(f"Decision files indexed for {len(file_index)} citations")

    extractor = None
    if not args.no_llm:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            extractor = LLMExtractor(api_key)
            print(f"LLM pass: enabled (effort={WC_REASONING_EFFORT})")
        else:
            print("LLM pass: DISABLED - OPENAI_API_KEY not set; rule-based values only")
    else:
        print("LLM pass: disabled by --no-llm")

    llm_cache = {} if args.refresh_llm else load_llm_cache(args.llm_cache)
    if llm_cache:
        counts = cache_pass_counts(llm_cache)
        print(f"LLM cache: {counts['entries']} entries; {counts['main_current']} current at "
              f"schema v{WC_SCHEMA_VERSION}, {counts['conduct_current']} at conduct "
              f"v{WC_CONDUCT_VERSION}")

    cache_lock = threading.Lock()
    counters = Counter()
    records = [record.to_dict() for _, record in selected.iterrows()]

    if args.relief_pilot:
        run_relief_pilot(args, records, file_index, extractor, llm_cache, cache_lock)
        return

    if args.relief_revote:
        run_relief_revote(args, records, file_index, extractor, llm_cache, cache_lock)
        return

    def process(csv_row):
        """One case end to end. Runs in a worker thread: the HTML parse is
        CPU-bound and the LLM call is I/O-bound, so both benefit from the pool.
        """
        url = csv_row.get("URL")
        case_id, year, number = case_id_from_url(url)
        entry = file_index.get((year, number), {"canonical": None, "all": []})
        text = load_text(url, entry["canonical"], args.decisions, args.cache)
        if not text:
            counters["missing"] += 1
        row = build_row(csv_row, text, entry["canonical"], entry["all"])

        parsed, error = None, None
        votes, statuses = {}, {}
        if text and (extractor is not None or url in llm_cache):
            parsed, error, was_cached = cached_llm_extract(
                extractor, llm_cache, url, text, context=case_id, lock=cache_lock)
            counters["cache_hits"] += int(was_cached)
            if parsed is None and extractor is not None:
                counters["llm_errors"] += 1
            if parsed is not None and args.stability_gate:
                votes, statuses = stability_gate(
                    extractor, llm_cache, url, text, parsed,
                    context=case_id, lock=cache_lock)
                for status in statuses.values():
                    counters[f"gate_{status}"] += 1
        row["llm_cached"] = bool(parsed is not None and error is None
                                 and url in llm_cache)
        for field in STABILITY_FIELDS:
            row[f"{field}_votes"] = ",".join(votes.get(field, []))
            row[f"{field}_stability"] = statuses.get(field, "not_gated")
        merge_llm_into_row(row, parsed, error)

        conduct, conduct_error = None, "not run"
        if args.conduct_pass and text and (extractor is not None or url in llm_cache):
            conduct, conduct_error, conduct_cached = cached_conduct_extract(
                extractor, llm_cache, url, text, context=case_id, lock=cache_lock)
            counters["conduct_cache_hits"] += int(conduct_cached)
            if conduct is None and extractor is not None:
                counters["conduct_errors"] += 1
        apply_conduct(row, conduct, conduct_error)

        relief, relief_error = None, "not run"
        if args.relief_pass and text and (extractor is not None or url in llm_cache):
            relief, relief_error, relief_cached = cached_relief_extract(
                extractor, llm_cache, url, text, context=case_id, lock=cache_lock)
            counters["relief_cache_hits"] += int(relief_cached)
            if relief is None and extractor is not None:
                counters["relief_errors"] += 1
        apply_relief(row, relief, relief_error,
                     resolution=resolve_relief_votes(llm_cache.get(url)))
        derive_review_flags(row)
        normalise_row(row)
        return row

    workers = max(1, min(args.workers, len(records)))
    print(f"Concurrency: {workers} worker(s)")
    rows = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, record): position
                   for position, record in enumerate(records)}
        done = 0
        for future in as_completed(futures):
            position = futures[future]
            try:
                rows[position] = future.result()
            except Exception as error:  # one bad case must not sink the run
                print(f"  ERROR on row {position}: {error}")
                rows[position] = {"case_id": "", "llm_status": f"failed: {error}"}
                counters["failed"] += 1
            done += 1
            # Checkpoint so an interrupted long run keeps the calls it paid for.
            if args.all and done % 100 == 0:
                print(f"  {done}/{len(records)} (cache hits {counters['cache_hits']})")
                with cache_lock:
                    save_llm_cache(args.llm_cache, dict(llm_cache))

    missing, llm_errors, cache_hits = (
        counters["missing"], counters["llm_errors"], counters["cache_hits"])
    save_llm_cache(args.llm_cache, llm_cache)

    extract = pd.DataFrame(rows)
    for column in extract.columns:
        if extract[column].dtype == object:
            extract[column] = extract[column].apply(
                lambda v: v[:CELL_CHAR_CAP] if isinstance(v, str) else v)

    out_path = args.out
    if not args.all:
        stem, ext = os.path.splitext(args.out)
        out_path = f"{stem}_sample{ext}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        extract.to_excel(writer, sheet_name="cases", index=False)
        build_dictionary(list(extract.columns)).to_excel(
            writer, sheet_name="data_dictionary", index=False)

    dictionary = write_data_dictionary(args.dictionary_out, list(extract.columns),
                                       corpus=frame, extract=extract)
    worksheets = write_reference_sets(args.reference_out, extract, size=args.reference_size)
    frozen = freeze_definitional_set(extract)
    print(f"Wrote {args.reference_out}: "
          + ", ".join(f"{k} n={len(v)}" for k, v in worksheets.items())
          + f", definitional_outcome_set n={len(frozen)}")
    undocumented = (dictionary["group"] == "UNDOCUMENTED").sum()
    print(f"Wrote {args.dictionary_out}: {len(dictionary)} field definitions"
          + (f" -- WARNING {undocumented} UNDOCUMENTED" if undocumented else ""))

    if not args.quiet:
        print_review(extract)

    spend = COST.total_cost()
    per_case = spend / len(extract) if len(extract) else 0
    print(f"\nLLM spend: ${spend:.4f} over {COST.calls} calls "
          f"(${per_case:.4f}/case; 2,385-case corpus projects to ${per_case * 2385:.2f})")

    print(f"\nWrote {out_path}: {len(extract)} rows x {len(extract.columns)} columns")
    if args.llm_cache:
        print(f"LLM cache: {cache_hits} hits, {len(llm_cache)} entries -> {args.llm_cache}")
    if missing:
        print(f"WARNING: {missing} rows had no readable HTML; text-derived fields are blank")
    if llm_errors:
        print(f"WARNING: {llm_errors} rows failed the LLM pass; they fall back to rule values")

    disagreements = [c for c in extract.columns if c.endswith("_agreement")]
    if disagreements:
        print("\nRule vs LLM agreement (fields with a real rule baseline):")
        for column in disagreements:
            field = column[:-len("_agreement")]
            if field in NO_RULE_BASELINE:
                continue
            frame = extract
            note = ""
            if field == "outcome" and "outcome_analysable" in extract.columns:
                # Procedural matters have no substantive winner, so a
                # disagreement there is not evidence about outcome quality.
                frame = extract[extract["outcome_analysable"] == True]
                note = f"  (analysable rows only, n={len(frame)})"
            counts = dict(frame[column].value_counts())
            if field in STRUCTURALLY_SPARSE and set(counts) <= {"both_empty"}:
                note = "   <- structurally absent, not a rule failure"
            print(f"  {field:<32} {counts}{note}")
        skipped = sorted(f for f in NO_RULE_BASELINE if f"{f}_agreement" in extract.columns)
        if skipped:
            print(f"  (no rule baseline, not compared: {', '.join(skipped)})")

    if "outcome" in extract.columns:
        analysable = extract[extract.get("outcome_analysable", True) == True]
        print(f"\noutcome (all {len(extract)}):", dict(extract["outcome"].value_counts()))
        print(f"outcome (analysable {len(analysable)}, Procedural excluded):",
              dict(analysable["outcome"].value_counts()))

    gate_counts = {k[len("gate_"):]: v for k, v in counters.items() if k.startswith("gate_")}
    if gate_counts:
        print("\nstability gate:", gate_counts)
        for field in STABILITY_FIELDS:
            column = f"{field}_stability"
            if column in extract.columns:
                print(f"  {field:<20}", dict(extract[column].value_counts()))

    if "review_flags" in extract.columns:
        flags = Counter()
        for value in extract["review_flags"].fillna(""):
            for flag in str(value).split("; "):
                if flag:
                    flags[flag] += 1
        if flags:
            print("review flags:", dict(flags.most_common()))


if __name__ == "__main__":
    main()
