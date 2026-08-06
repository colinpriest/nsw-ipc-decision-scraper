"""
Damages-breakdown extraction (spec v1, 2026-07-27 downstream request).

The combined extraction in `nsw_court_scraper.py` captures the LUMP SUM and two
damages heads (non-economic loss, future economic loss). The downstream CTP
simulator showed that this leaves ~73% of awards with an unexplained positive
residual, because the largest head — PAST economic loss — and the DEDUCTIONS
that make the lump sum a NET figure were never extracted.

This module adds a dedicated second LLM pass that extracts the full award
breakdown, its deductions, per-field provenance, and a reconciliation flag,
plus the P1 classification fields (accident mechanism, multi-label injury,
split WPI, workers-compensation overlap).

Design notes
------------
* It is a SEPARATE pass, not extra fields bolted onto CombinedSchema. The
  existing eight high-value fields are load-bearing for the consumer and must
  not regress; a focused prompt on a focused schema is also markedly better at
  the claimed-vs-allowed distinction that the spec calls out as the main trap.
* Nothing here imports from `nsw_court_scraper` — the dependency runs the other
  way — so this module stays importable and testable on its own.
* Every deterministic rule (status discipline, provenance defaults, gross
  derivation, reconciliation) lives in pure functions so it can be unit tested
  without the API. See `test_damages_extraction.py`.
* The reconciliation deliberately reports `insufficient data` rather than a
  flattering `yes` whenever the gross figure was itself derived from the sum of
  the heads — otherwise the identity would close by construction and the
  consumer's acceptance check would be meaningless.
"""

import os
import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Literal


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Reasoning effort for the damages pass. Higher than the main pass: this call is
# small, arithmetic-sensitive, and has to hold the claimed/allowed/refused
# distinction across a long quantum section.
DAMAGES_REASONING_EFFORT = os.getenv("NSW_DAMAGES_REASONING_EFFORT", "medium")

# Run the damages pass in the live extraction path. On by default.
DAMAGES_PASS_ENABLED = os.getenv("NSW_DAMAGES_PASS", "1") not in ("0", "false", "False", "")

# Case types the live damages pass runs for. The workbook the spec targets is
# CTP-only; widen with e.g. NSW_DAMAGES_CASE_TYPES="CTP,Workers Compensation".
DAMAGES_CASE_TYPES = tuple(
    t.strip() for t in os.getenv("NSW_DAMAGES_CASE_TYPES", "CTP").split(",") if t.strip()
)

# Char budget for the damages context window (see build_damages_context).
DAMAGES_CONTEXT_CHARS = int(os.getenv("NSW_DAMAGES_CONTEXT_CHARS", "120000"))

# Tolerance (AUD) for the reconciliation identities. Matches the spec's $1,000.
RECONCILE_TOLERANCE = float(os.getenv("NSW_DAMAGES_TOLERANCE", "1000"))


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------

class ProvenanceEnum(str, Enum):
    """How a money figure was obtained. The consumer excludes or down-weights
    `inferred`, so the distinction has to be honest: `stated` means the figure
    is in the decision, `derived` means we did arithmetic on figures that are."""
    STATED = "stated"
    DERIVED = "derived"
    INFERRED = "inferred"
    ABSENT = "absent"


class HeadStatusEnum(str, Enum):
    """Three-valued disposition of a damages head — the convention already used
    for non-economic / future economic loss, reused verbatim for every new
    money field. `Not addressed` must NEVER be collapsed to zero."""
    AWARDED = "Awarded"
    NIL = "Nil"
    NOT_ADDRESSED = "Not addressed"


class LumpSumBasisEnum(str, Enum):
    NET = "net of deductions"
    GROSS = "gross"
    UNCLEAR = "unclear"


class ReconciledEnum(str, Enum):
    YES = "yes"
    NO = "no"
    INSUFFICIENT = "insufficient data"


class AccidentMechanismEnum(str, Enum):
    VEHICLE_COLLISION = "vehicle collision"
    PEDESTRIAN_STRUCK = "pedestrian struck"
    MOTORCYCLIST = "motorcyclist"
    CYCLIST = "cyclist"
    PASSENGER = "passenger"
    SINGLE_VEHICLE = "single vehicle"
    OTHER = "other"
    UNCLEAR = "unclear"


class RoadRoleEnum(str, Enum):
    DRIVER = "driver"
    PASSENGER = "passenger"
    PEDESTRIAN = "pedestrian"
    MOTORCYCLIST = "motorcyclist"
    CYCLIST = "cyclist"
    OTHER = "other"


class InjuryCategoryEnum(str, Enum):
    """Multi-label injury taxonomy. MAIA assesses physical and psychiatric
    impairment separately for the >10% WPI threshold, so `psychiatric` is a
    peer category, never a runner-up to a physical one."""
    BRAIN_INJURY = "brain injury"
    SPINAL = "spinal"
    UPPER_LIMB = "upper limb"
    LOWER_LIMB = "lower limb"
    CHEST_OR_ABDOMINAL = "chest or abdominal"
    HEAD_OR_FACIAL = "head or facial"
    PSYCHIATRIC = "psychiatric"
    CHRONIC_PAIN = "chronic pain"
    SCARRING_OR_DISFIGUREMENT = "scarring or disfigurement"
    SOFT_TISSUE = "soft tissue"
    FATALITY = "fatality"
    OTHER = "other"


# ----------------------------------------------------------------------
# LLM schema
# ----------------------------------------------------------------------

class MoneyField(BaseModel):
    """A money figure together with how it was obtained and the text that
    establishes it. `amount` is EMPTY whenever there is no such figure — a
    missing figure is more useful to the consumer than a plausible guess."""
    amount: str = Field(description=(
        "The figure as a plain number: digits and at most one decimal point. "
        "NO $ sign, NO commas, NO words. EMPTY STRING if the decision does not "
        "support a figure. Never write 0 to mean 'unknown'."
    ))
    provenance: ProvenanceEnum = Field(description=(
        "'stated' = the figure appears verbatim in the decision; 'derived' = "
        "you computed it by arithmetic from figures that do appear (say so in "
        "the quote); 'inferred' = your judgement rather than the document's; "
        "'absent' = amount is empty. Use 'inferred' honestly — the consumer "
        "excludes inferred values, and a wrong number is worse than none."
    ))
    quote: str = Field(description=(
        "VERBATIM snippet (<=200 chars) copied from the decision that "
        "establishes this figure, or the arithmetic you performed when "
        "provenance is 'derived'. EMPTY when provenance is 'absent'."
    ))


class DamagesSchema(BaseModel):
    # ---- The heads of damage ----
    past_economic_loss: MoneyField = Field(description=(
        "PAST economic loss ALLOWED (past loss of earnings/wages to the date of "
        "assessment, including past superannuation if awarded as part of it). "
        "The amount the tribunal ALLOWED, not the amount claimed and not the "
        "total damages."
    ))
    past_economic_loss_status: HeadStatusEnum = Field(description=(
        "'Awarded' if a positive past economic loss was allowed; 'Nil' if the "
        "head was pressed and expressly refused or assessed at zero; "
        "'Not addressed' if past economic loss was never before the tribunal."
    ))

    non_economic_loss: MoneyField = Field(description=(
        "NON-ECONOMIC LOSS ALLOWED (general damages / pain and suffering). "
        "Independent re-extraction — do not copy any figure supplied to you."
    ))
    non_economic_loss_status: HeadStatusEnum

    future_economic_loss: MoneyField = Field(description=(
        "FUTURE economic loss ALLOWED (future loss of earning capacity and "
        "future superannuation), INCLUDING a buffer awarded for future economic "
        "loss. A buffer for something else (e.g. future treatment) does NOT "
        "belong here — put it in buffer_amount."
    ))
    future_economic_loss_status: HeadStatusEnum

    buffer_amount: MoneyField = Field(description=(
        "A buffer or global allowance you could NOT confidently assign to a "
        "named head (e.g. 'a buffer of 75,000 for future treatment'). MUST NOT "
        "duplicate an amount you already reported in another head — if the "
        "buffer IS the future economic loss award, report it there and leave "
        "this EMPTY."
    ))
    buffer_basis: str = Field(description=(
        "Short free text naming what the buffer is for, e.g. 'future treatment "
        "and care'. EMPTY if buffer_amount is empty."
    ))

    other_damages_heads: MoneyField = Field(description=(
        "The TOTAL of any other heads of DAMAGES allowed that are not "
        "non-economic loss, past economic loss or future economic loss — e.g. "
        "out-of-pocket expenses awarded as damages, gratuitous care damages, "
        "interest, past/future domestic assistance. EMPTY if there are none. "
        "Do NOT include treatment and care paid as STATUTORY BENEFITS (under "
        "MAIA 2017 those are not damages)."
    ))
    other_damages_heads_basis: str = Field(description=(
        "Short free text listing what other_damages_heads covers, e.g. "
        "'interest 12,430; out-of-pockets 3,110'. EMPTY if none."
    ))

    # ---- Gross vs net ----
    total_damages_gross: MoneyField = Field(description=(
        "Total damages BEFORE any deduction or reduction — the sum of the heads "
        "as assessed. Use 'stated' only if the decision itself states a total "
        "(e.g. 'assessed total damages at 476,407'). EMPTY if the decision "
        "gives no total and you cannot compute one from stated figures."
    ))
    lump_sum_net: MoneyField = Field(description=(
        "The amount actually PAYABLE to the claimant after every deduction and "
        "reduction — the settlement sum approved, or the judgment sum ordered. "
        "EMPTY if the decision does not state one."
    ))
    lump_sum_basis: LumpSumBasisEnum = Field(description=(
        "Is the payable sum you reported NET of deductions, GROSS (no deduction "
        "applies or none was made), or UNCLEAR from the decision?"
    ))

    # ---- Deductions and reductions ----
    contributory_negligence_percent: MoneyField = Field(description=(
        "The PERCENTAGE reduction for contributory negligence actually applied "
        "(a number 0-100, e.g. '20' for 20%). EMPTY if no finding of "
        "contributory negligence reduced the award. A percentage merely "
        "alleged or argued for is NOT a finding."
    ))
    contributory_negligence_amount: MoneyField = Field(description=(
        "The DOLLAR reduction for contributory negligence, if the decision "
        "states or lets you compute it. EMPTY otherwise."
    ))
    statutory_benefits_repaid: MoneyField = Field(description=(
        "The deduction for statutory benefits already paid — the insurer's "
        "entitlement to deduct under s 3.40 of the Motor Accident Injuries Act "
        "2017, or an equivalent repayment/credit taken out of the settlement. "
        "EMPTY if none."
    ))
    other_deductions: MoneyField = Field(description=(
        "The TOTAL of any other deduction from the payable sum: Medicare/"
        "Services Australia charge, Centrelink repayment, workers compensation "
        "s 151Z recovery, hospital charges. EMPTY if none. Legal costs ordered "
        "separately are NOT a deduction from damages."
    ))
    deductions_basis: str = Field(description=(
        "Short free text naming the deductions and their source, e.g. "
        "'s 3.40(1)(b) statutory benefits; Medicare charge'. EMPTY if there "
        "were no deductions."
    ))

    # ---- Statutory benefits (OUTSIDE the damages reconciliation) ----
    statutory_benefits_paid: MoneyField = Field(description=(
        "Total statutory benefits PAID to date (weekly payments plus treatment "
        "and care), where the decision quantifies them. Under MAIA 2017 these "
        "are statutory benefits, NOT damages — they sit outside the damages "
        "reconciliation. EMPTY if not quantified."
    ))
    treatment_and_care_paid: MoneyField = Field(description=(
        "Treatment and care expenses paid or ordered, where quantified. EMPTY "
        "if not quantified."
    ))
    weekly_statutory_benefit: MoneyField = Field(description=(
        "The weekly statutory benefit rate, as a weekly number. If several "
        "periods are stated, use the LATEST. EMPTY if not stated."
    ))

    # ---- Award breakdown prose (spec 4.2) ----
    award_breakdown_sentences: str = Field(description=(
        "TWO TO FOUR sentences, in the register the decision itself uses, "
        "stating the award breakdown WITH FIGURES: each head and its amount, "
        "any reduction for contributory negligence, any deduction, and the "
        "net sum payable. Write '$' amounts with thousands separators, e.g. "
        "'The Member assessed non-economic loss at $180,000, past economic "
        "loss at $64,500 and future economic loss at $250,000, a total of "
        "$494,500. After a deduction of $23,670.57 for statutory benefits "
        "paid, the sum payable was $470,829.43.' State ONLY figures the "
        "decision supports; if the decision gives no breakdown, state the "
        "total and say the decision did not apportion it. Do NOT name any "
        "party, Member, doctor or firm. ASCII only."
    ))

    # ---- Accident mechanism (spec 4.3) ----
    accident_mechanism: AccidentMechanismEnum = Field(description=(
        "How the accident happened. 'single vehicle' = the claimant's vehicle "
        "alone (run off road, hit an object). 'vehicle collision' = two or more "
        "vehicles. 'pedestrian struck'/'motorcyclist'/'cyclist' by the "
        "claimant's involvement. 'passenger' only when the claimant was a "
        "passenger and the mechanism is not otherwise classifiable. 'other' for "
        "a genuine motor-accident mechanism outside these; 'unclear' when the "
        "decision does not say. Do NOT guess."
    ))
    claimant_road_role: RoadRoleEnum = Field(description=(
        "What the claimant was doing at the time: driver / passenger / "
        "pedestrian / motorcyclist / cyclist / other."
    ))

    # ---- Multi-label injury (spec 4.4) ----
    injury_categories: List[InjuryCategoryEnum] = Field(description=(
        "EVERY injury category the decision establishes for this claimant — not "
        "first-match-wins. A claimant with a lumbar spine injury and an adjustment "
        "disorder gets BOTH 'spinal' and 'psychiatric'. Deduplicate."
    ))
    primary_injury_category: InjuryCategoryEnum = Field(description=(
        "The dominant category — the one driving the impairment and the award."
    ))
    has_psychiatric_injury: bool = Field(description=(
        "True if the decision establishes a psychiatric or psychological injury "
        "(diagnosed condition, psychiatric impairment assessment, or treatment "
        "for one). False if psychiatric injury is merely alleged and rejected, "
        "or not raised."
    ))

    # ---- Split WPI (spec 4.5) ----
    wpi_physical_percent: str = Field(description=(
        "The PHYSICAL whole person impairment percentage, ONLY if the decision "
        "states physical and psychiatric impairment SEPARATELY. Number only. "
        "EMPTY otherwise — do NOT split a combined figure yourself."
    ))
    wpi_psychiatric_percent: str = Field(description=(
        "The PSYCHIATRIC whole person impairment percentage, ONLY if separately "
        "stated. Number only. EMPTY otherwise. Note that a psychiatric "
        "impairment recited as not satisfying the 10% threshold IS a stated "
        "figure; a bare recitation of the statutory threshold is not."
    ))

    # ---- Workers-compensation overlap (spec 4.6) ----
    wc_overlap: Literal[0, 1, 2] = Field(description=(
        "0 = no workers compensation claim referred to at all; 1 = a parallel "
        "workers compensation claim is mentioned, or its payments are deducted, "
        "without shaping the damages reasoning; 2 = substantial interaction — "
        "s 151Z recovery, the WC claim shapes the assessment of economic loss "
        "or care, or the heads are adjusted for WC entitlements."
    ))

    # ---- Claim pathway (spec 5.5) ----
    is_fatality_or_dependency_claim: bool = Field(description=(
        "True if this is a death/dependency claim on the Compensation to "
        "Relatives pathway. Those claims do not have the ordinary heads, so "
        "the heads must be 'Not addressed' rather than 'Nil'."
    ))


# ----------------------------------------------------------------------
# System instruction
# ----------------------------------------------------------------------

DAMAGES_SYSTEM_INSTRUCTION = """\
You are a senior legal analyst reading a NSW Personal Injury Commission
decision to reconstruct the AWARD BREAKDOWN exactly as the tribunal made it.

Your output feeds a reconciliation:

    total damages (gross)
      - contributory negligence reduction
      - statutory benefits repaid
      - other deductions
      = the sum actually payable

so every figure has to sit in the right place for that identity to close.

NON-NEGOTIABLE RULES

1. ALLOWED, NOT CLAIMED. Decisions discuss amounts SOUGHT, amounts ALLOWED and
   amounts REFUSED. Report only what was ALLOWED. Watch the sentence shape
   "...past economic loss, and future economic loss, and assessed total damages
   at 476,407" - 476,407 is the TOTAL, not the past economic loss. If a
   sentence attaches one figure to several heads, that figure is a TOTAL; do
   not assign it to a single head.

2. DO NOT INFER TO FILL A GAP. An empty amount with status 'Not addressed' and
   provenance 'absent' is MORE useful than a plausible guess. A wrong number is
   worse than a missing one. Use provenance 'inferred' only when you really are
   exercising judgement, and expect that value to be discarded.

3. THE THREE-VALUED STATUS IS LOAD-BEARING:
     Awarded       - the head was assessed and an amount allowed.
     Nil           - the head was before the tribunal and NOTHING was allowed
                     (a genuine zero: refused, or assessed at zero).
     Not addressed - the head was NEVER before the tribunal (missing, NOT zero).
   Never collapse 'Not addressed' to zero.

4. BUFFERS AND GLOBAL ALLOWANCES. A buffer awarded FOR future economic loss is
   future economic loss - report it there. A buffer for anything else goes in
   buffer_amount with buffer_basis. Never let the same dollar appear twice.

5. FATALITY / DEPENDENCY CLAIMS follow the Compensation to Relatives pathway
   and do not carry the ordinary heads: mark them 'Not addressed', not 'Nil',
   and set is_fatality_or_dependency_claim true.

6. DEDUCTIONS. The sum payable is usually NET. Look specifically for:
     - "Pursuant to s 3.40(1)(b) of the Motor Accident Injuries Act 2017 the
       insurer is entitled to deduct the sum of $X" -> statutory_benefits_repaid
     - a percentage reduction for contributory negligence actually FOUND (not
       merely alleged) -> contributory_negligence_percent, and the dollar
       reduction if stated -> contributory_negligence_amount
     - Medicare / Services Australia / Centrelink / s 151Z workers compensation
       recovery -> other_deductions
   Legal costs ordered separately are NOT a deduction from damages.

7. TREATMENT AND CARE. Under the Motor Accident Injuries Act 2017 treatment and
   care are STATUTORY BENEFITS, not damages. Keep them out of the damages heads;
   report them in statutory_benefits_paid / treatment_and_care_paid.

8. MONEY FORMAT. Plain numbers: digits and at most one decimal point. No $, no
   commas, no words. The one exception is award_breakdown_sentences, which is
   prose and DOES use "$180,000" formatting.

9. PROVENANCE ON EVERY MONEY FIELD. 'stated' = the figure is verbatim in the
   decision; 'derived' = you computed it by arithmetic from figures that are
   (put the arithmetic in the quote); 'inferred' = judgement; 'absent' = no
   amount. Copy the establishing text verbatim into `quote`, at most 200 chars.

Search the WHOLE document before deciding a figure is absent - the quantum
assessment and the orders are usually near the end.
"""


# ----------------------------------------------------------------------
# Context building
# ----------------------------------------------------------------------

# Where the award breakdown lives. Used to build a damages-focused window when
# a decision overflows the context budget, so the quantum section survives even
# if the narrative does not.
_DAMAGES_KEYWORDS = (
    "past economic loss", "future economic loss", "economic loss",
    "non-economic loss", "non economic loss", "general damages",
    "loss of earning capacity", "buffer", "gratuitous", "griffiths",
    "total damages", "damages are assessed", "assessed at", "settlement sum",
    "s 3.40", "section 3.40", "statutory benefits", "entitled to deduct",
    "contributory negligence", "medicare", "centrelink", "151z",
    "orders", "determination", "i determine", "i assess", "interest",
    "superannuation", "out of pocket", "treatment and care",
)
_DAMAGES_BEFORE, _DAMAGES_AFTER = 2500, 9000


def _merge_ranges(ranges):
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


def build_damages_context(text, cap=None):
    """Return the text to send to the damages pass.

    Short decisions go whole. Long ones are reduced to the tail (where the
    orders live) plus windows around every damages keyword, anchored on the
    LAST occurrence first — the operative assessment is almost always the later
    mention, and the earlier ones are usually the parties' submissions, which
    are exactly the claimed-not-allowed figures we must not pick up.
    """
    text = text or ""
    cap = DAMAGES_CONTEXT_CHARS if cap is None else cap
    if len(text) <= cap:
        return text
    lowered = text.lower()
    n = len(text)

    # The orders/determination section is at the very end; take it first, and
    # size it so it never needs trimming (windows are trimmed from the BACK,
    # which would drop exactly the orders we came for). Capped at a third of
    # the budget so the keyword windows still get a share.
    tail = min(25000, max(4000, cap // 3))
    candidates = [(max(0, n - tail), n)]
    for kw in _DAMAGES_KEYWORDS:
        last = lowered.rfind(kw)
        if last == -1:
            continue
        candidates.append((max(0, last - _DAMAGES_BEFORE), min(n, last + _DAMAGES_AFTER)))
        first = lowered.find(kw)
        if first != -1 and abs(last - first) > 8000:
            candidates.append((max(0, first - _DAMAGES_BEFORE), min(n, first + _DAMAGES_AFTER)))
    # A little of the opening, for the accident mechanism and the road role.
    candidates.append((0, 12000))

    accepted, total = [], 0
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
# Deterministic normalisation
# ----------------------------------------------------------------------

_MONEY_SENTINELS = {
    "", "nil", "none", "n/a", "na", "not stated", "not addressed",
    "not applicable", "unknown", "-",
}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def to_money(val):
    """Coerce an LLM money string to a clean numeric string, or "" .

    Mirrors nsw_court_scraper.coerce_money (kept local so this module has no
    dependency on the scraper); test_damages_extraction asserts they agree.
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in _MONEY_SENTINELS:
        return ""
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    m = _NUMBER_RE.search(s)
    if not m:
        return ""
    out = m.group(0)
    # A bare "0" is a real zero here (a Nil head), unlike the WPI case.
    return out


def to_float(val):
    s = to_money(val)
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _money_of(mf):
    """(amount_str, provenance_str, quote) from a MoneyField-ish object."""
    if mf is None:
        return "", ProvenanceEnum.ABSENT.value, ""
    amount = to_money(getattr(mf, "amount", ""))
    prov = getattr(mf, "provenance", None)
    prov = getattr(prov, "value", prov) or ProvenanceEnum.ABSENT.value
    quote = (getattr(mf, "quote", "") or "").strip()[:200]
    if amount == "":
        return "", ProvenanceEnum.ABSENT.value, quote
    if prov == ProvenanceEnum.ABSENT.value:
        # A figure with 'absent' provenance is a contradiction; the safe reading
        # is that the model produced it without anchoring it to the text.
        prov = ProvenanceEnum.INFERRED.value
    return amount, prov, quote


def _enum_value(v, default=""):
    return getattr(v, "value", v) if v is not None else default


def _drop_zero(amount, provenance):
    """Deductions and reductions are 'null if none' per the spec, so a model
    that says 0 is saying there is none. Arithmetically identical, but it keeps
    the column honest: a 0 would otherwise read as a quantified nil finding."""
    if to_float(amount) == 0:
        return "", ProvenanceEnum.ABSENT.value
    return amount, provenance


def apply_head_status(amount, provenance, status, *, fatality=False):
    """Enforce the status/amount contract (spec acceptance criterion 4).

    - 'Not addressed'  -> amount null, provenance 'absent'
    - 'Nil'            -> amount 0, provenance kept ('absent' becomes 'derived',
                          since the zero follows from the refusal finding)
    - 'Awarded'        -> amount kept; an awarded head with no amount is a
                          contradiction and is reported as an issue.

    A fatality/dependency claim never carries an ordinary head, so 'Nil' is
    rewritten to 'Not addressed' (spec 5.5).

    Returns (amount, provenance, status, issues).
    """
    issues = []
    status = _enum_value(status, HeadStatusEnum.NOT_ADDRESSED.value)

    if fatality and status == HeadStatusEnum.NIL.value:
        status = HeadStatusEnum.NOT_ADDRESSED.value
        issues.append("fatality claim: Nil rewritten to Not addressed")

    if status == HeadStatusEnum.NOT_ADDRESSED.value:
        if amount not in ("", None):
            issues.append(f"Not addressed head carried amount {amount}; dropped")
        return "", ProvenanceEnum.ABSENT.value, status, issues

    if status == HeadStatusEnum.NIL.value:
        if amount not in ("", None, "0", "0.0") and to_float(amount):
            issues.append(f"Nil head carried non-zero amount {amount}; forced to 0")
        prov = provenance if provenance != ProvenanceEnum.ABSENT.value else ProvenanceEnum.DERIVED.value
        return "0", prov, status, issues

    # Awarded
    if amount in ("", None):
        issues.append("Awarded head has no amount")
        return "", ProvenanceEnum.ABSENT.value, status, issues
    if to_float(amount) == 0:
        # Awarded-with-zero is really a Nil finding.
        issues.append("Awarded head with amount 0; treated as Nil")
        return "0", provenance, HeadStatusEnum.NIL.value, issues
    return amount, provenance, status, issues


def _sum_known(values):
    """Sum a list of (amount_str, known_bool). Returns (total, all_known)."""
    total = 0.0
    all_known = True
    for amount, known in values:
        if not known:
            all_known = False
            continue
        f = to_float(amount)
        total += f if f is not None else 0.0
    return total, all_known


def _fmt(n):
    if n is None:
        return ""
    return f"{n:.2f}".rstrip("0").rstrip(".") if n != int(n) else str(int(n))


def normalise_damages(parsed, *, existing=None):
    """Turn a parsed DamagesSchema into flat CSV columns + a sidecar record.

    `existing` is the trusted flat row already on the record (`Lump Sum`,
    `Non-Economic Loss`, `Future Economic Loss` and their statuses). Those
    columns are load-bearing for the consumer and are NEVER overwritten here:
    the existing lump sum is used as the net figure in the gross/net identity,
    and any disagreement with this pass's independent reading is recorded in
    `Damages Notes` rather than resolved silently.

    Returns (flat, sidecar, issues).
    """
    existing = existing or {}
    existing_lump_sum = existing.get("Lump Sum", "")
    issues = []
    quotes = {}
    fatality = bool(getattr(parsed, "is_fatality_or_dependency_claim", False))

    def head(name, status_attr):
        amount, prov, quote = _money_of(getattr(parsed, name, None))
        amount, prov, status, hissues = apply_head_status(
            amount, prov, getattr(parsed, status_attr, None), fatality=fatality)
        quotes[name] = quote
        issues.extend(f"{name}: {m}" for m in hissues)
        return amount, prov, status

    nel, nel_prov, nel_status = head("non_economic_loss", "non_economic_loss_status")
    pel, pel_prov, pel_status = head("past_economic_loss", "past_economic_loss_status")
    fel, fel_prov, fel_status = head("future_economic_loss", "future_economic_loss_status")

    buffer_amt, buffer_prov, quotes["buffer_amount"] = _money_of(getattr(parsed, "buffer_amount", None))
    other_amt, other_prov, quotes["other_damages_heads"] = _money_of(getattr(parsed, "other_damages_heads", None))

    cn_pct, cn_pct_prov, quotes["contributory_negligence_percent"] = _money_of(
        getattr(parsed, "contributory_negligence_percent", None))
    cn_amt, cn_amt_prov, quotes["contributory_negligence_amount"] = _money_of(
        getattr(parsed, "contributory_negligence_amount", None))
    sbr, sbr_prov, quotes["statutory_benefits_repaid"] = _money_of(
        getattr(parsed, "statutory_benefits_repaid", None))
    odd, odd_prov, quotes["other_deductions"] = _money_of(getattr(parsed, "other_deductions", None))

    cn_pct, cn_pct_prov = _drop_zero(cn_pct, cn_pct_prov)
    cn_amt, cn_amt_prov = _drop_zero(cn_amt, cn_amt_prov)
    sbr, sbr_prov = _drop_zero(sbr, sbr_prov)
    odd, odd_prov = _drop_zero(odd, odd_prov)

    gross, gross_prov, quotes["total_damages_gross"] = _money_of(
        getattr(parsed, "total_damages_gross", None))
    net, net_prov, quotes["lump_sum_net"] = _money_of(getattr(parsed, "lump_sum_net", None))

    sbp, sbp_prov, quotes["statutory_benefits_paid"] = _money_of(
        getattr(parsed, "statutory_benefits_paid", None))
    tcp, tcp_prov, quotes["treatment_and_care_paid"] = _money_of(
        getattr(parsed, "treatment_and_care_paid", None))
    wsb, wsb_prov, quotes["weekly_statutory_benefit"] = _money_of(
        getattr(parsed, "weekly_statutory_benefit", None))

    # Contributory-negligence sanity: a percent outside 0-100 is a unit error.
    cn_pct_f = to_float(cn_pct)
    if cn_pct_f is not None and not (0 <= cn_pct_f <= 100):
        issues.append(f"contributory negligence percent {cn_pct} outside 0-100; dropped")
        cn_pct, cn_pct_prov = "", ProvenanceEnum.ABSENT.value
        cn_pct_f = None

    # ---- Deductions total ----
    deduction_parts = [cn_amt, sbr, odd]
    deductions_total = sum(to_float(x) or 0.0 for x in deduction_parts)
    has_any_deduction = any(to_float(x) is not None for x in deduction_parts)

    # ---- Gross derivation, if not stated ----
    # Priority: stated > net + deductions > sum of heads. The last one is
    # tracked separately because it makes the head identity close by
    # construction, so it must NOT count as a reconciliation.
    gross_source = "stated" if gross else ""
    lump_for_identity = to_float(existing_lump_sum)
    if lump_for_identity is None:
        lump_for_identity = to_float(net)

    heads_pairs = [
        (nel, nel_status != HeadStatusEnum.AWARDED.value or nel != ""),
        (pel, pel_status != HeadStatusEnum.AWARDED.value or pel != ""),
        (fel, fel_status != HeadStatusEnum.AWARDED.value or fel != ""),
        (buffer_amt, True),
        (other_amt, True),
    ]
    heads_total, heads_known = _sum_known(heads_pairs)

    if not gross:
        if lump_for_identity is not None and has_any_deduction:
            gross = _fmt(lump_for_identity + deductions_total)
            gross_prov = ProvenanceEnum.DERIVED.value
            gross_source = "net plus deductions"
        elif lump_for_identity is not None and not has_any_deduction:
            gross = _fmt(lump_for_identity)
            gross_prov = ProvenanceEnum.DERIVED.value
            gross_source = "net, no deductions found"
        elif heads_known and heads_total > 0:
            gross = _fmt(heads_total)
            gross_prov = ProvenanceEnum.DERIVED.value
            gross_source = "sum of heads"

    # ---- Reconciliation: gross vs the sum of the heads ----
    gross_f = to_float(gross)
    if gross_f is None or not heads_known or gross_source == "sum of heads":
        reconciled = ReconciledEnum.INSUFFICIENT.value
        residual = ""
    else:
        resid = gross_f - heads_total
        residual = _fmt(round(resid, 2))
        reconciled = (ReconciledEnum.YES.value if abs(resid) <= RECONCILE_TOLERANCE
                      else ReconciledEnum.NO.value)

    # ---- Reconciliation: the payable sum vs gross minus deductions ----
    if gross_f is None or lump_for_identity is None or gross_source.startswith("net"):
        net_reconciled = ReconciledEnum.INSUFFICIENT.value
        net_residual = ""
    else:
        nresid = lump_for_identity - (gross_f - deductions_total)
        net_residual = _fmt(round(nresid, 2))
        net_reconciled = (ReconciledEnum.YES.value if abs(nresid) <= RECONCILE_TOLERANCE
                          else ReconciledEnum.NO.value)

    # ---- Lump-sum basis cross-check (spec 3.3) ----
    # ---- Lump-sum basis (spec 3.3): decided by arithmetic, not self-report ----
    # The consumer's reconciliation hinges on whether the delivered `Lump Sum`
    # is net or gross, so where a STATED gross exists we answer it by testing
    # both identities against the delivered figure. The model's own answer is
    # only a fallback (and is kept in the sidecar either way), because it
    # describes the figure IT found, which need not be the delivered column.
    basis_model = _enum_value(getattr(parsed, "lump_sum_basis", None),
                              LumpSumBasisEnum.UNCLEAR.value)
    basis = basis_model
    basis_source = "model"
    existing_f = to_float(existing_lump_sum)
    net_f = to_float(net)
    if existing_f is not None and gross_source == "stated" and gross_f is not None:
        basis_source = "arithmetic"
        if abs(existing_f - (gross_f - deductions_total)) <= RECONCILE_TOLERANCE:
            basis = LumpSumBasisEnum.NET.value
        elif abs(existing_f - gross_f) <= RECONCILE_TOLERANCE:
            # Identical to the net test when there is nothing to deduct.
            basis = (LumpSumBasisEnum.GROSS.value if deductions_total > 0
                     else LumpSumBasisEnum.NET.value)
        else:
            basis = LumpSumBasisEnum.UNCLEAR.value

    # Provenance of the DELIVERED Lump Sum column: whichever figure this pass
    # found that it matches. A figure matching neither is not anchored to
    # anything we read, so it is reported as inferred rather than stated.
    if existing_f is None:
        lump_prov = ProvenanceEnum.ABSENT.value
    elif net_f is not None and abs(existing_f - net_f) <= RECONCILE_TOLERANCE:
        lump_prov = net_prov
    elif gross_f is not None and abs(existing_f - gross_f) <= RECONCILE_TOLERANCE:
        lump_prov = gross_prov
    else:
        lump_prov = ProvenanceEnum.INFERRED.value

    # ---- Disagreement with the trusted columns (spec acceptance criterion 2) ----
    # Recorded, never resolved: the consumer measures extractor accuracy against
    # these columns, so quietly conforming to them would destroy the signal.
    for label, recheck, current in (
        ("Lump Sum", net, existing.get("Lump Sum", "")),
        ("Non-Economic Loss", nel, existing.get("Non-Economic Loss", "")),
        ("Future Economic Loss", fel, existing.get("Future Economic Loss", "")),
    ):
        a, b = to_float(recheck), to_float(current)
        if a is not None and b is not None and abs(a - b) > RECONCILE_TOLERANCE:
            issues.append(f"{label} disagreement: existing {current} vs damages pass {recheck}")

    injury_cats = [_enum_value(c) for c in (getattr(parsed, "injury_categories", None) or [])]
    injury_cats = sorted(dict.fromkeys([c for c in injury_cats if c]))

    wpi_phys = to_money(getattr(parsed, "wpi_physical_percent", ""))
    wpi_psych = to_money(getattr(parsed, "wpi_psychiatric_percent", ""))
    for label, val in (("physical", wpi_phys), ("psychiatric", wpi_psych)):
        f = to_float(val)
        if f is not None and not (0 <= f <= 100):
            issues.append(f"{label} WPI {val} outside 0-100; dropped")
            if label == "physical":
                wpi_phys = ""
            else:
                wpi_psych = ""

    flat = {
        # --- P0 3.1 past economic loss ---
        "Past Economic Loss": pel,
        "Past Economic Loss Status": pel_status,
        # --- P0 3.2 deductions ---
        "Contributory Negligence Percent": cn_pct,
        "Contributory Negligence Amount": cn_amt,
        "Statutory Benefits Repaid": sbr,
        "Other Deductions": odd,
        "Deductions Basis": (getattr(parsed, "deductions_basis", "") or "").strip(),
        # --- P0 3.3 gross vs net ---
        "Total Damages Gross": gross,
        "Lump Sum Basis": basis,
        # --- buffers / other heads (spec 5.4) ---
        "Buffer Amount": buffer_amt,
        "Buffer Basis": (getattr(parsed, "buffer_basis", "") or "").strip(),
        "Other Damages Heads": other_amt,
        "Other Damages Heads Basis": (getattr(parsed, "other_damages_heads_basis", "") or "").strip(),
        # --- independent re-extraction of the two trusted heads (criterion 2) ---
        "Non-Economic Loss (Recheck)": nel,
        "Non-Economic Loss Status (Recheck)": nel_status,
        "Future Economic Loss (Recheck)": fel,
        "Future Economic Loss Status (Recheck)": fel_status,
        # The amount actually payable after every deduction. Kept separate from
        # `Lump Sum`, which the arithmetic shows is often the GROSS sum.
        "Net Sum Payable": net,
        "Net Sum Payable Provenance": net_prov,
        # --- P0 3.4 provenance ---
        "Lump Sum Provenance": lump_prov,
        "Non-Economic Loss Provenance": nel_prov,
        "Future Economic Loss Provenance": fel_prov,
        "Past Economic Loss Provenance": pel_prov,
        "Total Damages Gross Provenance": gross_prov,
        "Contributory Negligence Percent Provenance": cn_pct_prov,
        "Contributory Negligence Amount Provenance": cn_amt_prov,
        "Statutory Benefits Repaid Provenance": sbr_prov,
        "Other Deductions Provenance": odd_prov,
        "Buffer Amount Provenance": buffer_prov,
        "Other Damages Heads Provenance": other_prov,
        # --- P0 3.5 reconciliation ---
        "Damages Reconciled": reconciled,
        "Damages Residual": residual,
        "Net Reconciled": net_reconciled,
        "Net Residual": net_residual,
        "Damages Gross Derivation": gross_source,
        # --- P1 4.1 the three previously-empty columns ---
        "Statutory Benefits Paid": sbp,
        "Statutory Benefits Paid Provenance": sbp_prov,
        "Treatment And Care Paid": tcp,
        "Treatment And Care Paid Provenance": tcp_prov,
        "Weekly Statutory Benefit": wsb,
        "Weekly Statutory Benefit Provenance": wsb_prov,
        # --- P1 4.2 figures in prose ---
        "Award Breakdown": (getattr(parsed, "award_breakdown_sentences", "") or "").strip(),
        # --- P1 4.3 accident mechanism ---
        "Accident Mechanism": _enum_value(getattr(parsed, "accident_mechanism", None),
                                          AccidentMechanismEnum.UNCLEAR.value),
        "Claimant Road Role": _enum_value(getattr(parsed, "claimant_road_role", None),
                                          RoadRoleEnum.OTHER.value),
        # --- P1 4.4 multi-label injury ---
        "Injury Categories": " | ".join(injury_cats),
        "Primary Injury Category": _enum_value(getattr(parsed, "primary_injury_category", None),
                                               InjuryCategoryEnum.OTHER.value),
        "Has Psychiatric Injury": "Yes" if getattr(parsed, "has_psychiatric_injury", False) else "No",
        # --- P1 4.5 split WPI ---
        "WPI Physical %": wpi_phys,
        "WPI Physical % Provenance": ProvenanceEnum.STATED.value if wpi_phys else ProvenanceEnum.ABSENT.value,
        "WPI Psychiatric %": wpi_psych,
        "WPI Psychiatric % Provenance": ProvenanceEnum.STATED.value if wpi_psych else ProvenanceEnum.ABSENT.value,
        # --- P1 4.6 WC overlap ---
        "WC Overlap": str(getattr(parsed, "wc_overlap", 0) or 0),
        # --- claim pathway ---
        "Fatality Or Dependency Claim": "Yes" if fatality else "No",
        "Damages Notes": "; ".join(issues)[:1000],
    }

    sidecar = {
        "quotes": quotes,
        "issues": issues,
        "gross_derivation": gross_source,
        "lump_sum_basis_model": basis_model,
        "lump_sum_basis_source": basis_source,
        "deductions_total": _fmt(deductions_total) if has_any_deduction else "",
        "heads_total": _fmt(heads_total) if heads_known else "",
        "heads_known": heads_known,
    }
    return flat, sidecar, issues


def compose_description_with_figures(description, award_breakdown):
    """Spec 4.2: a parallel Description that keeps the money in.

    `Description` itself is load-bearing for the consumer and is left alone;
    this appends the award breakdown sentences so the downstream generator has
    a template for quantifying an award instead of only discussing one.
    """
    desc = (description or "").strip()
    breakdown = (award_breakdown or "").strip()
    if not breakdown:
        return desc
    if not desc:
        return breakdown
    joiner = " " if desc.endswith((".", "!", "?")) else ". "
    return f"{desc}{joiner}{breakdown}"


# Flat column names contributed by this module, in output order. Consumed by
# nsw_court_scraper.RESULT_FIELDS.
DAMAGES_FIELDS = [
    "Past Economic Loss", "Past Economic Loss Status",
    "Contributory Negligence Percent", "Contributory Negligence Amount",
    "Statutory Benefits Repaid", "Other Deductions", "Deductions Basis",
    "Total Damages Gross", "Lump Sum Basis",
    "Buffer Amount", "Buffer Basis",
    "Other Damages Heads", "Other Damages Heads Basis",
    "Non-Economic Loss (Recheck)", "Non-Economic Loss Status (Recheck)",
    "Future Economic Loss (Recheck)", "Future Economic Loss Status (Recheck)",
    "Net Sum Payable", "Net Sum Payable Provenance",
    "Lump Sum Provenance",
    "Non-Economic Loss Provenance", "Future Economic Loss Provenance",
    "Past Economic Loss Provenance", "Total Damages Gross Provenance",
    "Contributory Negligence Percent Provenance",
    "Contributory Negligence Amount Provenance",
    "Statutory Benefits Repaid Provenance", "Other Deductions Provenance",
    "Buffer Amount Provenance", "Other Damages Heads Provenance",
    "Damages Reconciled", "Damages Residual",
    "Net Reconciled", "Net Residual", "Damages Gross Derivation",
    "Statutory Benefits Paid", "Statutory Benefits Paid Provenance",
    "Treatment And Care Paid", "Treatment And Care Paid Provenance",
    "Weekly Statutory Benefit", "Weekly Statutory Benefit Provenance",
    "Award Breakdown", "Description With Figures",
    "Accident Mechanism", "Claimant Road Role",
    "Injury Categories", "Primary Injury Category", "Has Psychiatric Injury",
    "WPI Physical %", "WPI Physical % Provenance",
    "WPI Psychiatric %", "WPI Psychiatric % Provenance",
    "WC Overlap", "Fatality Or Dependency Claim",
    "Damages Extraction Status", "Damages Notes",
]

# Columns whose value is legitimately SIGNED. The shared coercion in
# nsw_court_scraper rejects negatives on purpose (an age or an award is never
# negative), which would silently blank exactly the deduction-heavy rows the
# consumer cares about, so the workbook must coerce these separately.
DAMAGES_SIGNED_FIELDS = {"Damages Residual", "Net Residual"}

# Numeric columns among the above — used by the workbook export.
DAMAGES_NUMERIC_FIELDS = [
    "Past Economic Loss", "Contributory Negligence Percent",
    "Contributory Negligence Amount", "Statutory Benefits Repaid",
    "Other Deductions", "Total Damages Gross", "Buffer Amount",
    "Other Damages Heads", "Non-Economic Loss (Recheck)",
    "Future Economic Loss (Recheck)", "Net Sum Payable",
    "Damages Residual", "Net Residual",
    "Statutory Benefits Paid", "Treatment And Care Paid",
    "Weekly Statutory Benefit",
    "WPI Physical %", "WPI Psychiatric %", "WC Overlap",
]

# Money fields that must carry a provenance value (acceptance criterion 5).
MONEY_PROVENANCE_PAIRS = [
    ("Lump Sum", "Lump Sum Provenance"),
    ("Net Sum Payable", "Net Sum Payable Provenance"),
    ("Non-Economic Loss", "Non-Economic Loss Provenance"),
    ("Future Economic Loss", "Future Economic Loss Provenance"),
    ("Past Economic Loss", "Past Economic Loss Provenance"),
    ("Total Damages Gross", "Total Damages Gross Provenance"),
    ("Contributory Negligence Percent", "Contributory Negligence Percent Provenance"),
    ("Contributory Negligence Amount", "Contributory Negligence Amount Provenance"),
    ("Statutory Benefits Repaid", "Statutory Benefits Repaid Provenance"),
    ("Other Deductions", "Other Deductions Provenance"),
    ("Buffer Amount", "Buffer Amount Provenance"),
    ("Other Damages Heads", "Other Damages Heads Provenance"),
    ("Statutory Benefits Paid", "Statutory Benefits Paid Provenance"),
    ("Treatment And Care Paid", "Treatment And Care Paid Provenance"),
    ("Weekly Statutory Benefit", "Weekly Statutory Benefit Provenance"),
]

# Head status columns and their amount columns (acceptance criterion 4).
STATUS_AMOUNT_PAIRS = [
    ("Past Economic Loss Status", "Past Economic Loss"),
    ("Non-Economic Loss Status", "Non-Economic Loss"),
    ("Future Economic Loss Status", "Future Economic Loss"),
    ("Non-Economic Loss Status (Recheck)", "Non-Economic Loss (Recheck)"),
    ("Future Economic Loss Status (Recheck)", "Future Economic Loss (Recheck)"),
]


def empty_damages_row(status="not run"):
    """Flat defaults so every row carries the new columns, with statuses at the
    honest 'we do not know' value rather than a fabricated zero."""
    row = {f: "" for f in DAMAGES_FIELDS}
    row["Past Economic Loss Status"] = HeadStatusEnum.NOT_ADDRESSED.value
    row["Damages Reconciled"] = ReconciledEnum.INSUFFICIENT.value
    row["Net Reconciled"] = ReconciledEnum.INSUFFICIENT.value
    row["Lump Sum Basis"] = LumpSumBasisEnum.UNCLEAR.value
    for _, prov_col in MONEY_PROVENANCE_PAIRS:
        row[prov_col] = ProvenanceEnum.ABSENT.value
    row["WPI Physical % Provenance"] = ProvenanceEnum.ABSENT.value
    row["WPI Psychiatric % Provenance"] = ProvenanceEnum.ABSENT.value
    row["Damages Extraction Status"] = status
    return row


def damages_row_from_parsed(parsed, *, existing=None, description=""):
    """Full flat row for a successful damages pass, including the composed
    Description With Figures. Returns (flat, sidecar, issues)."""
    flat, sidecar, issues = normalise_damages(parsed, existing=existing)
    row = empty_damages_row(status="ok")
    row.update(flat)
    row["Description With Figures"] = compose_description_with_figures(
        description, row.get("Award Breakdown", ""))
    row["Damages Extraction Status"] = "ok"
    return row, sidecar, issues
