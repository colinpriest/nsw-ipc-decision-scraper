"""
WPI resolution: work out the impairment percentage a decision actually supports
when the source holds MORE THAN ONE whole-person-impairment figure.

Why this exists
---------------
`find_wpi_candidates` in nsw_court_scraper is deliberately strict: it fills the
WPI field only when the source contains exactly ONE distinct non-zero value, and
bails otherwise, because a regex picking between several numbers gets the wrong
one often enough to be dangerous. That is the right instinct and the wrong
outcome, because it throws away three genuinely different situations:

  * COMPONENTS with a stated total  - "neck 5%, back 5%, ankle 7% ... these
    combined give a 16% WPI". Nobody disagreed; one assessor added up his own
    findings. The total is right there.
  * COMPONENTS with no stated total - the total is computable.
  * COMPETING ASSESSMENTS           - Dr A says 14%, Dr B says 19%, and the
    tribunal never picked. A central estimate beats a blank.

Telling those apart is a CLASSIFICATION problem, not a pattern-matching one:
"a combined 7% WPI ... assessing the claimant with a total of 9% WPI" contains
two aggregate-sounding phrases and only one answer. So the LLM classifies each
mention and the deterministic ladder in `resolve_wpi` does the arithmetic and
the selection - the judgement is the model's, the maths is testable.

Two domain rules drive the arithmetic:

1. WPI components COMBINE, they do not add. The AMA Guides Combined Values
   Chart applies A + B(1 - A) from the largest component down. Summing
   overstates, badly, once the values get large (five 20% components sum to
   100% but combine to 67%).
2. Physical and psychiatric impairment are assessed SEPARATELY under the Motor
   Accident Guidelines and are NOT combined with each other. Where a decision
   states both, the governing figure is the HIGHER. (Measured on the 43
   workbook rows that state both: the accepted WPI equals max(physical,
   psychiatric) on 88% and their sum on 23%.)

Provenance is never overstated: a figure copied from the decision is `stated`,
one we computed by combining components is `derived`, and a central estimate of
competing assessments is `inferred` - which the downstream consumer excludes.
"""

import os
import re
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

WPI_RESOLUTION_REASONING_EFFORT = os.getenv("NSW_WPI_RESOLUTION_EFFORT", "medium")
WPI_RESOLUTION_ENABLED = os.getenv("NSW_WPI_RESOLUTION", "1") not in ("0", "false", "False", "")
# Central-estimate rule for competing assessments: median (robust to a single
# outlier expert) or mean. Median == mean for the common two-assessment case.
WPI_CENTRAL_ESTIMATE = os.getenv("NSW_WPI_CENTRAL_ESTIMATE", "median").strip().lower()


# ----------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------

class WpiKindEnum(str, Enum):
    """What a WPI number in the text actually IS."""
    COMPONENT = "component"
    ASSESSOR_TOTAL = "assessor total"
    MAS_CERTIFICATE = "MAS certificate"
    TRIBUNAL_FINDING = "tribunal finding"
    THRESHOLD_RECITAL = "threshold recital"
    CLAIMED_OR_REJECTED = "claimed or rejected"
    OTHER = "other"


class BodySystemEnum(str, Enum):
    PHYSICAL = "physical"
    PSYCHIATRIC = "psychiatric"
    COMBINED = "combined"
    UNCLEAR = "unclear"


class ThresholdFindingEnum(str, Enum):
    """The legally operative fact under s 4.11 of the Motor Accident Injuries
    Act 2017: non-economic loss is available only where impairment EXCEEDS 10%.
    The decision often settles this without ever stating a percentage."""
    ABOVE = "above 10%"
    BELOW = "not above 10%"
    NONE = "not determined"


class WpiProvenanceEnum(str, Enum):
    STATED = "stated"
    DERIVED = "derived"
    INFERRED = "inferred"
    ABSENT = "absent"


# ----------------------------------------------------------------------
# LLM schema
# ----------------------------------------------------------------------

class WpiMention(BaseModel):
    value: str = Field(description=(
        "The percentage as a plain number, e.g. '16' or '7.5'. No % sign."
    ))
    kind: WpiKindEnum = Field(description=(
        "'component' = one body part / one impairment element that forms part of "
        "a larger assessment (e.g. 'DRE II Neck Impairment, 5%').\n"
        "'assessor total' = a doctor's OWN overall figure for the claimant, "
        "whether stated directly or as the combination of their components "
        "(e.g. 'these combined give a 16% WPI', 'Dr Bodel assessed a 19% WPI').\n"
        "'MAS certificate' = a figure certified by a Medical Assessor or a MAS "
        "combined certificate. This OUTRANKS a party's medico-legal report.\n"
        "'tribunal finding' = the Member's OWN finding or express acceptance of a "
        "figure as the basis of the award.\n"
        "'threshold recital' = the statutory 10% bar, NOT a finding about this "
        "claimant (e.g. 'injuries do not exceed the threshold of 10% WPI', "
        "'conceded greater than 10% WPI'). Always classify these here.\n"
        "'claimed or rejected' = a figure advanced by a party and not accepted, "
        "or expressly rejected by the Member or a later assessment.\n"
        "'other' = anything else, e.g. a percentage that is not a WPI at all."
    ))
    body_system: BodySystemEnum = Field(description=(
        "'physical' for musculoskeletal/neurological/other bodily impairment; "
        "'psychiatric' for psychiatric or psychological impairment; 'combined' "
        "only if the figure expressly combines both; 'unclear' if not stated."
    ))
    assessor: str = Field(description=(
        "Short label for whose assessment this is, so figures can be grouped: "
        "'Dr Bodel', 'MAS Cameron', 'the Member', 'the insurer'. 'unknown' if "
        "not attributable."
    ))
    superseded: bool = Field(description=(
        "True if this figure was later corrected, replaced, reassessed or "
        "withdrawn - e.g. an initial assessment revised on review, or a figure "
        "before a pre-existing-condition deduction was applied."
    ))
    about_claimant: bool = Field(description=(
        "True if this number is an impairment OF THIS CLAIMANT. False for "
        "statutory thresholds, figures from cited authorities, or another "
        "person's impairment."
    ))
    quote: str = Field(description="Verbatim snippet (<=160 chars) containing this figure.")


class WpiResolution(BaseModel):
    mentions: List[WpiMention] = Field(description=(
        "EVERY whole-person-impairment percentage appearing anywhere in the "
        "decision, classified. Include statutory-threshold recitals and rejected "
        "figures - they are classified out, not omitted. Do not invent figures."
    ))
    tribunal_selected_value: str = Field(description=(
        "The WPI the tribunal ACTUALLY relied on as the basis of the award, if "
        "the decision makes that clear (a Member's finding, an accepted MAS "
        "certificate, or an agreed figure). Plain number, no % sign. EMPTY if "
        "the decision never settles on one - do NOT guess here; the caller has "
        "a separate rule for that case."
    ))
    tribunal_selected_quote: str = Field(description=(
        "Verbatim snippet establishing tribunal_selected_value. EMPTY if none."
    ))
    components_share_one_assessment: bool = Field(description=(
        "True if the 'component' mentions are parts of ONE assessment that can "
        "legitimately be combined into a single total. False if components come "
        "from rival assessments that must not be pooled."
    ))
    totals_are_rival_assessments: bool = Field(description=(
        "True if the assessor totals / certificates are COMPETING assessments "
        "of the SAME impairment (Dr A says 14%, Dr B says 19% about the same "
        "injuries). False if they cover DIFFERENT injuries or body parts and "
        "are meant to be combined into one figure (e.g. one certificate for "
        "scarring and nerve injury, another for a brain injury). Getting this "
        "wrong is the difference between averaging and combining."
    ))
    threshold_finding: ThresholdFindingEnum = Field(description=(
        "Does the decision establish whether the claimant's whole person "
        "impairment EXCEEDS the 10% statutory threshold in s 4.11? 'above 10%' "
        "if found, conceded or certified as greater than 10%; 'not above 10%' "
        "if found or agreed not to exceed it; 'not determined' if the decision "
        "does not settle it. This is often decided WITHOUT any percentage being "
        "stated, so answer it independently of the mentions above."
    ))
    settlement_approval_without_wpi: bool = Field(description=(
        "True if this is a settlement approval in which no exact WPI is quoted "
        "for the claimant - only the statutory threshold is discussed. In that "
        "case leaving the WPI blank is the correct outcome."
    ))
    notes: str = Field(description="One short sentence on how the figures relate. Empty if trivial.")


WPI_SYSTEM_INSTRUCTION = """\
You are reading a NSW Personal Injury Commission decision to work out the WHOLE
PERSON IMPAIRMENT percentage it supports. The decision may contain several
percentages that look alike but mean very different things, and telling them
apart is the whole task.

CLASSIFY EVERY WPI FIGURE IN THE DECISION:

  component          - one body part or element within a larger assessment.
                       "Table 73, DRE II Neck Impairment, 5% WPI"
  assessor total     - a doctor's own overall figure for the claimant.
                       "these combined, he calculated to give a 16% WPI"
  MAS certificate    - certified by a Medical Assessor / MAS combined
                       certificate. Outranks a party's medico-legal report.
  tribunal finding   - the Member's own finding, or express acceptance of a
                       figure as the basis of the award.
  threshold recital  - the statutory 10% bar under s 4.11 of the Motor Accident
                       Injuries Act 2017. "injuries do not exceed the threshold
                       of 10% whole person impairment"; "conceded a WPI greater
                       than 10%". These describe the LEGISLATIVE TEST, not this
                       claimant's impairment. Never treat one as a finding.
  claimed or rejected- advanced by a party and not accepted, or superseded by a
                       later/corrected assessment.

ALSO RECORD, for each figure:
  * body_system - physical vs psychiatric. These are assessed SEPARATELY under
    the Motor Accident Guidelines and must never be merged by you.
  * assessor - whose figure it is, so rival assessments can be grouped.
  * superseded - true if corrected, revised, or pre-deduction.
  * about_claimant - false for thresholds and figures from cited authorities.

THEN, SEPARATELY, answer whether the TRIBUNAL SETTLED ON A FIGURE. Only fill
tribunal_selected_value when the decision genuinely shows one being adopted as
the basis of the award. If two doctors disagree and the Member never chose,
leave it EMPTY - the caller handles that case with its own rule. Do not invent
a selection, and do not copy a threshold figure into it.

A 0% assessment IS a real finding when a doctor assessed this claimant at 0%
("he opined 0% whole person impairment"). Classify it as an assessor total, not
a threshold recital. Only the statutory 10% bar and minor-injury definitions are
threshold recitals.

Do NOT perform any arithmetic. Do not add or combine components - report them as
you find them and the caller will combine them correctly.
"""


# ----------------------------------------------------------------------
# Deterministic resolution
# ----------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def to_pct(value):
    """Parse a percentage to float, or None if absent/implausible."""
    if value is None:
        return None
    m = _NUM_RE.search(str(value).replace("%", "").strip())
    if not m:
        return None
    try:
        f = float(m.group(0))
    except ValueError:
        return None
    return f if 0 <= f <= 100 else None


def combine_wpi(values):
    """Combine WPI components per the AMA Guides Combined Values Chart.

    Impairments do NOT add: each successive impairment applies to the ability
    REMAINING after the previous one, so A and B combine to A + B(1 - A),
    applied from the largest component down. Returns a value rounded to the
    nearest whole percent (the convention the Guides use), or None.

    Sanity check from the corpus: Vanzanella's components 5, 5, 3, 3, 1 combine
    to 15.94 -> 16, exactly reproducing the assessor's own stated 16% total.
    Simple addition would have given 17.
    """
    vals = sorted((v for v in (to_pct(x) for x in values) if v is not None and v > 0),
                  reverse=True)
    if not vals:
        return None
    acc = vals[0] / 100.0
    for v in vals[1:]:
        b = v / 100.0
        acc = acc + b * (1.0 - acc)
    return round(acc * 100.0)


def central_estimate(values):
    """Central estimate of competing assessments. Median by default: robust to
    a single outlier expert, and identical to the mean for the common
    two-assessment case."""
    vals = sorted(v for v in (to_pct(x) for x in values) if v is not None)
    if not vals:
        return None
    if WPI_CENTRAL_ESTIMATE == "mean":
        out = sum(vals) / len(vals)
    else:
        mid = len(vals) // 2
        out = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
    return round(out, 1)


def _fmt_pct(v):
    if v is None:
        return ""
    return str(int(v)) if float(v) == int(v) else str(round(float(v), 1))


def _usable(mentions):
    """Mentions that say something about THIS claimant's impairment."""
    out = []
    for m in mentions or []:
        kind = getattr(getattr(m, "kind", None), "value", getattr(m, "kind", ""))
        if kind in (WpiKindEnum.THRESHOLD_RECITAL.value,
                    WpiKindEnum.CLAIMED_OR_REJECTED.value,
                    WpiKindEnum.OTHER.value):
            continue
        if getattr(m, "superseded", False) or not getattr(m, "about_claimant", True):
            continue
        if to_pct(getattr(m, "value", None)) is None:
            continue
        out.append(m)
    return out


def _enum(v):
    return getattr(v, "value", v) if v is not None else ""


def _kind(m):
    return getattr(getattr(m, "kind", None), "value", getattr(m, "kind", ""))


def _system(m):
    return getattr(getattr(m, "body_system", None), "value", getattr(m, "body_system", "unclear"))


def resolve_wpi(parsed, *, existing="", nel_status=""):
    """Resolve the WPI a decision supports. Returns a flat column dict.

    The ladder, in order:
      1. A figure the TRIBUNAL selected            -> stated
      2. A single MAS certificate / assessor total -> stated
      3. Components of one assessment, no total    -> combined per AMA -> derived
      4. Competing totals, no selection            -> central estimate -> inferred
      5. Nothing usable                            -> blank, absent

    Physical and psychiatric are resolved independently and the HIGHER governs;
    they are never combined with each other.
    """
    # Non-economic loss can only be awarded above the 10% threshold (s 4.11),
    # so an award is harder evidence of the threshold than anything the model
    # infers. NOTE: economic loss carries NO threshold — 27 of the 33 decisions
    # assessing 0% WPI still awarded future economic loss — so an FEL award
    # must never be read this way.
    nel_threshold = (ThresholdFindingEnum.ABOVE.value
                     if str(nel_status or "").strip() == "Awarded" else "")

    mentions = _usable(getattr(parsed, "mentions", None))
    all_values = sorted({to_pct(getattr(m, "value", None))
                         for m in (getattr(parsed, "mentions", None) or [])
                         if to_pct(getattr(m, "value", None)) is not None})

    def row(value, provenance, basis, notes=""):
        return {
            "Impairment % (Accepted)": _fmt_pct(value),
            "WPI Provenance": provenance,
            "WPI Basis": basis,
            "WPI Candidates": " | ".join(_fmt_pct(v) for v in all_values),
            "WPI Threshold Finding": _enum(getattr(parsed, "threshold_finding", None))
                                     or ThresholdFindingEnum.NONE.value,
            "WPI Resolution Notes": (notes or getattr(parsed, "notes", "") or "")[:300],
        }

    # --- 1. the tribunal actually settled on a figure ---
    selected = to_pct(getattr(parsed, "tribunal_selected_value", None))
    if selected is not None:
        return row(selected, WpiProvenanceEnum.STATED.value, "tribunal selected")

    if not mentions:
        if getattr(parsed, "settlement_approval_without_wpi", False):
            return row(None, WpiProvenanceEnum.ABSENT.value,
                       "settlement approval, no WPI quoted")
        return row(None, WpiProvenanceEnum.ABSENT.value, "no WPI stated")

    # --- resolve each body system independently ---
    share_one = bool(getattr(parsed, "components_share_one_assessment", False))
    rival = bool(getattr(parsed, "totals_are_rival_assessments", True))
    systems, worst_prov, bases = {}, WpiProvenanceEnum.STATED.value, []
    all_assessor_values = []

    def rank(p):
        return {WpiProvenanceEnum.STATED.value: 0, WpiProvenanceEnum.DERIVED.value: 1,
                WpiProvenanceEnum.INFERRED.value: 2}.get(p, 3)

    for system in {_system(m) for m in mentions}:
        group = [m for m in mentions if _system(m) == system]
        finds = [m for m in group if _kind(m) == WpiKindEnum.TRIBUNAL_FINDING.value]
        certs = [m for m in group if _kind(m) == WpiKindEnum.MAS_CERTIFICATE.value]
        totals = [m for m in group if _kind(m) == WpiKindEnum.ASSESSOR_TOTAL.value]
        comps = [m for m in group if _kind(m) == WpiKindEnum.COMPONENT.value]

        # Reduce each ASSESSOR to one figure first: their own stated total if
        # they gave one, otherwise their components combined. Doing this before
        # any comparison stops an assessor who only itemised (Seaman's
        # Dr Bentivoglio, 3% + 2% + 2%) from being discarded merely because a
        # rival assessor happened to state a total.
        derived_any = False

        def assessor_totals(members, allow_components):
            nonlocal derived_any
            out, by_assessor = {}, {}
            for m in members:
                by_assessor.setdefault(getattr(m, "assessor", "unknown"), []).append(m)
            for who, ms in by_assessor.items():
                stated = [to_pct(m.value) for m in ms if _kind(m) != WpiKindEnum.COMPONENT.value]
                stated = [v for v in stated if v is not None]
                if stated:
                    out[who] = max(stated) if len(set(stated)) > 1 else stated[0]
                elif allow_components:
                    combined = combine_wpi([m.value for m in ms])
                    if combined is not None:
                        out[who] = combined
                        derived_any = True
            return out

        # A tribunal finding, then a MAS certificate, outranks a party report.
        if finds:
            per_assessor, source = assessor_totals(finds, False), "tribunal finding"
        elif certs:
            per_assessor, source = assessor_totals(certs, False), "MAS certificate"
        else:
            pool = totals + comps
            if share_one and comps and not totals:
                # One assessment itemised across several entries.
                combined = combine_wpi([m.value for m in comps])
                per_assessor = {"combined": combined} if combined is not None else {}
                derived_any = combined is not None
                source = f"combined from {len(comps)} components (AMA Combined Values)"
            else:
                per_assessor = assessor_totals(pool, True)
                source = "assessor total"

        # One vote per ASSESSOR — deliberately not deduplicated. Vanzanella has
        # Dr Conrad at 16%, Dr Dryson at 16% and Dr Ugwu (ankle only) at 6%;
        # collapsing the two 16s to a single value turns the median into 11
        # and throws away the fact that two of three assessors agree.
        vals = sorted(per_assessor.values())
        if not vals:
            continue
        all_assessor_values.extend(vals)

        if len(set(vals)) == 1:
            value = vals[0]
            prov = (WpiProvenanceEnum.DERIVED.value if derived_any
                    else WpiProvenanceEnum.STATED.value)
            basis = (source if not derived_any or "components" in source
                     else f"{source}, combined from components")
        elif rival:
            value = central_estimate(vals)
            prov = WpiProvenanceEnum.INFERRED.value
            basis = f"{WPI_CENTRAL_ESTIMATE} of {len(vals)} competing assessments"
        else:
            value = combine_wpi(sorted(set(vals)))
            prov = WpiProvenanceEnum.DERIVED.value
            basis = (f"combined {len(set(vals))} assessments covering "
                     f"different injuries")

        if value is None:
            continue
        systems[system] = value
        bases.append(f"{system}: {basis}")
        if rank(prov) > rank(worst_prov):
            worst_prov = prov

    if not systems:
        return row(None, WpiProvenanceEnum.ABSENT.value, "no usable WPI figure")

    # Physical and psychiatric are assessed separately; the higher governs.
    value = max(systems.values())
    basis = "; ".join(bases)
    if len(systems) > 1:
        winner = max(systems, key=lambda k: systems[k])
        basis += f"; higher of {len(systems)} body systems ({winner})"

    # Threshold sanity check. A statutory recital is not a finding about the
    # claimant, but "the insurer conceded greater than 10%" still CONSTRAINS
    # the answer. A resolved value contradicting a settled threshold is wrong on
    # the face of the document - including a 'stated' one, which just means we
    # picked the wrong quoted figure - so we withhold it rather than publish it.
    threshold = _enum(getattr(parsed, "threshold_finding", None)) or nel_threshold
    if nel_threshold == ThresholdFindingEnum.ABOVE.value:
        # Non-economic loss was AWARDED, which s 4.11 permits only above 10%.
        # That is a legal fact, so it outranks the model's own reading.
        threshold = nel_threshold

    if threshold == ThresholdFindingEnum.ABOVE.value and value <= 10:
        # The decision establishes impairment above 10% but the assessments
        # average out below it. Rather than withhold, drop the assessments the
        # established threshold rules out and average what remains.
        above = [v for v in all_assessor_values if v > 10]
        if above:
            value = round(sum(above) / len(above), 1)
            return row(value, WpiProvenanceEnum.INFERRED.value,
                       f"mean of the {len(above)} assessment(s) above the 10% threshold "
                       f"(lower assessments excluded: impairment found above 10%)",
                       notes=basis)
        return row(None, WpiProvenanceEnum.ABSENT.value,
                   f"withheld: {_fmt_pct(value)} contradicts a finding of impairment "
                   f"above the 10% threshold, and no assessment exceeds it", notes=basis)

    if threshold == ThresholdFindingEnum.BELOW.value and value > 10:
        at_or_below = [v for v in all_assessor_values if v <= 10]
        if at_or_below:
            value = round(sum(at_or_below) / len(at_or_below), 1)
            return row(value, WpiProvenanceEnum.INFERRED.value,
                       f"mean of the {len(at_or_below)} assessment(s) at or below the 10% "
                       f"threshold (higher assessments excluded: impairment found not "
                       f"above 10%)", notes=basis)
        return row(None, WpiProvenanceEnum.ABSENT.value,
                   f"withheld: {_fmt_pct(value)} contradicts a finding of impairment "
                   f"not above the 10% threshold", notes=basis)
    return row(value, worst_prov, basis)


# Flat columns contributed by this module.
WPI_FIELDS = [
    "WPI Provenance",
    "WPI Threshold Finding",
    "WPI Basis",
    "WPI Candidates",
    "WPI Resolution Notes",
]


def empty_wpi_row(existing_value=""):
    """Defaults for a row the resolution pass has not seen. Provenance follows
    the value that is already there, so a pre-existing WPI is not mislabelled."""
    return {
        "WPI Provenance": (WpiProvenanceEnum.STATED.value if str(existing_value or "").strip()
                           else WpiProvenanceEnum.ABSENT.value),
        "WPI Threshold Finding": ThresholdFindingEnum.NONE.value,
        "WPI Basis": "",
        "WPI Candidates": "",
        "WPI Resolution Notes": "",
    }
