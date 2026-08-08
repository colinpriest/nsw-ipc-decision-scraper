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

# A medical review panel reviews an existing MAS certificate and may revoke it
# (s 7.26 MAI Act), so its figure supersedes rather than competes.
_REVIEW_PANEL_RE = re.compile(r"review\s*panel|medical\s*panel|\bpanel\b", re.I)


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
    """Deliberately the same vocabulary as `damages_extraction.ProvenanceEnum`,
    including the round-2 §10.1 split of `absent` into WHY it is absent. The two
    enums are kept parallel rather than shared so neither module depends on the
    other; `test_round2_applicability` asserts they stay in step."""
    STATED = "stated"
    DERIVED = "derived"
    INFERRED = "inferred"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"
    NOT_STATED = "not_stated"
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

    systems_out = {}

    def row(value, provenance, basis, notes="", governing=None):
        return {
            "Impairment % (Accepted)": _fmt_pct(value),
            "WPI Provenance": provenance,
            "WPI Basis": basis,
            "WPI Candidates": " | ".join(_fmt_pct(v) for v in all_values),
            "WPI Threshold Finding": _enum(getattr(parsed, "threshold_finding", None))
                                     or ThresholdFindingEnum.NONE.value,
            "WPI Resolution Notes": (notes or getattr(parsed, "notes", "") or "")[:300],
            # Round 5 §13: emitted HERE, from the same `systems` dict that
            # writes "higher of N body systems (x)" into the basis, rather than
            # re-derived downstream from which cells happen to be populated.
            # That derivation was circular — with one component captured it
            # could only ever name the component we held.
            "WPI Governing System": governing or GoverningSystemEnum.NOT_STATED.value,
            # Round 6 §14.1: the per-system figures the comparison was made
            # from. A governing-system label asserts that a comparison happened,
            # so the components it compared cannot simultaneously be `absent` —
            # carrying them here lets the split columns be filled from the same
            # authority that names the winner, instead of being reconstructed.
            # Underscore-prefixed, so it stays out of the CSV.
            "_wpi_systems": dict(systems_out or {}),
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
            # A REVIEW PANEL certificate is not a rival opinion — under s 7.26
            # a panel may revoke the original certificate and issue its own, so
            # the earlier one is superseded, not competing. Averaging the two
            # produced a figure neither assessor ever gave: Tiwari [2026] NSWPIC
            # 251 had MAS Oates at 0% and the Review Panel at 12%, and the
            # median of 6 made psychiatric (7%) look like the governing system
            # when physical governs at 12.
            # Only where the certificates are RIVAL readings of the same
            # impairment. Where they cover DIFFERENT injuries they combine, and
            # a panel that reviewed one of them does not supersede the other:
            # Quigley has MAS Curtin at 4% (scarring, nerve) and a Review Panel
            # at 8% (brain injury, shoulder), which combine to 12.
            panel = [m for m in certs
                     if _REVIEW_PANEL_RE.search(str(getattr(m, "assessor", "") or ""))]
            if rival and panel and len(panel) < len(certs):
                per_assessor = assessor_totals(panel, False)
                source = "review panel certificate (supersedes the certificate reviewed)"
            else:
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
        systems_out[system] = (_fmt_pct(value), prov, basis)
        bases.append(f"{system}: {basis}")
        if rank(prov) > rank(worst_prov):
            worst_prov = prov

    if not systems:
        return row(None, WpiProvenanceEnum.ABSENT.value, "no usable WPI figure")

    # Physical and psychiatric are assessed separately; the higher governs.
    value = max(systems.values())
    basis = "; ".join(bases)
    # `not determined` where only ONE system was ever quantified: which system
    # governs is a comparison, and there is nothing to compare against. Naming
    # the one we hold would assert a finding the decision never made.
    governing = GoverningSystemEnum.NOT_DETERMINED.value
    if len(systems) > 1:
        winner = max(systems, key=lambda k: systems[k])
        basis += f"; higher of {len(systems)} body systems ({winner})"
        governing = (GoverningSystemEnum.PSYCHIATRIC.value
                     if winner == BodySystemEnum.PSYCHIATRIC.value
                     else GoverningSystemEnum.PHYSICAL.value
                     if winner == BodySystemEnum.PHYSICAL.value
                     else GoverningSystemEnum.COMBINED.value)
    elif set(systems) == {BodySystemEnum.COMBINED.value}:
        # A single figure the decision itself says covers both systems.
        governing = GoverningSystemEnum.COMBINED.value

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
                       notes=basis, governing=governing)
        return row(None, WpiProvenanceEnum.ABSENT.value,
                   f"withheld: {_fmt_pct(value)} contradicts a finding of impairment "
                   f"above the 10% threshold, and no assessment exceeds it",
                   notes=basis, governing=governing)

    if threshold == ThresholdFindingEnum.BELOW.value and value > 10:
        at_or_below = [v for v in all_assessor_values if v <= 10]
        if at_or_below:
            value = round(sum(at_or_below) / len(at_or_below), 1)
            return row(value, WpiProvenanceEnum.INFERRED.value,
                       f"mean of the {len(at_or_below)} assessment(s) at or below the 10% "
                       f"threshold (higher assessments excluded: impairment found not "
                       f"above 10%)", notes=basis, governing=governing)
        return row(None, WpiProvenanceEnum.ABSENT.value,
                   f"withheld: {_fmt_pct(value)} contradicts a finding of impairment "
                   f"not above the 10% threshold", notes=basis, governing=governing)
    return row(value, worst_prov, basis, governing=governing)


# ----------------------------------------------------------------------
# The ex gratia carve-out to s 4.11
# ----------------------------------------------------------------------

# An insurer may pay non-economic loss it does not owe. QBE v Silcocks [2023]
# NSWPIC 24 is the pattern: 9% WPI, no entitlement, and the Member approved
# $120,000 anyway as "an appropriate compromise ... where no legal obligation on
# insurer to make any allowance for non-economic loss". Such a row is CORRECT -
# a real exception to s 4.11, not a bad extraction - and must survive the
# quarantine that catches genuine WPI errors, which would otherwise blank the
# one figure in it that is right.
#
# The signal is the decision saying the entitlement was absent while the money
# was paid anyway. Deliberately narrow: it is only ever consulted for rows that
# already pay non-economic loss on a WPI at or below 10, so it cannot fire on an
# ordinary case that merely recites the threshold.
_EX_GRATIA_PATTERNS = [
    # "no legal obligation on insurer to make any allowance for non-economic loss"
    r"no\s+legal\s+obligation[^.]{0,120}non-?economic\s+loss",
    r"non-?economic\s+loss[^.]{0,120}no\s+legal\s+obligation",
    # "not obliged / not liable / under no obligation to pay non-economic loss"
    r"(?:not\s+(?:legally\s+)?(?:obliged|liable|required|bound)|under\s+no\s+"
    r"obligation)[^.]{0,120}non-?economic\s+loss",
    # "no entitlement to non-economic loss" / "no legal entitlement ... yet paid"
    r"no\s+(?:legal\s+)?entitlement[^.]{0,80}non-?economic\s+loss",
    r"non-?economic\s+loss[^.]{0,60}no\s+(?:legal\s+)?entitlement",
    # "unable to establish / cannot demonstrate an entitlement to NEL"
    r"(?:unable\s+to\s+establish|cannot\s+(?:demonstrate|establish)|has\s+not\s+"
    r"established)[^.]{0,100}entitlement[^.]{0,60}non-?economic\s+loss",
    # "absence of legal entitlement" in the same breath as the offer
    r"absence\s+of\s+(?:any\s+)?legal\s+entitlement",
]

_EX_GRATIA_RE = re.compile("|".join(_EX_GRATIA_PATTERNS), re.IGNORECASE | re.DOTALL)


def nel_paid_without_entitlement(*texts):
    """True if any text says non-economic loss was paid despite no entitlement.

    Callers pass whatever they have - catchwords, the Member's reasoning, the
    full decision. Absence of the phrase is not evidence either way; it just
    means the row goes through the normal contradiction check.
    """
    for text in texts:
        if not text:
            continue
        collapsed = re.sub(r"\s+", " ", str(text))
        if _EX_GRATIA_RE.search(collapsed):
            return True
    return False


# ----------------------------------------------------------------------
# Why a split-WPI figure is missing (round 2 §10.1)
# ----------------------------------------------------------------------

def _has_mention(mentions, system):
    """True if the classified mentions hold a usable figure for this body
    system — i.e. the decision DID quantify it and we failed to carry it."""
    for m in mentions or ():
        if str(m.get("body_system") or "") != system:
            continue
        if not m.get("about_claimant", True) or m.get("superseded"):
            continue
        if str(m.get("kind") or "") in (WpiKindEnum.THRESHOLD_RECITAL.value,
                                        WpiKindEnum.CLAIMED_OR_REJECTED.value,
                                        WpiKindEnum.OTHER.value):
            continue
        if to_pct(m.get("value")) is not None:
            return True
    return False


def classify_split_wpi_absence(*, system, has_psychiatric, total_present,
                               mentions=(), psychiatric_only=False):
    """Why is `WPI {system} %` empty? Returns a WpiProvenanceEnum string.

    Deterministic — no model call. The four states are decidable from columns
    the row already carries, which is the whole point of §10.1: applicability
    becomes a lookup rather than an inference from a flag and a threshold.

      not_applicable  the body system does not arise for this claimant
      absent          the decision quantified it and we did not capture it
      not_stated      it was assessed, but not given as a separate figure
      not_assessed    nobody quantified it at all

    The two systems are NOT symmetric, because the columns are not. A
    psychiatric percentage is separately stated by its nature, so a psychiatric
    assessment in the text with an empty column is a defect. The physical
    column is defined as "only if the decision states physical and psychiatric
    SEPARATELY", so physical figures on a decision that never quantified
    psychiatric are `not_stated`, not a miss — there was no split to capture.
    Reading that asymmetry the other way put 91 rows in `absent` that were
    behaving exactly as specified.
    """
    if system == BodySystemEnum.PSYCHIATRIC.value and not has_psychiatric:
        return WpiProvenanceEnum.NOT_APPLICABLE.value
    if system == BodySystemEnum.PHYSICAL.value and psychiatric_only:
        return WpiProvenanceEnum.NOT_APPLICABLE.value

    mine = _has_mention(mentions, system)
    if system == BodySystemEnum.PHYSICAL.value:
        # A split only existed if the counterpart was quantified too.
        split_existed = mine and _has_mention(
            mentions, BodySystemEnum.PSYCHIATRIC.value)
    else:
        split_existed = mine
    if split_existed:
        return WpiProvenanceEnum.ABSENT.value

    # `not_stated` claims an assessment happened, so it needs a USABLE figure
    # somewhere. A decision whose only percentage is the statutory 10% recital
    # has assessed nothing, and counting it would relabel 'nobody measured
    # this' as 'measured but not broken down'.
    if total_present or any(
            _has_mention(mentions, s.value) for s in BodySystemEnum):
        return WpiProvenanceEnum.NOT_STATED.value
    return WpiProvenanceEnum.NOT_ASSESSED.value


# ----------------------------------------------------------------------
# Which body system governs (round 2 §10.4 rule 1)
# ----------------------------------------------------------------------

class GoverningSystemEnum(str, Enum):
    """Which assessment the WPI figure is taken from.

    Physical and psychiatric impairment are assessed SEPARATELY under the Motor
    Accident Guidelines and are not combined with each other, so where both are
    stated the GREATER governs the s 4.11 question. `resolve_wpi` already
    computes this ("higher of 2 body systems (psychiatric)"); this promotes it
    from prose in `WPI Resolution Notes` to a column the consumer can filter on.
    """
    PHYSICAL = "physical"
    PSYCHIATRIC = "psychiatric"
    COMBINED = "combined"          # one figure expressly covering both
    NOT_DETERMINED = "not determined"   # only one system was ever quantified
    NOT_STATED = "not stated"           # neither system was quantified


def governing_system(physical, psychiatric, total=None):
    """Which system the governing WPI comes from, from the split components.

    `total` disambiguates the case where the components tie, and catches the
    row whose stated total matches neither component — which means the total
    was combined across systems rather than selected between them.
    """
    p, q = to_pct(physical), to_pct(psychiatric)
    if p is None and q is None:
        return GoverningSystemEnum.NOT_STATED.value
    if p is None or q is None:
        # Round 5 §13. Which system governs is a COMPARISON, so one component
        # cannot answer it. Returning the captured one made the column a
        # tautology — "the system we happen to hold is the system that governs"
        # — on 145 rows, and it was demonstrably wrong wherever the missing
        # component was the larger. Row 7: total 12, psychiatric 1, physical
        # uncaptured; the label read `psychiatric` while the resolution's own
        # notes said `higher of 2 body systems (physical)`.
        return GoverningSystemEnum.NOT_DETERMINED.value

    t = to_pct(total)
    if t is not None and t > max(p, q):
        # Neither component alone reaches the stated total, so the assessor
        # combined across systems (Mason [2024] NSWPIC 348: 7% shoulder + 6%
        # emotional/behavioural within ONE brain-injury assessment = 13%).
        return GoverningSystemEnum.COMBINED.value
    if q > p:
        return GoverningSystemEnum.PSYCHIATRIC.value
    if p > q:
        return GoverningSystemEnum.PHYSICAL.value
    return GoverningSystemEnum.COMBINED.value if t is None or t == p else (
        GoverningSystemEnum.PHYSICAL.value)


# ----------------------------------------------------------------------
# Non-economic loss vs the threshold (round 2 §10.4 rule 2)
# ----------------------------------------------------------------------

class NelConsistencyEnum(str, Enum):
    """Does the award of non-economic loss agree with the impairment finding?

    s 4.11 gates non-economic loss on impairment ABOVE 10%; economic loss has
    no such threshold. Where the two disagree, either the head is misclassified
    (a buffer, or a Compensation to Relatives award, booked as non-economic
    loss) or the finding is misattributed — or, occasionally, the insurer paid
    what it did not owe and the Member approved it.

    This column reports the disagreement; it does not resolve it. Nothing is
    edited on the strength of it.
    """
    YES = "yes"
    NO = "no"
    UNKNOWN = "cannot determine"


def nel_threshold_consistency(*, nel_status, threshold_finding, wpi):
    """Compare the non-economic loss award against the threshold evidence.

    The explicit `WPI Threshold Finding` outranks any percentage, because
    physical and psychiatric are assessed separately and the greater governs —
    so a `WPI %` holding one body system cannot be compared to 10 directly.
    """
    if str(nel_status or "").strip() != "Awarded":
        return NelConsistencyEnum.UNKNOWN.value

    finding = str(threshold_finding or "").strip()
    if finding == ThresholdFindingEnum.ABOVE.value:
        return NelConsistencyEnum.YES.value
    if finding == ThresholdFindingEnum.BELOW.value:
        return NelConsistencyEnum.NO.value

    value = to_pct(wpi)
    if value is None:
        return NelConsistencyEnum.UNKNOWN.value
    return (NelConsistencyEnum.YES.value if value > 10
            else NelConsistencyEnum.NO.value)


# Flat columns contributed by this module.
WPI_FIELDS = [
    "WPI Provenance",
    "WPI Threshold Finding",
    "WPI Basis",
    "WPI Candidates",
    "WPI Resolution Notes",
    # --- round 2 ---
    "WPI Governing System",         # §10.4 rule 1
    "NEL Threshold Consistent",     # §10.4 rule 2
    "WPI Threshold Finding Basis",  # §10.3
]


class ThresholdBasisEnum(str, Enum):
    """Where `WPI Threshold Finding` came from.

    §10.3 asked for coverage well above 38.7% while keeping `not determined`
    distinct from empty. Coverage can be raised because s 4.11 makes some
    findings deducible — an award of non-economic loss is only lawful above
    10%, so the award itself settles the threshold. But a deduction is not a
    finding the court made, and the consumer's own rule is that an explicit
    finding outranks anything computed. So the finding is filled and this
    column says how, letting a consumer that wants only judicial findings
    filter to `decision`.
    """
    DECISION = "decision"
    FROM_NEL_AWARD = "implied by non-economic loss award"
    FROM_WPI = "implied by stated WPI"
    NONE = "not determined"


def derive_threshold_finding(*, nel_status, wpi, ex_gratia=False):
    """Deduce the s 4.11 threshold where the decision did not state it.

    Returns (finding, basis). Only two deductions are safe:

      * non-economic loss was awarded, which s 4.11 permits only above 10% —
        unless the insurer paid without any obligation to, which is the one
        way a lawful award can sit below the threshold; and
      * a stated whole-person impairment, compared to 10 directly.

    Nothing is deduced from a `Nil` award: a head can be refused for many
    reasons besides the threshold, and guessing which would manufacture a
    judicial finding out of silence.
    """
    if str(nel_status or "").strip() == "Awarded" and not ex_gratia:
        return ThresholdFindingEnum.ABOVE.value, ThresholdBasisEnum.FROM_NEL_AWARD.value

    value = to_pct(wpi)
    if value is not None:
        finding = (ThresholdFindingEnum.ABOVE.value if value > 10
                   else ThresholdFindingEnum.BELOW.value)
        return finding, ThresholdBasisEnum.FROM_WPI.value

    return ThresholdFindingEnum.NONE.value, ThresholdBasisEnum.NONE.value


def empty_wpi_row(existing_value=""):
    """Defaults for a row the resolution pass has not seen. Provenance follows
    the value that is already there, so a pre-existing WPI is not mislabelled.

    An unseen row's absence is `not_assessed`, not `absent`: we have no evidence
    the decision quantified anything, and `absent` now means "it was there and
    we missed it", which is a claim this default cannot support."""
    return {
        "WPI Provenance": (WpiProvenanceEnum.STATED.value if str(existing_value or "").strip()
                           else WpiProvenanceEnum.NOT_ASSESSED.value),
        "WPI Threshold Finding": ThresholdFindingEnum.NONE.value,
        "WPI Basis": "",
        "WPI Candidates": "",
        "WPI Resolution Notes": "",
        "WPI Governing System": GoverningSystemEnum.NOT_STATED.value,
        "NEL Threshold Consistent": NelConsistencyEnum.UNKNOWN.value,
        "WPI Threshold Finding Basis": ThresholdBasisEnum.NONE.value,
    }
