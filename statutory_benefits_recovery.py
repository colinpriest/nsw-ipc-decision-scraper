"""
Recover a stated `Statutory Benefits Paid` total, in MAIA *and* MACA wording.

Round 4 §12.3 found MACA-era language 11x enriched among the rows where this
field failed to extract: 16.1% of the 62 failures against 1.4% of the successes.
The predecessor scheme says the same thing in different words —

    MAIA 2017     "statutory benefits", "s 3.40(1)(b) deduction"
    MACA 1999     "s 83 payments", "s 130 credit"

— so an extractor keyed on MAIA vocabulary misses the MACA equivalents by
construction, however good it is. Greer [2026] NSWPIC 279 is the example the
request cites: "s 130 MACA credit for s 83 payments".

Two things matter for correctness here, both from §12.1:

1. `Statutory Benefits Paid` and `Statutory Benefits Repaid` are DIFFERENT
   fields. Paid is everything the claimant received (weekly plus treatment and
   care); Repaid is the s 3.40 / s 130 deduction, which reaches only the
   recoverable categories. They coincide on 96% of rows only because treatment
   and care is 98.1% not applicable. So this module never derives one from the
   other — it reads a figure the decision states, and returns nothing when the
   decision states only the deduction.
2. A figure the decision states IS `stated` even when it equals the deduction.
   Filtering those out as duplicates hides genuinely stated totals: Mepani
   ("the insurer has allowed the sum of $212,752.78 in benefits paid to date")
   states a paid total that happens to match the repayment exactly.
"""

import re

MONEY = r'\$\s?([\d]{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)'

# MAIA phrasings. These are ambiguous by nature: "a deduction of $X for weekly
# payments of statutory benefits paid" states the DEDUCTION, and under §12.1 the
# deduction reaches only the recoverable categories — so it is `Repaid`, not
# `Paid`, and reading it as Paid is the exact conflation the request warns
# against. They are accepted only when the surrounding clause is not framing
# the figure as a deduction.
MAIA_PAID_PATTERNS = [
    rf'(?:weekly |statutory )*benefits (?:already )?paid(?: to date| to the claimant| to [A-Z][a-z]+)?[^.$]{{0,45}}{MONEY}',
    rf'{MONEY}[^.$]{{0,30}}(?:in|of) (?:weekly |statutory )*benefits paid',
    rf'(?:has|have) been paid[^.$]{{0,40}}(?:weekly payments of statutory benefits|statutory benefits)[^.$]{{0,40}}{MONEY}',
]

# MACA phrasings. s 83 is the insurer's obligation to PAY; s 130 is the separate
# right to recover. So "Section 83 payments $215,552.18" states an amount paid
# by construction, even when it appears under a s 130 credit heading — which is
# why these are not subject to the deduction-framing rejection.
MACA_PAID_PATTERNS = [
    rf'(?:s|section)\s?83\s+(?:payments|expenses)[^.$]{{0,35}}(?:total(?:s|ling|led)?|of|is|are|amount(?:ing)?\s+to)?[^.$]{{0,20}}{MONEY}',
    rf'{MONEY}[^.$]{{0,30}}in\s+(?:s|section)\s?83\s+(?:payments|expenses)',
    rf'(?:has|have) paid[^.$]{{0,30}}(?:s|section)\s?83[^.$]{{0,40}}{MONEY}',
]

PAID_RE = [(re.compile(p, re.I), True) for p in MAIA_PAID_PATTERNS] + \
          [(re.compile(p, re.I), False) for p in MACA_PAID_PATTERNS]

# A figure introduced as a deduction is the s 3.40 repayment, not the total
# paid. Checked in the run-up to a MAIA match only.
DEDUCTION_FRAMING = re.compile(
    r'\bdeduct(?:ion|ed|ing)?\b|\bless\b|\bcredit(?:ed)?\b|\brecovery\b|'
    r'\brepay(?:ment)?\b|\brefund\b|\bentitled to\b', re.I)

# A match is rejected when the span shows the figure is something else wearing
# similar words. Each of these produced a false positive on the real corpus:
#   "tax paid on statutory benefits $868"      -> tax, not benefits
#   "$22,000 less deduction of statutory ..."  -> the settlement, not the paid total
#   "... and a $150,000 buffer for ..."        -> a buffer
REJECT_IN_SPAN = re.compile(
    r'\btax(?:ation)?\b|\bless\b|\bbuffer\b|\bMedicare\b|\bCentrelink\b|'
    r'\bsuperannuation\b|\bFox v Wood\b', re.I)

# MACA-era vocabulary, for detecting which scheme a decision speaks.
MACA_LANGUAGE = re.compile(
    r'\bMACA\b|Motor Accidents Compensation Act|(?:s|section)\s?83\b|'
    r'(?:s|section)\s?130\b', re.I)


def find_statutory_benefits_paid(text):
    """Return (amount, quote) for a stated benefits-paid total, else (None, "").

    Picks the LARGEST credible figure: where a decision itemises weekly
    payments and treatment separately and then gives a combined total, the
    total is the field's definition.
    """
    if not text:
        return None, ""
    flat = re.sub(r'\s+', ' ', str(text))
    best = None
    for pattern, is_maia in PAID_RE:
        for match in pattern.finditer(flat):
            groups = [g for g in match.groups() if g]
            if not groups:
                continue
            before = flat[max(0, match.start() - 95):match.start()]
            after = flat[match.end():match.end() + 45]
            span = flat[max(0, match.start() - 40):match.end() + 30]
            # "Past economic loss (incl s 83 payments) $222,124" is a
            # PARENTHETICAL inside another head — the figure is that head's,
            # not the benefits total.
            if re.search(r'\bincl(?:uding|\.)?\b', before, re.I):
                continue
            # "...the only deduction to be made was the sum of $23,245.66
            # representing weekly payments of statutory benefits" states the
            # DEDUCTION, and the framing can sit either side of the figure.
            if is_maia and DEDUCTION_FRAMING.search(before + match.group(0) + after):
                continue
            if REJECT_IN_SPAN.search(match.group(0) + flat[match.end():match.end() + 25]) \
                    or _tax_context(span):
                continue
            try:
                value = float(groups[0].replace(',', ''))
            except ValueError:
                continue
            if value < 100:                      # not a benefits total
                continue
            quote = flat[max(0, match.start() - 70):match.end() + 40].strip()
            if best is None or value > best[0]:
                best = (value, quote)
    return best if best else (None, "")


def _tax_context(span):
    """`tax paid on statutory benefits $16,259` is a Fox v Wood component, not
    a benefits total, and it appears in exactly this shape."""
    return bool(re.search(r'tax(?:ation)?\s+(?:paid|withheld)', span, re.I))


def uses_maca_language(*texts):
    """True if any text speaks the predecessor scheme's vocabulary."""
    return any(MACA_LANGUAGE.search(str(t)) for t in texts if t)
