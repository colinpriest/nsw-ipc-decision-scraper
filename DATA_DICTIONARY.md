# Data dictionary — `ctp_impairment_lump_sum.xlsx`

**Delivery:** damages-breakdown spec v1 (request dated 2026-07-27)
**Generated:** 2026-08-06 · **schema version** 3 · **damages version** 1
**Shape:** 540 rows × 128 columns · one row per decision
**Population:** analysis-ready CTP decisions with a **positive lump sum**. A WPI is *not* required.

> **Row count.** 540, matching the count the spec refers to. An interim version of this workbook also required a positive accepted WPI, which cut it to 287 by dropping awards whose decision never states a WPI — a fact about the decision, not a defect in the row. Those 253 rows are back, with `WPI %` blank and a `WPI % Provenance` that says *why* it is blank (round 2 §10.1). Filter on `WPI % Provenance = 'stated'` for any WPI-conditional analysis.
>
> Separately, a WPI **correctness** fix remains in place and is unrelated to the population rule: decisions reciting the MAI Act's ">10% WPI" statutory bar were being read as a 10% finding about the claimant. Those false positives stay suppressed; spot-checked on 10 cases, all 10 correct.

Coverage percentages below are **measured on this delivery**, not targets.

---

## Conventions

### Null semantics — read this first

A blank cell means **"no value"**, never zero. Which kind of "no value" is recorded in the paired `... Status` and `... Provenance` columns. Do not impute.

### Three-valued status (`... Status` columns)

| value | meaning |
|---|---|
| `Awarded` | the head was assessed and an amount awarded |
| `Nil` | the head was before the tribunal and **nothing** was awarded — a genuine zero, amount is `0` |
| `Not addressed` | the head was **never before the tribunal** — missing, **not** zero, amount is blank |

Enforced deterministically, not trusted to the model: a `Not addressed` row never carries an amount, a `Nil` row always carries `0`, an `Awarded` head with amount `0` is reclassified `Nil`, and a fatality/dependency claim (Compensation to Relatives pathway) gets `Not addressed` rather than `Nil`.

### Provenance (`... Provenance` columns)

Every money and WPI field has one. Populated on 100% of rows.

| value | meaning | a defect? |
|---|---|---|
| `stated` | the figure appears verbatim in the decision | no |
| `derived` | computed by arithmetic from figures that do appear | no |
| `inferred` | model judgement rather than the document's — **exclude or down-weight** | no |
| `not_applicable` | the precondition does not arise — no psychiatric injury, so no psychiatric WPI to assess | no |
| `not_assessed` | the precondition arises, but nobody quantified it: no MAS certificate, or the head was never in issue | no |
| `not_stated` | it *was* assessed, but the decision does not give the number — typically a combined total with no body-system split | no |
| `absent` | it should have been recoverable from this text and we did not get it | **yes** |

> **Changed (round 2 §10.1).** The last four used to be one word, `absent`. That made "the answer is no" indistinguishable from "we don't know", so a missingness check on `WPI Psychiatric %` had to either flag all 448 blanks or none — when exactly 13 are defects. **`absent` is now the only value that means a defect**, and a consumer testing `== 'absent'` gets precisely the defect case, which is what that test always meant. Nothing was renamed and no positive value changed, so the split is strictly additive.

The verbatim source snippet behind each figure is in `processed_sidecar.json` under `damages.quotes`, keyed by decision URL.

### `Has Psychiatric Injury` vs `Psychological Injury Emphasis`

These measure different things, and the names invite the assumption that they should agree. They should not.

| column | what it records |
|---|---|
| `Has Psychiatric Injury` | a recognised psychiatric injury **established** by the decision — diagnosed, assessed, or accepted |
| `Psychological Injury Emphasis` | how much the **narrative** dwells on psychological harm, including a claimant who advances it and fails |

So `emphasis = 1 or 2` with `Has Psychiatric Injury = No` is a normal, non-contradictory combination: the decision discussed psychological harm at length and established none. NRMA v Taylor [2024] NSWPIC 301 is the clean example — Ms Taylor's driving anxiety and lost confidence run through the decision (emphasis 2), and the Member expressly held "there was not a need to refer Ms Taylor for WPI assessment of her psychological symptoms". Nothing was diagnosed or assessed, so the flag is `No` and `WPI Psychiatric % Provenance` is `not_applicable`.

Use `Has Psychiatric Injury` as the applicability gate. Do not use emphasis as one — it splits exactly evenly at emphasis 1 (90 / 90) and cannot separate the two states.

### Money and dates

All money is AUD as a bare number — no `$`, no thousands separators. Dates are `YYYY-MM-DD`. Percentages are numbers (`14`, not `14%`).

---

## 1. Identity and decision metadata

| column | type | coverage | notes |
|---|---|---|---|
| `Case Name` | text | 100% | AustLII case title |
| `URL` | text | 100% | **Join key.** AustLII permalink; also the key into `processed_sidecar.json` |
| `File Saved` | text | 100% | Local filename under `nsw_pic_decisions/` |
| `Jurisdiction` | enum | 100% | `NSW` for this workbook |
| `Case Type` | enum | 100% | `CTP` for this workbook |
| `Decision Date` | date | 100% | |
| `Injury Date` | date | 100% | `Unknown` where the decision does not state it |
| `Applicant` / `Respondent` | text | 100% | Party names. Suppressible — see "Privacy" below |

## 2. Claimant

| column | type | coverage | notes |
|---|---|---|---|
| `Claimant Age` | number | 91.9% | Age **at injury**; derived from DOB or age-at-decision where not stated directly |
| `Claimant Age At Decision` | number | 91.9% | |
| `Claimant Gender` | enum | 100% | `Male` / `Female` / `Other` / `Not stated` |
| `Claimant Occupation` | text | 100% | `Not stated` where absent |
| `Claimant Weekly Income` | number | 58.1% | Normalised to a **weekly** figure; pre-injury (PIAWE) and gross preferred |
| `Claimant Weekly Income Basis` | text | 100% | What the figure represents and any conversion performed — makes the number auditable |
| `Employer Name` | text | 100% | `Not applicable` for CTP |
| `Accident/Injury Location` | text | 99.6% | Free text as stated |
| `Location Locality` / `Location State` | text | 100% | Normalised |

## 3. Outcome and the trusted award columns

**Unchanged by this delivery.** These are the columns the consumer already relies on; they were not overwritten.

| column | type | coverage | notes |
|---|---|---|---|
| `Lump Sum` | number | 100% | **This is the GROSS sum, not net — see §5.** |
| `WPI %` | number | **59.3%** | The WPI the award is calibrated against (accepted, whether assessed here or in a prior MAS certificate). Blank where the decision does not state one, and blank where the row's own award of non-economic loss proves the captured figure wrong — see §3.1 |
| `WPI % Provenance` | enum | 100% | `stated` 309 · `not_assessed` 213 · `inferred` 9 · `absent` 5 · `derived` 2 · `not_stated` 1 · `not_applicable` 1. **`absent` now means a defect** — see Conventions. **Exclude `inferred` for WPI-conditional analysis** — those are central estimates of competing assessments, not figures the decision states |
| `WPI % Basis` | text | 100% | how the figure was resolved: `tribunal selected`, `MAS certificate`, `assessor total`, `combined from N components (AMA Combined Values)`, `median of N competing assessments`, `withheld: … contradicts …`, or `retained from main extraction` |
| `WPI % Candidates` | text | 29% | every distinct percentage found in the decision, ` \| `-delimited, so an outlier can be audited |
| `WPI Threshold Finding` | enum | **100%** | `above 10%` 263 · `not above 10%` 187 · `not determined` 90 — **the legally operative fact under s 4.11**, often settled without any percentage being stated. Never empty; pair with `WPI Threshold Finding Basis` to filter to judicial findings only |
| `WPI Threshold Finding Basis` | enum | 100% | `decision` 169 · `implied by non-economic loss award` 175 · `implied by stated WPI` 106 · `not determined` 90. **Filter to `decision` for findings the court actually made** |
| `WPI Governing System` | enum | 100% | `physical` 92 · `psychiatric` 52 · `combined` 3 · **`not determined` 116** · `not stated` 277 — which assessment the WPI is taken from. **Computed by the resolution pass** from the classified figures, not from which cells are populated — see §3.06. `not determined` = only one body system was ever quantified, so the comparison cannot be made; `not stated` = neither was |
| `NEL Threshold Consistent` | enum | 100% | `yes` 154 · `cannot determine` 386 — does the non-economic loss award agree with the threshold finding. `no` is now empty: the only row that produced one was an ex gratia payment, where the rule never applied (see §3.1) |
| `Non-Economic Loss` | number | 84.6% | General damages / pain and suffering |
| `Non-Economic Loss Status` | enum | 100% | |
| `Future Economic Loss` | number | 76.5% | Future loss of earning capacity. **Excludes future superannuation** — see §6 |
| `Future Economic Loss Status` | enum | 100% | |
| `Statutory Benefits` | text | 100% | Free-text status. The numeric equivalent is `Statutory Benefits Paid` (§7) |
| `Medical Costs` | enum | 100% | `Yes` / `No` / `Not addressed`. **Changed:** the old `N/A` sentinel is in pandas' default `na_values` and read back as null, which is why this column looked 0% populated |
| `Claimant Outcome` | enum | 100% | `For Claimant` / `Against Claimant` |
| `Nature` / `Result` | text | 100% | Dispute category and short legal summary |

> **Dropped:** `Weekly Benefit`. The spec asked us to populate it or drop it. We looked: a CTP damages assessment or settlement approval essentially never states a weekly statutory-benefit rate (1 row in 540). It is dropped from the workbook rather than left implying data we do not have. It remains in `output/detailed_payout_summary.csv`, where it is meaningful for workers-compensation rows.

### 3.0 Round 2 — the seven rows the request named

Each was checked against its source decision. **Three mean the opposite of what the row numbers suggested**, so they are recorded here rather than silently "fixed".

| row | case | finding |
|---|---|---|
| 22 | Antonio [2026] NSWPIC 213 | **Confirmed.** A Review Panel certified 3% for the *physical* injuries; MAS Sidorov separately certified 6% for Major Depressive Disorder. Both are right; the total took the lesser. `WPI Governing System = psychiatric` now says so. Neither figure crosses 10, so the s 4.11 answer is unaffected |
| 222 | Slyney [2024] NSWPIC 293 | **Confirmed.** Dr Home 3% physical, Dr George 5% psychiatric. Same shape as row 22 |
| 67 | Cowper [2025] NSWPIC 596 | **Confirmed and resolved.** Assessors Fitzsimons and Jeyasingam each found 0%. `WPI %` is now `0` (`derived`) — an assessment, not a null |
| 214 | Mason [2024] NSWPIC 348 | **The flag was right; the percentage was wrong.** Professor Cameron assessed a traumatic brain injury — mental status 0%, emotional and behavioural functioning 6%, left shoulder 7% — and certified 13%. He could only reach 13 by *combining* 6 with 7, and psychiatric impairment is never combined with physical under the Guidelines. So the 6% is a neurological component of the brain injury. `Injury Categories` agrees: brain injury \| head or facial \| lower limb \| upper limb, no psychiatric entry. The misattributed percentage is withdrawn; `Has Psychiatric Injury` stays `No` |
| 220 | Taylor [2024] NSWPIC 301 | **Not a contradiction.** See "Has Psychiatric Injury vs Psychological Injury Emphasis" above — emphasis 2 with flag `No` is the expected combination for a claimant whose psychological symptoms were discussed and never assessed |
| 97 | Washbourne [2025] NSWPIC 334 | **Already resolved.** The 8% was withdrawn by the s 4.11 quarantine below before this round; `WPI %` is blank, not 8. Psychiatric `0.0` is MAS Samuell's assessment of an adjustment disorder — assessed-at-zero, which is data |
| 407 | Silcocks [2023] NSWPIC 24 | **Real, and the WPI is now withheld.** The insurer paid $120,000 it did not owe and the Member approved it; see §3.1 |

**On `WPI Psychiatric % = 0.0`:** it means *assessed at zero*, never a null written as zero. Anything unassessed is blank with a `not_applicable` / `not_assessed` / `not_stated` provenance. That is the semantics the request asked for, and it is now enforced rather than assumed.

**On the one real defect class:** hand-decomposition found 1 uncaptured psychiatric certificate. Reading the classified mentions in `processed_sidecar.json` finds **13** — rows where a psychiatric MAS certificate or assessor total is in the text with no figure carried across (Tiwari 7%, Ratanasirilak 6%, CAC 18%, …). Those are the 13 `absent` values, and they are the whole of what a missingness check should fail on.

### 3.05 Round 3 — the money columns, and what the residual triage found

The four-way provenance vocabulary now covers **all 13 money columns**, so `FAIL on absent, INFO on everything else` can be applied everywhere with no rate heuristic. What a blank means depends on the kind of column:

| kind | columns | a blank means |
|---|---|---|
| **head** | `Non-Economic Loss`, `Past`/`Future Economic Loss`, `Other Damages Heads` | the paired `Status` decides: `Not addressed` → `not_applicable`; `Awarded` with no figure → `not_stated` (allowed but never broken out) |
| **event** | `Buffer Amount`, `Other Deductions`, the contributory-negligence pair, `Statutory Benefits Repaid` | it did not happen → `not_applicable` |
| **statutory** | `Statutory Benefits Paid`, `Treatment And Care Paid`, `Weekly Statutory Benefit` | under MAIA these are statutory benefits, not damages, so a damages determination that does not quantify one is complete → `not_applicable`. But where the row records a repayment, benefits demonstrably *were* paid, so the precondition arises and the value is `not_stated` — see §7.1 |
| **always** | `Net Sum Payable`, `Lump Sum`, `Total Damages Gross` | applies to every award, so never `not_applicable` → `not_stated` |

`absent` is reached only by **corroboration** — positive evidence the figure exists in the text. In practice one route survives: the accounting identity leaves a hole only one column can fill (`Other Damages Heads`, 8 rows). A stated repayment was briefly treated as corroborating `Statutory Benefits Paid`; round 4 §12.2 corrected that — see §7.1.

#### `Other Damages Heads` — a Status column, and 302 real zeros

This was the only money head shipped without a `Status` companion, so a considered-and-refused head and one never in issue both collapsed to null, and 71.3% of the column read as missing data. **`Other Damages Heads Status` is added** (`Nil` 302 · `Awarded` 155 · blank 83), and where the accounting identity closes the blank is written as the zero it is — `Other Damages Heads = 0`, provenance `derived`. Missingness falls from 71.3% to 15.4%. The Stage A workaround can be deleted.

New extractions read the status **from the decision** rather than deriving it: `other_damages_heads_status` is now part of the damages schema. The 540 existing rows keep the derived value.

#### The 17 high-residual rows are three different faults

Triaged against source. Only 8 are what the symptom looked like:

| class | n | what it is |
|---|---|---|
| **unapportioned global settlement** | 5 | Taaga: *"The parties agreed total damages at $1,900,000 … The decision did not apportion the $1,900,000 between heads of damage."* The whole gross shows up as residual and says nothing about other heads. Reading it as one would invent a $1.9m head that does not exist |
| **a named head holds the NET** | 3 | Pantelis: *"non-economic loss of $275,000. Contributory negligence of 20% reduced damages by $55,000"* — and $220,000 was recorded. The residual equals the reduction exactly. A real defect, but in `Non-Economic Loss`; now reported in `Damages Notes` |
| **genuinely uncaptured other head** | 8 | Macdonald (No 2): future superannuation of $26,244, which `Future Economic Loss` excludes by definition. These are the `absent` rows |

On an unapportioned award the named heads are marked `not_stated`, not `not_applicable` — they *were* allowed, just never broken out. A downstream `fel_applies` gate reading `not_applicable` there would drop a live head.

### 3.06 Round 5 — `WPI Governing System` was circular, and 11 physical figures were recoverable

**The column was a tautology on 145 rows.** It had been derived downstream from *which cells are populated*, so wherever only one component was captured it could only ever name that component — "the system we happen to hold is the system that governs". It was wrong wherever the missing component was the larger, and it contradicted the resolution's own notes on the rows that mattered: Quigley [2026] NSWPIC 280 read `psychiatric` (the only component captured, at 1%) while `WPI Resolution Notes` said *"higher of 2 body systems (physical)"*.

It is now emitted by `resolve_wpi` itself, from the same per-system comparison that writes those notes, with a new value:

| value | meaning |
|---|---|
| `physical` / `psychiatric` | both systems quantified; this one is the greater and governs |
| `combined` | one figure the decision itself says covers both |
| **`not determined`** | **only one system was ever quantified — the comparison cannot be made** |
| `not stated` | neither system was quantified |

Notes-vs-column mismatches: **2 → 0**.

#### The 11 recovered physical figures

Physical and psychiatric are assessed separately and the **greater governs**, so the accepted total *is* one of the two components. Where psychiatric is stated and the total **exceeds** it, physical must be the greater — and therefore equals the total. That is exact, not an estimate, and it fired on 11 rows (implied values 10 · 11 · 12 · 12 · 13 · 14 · 16.5 · 18 · 20 · 20 · 25). Each recovered figure inherits the total's provenance, so an inferred total cannot yield a stated component.

Confirmed against source. Quigley is the instructive one: MAS Curtin certified **4%** (scarring, nerve injury) and a Review Panel **8%** (brain injury, shoulder) — *different injuries*, so they combine to 12 rather than competing, and MAS Lahz's combined certificate independently certifies *"greater than 10%"*. The resolution ladder had read them as rival assessments and taken the median, 6. **The total was right and the ladder's per-system figure was not**, which is why the recovery works from the total rather than from the ladder's arithmetic.

> **The rule does not run backwards.** A total above the stated *physical* figure is usually further physical components combining, not a larger psychiatric one. Mason [2024] NSWPIC 348 — 7% shoulder and 6% emotional/behavioural combining to 13% inside one brain-injury assessment — would have gained a fabricated 13% psychiatric impairment on a claimant with no psychiatric injury at all.

#### What remains

`WPI Physical %` coverage 26.3% → **28.3%**; `absent` 23 → **12**, and those 12 are genuine:

| group | n | why it is still absent |
|---|---|---|
| A2 | 4 | the total *equals* psychiatric, so psychiatric governs and physical is known only to be ≤ it — bounded, not determined |
| B | 8 | both components absent, and the classified mentions show the decision quantified **both** systems, so the split was there and we lost it |

The 8 group-B rows are the same 8 that are `absent` in `WPI Psychiatric %`, so the two counts overlap: **17 distinct rows carry a split gap, not 25.** Their candidate figures are in `WPI % Candidates`. They were not auto-filled from the classified mentions because Quigley shows that per-system reduction can itself be wrong — a rival-vs-combine misclassification — and filling from a demonstrably shaky source is worse than flagging.

### 3.07 Round 6 — the split columns are filled from the resolution, and a review panel supersedes

A `WPI Governing System` label asserts that the two components were **compared**, so they cannot simultaneously be `absent`. That contradiction stood on 8 rows. Both components are now carried into `WPI Physical %` / `WPI Psychiatric %` from the same per-system reduction that names the winner — only where **both** systems resolved, since the split columns are defined as populated only when the decision states them separately.

`absent` across the whole workbook: **38 → 11.** `WPI Physical %` is now **0**; coverage 28.3% → **30.7%** physical, 16.9% → **18.5%** psychiatric.

#### A review panel supersedes the certificate it reviewed

Reconciling Tiwari [2026] NSWPIC 251 turned up a rule the ladder was missing. MAS Oates certified **0%** for the lumbar spine; a **Review Panel** then certified **12%**; MAS Roberts certified **7%** for PTSD. The ladder averaged Oates and the Panel to 6%, which made psychiatric (7%) look like the governing system — and that is exactly why the label read `psychiatric` while the total read 12.

A review panel is not offering a rival opinion. Under s 7.26 it may **revoke** the certificate it reviewed and issue its own, so the earlier figure is superseded, not competing. With that rule: physical **12**, psychiatric **7**, and **physical governs** — consistent with the accepted total.

> **The proposed fill would have been wrong here.** §14.1 suggested writing the total into whichever component the label named, which would have recorded psychiatric = 12 when psychiatric is 7. The label was the broken half, not the value.

The rule is scoped to **rival** certificates only. Where certificates cover *different injuries* they combine, and a panel that reviewed one does not supersede a separate certificate about others — Quigley's MAS Curtin 4% (scarring, nerve) and Review Panel 8% (brain injury, shoulder) still combine to 12.

#### `Other Damages Heads`

Triaged on the round-4 basis: `absent` now requires an **itemised** other head with a figure visible in the decision, not merely a residual. A residual localises a problem without identifying which column is wrong, and on these rows it usually indicted a different one — *"leaving $100,000 for future economic loss"* is an uncaptured FEL, and *"a 30% reduction for contributory negligence"* is a head holding the net. **8 → 2**, and both survivors are genuine (Macdonald's future superannuation of $26,244; a past-superannuation and Fox v Wood pair folded into past economic loss).

#### What remains

| column | `absent` | what they are |
|---|---|---|
| `WPI %` | 5 | genuine total-impairment misses, stable across four rounds |
| `WPI Psychiatric %` | 4 | psychiatric material and a percentage are present but no psychiatric total resolves |
| `Other Damages Heads` | 2 | an itemised other head was awarded and not captured |
| **total** | **11** | |

### 3.1 The s 4.11 quarantine — five blank WPIs

s 4.11 of the Motor Accident Injuries Act 2017 permits damages for non-economic loss **only where whole person impairment exceeds 10%**. So a row carrying `Non-Economic Loss Status = Awarded` alongside a `WPI %` at or below 10 is self-contradictory, and on audit the WPI is the wrong half nearly every time:

| case | captured | what it actually was |
|---|---|---|
| Washbourne [2025] NSWPIC 334 | 8% | ONE SHOULDER — "the shoulders were equally impaired … at 8% each and the cervical spine at 5%". Combines to ~20%; the Medical Panel's own total is never stated |
| Young [2023] NSWPIC 473 | 6% | Dr Wallace's figure, superseded by the insurer's concession that with scarring and muscle atrophy impairment "would likely to exceed the 10% threshold" |
| Bond [2024] NSWPIC 468 | 9% | Dr Lee's partial assessment. A settlement approval; "the parties agreed that entitlement … was enlivened", so nothing was ever certified |
| Singh [2024] NSWPIC 313 | 10% | the concession that injuries "**exceeded** 10%", plus a physical-only figure on a claimant who also had psychological injury |
| Ristevski [2023] NSWPIC 400 | 10% | Dr Gothelf's **shoulder-only** figure; the PTSD and major depressive disorder were assessed separately |

Because the governing total is usually never stated, these cannot be corrected — only withheld. Each row now has `WPI %` blank, `WPI % Provenance = absent`, and the withheld figure preserved in `WPI % Candidates` so it can be audited.

**The rows themselves stay in this workbook.** Only the impairment figure was wrong; the damages columns are sound, and Washbourne in particular is a complete $1,451,619 award with a full breakdown. A blank `WPI %` is already the exclusion that matters — 220 rows here have none, and WPI-conditional analysis filters on `WPI % Provenance = 'stated'`, which these five now fail. Evicting the whole row would cost ~120 good fields to suppress one bad one. `Needs Review` is deliberately **not** set for the same reason: it feeds the analysis-ready gate. The audit trail lives in `Review Notes`, `WPI % Basis` and `WPI % Candidates`.

> **The exception — an ex gratia payment.** Silcocks [2023] NSWPIC 24 awards $120,000 of non-economic loss on a certified 9% WPI, because the decision says the insurer paid what it did not owe: an "appropriate compromise … where **no legal obligation on insurer to make any allowance for non-economic loss**". The 9% is *correct*, unlike the five above — but it is withheld too, because publishing it makes every downstream s 4.11 check read the row as an impossible combination. A checker comparing WPI to 10 cannot see that the payment was never made under s 4.11 at all.
>
> What distinguishes it from a defect is the **provenance**: `WPI % Provenance = not_applicable` (the threshold question does not arise), against `absent` for the five genuine misses. `NEL Threshold Consistent` is `cannot determine` rather than `no` for the same reason — the award was not predicated on the impairment finding, so the rule has nothing to say about it. The figure survives in `WPI % Candidates` and `WPI Resolution Notes` records the reasoning.
>
> **Ingest rule:** an ex gratia row is `WPI %` blank + `WPI % Provenance = not_applicable`. It needs no exception list — there is no WPI to compare against 10.

Two scope limits worth knowing: the rule applies **only to `Case Type = CTP`** (workers compensation runs on s 66 of the Workers Compensation Act 1987, where 10% WPI with non-economic loss is unremarkable — Birleson and Tysoe v State of NSW are both untouched), and it tests **`> 10`, not `>= 10`**, because that is what the section says.

Re-run with `python backfill_wpi_nel_quarantine.py --dry-run`.

---

## 4. Heads of damage — NEW

The identity these columns are built to satisfy:

```
Total Damages Gross
  - Contributory Negligence Amount
  - Statutory Benefits Repaid
  - Other Deductions
  = Net Sum Payable
```

| column | type | coverage | notes |
|---|---|---|---|
| `Past Economic Loss` | number | 74.1% | **The head that was previously missing.** Past loss of earnings to assessment, including past superannuation where awarded as part of it |
| `Past Economic Loss Status` | enum | 100% | `Awarded` 387 · `Not addressed` 121 · `Nil` 32 |
| `Past Economic Loss Provenance` | enum | 100% | `stated` 342 · `absent` 140 · `derived` 58 |
| `Buffer Amount` | number | 2.6% | A buffer or global allowance that could **not** be assigned to a named head. A buffer awarded *for* future economic loss is reported in that head, not here — no dollar appears twice |
| `Buffer Basis` | text | 2.6% | What the buffer is for |
| `Other Damages Heads` | number | 28.7% | Total of other heads allowed as **damages**: interest, out-of-pockets, gratuitous/domestic care. Excludes treatment and care paid as statutory benefits |
| `Other Damages Heads Basis` | text | 28.7% | Itemisation |
| `Total Damages Gross` | number | 100% | Total **before** any deduction |
| `Total Damages Gross Provenance` | enum | 100% | `stated` 489 · `derived` 51 |
| `Damages Gross Derivation` | enum | 100% | How gross was obtained: `stated` (536) · `net plus deductions` · `net, no deductions found` (4) · `sum of heads`. **See the warning in §8** |

## 5. Deductions, and the gross-vs-net answer — NEW

| column | type | coverage | notes |
|---|---|---|---|
| `Contributory Negligence Percent` | number 0–100 | 8.0% | The reduction actually **found**, not one merely alleged. Blank where there is none — a `0` would read as a quantified nil finding |
| `Contributory Negligence Amount` | number | 7.4% | The dollar reduction |
| `Statutory Benefits Repaid` | number | 45.4% | The s 3.40 MAI Act deduction / insurer's entitlement to deduct |
| `Other Deductions` | number | 6.9% | Medicare, Centrelink, s 151Z workers-comp recovery, hospital charges. Legal costs ordered separately are **not** a deduction |
| `Deductions Basis` | text | 59.1% | e.g. `s 3.40(1)(b) statutory benefits` |
| `Net Sum Payable` | number | 85.4% | **The amount actually payable after every deduction** |
| `Net Sum Payable Provenance` | enum | 100% | `stated` 321 · `derived` 140 · `absent` 79 |
| `Lump Sum Provenance` | enum | 100% | Provenance of the delivered `Lump Sum`: `stated` 498 · `derived` 24 · `inferred` 18 |
| `Lump Sum Basis` | enum | 100% | Which reading applies **per row**: `net of deductions` 288 · `gross` 236 · `unclear` 16 |

### ⚠ `Lump Sum` is GROSS, not net

The spec asked us to confirm or correct the assumption that `Lump Sum` is net. **It is not.** On the 300 rows where the distinction is testable (a stated gross *and* a non-zero deduction):

| test | result |
|---|---|
| `Lump Sum` matches **gross** | **83.3%** |
| `Lump Sum` matches **net** | 17.3% |
| `Net Sum Payable` = gross − deductions | **99.8%** (457 testable rows) |

`Lump Sum` is unchanged, as requested. **Use `Net Sum Payable` for anything that needs the amount the claimant actually received**, and `Lump Sum Basis` to see which reading holds on a given row. Where `Net Sum Payable` is blank (15%), the decision did not state a payable sum distinct from the total.

## 6. Independent re-extraction of the trusted heads — NEW

`Non-Economic Loss` and `Future Economic Loss` were re-extracted independently rather than overwritten, so extractor accuracy stays measurable against columns already trusted.

| column | type | coverage |
|---|---|---|
| `Non-Economic Loss (Recheck)` | number | 83.3% |
| `Non-Economic Loss Status (Recheck)` | enum | 100% |
| `Future Economic Loss (Recheck)` | number | 73.7% |
| `Future Economic Loss Status (Recheck)` | enum | 100% |
| `Non-Economic Loss Provenance` / `Future Economic Loss Provenance` | enum | 100% |

**Agreement with the trusted columns:** NEL 98.6% of 444 comparable rows; FEL 95.2% of 394. Both clear the 95% bar. The disagreements are systematic, not noise — do not treat them as errors:

- **19 FEL disagreements** — recheck ≈ old × 1.115. The original captured future loss of earning capacity only; the recheck **includes future superannuation**, which belongs in the gross total. `Total Damages Gross` is built on the recheck value.
- **6 NEL disagreements** — the original recorded the head *after* a contributory-negligence reduction (e.g. $112,500 where the assessed head was $225,000 with a 50% finding). The recheck records the assessed head, with the reduction in `Contributory Negligence Percent`/`Amount`.

Every disagreement is named in `Damages Notes` on the row.

## 7. Statutory benefits — NEW (outside the damages reconciliation)

Under MAIA 2017 treatment and care are **statutory benefits, not damages**. These sit outside the identity in §4 and must not be added to the heads.

| column | type | coverage | notes |
|---|---|---|---|
| `Statutory Benefits Paid` | number | **43.0%** | Total statutory benefits paid to date, where quantified. `stated` 231 · `not_applicable` 257 · `not_stated` 51 · `derived` 1 |
| `Treatment And Care Paid` | number | 1.9% | Where quantified |
| `Weekly Statutory Benefit` | number | 0.2% | Latest weekly rate. Near-empty because CTP damages decisions genuinely do not state one — the paired provenance column says `not_applicable` on 539 of 540 rows, which is the honest signal, not a gap in extraction |
| `... Provenance` (each of the three) | enum | 100% | |

### 7.1 ⚠ `Statutory Benefits Paid` is NOT `Statutory Benefits Repaid`

The names invite the assumption that these are the same number. They are not, and the difference is information:

| column | what it is |
|---|---|
| `Statutory Benefits Paid` | **everything the claimant received** — weekly payments *plus* treatment and care |
| `Statutory Benefits Repaid` (§5) | **the s 3.40 deduction**, which reaches only the recoverable categories, in practice the weekly wage payments |
| the difference | the **non-deductible** categories, chiefly treatment and care |

On the rows where both are stated they are equal within $1 on 96–97%, which is exactly the trap: they coincide only because `Treatment And Care Paid` is 98.1% `not_applicable`. The rows that differ are systematic, and one settles the semantics outright — its difference of **$34,893.12 equals that row's `Treatment And Care Paid` to the cent**.

**So never impute one from the other.** Deriving Paid from Repaid would close every gap in this column at a stroke and would understate benefits paid in precisely the cases the field is most useful for. Where the decision states only the deduction, `Statutory Benefits Paid` is left blank with provenance `not_stated` — 51 rows.

### 7.2 MACA-era wording

The predecessor scheme says the same things in different words, and an extractor keyed on MAIA vocabulary misses them by construction:

| MAIA 2017 | MACA 1999 |
|---|---|
| "statutory benefits" | "s 83 payments" |
| "s 3.40(1)(b) deduction" | "s 130 credit" |

This was measurable: MACA-era language was **11× enriched** among the rows where this field failed (16.1% of failures against 1.4% of successes). Greer [2026] NSWPIC 279 — *"s 130 MACA credit for s 83 payments"* — is the shape of it.

Both halves are fixed. The field prompts now name the MACA equivalents, and — the structural half — `s 83` / `s 130` / `MACA` were added to the keyword list that selects which part of a long decision the model ever sees, so a MACA quantum section is no longer trimmed away before extraction. **22 rows were recovered from source, 18 of them in MACA wording.**

> A note on reading these: s 83 is the insurer's obligation to **pay**; s 130 is the separate right to **recover**. So an itemised "Section 83 payments $215,552.18" states an amount paid even when it appears under a s 130 credit heading. The MAIA phrasing is the ambiguous one — "a deduction of $X for weekly payments of statutory benefits paid" states the *deduction*, and is recorded as `Repaid`.

## 8. Reconciliation — NEW

| column | type | coverage | notes |
|---|---|---|---|
| `Damages Reconciled` | enum | 100% | `yes` 490 · `no` 28 · `insufficient data` 22 |
| `Damages Residual` | number, **signed** | 95.9% | `Total Damages Gross − Σ heads`. Negative where the heads exceed the stated gross |
| `Net Reconciled` | enum | 100% | The second identity: payable sum vs gross − deductions. `yes` 286 · `no` 250 · `insufficient` 4. The `no` count is the gross-vs-net finding in §5, not an extraction failure |
| `Net Residual` | number, **signed** | 99.3% | |

Σ heads = `Non-Economic Loss (Recheck)` + `Past Economic Loss` + `Future Economic Loss (Recheck)` + `Buffer Amount` + `Other Damages Heads`, with `Nil` and `Not addressed` contributing a known zero.

> **⚠ `Damages Reconciled` is deliberately conservative.** It reports `insufficient data` — never `yes` — when `Damages Gross Derivation` is `sum of heads`, because that identity would close by construction and tell you nothing. It also reports `insufficient data` when an `Awarded` head has no amount. Treat `yes` as a real check.

**Measured:** 97.3% of the 111 rows where all four figures are `stated` reconcile within $1,000. Across **all** rows: 90.7% `yes`, 5.2% `no`, 4.1% `insufficient`. The `no` rows split between small uncaptured items (interest, out-of-pockets: residuals of $2k–$4.5k) and decisions that state a total without apportioning it — in the latter, the residual equals the whole total, which is the correct signal that no breakdown is available.

## 9. Award breakdown prose — NEW

| column | type | coverage | notes |
|---|---|---|---|
| `Award Breakdown` | text | 100% | 2–4 sentences stating the breakdown **with `$` figures**, in the register the source decisions use. Anonymised (no party, Member, doctor or firm names) |
| `Description With Figures` | text | 100% | `Description` + `Award Breakdown`. **100% contain a `$` figure**, against 2.8% for `Description` |

`Description` and `Banded Description` are **unchanged**. `Description With Figures` is the parallel field the spec offered as an alternative — use it as the generator's training target.

## 10. Accident mechanism — NEW

| column | type | coverage | distribution |
|---|---|---|---|
| `Accident Mechanism` | enum | 100% | `vehicle collision` 255 · `pedestrian struck` 101 · `single vehicle` 61 · `motorcyclist` 56 · `cyclist` 36 · `unclear` 18 · `passenger` 10 · `other` 3 |
| `Claimant Road Role` | enum | 100% | `driver` 153 · `passenger` 127 · `pedestrian` 101 · `motorcyclist` 82 · `other` 40 · `cyclist` 37 |

`unclear` means the decision does not say — it is not a residual bucket for hard cases. **Every row is a motor accident**; this is the real distribution to constrain generation against.

## 11. Injury — NEW

| column | type | coverage | notes |
|---|---|---|---|
| `Injury Categories` | ` \| `-delimited list | 100% | All that apply, deduplicated, alphabetical. Vocabulary: `brain injury`, `spinal`, `upper limb`, `lower limb`, `chest or abdominal`, `head or facial`, `psychiatric`, `chronic pain`, `scarring or disfigurement`, `soft tissue`, `fatality`, `other` |
| `Primary Injury Category` | enum | 100% | The dominant one: `spinal` 130 · `lower limb` 130 · `upper limb` 127 · `psychiatric` 64 · others 89 |
| `Has Psychiatric Injury` | `Yes`/`No` | 100% | **`Yes` on 229 of 540 (42.4%)** — a large class the previous single-label treatment could not represent |
| `WPI Physical %` | number | 26.3% | Only where the decision states it **separately** |
| `WPI Psychiatric %` | number | 17.0% | Only where stated separately |
| `WPI Physical % Provenance` / `WPI Psychiatric % Provenance` | enum | 100% | `stated` or `absent` only — never `inferred` |

> **On the split-WPI coverage.** The spec expected ~8%; we deliver 30.7% / 18.5% across the full 540 (44% / 31% among the rows that state any WPI at all). This is **not** a combined figure being split. CTP matters routinely carry two separate MAS certificates — e.g. one assessor certifying 6% for PTSD and persistent depressive disorder, another certifying 4% for cervical/thoracic/lumbar soft tissue. Note that `WPI %` is generally the **higher** of the two, not their sum: MAIA assesses physical and psychiatric separately and the greater governs the threshold.

## 12. Workers-compensation overlap — NEW

| column | type | coverage | notes |
|---|---|---|---|
| `WC Overlap` | ordinal 0–2 | 100% | `0` none (494) · `1` deduction or passing mention (29) · `2` substantial interaction shaping the damages reasoning (17) |

Lower than the 33.5% narrative-mention rate the spec cites, because this scores whether a **parallel workers-compensation claim** exists, not whether the words appear.

## 13. Extraction status and QA

| column | type | coverage | notes |
|---|---|---|---|
| `Damages Extraction Status` | enum | 100% | `ok` (540) · `not run` · `not applicable` · `error`. **Anything other than `ok` means the damages columns on that row are defaults, not findings** |
| `Damages Notes` | text | 47.4% | Per-row issues: disagreements with the trusted columns, dropped out-of-range values, status corrections. Read this before trusting an outlier |
| `Fatality Or Dependency Claim` | `Yes`/`No` | 100% | `Yes` on 10. Compensation to Relatives claims lack the ordinary heads, so those rows carry `Not addressed`, never `Nil` |
| `Status` | enum | 100% | Row processing status, `ok` |
| `LLM Error` | text | 0% | Empty = no extraction error |
| `Needs Review` / `Review Notes` | enum / text | 100% / 23.5% | Field-loss gate output for the **eight original** high-value fields |
| `Analysis Ready` / `Analysis Exclusion Reason` | enum / text | 100% / 0% | `Yes` on all rows in this workbook by construction |

## 14. Ordinal analysis dimensions (unchanged)

All 100% populated, integer, scored against rubrics in `nsw_court_scraper.py`.

| column | range |
|---|---|
| `Injury Burden Intensity` | 0–4 (minimal → catastrophic) |
| `Psychological Injury Emphasis` | 0–2 |
| `Liability Clarity` | 0–2 (contested → clear) |
| `Causation Complexity` | 0–2 |
| `Treatment Burden` | 0–3 |
| `Work Impact Severity` | 0–3 |
| `Pre-existing Condition Salience` | 0–2 |
| `Legal Procedural Complexity` | 0–3 |

## 15. Narrative and verbatim fields (unchanged)

| column | coverage | notes |
|---|---|---|
| `Narrative: …` (10 columns) | 100% | 60–150 words each, anonymised. Profile, mechanism, injuries, treatment, functional impact, medical evidence, prior insurer actions, claimant submissions, insurer submissions, legal reasoning |
| `Slice (LLM): Catchwords` | 83.3% | Verbatim slices cut from the source by marker |
| `Slice (LLM): Determinations or Orders` | 98.9% | **Highest-value verbatim source for award figures** |
| `Slice (LLM): Introduction` | 98.0% | |
| `Catchwords` | 100% | Deterministic regex parse of the AustLII `CATCHWORDS:` block — independent of the LLM |
| `Key Paragraphs` | 100% | 4–8 numbered paragraphs from the Member's reasoning, with rationale and verbatim text |
| `Event History` | 100% | `date \| actor \| tag` per line, chronological |
| `Regulatory Sections` | 99.4% | ` \| `-delimited statutory provisions relied on |

---

## Related outputs

All generated artifacts live under `output/`.

| file | contents |
|---|---|
| `output/ctp_impairment_lump_sum.xlsx` | **This workbook.** 540 CTP rows × 128 columns |
| `output/detailed_payout_summary.csv` | All 3,501 cached decisions, all case types, same flat columns. Damages columns carry `Damages Extraction Status = not run` on the 2,961 rows outside the CTP workbook population |
| `output/analysis_ready_payout_summary.csv` | The analysis-ready subset of the above |
| `output/processed_sidecar.json` | Keyed by `URL`. Verbatim **provenance quotes** for every damages figure under `damages.quotes`, plus `damages.issues`, `damages.gross_derivation`, narrative sub-fields, slices, key paragraphs, event history |
| `output/run_manifest.json` | Provenance of the delivery: `schema_version`, `damages_version`, row counts, `damages_rows_extracted`, generation timestamp, privacy settings |

## Privacy

Defaults retain everything, because this is public NSWPIC material and identity/DOB/provenance are load-bearing for the payout-vs-WPI use case. A privacy-sensitive consumer can regenerate the outputs with `NSW_PRIVACY_DROP_IDENTITY=1`, `NSW_PRIVACY_DROP_DOB=1`, `NSW_PRIVACY_DROP_PROVENANCE=1`, or `NSW_PRIVACY_NAME_MODE=hash|redact`. `Description` and `Banded Description` are already anonymised by prompt design; so is `Award Breakdown`.

## Reproducing the checks

```bash
python check_damages_acceptance.py          # the spec's section 6 criteria
python test_damages_extraction.py           # 35 deterministic damages tests
python test_extraction_fields.py            # 40 field-reliability tests
```
