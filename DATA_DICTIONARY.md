# Data dictionary — `ctp_impairment_lump_sum.xlsx`

**Delivery:** damages-breakdown spec v1 (request dated 2026-07-27)
**Generated:** 2026-08-06 · **schema version** 3 · **damages version** 1
**Shape:** 540 rows × 124 columns · one row per decision
**Population:** analysis-ready CTP decisions with a **positive lump sum**. A WPI is *not* required.

> **Row count.** 540, matching the count the spec refers to. An interim version of this workbook also required a positive accepted WPI, which cut it to 287 by dropping awards whose decision never states a WPI — a fact about the decision, not a defect in the row. Those 253 rows are back, with `WPI %` blank and `WPI % Provenance = absent`. Filter on `WPI % Provenance = 'stated'` for any WPI-conditional analysis.
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

Every money field has one. Populated on 100% of rows.

| value | meaning |
|---|---|
| `stated` | the figure appears verbatim in the decision |
| `derived` | computed by arithmetic from figures that do appear |
| `inferred` | model judgement rather than the document's — **exclude or down-weight** |
| `absent` | no amount |

The verbatim source snippet behind each figure is in `processed_sidecar.json` under `damages.quotes`, keyed by decision URL.

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
| `WPI %` | number | **60.0%** | The WPI the award is calibrated against (accepted, whether assessed here or in a prior MAS certificate). Blank where the decision does not state one |
| `WPI % Provenance` | enum | 100% | `stated` 315 · `inferred` 9 · `absent` 216. **Exclude `inferred` for WPI-conditional analysis** — those are central estimates of competing assessments, not figures the decision states |
| `WPI % Basis` | text | 100% | how the figure was resolved: `tribunal selected`, `MAS certificate`, `assessor total`, `combined from N components (AMA Combined Values)`, `median of N competing assessments`, `withheld: … contradicts …`, or `retained from main extraction` |
| `WPI % Candidates` | text | 29% | every distinct percentage found in the decision, ` \| `-delimited, so an outlier can be audited |
| `WPI Threshold Finding` | enum | 29% | `above 10%` / `not above 10%` / `not determined` — **the legally operative fact under s 4.11**, often settled without any percentage being stated. Populated on the 158 rows the resolution pass examined |
| `Non-Economic Loss` | number | 84.6% | General damages / pain and suffering |
| `Non-Economic Loss Status` | enum | 100% | |
| `Future Economic Loss` | number | 76.5% | Future loss of earning capacity. **Excludes future superannuation** — see §6 |
| `Future Economic Loss Status` | enum | 100% | |
| `Statutory Benefits` | text | 100% | Free-text status. The numeric equivalent is `Statutory Benefits Paid` (§7) |
| `Medical Costs` | enum | 100% | `Yes` / `No` / `Not addressed`. **Changed:** the old `N/A` sentinel is in pandas' default `na_values` and read back as null, which is why this column looked 0% populated |
| `Claimant Outcome` | enum | 100% | `For Claimant` / `Against Claimant` |
| `Nature` / `Result` | text | 100% | Dispute category and short legal summary |

> **Dropped:** `Weekly Benefit`. The spec asked us to populate it or drop it. We looked: a CTP damages assessment or settlement approval essentially never states a weekly statutory-benefit rate (1 row in 540). It is dropped from the workbook rather than left implying data we do not have. It remains in `output/detailed_payout_summary.csv`, where it is meaningful for workers-compensation rows.

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
| `Statutory Benefits Paid` | number | 38.9% | Total statutory benefits paid to date, where quantified |
| `Treatment And Care Paid` | number | 1.9% | Where quantified |
| `Weekly Statutory Benefit` | number | 0.2% | Latest weekly rate. Near-empty because CTP damages decisions genuinely do not state one — the paired provenance column says `absent` on 539 of 540 rows, which is the honest signal, not a gap in extraction |
| `... Provenance` (each of the three) | enum | 100% | |

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

> **On the split-WPI coverage.** The spec expected ~8%; we deliver 26.3% / 17.0% across the full 540 (44% / 31% among the rows that state any WPI at all). This is **not** a combined figure being split. CTP matters routinely carry two separate MAS certificates — e.g. one assessor certifying 6% for PTSD and persistent depressive disorder, another certifying 4% for cervical/thoracic/lumbar soft tissue. Note that `WPI %` is generally the **higher** of the two, not their sum: MAIA assesses physical and psychiatric separately and the greater governs the threshold.

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
| `output/ctp_impairment_lump_sum.xlsx` | **This workbook.** 540 CTP rows × 124 columns |
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
