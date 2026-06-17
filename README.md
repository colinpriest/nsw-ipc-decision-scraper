# NSW Court Decision Extractor (AI Enhanced)

A Python tool that scrapes NSW Personal Injury Commission (NSWPIC) decisions from AustLII and uses OpenAI's `gpt-5` to extract structured legal information including payout amounts, injury details, dates, whole-person impairment, and case outcomes.

![NSW Court Decisions](nsw-court-decisions.png)

## Features

- **Automated Scraping**: Fetches court decisions from AustLII NSWPIC index
- **HTML and PDF Support**: Extracts text from both HTML and PDF decisions
- **AI-Powered Extraction**: Uses `gpt-5` with a single combined structured-output schema (Pydantic) to extract detailed case information
- **Threshold-Aware WPI Extraction**: Separately tracks the WPI *made* in the proceeding and the WPI *accepted* as the basis of the award. The high-precision regex + threshold-aware safeguards now run **in the live extraction path** (not just the backfill scripts), seeding/backfilling/cross-checking the WPI on every extraction (see "WPI extraction" below)
- **Field-loss safeguards (v3)**: A per-field provenance trail, a loss-detection gate, and a focused second pass protect the eight high-value fields (WPI, non-economic loss, weekly income, future economic loss, age, gender, occupation, location) against silent loss — see "Reliability & field-loss safeguards"
- **Intelligent Caching**: Caches extracted data (keyed by URL) to avoid re-processing decisions. Cache entries are stamped with a `_schema_version`; bumping it forces re-extraction of stale rows
- **Parallel Processing**: Processes multiple decisions concurrently (default 25 threads) for faster execution
- **Analysis-Ready Filtering**: Flags rows that are unsuitable for analysis and exports a filtered report
- **Comprehensive Data Extraction**: Extracts:
  - Applicant and Respondent names
  - Claimant demographics: age at injury, **age at decision**, **date of birth** (cross-check), gender (enum), occupation, weekly income and its **basis** (PIAWE/gross/net, conversion used), plus employer
  - Accident/injury location, with normalised **locality** and **state**
  - Claimant outcome (For/Against Claimant)
  - Case type (Workers Compensation, CTP, Other)
  - Impairment percentage (both "made" and "accepted")
  - Lump sum, weekly benefit, statutory benefits
  - Non-economic loss and future economic loss, each as a numeric **amount + status** (Awarded / Nil / Not addressed)
  - Medical costs awarded status
  - Decision nature and result
  - Case description with injury details, plus a PII-banded description
  - Verbatim catchwords (parsed without the LLM)
  - Ordinal analysis dimensions (injury burden, psychological emphasis, liability clarity, causation complexity, treatment burden, work impact, pre-existing condition salience, legal/procedural complexity) and regulatory sections
  - Dates (injury and decision)
  - Jurisdiction
  - Per-field provenance quotes and a Needs Review flag (see "Reliability & field-loss safeguards")

## Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key**: You must have a paid OpenAI account with access to `gpt-5`
3. **Internet Connection**: Required to access AustLII and OpenAI API

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** in the project root:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   ```

## Usage

Run the script:
```bash
python nsw_court_scraper.py
```

The script will:
1. Fetch the list of decisions from AustLII NSWPIC indexes for years 2021 to present
2. Process each decision (using cached data when available)
3. Save HTML files to `nsw_pic_decisions/`
4. Generate a full CSV report: `detailed_payout_summary.csv`
5. Generate an analysis-ready CSV report: `analysis_ready_payout_summary.csv`
6. Update the cache file: `processed_cache.json` (saved periodically during processing)
7. Print a data-quality summary plus an analysis-ready data summary to the console

## Output Files

### `detailed_payout_summary.csv`
A comprehensive CSV file containing all extracted data with columns:
- Case Name, URL, File Saved
- Jurisdiction, Case Type
- Decision Date, Injury Date
- Applicant, Respondent
- Claimant Age, Claimant Age At Decision, Claimant Gender, Claimant Occupation
- Claimant Weekly Income, Claimant Weekly Income Basis
- Employer Name, Accident/Injury Location, Location Locality, Location State
- Claimant Outcome
- Impairment % *(WPI made in this proceeding)*
- Lump Sum, Weekly Benefit
- Non-Economic Loss, Non-Economic Loss Status, Future Economic Loss, Future Economic Loss Status, Statutory Benefits
- Medical Costs
- Nature, Result
- Description, Banded Description, Catchwords
- Impairment % (Accepted) *(WPI the lump sum is calibrated against — see "WPI extraction")*
- Ordinal dimensions: Injury Burden Intensity, Psychological Injury Emphasis, Liability Clarity, Causation Complexity, Treatment Burden, Work Impact Severity, Pre-existing Condition Salience, Legal Procedural Complexity
- Regulatory Sections
- Status, LLM Error, Needs Review, Review Notes, Analysis Ready, Analysis Exclusion Reason

> **Note on damages heads:** `Non-Economic Loss` and `Future Economic Loss` now hold a pure numeric amount (no `$`/commas), with the disposition in the paired `... Status` column — `Awarded` (amount present), `Nil` (claimed but refused/zero, amount `0`), or `Not addressed` (empty). This stops "refused" being confused with "not dealt with".

### `analysis_ready_payout_summary.csv`
Filtered CSV export containing only rows that are suitable for downstream analysis. Rows are excluded if processing failed, the LLM extraction failed, the decision date is missing/invalid, or the row is flagged **Needs Review** (a high-value field the loss gate believes is in the source but was not captured even after the focused second pass). Set `NSW_EXCLUDE_NEEDS_REVIEW=0` to keep Needs Review rows in the analysis-ready set.

### `nsw_pic_decisions/`
Directory containing:
- Original HTML files for each decision
- Files are named using sanitized case titles

### `processed_cache.json`
JSON cache file storing all extracted data keyed by URL. This prevents re-processing decisions that have already been analyzed. Each row carries a `_schema_version`; when the schema is bumped, rows below the current version are re-extracted.

### `processed_sidecar.json`
Companion JSON (keyed by URL) holding the long/nested fields that don't fit a CSV cell: narrative sub-fields, LLM-marked verbatim slices, key paragraphs, event history, regulatory sections, token usage, banding validation, the **per-field provenance quotes** (`provenance`), and the **field-review record** (`field_review`: the loss-gate issues, the focused second-pass outcome, DOB and age-at-decision). Consumed by `ctp_lump_sum_impairment.py`.

### `ctp_impairment_lump_sum.xlsx`
Filtered Excel export containing only CTP cases that have both an Impairment % and a Lump Sum value. Generated by running `ctp_lump_sum_impairment.py` (see below).

### `scraper.log`
Log file containing execution details, errors, and processing status.

## How It Works

1. **Multi-Year Index Scraping**: Fetches AustLII NSWPIC index pages for years 2021 to present, identifying decision links using regex pattern matching (`/NSWPIC/YYYY/NUMBER.html` and PDF equivalents)

2. **Decision Download with Retry Logic**: Downloads each decision with exponential backoff retry logic:
   - Handles rate limiting (403, 429) and server errors (500, 502, 503, 504)
   - Retries connection errors and timeouts
   - Uses random jitter to avoid thundering herd problems

3. **Text Extraction**: Extracts clean text from HTML (focusing on the main content area) and PDF decisions

4. **AI Extraction**: 
   - Uses `gpt-5` with a single combined structured-output schema via Pydantic models
   - Extracts financial highlights and WPI using regex patterns
   - Implements retry logic for inconsistent extractions and quota/rate-limit errors
   - Handles long documents by truncating to key sections

5. **Thread-Safe Caching**: 
   - Checks cache before processing (thread-safe)
   - Saves new extractions to cache with locking
   - Periodically saves cache every 20 completions to prevent data loss
   - Handles corrupted cache files by backing them up and starting fresh

6. **Parallel Processing**: Uses ThreadPoolExecutor (default 25 workers, configurable via the `EXTRACTION_WORKERS` env var) to process multiple decisions concurrently

7. **CSV Generation**: Writes a full audit CSV and a filtered analysis-ready CSV, both sorted by valid decision date

## WPI extraction

Whole Person Impairment is tracked in two columns:

- **`Impairment %`** — the WPI *made in this proceeding* (the Member's binding finding). Usually empty for CTP settlement approvals/damages assessments that merely accept a prior MAS certificate.
- **`Impairment % (Accepted)`** — the WPI the lump sum is actually *calibrated against*, regardless of who assessed it. This is the column used for downstream payout-vs-WPI analysis.

As of v3 the WPI safeguards run **in the live extraction path** (`reconcile_wpi` in `nsw_court_scraper.py`), not only in the backfill scripts. On every extraction the pipeline:

1. **Cleans** both WPI values — drops implausible values (outside 0–100%) and a lone `0` (almost always statutory-threshold / minor-injury framing, not a finding).
2. **Seeds** `Impairment % (Accepted)` from `Impairment %` when the lenient value is blank (a WPI made here is, by definition, relied on).
3. **Regex-backfills** the accepted value when the LLM left it blank and exactly one distinct non-zero WPI appears in the text. Statutory-threshold framing is deliberately ignored — under the MAI Act 2017 the >10% WPI bar gates non-economic loss, so phrases like "does not exceed the threshold of 10% whole person impairment" or "greater than 10% WPI" are *not* findings about the claimant. `find_wpi_candidates`/`_is_threshold_mention` drop these.
4. **Cross-checks** a populated accepted value against the lone-token regex value and flags any mismatch for review (may be a legitimate combined-vs-component case).

The standalone `backfill_wpi_accepted.py` / `reprocess_threshold_wpi.py` scripts remain for re-running these safeguards over an existing cache without a full re-extraction.

## Reliability & field-loss safeguards

Version 3 hardens the extraction of the eight high-value fields — **WPI %, Non-Economic Loss, Claimant Weekly Income, Future Economic Loss, Claimant Age, Gender, Occupation, Location** — so they are captured reliably and not silently dropped:

- **Per-field provenance (A2):** the model returns a verbatim source quote for each of the eight fields. A non-empty quote next to an empty value is treated as a contradiction. Stored in the sidecar under `provenance`.
- **Field-loss gate (A1):** after the first pass, `detect_field_losses` flags any field that is empty but whose value is clearly present in the source (a strong textual signal, e.g. a `% WPI`/`PIAWE`/`aged NN` token near a dollar amount, or a provenance quote the model itself supplied). It deliberately skips NEL/FEL marked `Nil` (a real determination).
- **Focused second pass (A6):** when a high-value field looks lost, `extract_focused` re-asks **only** for the suspect fields, at a higher reasoning effort (`NSW_FOCUSED_REASONING_EFFORT`, default `medium`). Recovered values are merged without overwriting anything the first pass already captured.
- **Needs Review (C4):** if a high-severity loss survives the second pass, the row is flagged `Needs Review = Yes` and held out of the analysis-ready set, with a human-readable `Review Notes` summary.
- **Coercion & plausibility (B2/B3):** money fields are coerced to clean numbers; weekly income is range-checked (`NSW_INCOME_WEEKLY_MIN`/`MAX`, default 50–15000) to catch unit errors (an annual figure not converted to weekly).
- **Age cross-check (B4):** the DOB-derived age is compared against the stated age and a mismatch is flagged.
- **Truncation anchoring (A3):** for long decisions, `_narrative_truncate` keeps the quantum/claimant sections (non-economic loss, PIAWE, WPI, DOB, occupation, …) and anchors on both the first and last occurrence of each keyword, so the section where these fields live survives. The single-pass budget is raised to 200k chars (`NSW_SINGLE_PASS_LIMIT_CHARS`).

### `test_extraction_fields.py`
Golden-fixture regression tests for the deterministic helpers above (coercion, WPI reconciliation incl. the threshold trap, the loss gate, the focused-pass merge, age/DOB cross-check, truncation anchoring). Run them whenever the prompt or `SCHEMA_VERSION` changes:

```bash
python test_extraction_fields.py   # or: pytest test_extraction_fields.py
```

## Additional Scripts

### `ctp_lump_sum_impairment.py`
A standalone filtering script that reads the analysis-ready CSV and produces an Excel workbook (`ctp_impairment_lump_sum.xlsx`) containing only CTP cases where both Impairment % and Lump Sum are present. This workbook is the basis of the payout-vs-WPI analysis.

```bash
python ctp_lump_sum_impairment.py
```

This script prefers `analysis_ready_payout_summary.csv` when available, and falls back to `detailed_payout_summary.csv` while still filtering out rows that are not analysis-ready. Requires `pandas` and `openpyxl` (included in `requirements.txt`).

> **Note:** the backfill/reprocess scripts below rewrite the CSVs but **not** `ctp_impairment_lump_sum.xlsx`. After running any of them, re-run `ctp_lump_sum_impairment.py` to refresh the workbook.

### `backfill_wpi_accepted.py`
Backfills the `Impairment % (Accepted)` column for CTP cases (regex stage, then focused-LLM stage), seeds it from `Impairment %` where already set, and regenerates both CSVs.

### `reprocess_threshold_wpi.py`
One-off maintenance: re-extracts `Impairment % (Accepted)` for cases whose value was a statutory-threshold false positive (e.g. a bare `10` read from threshold language) or a rejected/disputed claimant figure. Backs up the cache, re-runs the threshold-aware regex and hardened LLM, and regenerates both CSVs.

### `backfill_catchwords.py`
Backfills the verbatim `Catchwords` column by parsing the AustLII `CATCHWORDS:` block directly (no LLM).

### `backfill_age_from_dob.py`
Backfills/cleans `Claimant Age`, deriving it from a stated date of birth and the injury date where possible.

### `reprocess_offline.py`
Re-runs extraction from already-downloaded local HTML/PDF files without re-scraping AustLII.

## Configuration

You can modify these settings in `nsw_court_scraper.py`:

- **Years to process**: Modify the `years` list in `main()` to change the range (currently 2021 to present)
- **OUTPUT_DIR**: Change the output directory name
- **CSV_REPORT**: Change the CSV filename
- **ThreadPoolExecutor max_workers**: Adjust the number of parallel threads (default: 25, or set the `EXTRACTION_WORKERS` env var)
- **Retry settings**: Adjust `max_retries` in `_make_request_with_retry()` (default: 5)
- **Cache save frequency**: Change the interval in `main()` where cache is saved (currently every 20 completions)
- **AUSTLII_INDEX_DELAY**: Seconds to wait between index page requests (default: 2)
- **AUSTLII_RATE_LIMIT_DELAY**: Seconds to throttle decision requests after rate limiting is detected (default: 5)

### v3 field-reliability env vars

- **NSW_SINGLE_PASS_LIMIT_CHARS**: Max chars sent to the model in one pass before keyword-anchored truncation (default: 200000)
- **NSW_FOCUSED_SECOND_PASS**: Set `0` to disable the focused second pass (default: on)
- **NSW_FOCUSED_REASONING_EFFORT**: Reasoning effort for the focused second pass (default: `medium`)
- **NSW_EXCLUDE_NEEDS_REVIEW**: Set `0` to keep `Needs Review` rows in the analysis-ready set (default: exclude)
- **NSW_INCOME_WEEKLY_MIN** / **NSW_INCOME_WEEKLY_MAX**: Plausible weekly-income range for the unit-error check (defaults: 50 / 15000)

## Important Notes

### Rate Limiting
- AustLII has anti-scraping measures
- The script automatically handles 403/429 errors with exponential backoff retry logic
- Retries up to 5 times with increasing delays (2^attempt seconds + random jitter)
- If you encounter persistent rate limiting, consider:
  - Reducing `max_workers` (default: 10)
  - Increasing delays between index page requests
  - Processing fewer years at once

### Testing Mode
To test with only the latest 8 decisions, modify the `main()` function:
```python
# Process only the latest 8 decisions for testing
target_links = all_links[-8:] if len(all_links) > 8 else all_links
```

Or to test with only a specific year:
```python
# Process only 2024 decisions for testing
years = [2024]
```

### Cache Management
- The cache file (`processed_cache.json`) stores all processed decisions
- Cache is saved periodically (every 20 completions) to prevent data loss
- Thread-safe operations ensure cache integrity during parallel processing
- If the cache file becomes corrupted, it's automatically backed up (`.corrupted` extension) and a new cache is started
- To reprocess all decisions, delete or rename the cache file
- The cache prevents unnecessary API calls and saves costs

### API Costs
- Uses `gpt-5` for higher accuracy; the focused WPI backfill uses a much cheaper small call
- Costs depend on document length and number of decisions
- Caching significantly reduces API usage on subsequent runs

## Data Schema

The extraction uses a structured Pydantic model with (among others) the following fields:

- `applicant_name`: Name of the Applicant/Claimant
- `respondent_name`: Name of the Respondent (usually insurer or employer)
- `claimant_age` (at injury) / `claimant_age_at_decision` / `claimant_date_of_birth`
- `claimant_gender`: Enum (Male / Female / Other / Not stated)
- `claimant_occupation`
- `claimant_weekly_income` / `claimant_weekly_income_basis`
- `employer_name`, `location_of_accident_or_injury`, `location_locality`, `location_state`
- `claimant_outcome`: Enum (For Claimant / Against Claimant)
- `case_type`: Enum (Workers Compensation / CTP / Other)
- `impairment_percentage`: WPI made in this proceeding (if assessed)
- `impairment_percentage_accepted`: WPI used as the basis of the award
- `lump_sum_amount`, `weekly_benefit_amount`, `statutory_benefits`
- `non_economic_loss` / `non_economic_loss_status`, `future_economic_loss` / `future_economic_loss_status`: amount + Enum (Awarded / Nil / Not addressed)
- `medical_costs_awarded`: Enum (Yes / No / N/A)
- `provenance`: per-field verbatim source quotes for the eight high-value fields
- `decision_nature`: Primary category (e.g., Liability Dispute, Permanent Impairment)
- `decision_result`: Short legal summary
- `case_description`: Paragraph summarizing injury, claimant, and reasoning
- `banded_case_description`: Case description with PII/specifics banded out
- ordinal analysis dimensions (injury burden, psychological emphasis, liability clarity, causation complexity, treatment burden, work impact, pre-existing salience, legal/procedural complexity)
- `date_of_injury` / `date_of_decision`: YYYY-MM-DD format
- `jurisdiction`: Enum (default: NSW)

## Troubleshooting

### "OPENAI_API_KEY not found"
- Ensure you have created a `.env` file with your API key
- Check that the `.env` file is in the same directory as the script

### "403 Forbidden" or "429 Too Many Requests"
- The script automatically retries with exponential backoff
- If retries are exhausted, the decision is skipped and logged
- Check `scraper.log` to see which decisions failed
- Consider reducing `max_workers` or processing fewer years at once

### Empty or incomplete extractions
- Check `scraper.log` for detailed error messages
- Some decisions may not contain all expected fields
- The retry logic should catch most inconsistencies

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### Corrupted cache file
- If you see "Cache file is corrupted" in the logs, the script automatically backs up the corrupted file (`.corrupted` extension) and starts fresh
- You can manually inspect the corrupted backup if needed
- The script will continue processing with an empty cache

## License

This project is provided as-is for educational and research purposes. Please respect AustLII's terms of service and rate limits when using this tool.

## Contributing

Feel free to submit issues or pull requests for improvements.
