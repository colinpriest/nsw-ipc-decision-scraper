# NSW Court Decision Extractor (AI Enhanced)

A Python tool that scrapes NSW Personal Injury Commission (NSWPIC) decisions from AustLII and uses OpenAI's GPT-4o to extract structured legal information including payout amounts, injury details, dates, and case outcomes.

![NSW Court Decisions](nsw-court-decisions.png)

## Features

- **Automated Scraping**: Fetches court decisions from AustLII NSWPIC index
- **AI-Powered Extraction**: Uses GPT-4o with structured output (Pydantic) to extract detailed case information
- **Intelligent Caching**: Caches extracted data to avoid re-processing decisions
- **Parallel Processing**: Processes multiple decisions concurrently (10 threads) for faster execution
- **Comprehensive Data Extraction**: Extracts:
  - Applicant and Respondent names
  - Claimant outcome (For/Against Claimant)
  - Case type (Workers Compensation, CTP, Other)
  - Impairment percentage
  - Lump sum and weekly benefit amounts
  - Medical costs awarded status
  - Decision nature and result
  - Case description with injury details
  - Dates (injury and decision)
  - Jurisdiction

## Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key**: You must have a paid OpenAI account with access to GPT-4o
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
4. Generate a CSV report: `detailed_payout_summary.csv`
5. Update the cache file: `processed_cache.json` (saved periodically during processing)
6. Print a data summary to the console (counts of lump sums, impairment percentages, and date ranges, split by Case Type)

## Output Files

### `detailed_payout_summary.csv`
A comprehensive CSV file containing all extracted data with columns:
- Case Name
- URL
- File Saved
- Jurisdiction
- Case Type
- Decision Date
- Injury Date
- Applicant
- Respondent
- Claimant Outcome
- Impairment %
- Lump Sum
- Weekly Benefit
- Medical Costs
- Nature
- Result
- Description

### `nsw_pic_decisions/`
Directory containing:
- Original HTML files for each decision
- Files are named using sanitized case titles

### `processed_cache.json`
JSON cache file storing all extracted data keyed by URL. This prevents re-processing decisions that have already been analyzed.

### `scraper.log`
Log file containing execution details, errors, and processing status.

## How It Works

1. **Multi-Year Index Scraping**: Fetches AustLII NSWPIC index pages for years 2021 to present, identifying decision links using regex pattern matching (`/NSWPIC/YYYY/NUMBER.html`)

2. **HTML Download with Retry Logic**: Downloads each decision's HTML file with exponential backoff retry logic:
   - Handles rate limiting (403, 429) and server errors (500, 502, 503, 504)
   - Retries connection errors and timeouts
   - Uses random jitter to avoid thundering herd problems

3. **Text Extraction**: Extracts clean text from HTML, focusing on the main content area (article, document, or body)

4. **AI Extraction**: 
   - Uses GPT-4o with structured output via Pydantic models
   - Extracts financial highlights using regex patterns
   - Implements retry logic for inconsistent extractions
   - Handles long documents by truncating to key sections

5. **Thread-Safe Caching**: 
   - Checks cache before processing (thread-safe)
   - Saves new extractions to cache with locking
   - Periodically saves cache every 20 completions to prevent data loss
   - Handles corrupted cache files by backing them up and starting fresh

6. **Parallel Processing**: Uses ThreadPoolExecutor (10 workers) to process multiple decisions concurrently

7. **CSV Generation**: Combines all cached and newly extracted data into a single CSV report, sorted by decision date

## Configuration

You can modify these settings in `nsw_court_scraper.py`:

- **Years to process**: Modify the `years` list in `main()` to change the range (currently 2021 to present)
- **OUTPUT_DIR**: Change the output directory name
- **CSV_REPORT**: Change the CSV filename
- **ThreadPoolExecutor max_workers**: Adjust the number of parallel threads (default: 10)
- **Retry settings**: Adjust `max_retries` in `_make_request_with_retry()` (default: 5)
- **Cache save frequency**: Change the interval in `main()` where cache is saved (currently every 20 completions)
- **AUSTLII_INDEX_DELAY**: Seconds to wait between index page requests (default: 2)
- **AUSTLII_RATE_LIMIT_DELAY**: Seconds to throttle decision requests after rate limiting is detected (default: 5)

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
- Uses GPT-4o (not GPT-4o-mini) for higher accuracy
- Costs depend on document length and number of decisions
- Caching significantly reduces API usage on subsequent runs

## Data Schema

The extraction uses a structured Pydantic model (`DecisionSchema`) with the following fields:

- `applicant_name`: Name of the Applicant/Claimant
- `respondent_name`: Name of the Respondent (usually insurer or employer)
- `claimant_outcome`: Enum (For Claimant / Against Claimant)
- `case_type`: Enum (Workers Compensation / CTP / Other)
- `impairment_percentage`: WPI percentage (if assessed)
- `lump_sum_amount`: Nominal number (no currency symbols)
- `weekly_benefit_amount`: Nominal number (no currency symbols)
- `medical_costs_awarded`: Enum (Yes / No / N/A)
- `decision_nature`: Primary category (e.g., Liability Dispute, Permanent Impairment)
- `decision_result`: Short legal summary
- `case_description`: Paragraph summarizing injury, claimant, and reasoning
- `date_of_injury`: YYYY-MM-DD format
- `date_of_decision`: YYYY-MM-DD format
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
