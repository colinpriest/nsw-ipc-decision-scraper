# NSW Court Decision Extractor (AI Enhanced)

A Python tool that scrapes NSW Personal Injury Commission (NSWPIC) decisions from AustLII and uses OpenAI's GPT-4o to extract structured legal information including payout amounts, injury details, dates, and case outcomes.

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
1. Fetch the list of decisions from the AustLII NSWPIC index
2. Process each decision (using cached data when available)
3. Save HTML files to `nsw_pic_decisions/`
4. Generate a CSV report: `detailed_payout_summary.csv`
5. Update the cache file: `processed_cache.json`

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

1. **Index Scraping**: Fetches the AustLII NSWPIC index page and identifies decision links using regex pattern matching (`/NSWPIC/YYYY/NUMBER.html`)

2. **HTML Download**: Downloads each decision's HTML file and saves it locally

3. **Text Extraction**: Extracts clean text from HTML, focusing on the main content area (article, document, or body)

4. **AI Extraction**: 
   - Uses GPT-4o with structured output via Pydantic models
   - Extracts financial highlights using regex patterns
   - Implements retry logic for inconsistent extractions
   - Handles long documents by truncating to key sections

5. **Caching**: Checks cache before processing; saves new extractions to cache

6. **Parallel Processing**: Uses ThreadPoolExecutor (10 workers) to process multiple decisions concurrently

7. **CSV Generation**: Combines all cached and newly extracted data into a single CSV report

## Configuration

You can modify these settings in `nsw_court_scraper.py`:

- **TARGET_INDEX_URL**: Change the AustLII index URL to scrape different years or jurisdictions
- **OUTPUT_DIR**: Change the output directory name
- **CSV_REPORT**: Change the CSV filename
- **ThreadPoolExecutor max_workers**: Adjust the number of parallel threads (default: 10)

## Important Notes

### Rate Limiting
- AustLII has anti-scraping measures
- The script handles 403/429 errors gracefully
- If you encounter rate limiting, wait before retrying
- Consider reducing `max_workers` if you encounter frequent rate limits

### Testing Mode
To test with only the latest 8 decisions, modify the `main()` function:
```python
# Process only the latest 8 decisions for testing
latest_links = links[-8:] if len(links) > 8 else links
target_links = latest_links
```

### Cache Management
- The cache file (`processed_cache.json`) stores all processed decisions
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
- AustLII is rate limiting your requests
- Wait an hour before retrying
- Reduce the number of parallel workers

### Empty or incomplete extractions
- Check `scraper.log` for detailed error messages
- Some decisions may not contain all expected fields
- The retry logic should catch most inconsistencies

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## License

This project is provided as-is for educational and research purposes. Please respect AustLII's terms of service and rate limits when using this tool.

## Contributing

Feel free to submit issues or pull requests for improvements.

