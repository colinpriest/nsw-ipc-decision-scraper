
# NSW Court Decision Extractor (AI Enhanced)

This tool scrapes**AustLII** for NSW Personal Injury Commission (NSWPIC) decisions and uses **OpenAI's GPT-4o-mini** to extract detailed legal** information.**

## Changes in Version 2

* **Source Change** **: Now targets the AustLII NSWPIC index instead of the PIC website landing pages. This ensures we find ***actual* decision texts rather than summary lists.
* **Refined Selector** **: Scraper now looks specifically for case citations (e.g., **`<span class="highlight-diff-selection">[2024] NSWPIC 1</span>`).
* **Politeness** **: Added a 5-second delay between requests to respect AustLII's servers.**

## Prerequisites

1. **Python 3.x**
2. **OpenAI API Key** : You must have a paid OpenAI account.

## Setup

1. **Install dependencies** :

```
   pip install -r requirements.txt

```

1. Create a .env file:
   Create a file named .env in the same folder as the script:
   ```
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here

   ```

## Usage

Run the script:

```
python nsw_court_scraper.py

```

## How it Works

1. **Index Scraping** **: Fetches the "Recent Decisions" list from AustLII.**
2. **Filtering** **: Identifies links that look like case citations (**`<span class="highlight-diff-selection">[YYYY] NSWPIC ...</span>`).
3. **Extraction** **:**

* **Downloads the decision HTML.**
* **Uses **`<span class="highlight-diff-selection">gpt-4o-mini</span>` to extract Payout, Injury details, Dates, and Logic.
* **Chunking** **: If the decision is long, it splits it into parts to ensure no details are lost.**

1. **Reporting** **: Saves **`detailed_payout_summary.csv`.

## ⚠️ Important: AustLII Rate Limiting

**AustLII has strict anti-scraping measures.**

* **The script includes a ****5-second delay** between requests. **Do not remove this.** * If you receive "403 Forbidden" errors, stop the script and wait an hour before trying again.
