import os
import time
import csv
import logging
import json
import re
import shutil
from enum import Enum
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import random
from threading import Lock
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

class JurisdictionEnum(str, Enum):
    NSW = "NSW"
    VIC = "VIC"
    QLD = "QLD"
    WA = "WA"
    SA = "SA"
    TAS = "TAS"
    ACT = "ACT"
    NT = "NT"
    FEDERAL = "Federal"

class ClaimantOutcomeEnum(str, Enum):
    FOR_CLAIMANT = "For Claimant"
    AGAINST_CLAIMANT = "Against Claimant"

class CaseCategoryEnum(str, Enum):
    WORKERS_COMPENSATION = "Workers Compensation"
    CTP = "CTP"
    OTHER = "Other"

class MedicalCostsEnum(str, Enum):
    YES = "Yes"
    NO = "No"
    NA = "N/A"

class DecisionSchema(BaseModel):
    applicant_name: str = Field(description="Name of the Applicant/Claimant.")
    respondent_name: str = Field(description="Name of the Respondent (usually insurer or employer).")
    
    claimant_outcome: ClaimantOutcomeEnum = Field(description="Did the decision favor the claimant? If procedural (e.g. allowed a review), classify as 'For Claimant'.")
    
    case_type: CaseCategoryEnum = Field(description="Whether the case is Workers Compensation, CTP (Motor Accidents), or Other.")
    
    impairment_percentage: str = Field(description="The percentage of Whole Person Impairment (WPI) ONLY if a final assessment is made (e.g. '15'). IMPORTANT: If the decision is to ALLOW a reassessment or remit to a medical assessor, LEAVE THIS EMPTY.")
    
    lump_sum_amount: str = Field(description="Total lump sum payout as a NOMINAL NUMBER (e.g. '150000.00'). Do not use currency symbols like '$' or commas. Leave empty if none.")
    
    weekly_benefit_amount: str = Field(description="Weekly benefit amount as a NOMINAL NUMBER (e.g. '540.50'). Do not use currency symbols like '$' or text. If multiple periods, use the amount at the LATEST date. If deductions are quantified, use NET amount.")

    medical_costs_awarded: MedicalCostsEnum = Field(description="Were medical costs explicitly awarded/ordered? Select 'Yes' if ordered to pay. Select 'No' if explicitly denied. Select 'N/A' if medical costs were not discussed OR if the order is silent/parties to agree.")

    decision_nature: str = Field(description="PRIMARY category of the dispute. Simplify to: 'Liability Dispute', 'Permanent Impairment', 'Medical Dispute', 'Death Benefit', 'Damages', 'Procedural'. Do NOT prefix with 'Workers Compensation - ...'. Choose ONE primary category.")
    
    decision_result: str = Field(description="Short legal summary (e.g., Award for Applicant, Matter Remitted).")
    case_description: str = Field(description="A paragraph summarizing injury details, claimant info (age/occupation), and the judge's reasoning.")
    date_of_injury: str = Field(description="YYYY-MM-DD or 'Unknown'")
    date_of_decision: str = Field(description="YYYY-MM-DD or 'Unknown'")
    jurisdiction: JurisdictionEnum = Field(default=JurisdictionEnum.NSW, description="The legal jurisdiction of the decision.")

class LLMExtractor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.SINGLE_PASS_LIMIT_CHARS = 100000 
        self.financial_pattern = re.compile(r"(?:order|award|pay|judgment|sum of|costs|medical expenses)\s*(?:of)?\s*[\$£€]?", re.IGNORECASE)

    def _get_financial_highlights(self, text, window=300):
        highlights = []
        for match in self.financial_pattern.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            highlights.append(f"...{text[start:end].strip()}...")
        return "\n".join(highlights[:5])

    def extract_data(self, text, context=None):
        if not text:
            return self._get_empty_schema(), None
            
        highlights = self._get_financial_highlights(text)
        data, error = self._call_llm(text, highlights, context=context)
        
        # Retry logic
        if not error and data.claimant_outcome == ClaimantOutcomeEnum.FOR_CLAIMANT and \
           not data.lump_sum_amount and \
           not data.weekly_benefit_amount and \
           data.medical_costs_awarded == MedicalCostsEnum.NO:

            if "$" in data.case_description or "compensation" in data.decision_nature.lower():
                data, error = self._call_llm(text, highlights, retry_mode=True, context=context)
            
        return data.model_dump(mode="json"), error

    def _extract_relevant_sections(self, text):
        keywords = [
            "Orders",
            "Conclusion",
            "Decision",
            "Findings",
            "Reasons",
        ]
        lowered = text.lower()
        segments = []
        for keyword in keywords:
            index = lowered.rfind(keyword.lower())
            if index != -1:
                start = max(0, index - 8000)
                end = min(len(text), index + 12000)
                segments.append(text[start:end].strip())

        intro = text[:20000].strip()
        if intro:
            segments.insert(0, intro)

        seen = set()
        unique_segments = []
        for segment in segments:
            if segment and segment not in seen:
                seen.add(segment)
                unique_segments.append(segment)

        combined = "\n\n...[SECTION BREAK]...\n\n".join(unique_segments)
        if len(combined) > self.SINGLE_PASS_LIMIT_CHARS:
            combined = combined[:self.SINGLE_PASS_LIMIT_CHARS]
        return combined

    def _call_llm(self, text, highlights, retry_mode=False, context=None):
        if len(text) > self.SINGLE_PASS_LIMIT_CHARS:
            processed_text = self._extract_relevant_sections(text)
        else:
            processed_text = text

        system_instruction = """
        You are a senior legal data analyst. Extract specific details from the court decision.
        
        CRITICAL INSTRUCTIONS:
        1. **Amounts**: Extract NOMINAL NUMBERS ONLY (e.g. '150000.00', '540.50'). Do NOT use '$' or commas.
        
        2. **Medical Costs**: 
           - 'Yes': Explicit order to pay (e.g. "Respondent to pay section 60 expenses").
           - 'No': Explicitly denied/dismissed.
           - 'N/A': Not mentioned OR no specific order made (e.g. "parties to agree on quantum").
        
        3. **Weekly Benefits**: 
           - Use the amount for the LATEST period.
           - If deductions quantified, return NET amount.
        
        4. **Payouts**: If allowing a late claim/medical review with no money awarded *yet*, leave amounts EMPTY.
        """
        
        if retry_mode:
            system_instruction += " REVIEW MODE: Search specifically for the 'Orders' section to confirm if any monetary sum exists."

        user_content = f"""
        Full Decision Text:
        {processed_text}
        
        ---
        HIGH PRIORITY FINANCIAL SEGMENTS:
        {highlights}
        ---
        """

        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                response_format=DecisionSchema,
            )
            return completion.choices[0].message.parsed, None
            
        except Exception as e:
            context_msg = f" ({context})" if context else ""
            logging.error(f"LLM Error{context_msg}: {e}")
            return DecisionSchema(
                applicant_name="Error", respondent_name="Error", 
                claimant_outcome=ClaimantOutcomeEnum.AGAINST_CLAIMANT, 
                case_type=CaseCategoryEnum.OTHER,
                impairment_percentage="", lump_sum_amount="", weekly_benefit_amount="",
                medical_costs_awarded=MedicalCostsEnum.NA,
                decision_nature="Error", decision_result="Error", case_description="Failed extraction", 
                date_of_injury="", date_of_decision="", jurisdiction=JurisdictionEnum.NSW
            ), str(e)

    def _get_empty_schema(self):
        return {
            "applicant_name": "", "respondent_name": "", 
            "claimant_outcome": ClaimantOutcomeEnum.AGAINST_CLAIMANT.value, 
            "case_type": CaseCategoryEnum.OTHER.value,
            "impairment_percentage": "", "lump_sum_amount": "", "weekly_benefit_amount": "",
            "medical_costs_awarded": MedicalCostsEnum.NA.value,
            "decision_nature": "Unknown", "decision_result": "Unknown", "case_description": "Not found",
            "date_of_injury": "", "date_of_decision": "", "jurisdiction": "NSW"
        }

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        main_content = soup.find('article') or soup.find(class_='the-document') or soup.find(class_='austlii-doc')
        
        if not main_content:
            main_content = soup.body
        
        if main_content:
            for garbage in main_content.find_all(['div'], class_=['austlii-header', 'breadcrumb', 'page-footer', 'nav']):
                garbage.decompose()
            return main_content.get_text(separator='\n').strip()
            
        return soup.get_text(separator='\n').strip()

class DecisionScraper:
    def __init__(self, base_url, output_folder="nsw_decisions", api_key=None):
        self.base_url = base_url
        self.output_folder = output_folder
        self.extractor = LLMExtractor(api_key) if api_key else None
        self.cache_file = "processed_cache.json"
        self.cache_lock = Lock() # Thread lock for cache operations
        self.cache = self._load_cache()
        self.rate_limit_lock = Lock()
        self.rate_limit_triggered = False
        self.next_request_time = 0.0
        self.rate_limit_delay = float(os.getenv("AUSTLII_RATE_LIMIT_DELAY", "5"))
        self.rate_limit_success_count = 0
        self.rate_limit_reset_threshold = int(os.getenv("AUSTLII_RATE_LIMIT_RESET_THRESHOLD", "3"))
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        }
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _load_cache(self):
        """Loads cache safely. If corrupted, backs up and starts empty."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Cache file {self.cache_file} is corrupted. Starting with empty cache.")
                shutil.move(self.cache_file, self.cache_file + ".corrupted")
                return {}
        return {}

    def _save_cache(self, max_retries=3):
        """
        Thread-safe atomic write.
        Copies cache under lock, dumps to temp file, then renames.
        """
        temp_file = self.cache_file + ".tmp"
        
        # Create a thread-safe snapshot of the data
        with self.cache_lock:
            cache_copy = self.cache.copy()

        for attempt in range(1, max_retries + 1):
            try:
                with open(temp_file, 'w') as f:
                    json.dump(cache_copy, f, indent=2, default=str)
                os.replace(temp_file, self.cache_file)
                return True
            except Exception as e:
                logging.error(f"Failed to save cache (attempt {attempt}/{max_retries}): {e}")
                time.sleep(min(2 ** attempt, 5))
        return False

    def update_cache(self, url, data):
        """Thread-safe cache update helper"""
        with self.cache_lock:
            self.cache[url] = data

    def _throttle_if_rate_limited(self):
        delay = max(self.rate_limit_delay, 0)
        if delay == 0:
            return
        with self.rate_limit_lock:
            if not self.rate_limit_triggered:
                return
            now = time.monotonic()
            if now < self.next_request_time:
                time.sleep(self.next_request_time - now)
                now = time.monotonic()
            self.next_request_time = now + delay

    def _make_request_with_retry(self, url, max_retries=5):
        for attempt in range(max_retries):
            self._throttle_if_rate_limited()
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                
                if response.status_code == 200:
                    with self.rate_limit_lock:
                        if self.rate_limit_triggered:
                            self.rate_limit_success_count += 1
                            if self.rate_limit_success_count >= self.rate_limit_reset_threshold:
                                self.rate_limit_triggered = False
                                self.rate_limit_success_count = 0
                                self.next_request_time = 0.0
                    return response
                
                if response.status_code in [403, 429, 500, 502, 503, 504]:
                    if response.status_code in [403, 429]:
                        with self.rate_limit_lock:
                            self.rate_limit_triggered = True
                            self.rate_limit_success_count = 0
                    sleep_time = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Request failed ({response.status_code}) for {url}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Request failed ({response.status_code}) for {url}. No retry.")
                    return None
                    
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                 sleep_time = (2 ** attempt) + random.uniform(0, 1)
                 logging.warning(f"Connection error ({e}) for {url}. Retrying in {sleep_time:.2f}s...")
                 time.sleep(sleep_time)
        
        logging.error(f"Max retries exceeded for {url}")
        return None

    def get_decision_links(self, index_url):
        logging.info(f"Fetching index: {index_url}")
        
        response = self._make_request_with_retry(index_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        decision_url_pattern = re.compile(r"\/NSWPIC\/\d{4}\/[^\/]+\.(?:html|pdf)", re.IGNORECASE)
        nswpic_pattern = re.compile(r"\/NSWPIC\/\d{4}\/", re.IGNORECASE)

        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(self.base_url, href)
            
            if decision_url_pattern.search(full_url):
                title = a.get_text(" ", strip=True)
                if 5 < len(title) < 250:
                    links.append((title, full_url))
            elif nswpic_pattern.search(full_url):
                logging.warning(f"Potential decision link did not match expected pattern: {full_url}")
        
        unique_links = {}
        for title, url in links:
            if url not in unique_links:
                unique_links[url] = title
        
        return [(title, url) for url, title in unique_links.items()]

    def process_decision(self, title, url):
        if not self.extractor:
            raise RuntimeError("LLM extractor is not configured. Ensure OPENAI_API_KEY is set.")
        safe_title = re.sub(r'[^\w\s\-\.]', '', title)
        safe_title = re.sub(r'[\s]+', '_', safe_title)
        case_id_match = re.search(r"/NSWPIC/(\d{4})/(\d+)\.html", url)
        case_id = None
        if case_id_match:
            case_id = f"{case_id_match.group(1)}_{case_id_match.group(2)}"
        if case_id:
            suffix = f"_{case_id}"
        else:
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
            suffix = f"_unknown_case_{url_hash}"
        safe_title = f"{safe_title[:100]}{suffix}.html"
        
        log_title = (title[:75] + '...') if len(title) > 75 else title

        # Check cache under lock is safer, though read-only access to dict is atomic in GIL
        # But to be safe and consistent:
        with self.cache_lock:
            if url in self.cache:
                return self.cache[url]

        logging.info(f"Processing: {log_title}")
        
        response = self._make_request_with_retry(url)
        if not response:
            logging.error(f"Failed to fetch content for {log_title} after retries.")
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if url.lower().endswith(".pdf") or (content_type and "text/html" not in content_type):
            logging.warning(f"Non-HTML decision content skipped: {url}")
            result_data = {
                "Case Name": title,
                "URL": url,
                "File Saved": "",
                "Jurisdiction": "",
                "Case Type": "",
                "Decision Date": "",
                "Injury Date": "",
                "Applicant": "",
                "Respondent": "",
                "Claimant Outcome": "",
                "Impairment %": "",
                "Lump Sum": "",
                "Weekly Benefit": "",
                "Medical Costs": "",
                "Nature": "",
                "Result": "",
                "Description": "",
                "Status": "skipped_non_html",
                "LLM Error": "",
            }
            self.update_cache(url, result_data)
            return result_data
            
        decision_text = self.extractor.extract_text_from_html(response.content)
        
        if len(decision_text) < 500:
            logging.warning(f"Decision text too short; skipping {log_title}")
            result_data = {
                "Case Name": title,
                "URL": url,
                "File Saved": "",
                "Jurisdiction": "",
                "Case Type": "",
                "Decision Date": "",
                "Injury Date": "",
                "Applicant": "",
                "Respondent": "",
                "Claimant Outcome": "",
                "Impairment %": "",
                "Lump Sum": "",
                "Weekly Benefit": "",
                "Medical Costs": "",
                "Nature": "",
                "Result": "",
                "Description": "",
                "Status": "skipped_short_text",
                "LLM Error": "",
            }
            self.update_cache(url, result_data)
            return result_data

        full_path = os.path.join(self.output_folder, safe_title)
        if os.path.exists(full_path):
            url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
            safe_title = f"{safe_title[:-5]}_{url_hash}.html"
            full_path = os.path.join(self.output_folder, safe_title)
        with open(full_path, 'wb') as f:
            f.write(response.content)

        result_data = None
        if self.extractor:
            llm_data, llm_error = self.extractor.extract_data(
                decision_text,
                context=f"title={log_title}, url={url}",
            )
            status = "ok" if not llm_error else "llm_error"
            
            result_data = {
                "Case Name": title,
                "URL": url,
                "File Saved": safe_title,
                "Jurisdiction": llm_data.get("jurisdiction"),
                "Case Type": llm_data.get("case_type"),
                "Decision Date": llm_data.get("date_of_decision"),
                "Injury Date": llm_data.get("date_of_injury"),
                "Applicant": llm_data.get("applicant_name"),
                "Respondent": llm_data.get("respondent_name"),
                "Claimant Outcome": llm_data.get("claimant_outcome"),
                "Impairment %": llm_data.get("impairment_percentage", "").replace("%", "").strip(),
                "Lump Sum": llm_data.get("lump_sum_amount", "").replace("$", "").replace(",", "").strip(),
                "Weekly Benefit": llm_data.get("weekly_benefit_amount"),
                "Medical Costs": llm_data.get("medical_costs_awarded"),
                "Nature": llm_data.get("decision_nature"),
                "Result": llm_data.get("decision_result"),
                "Description": llm_data.get("case_description"),
                "Status": status,
                "LLM Error": llm_error or ""
            }
            
            # Update cache safely
            self.update_cache(url, result_data)
            
        return result_data

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("⚠️ OPENAI_API_KEY not found in .env file.")
        return

    BASE_DOMAIN = "https://www.austlii.edu.au"
    OUTPUT_DIR = "nsw_pic_decisions"
    CSV_REPORT = "detailed_payout_summary.csv"
    scraper = DecisionScraper(BASE_DOMAIN, OUTPUT_DIR, api_key)
    index_delay_seconds = float(os.getenv("AUSTLII_INDEX_DELAY", "2"))
    
    years = list(range(2021, datetime.datetime.now().year + 1))
    all_links = []

    for year in years:
        index_url = f"https://www.austlii.edu.au/cgi-bin/viewdb/au/cases/nsw/NSWPIC/{year}/"
        logging.info(f"Scanning Year: {year} ...")
        links = scraper.get_decision_links(index_url)
        all_links.extend(links)
        time.sleep(max(index_delay_seconds, 0))

    logging.info(f"Total decisions found (2021-Present): {len(all_links)}")
    
    target_links = all_links 
    
    logging.info(f"Starting parallel processing of {len(target_links)} decisions (5 threads)...")
    results = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(scraper.process_decision, title, url): url
            for title, url in target_links
        }
        
        for i, future in enumerate(as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception as e:
                logging.error(f"Unhandled exception while processing {url}: {e}")
            
            # Save cache periodically (e.g., every 20 completions)
            if i > 0 and i % 10 == 0:
                scraper._save_cache()
    
    # Final save
    scraper._save_cache()

    # Use thread-safe snapshot for final report generation
    with scraper.cache_lock:
        all_data = list(scraper.cache.values())
    
    all_data.sort(key=lambda x: x.get("Decision Date") or "0000-00-00", reverse=True)
    
    if all_data:
        keys = [
            "Case Name", "URL", "File Saved", "Jurisdiction", "Case Type", "Decision Date", 
            "Injury Date", "Applicant", "Respondent", "Claimant Outcome", "Impairment %", 
            "Lump Sum", "Weekly Benefit", "Medical Costs", "Nature", "Result", "Description",
            "Status", "LLM Error"
        ]
        with open(CSV_REPORT, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_data)
        logging.info(f"Summary report saved to {CSV_REPORT}")

    print_summary(all_data)

def print_summary(all_data):
    """Print data summary to console, split by Case Type."""
    if not all_data:
        print("\nNo data to summarise.")
        return

    lump_sum_counts = defaultdict(int)
    impairment_counts = defaultdict(int)
    both_counts = defaultdict(int)
    injury_dates = defaultdict(list)
    decision_dates = defaultdict(list)

    def _has_numeric_value(val):
        """Check if a value is a non-empty numeric string."""
        if not val or not isinstance(val, str):
            return False
        val = val.strip()
        if not val or val.lower() in ("n/a", "unknown", "none", "nan"):
            return False
        try:
            float(val)
            return True
        except ValueError:
            return False

    for row in all_data:
        case_type = row.get("Case Type") or "Unknown"
        has_lump = _has_numeric_value(row.get("Lump Sum", ""))
        has_impairment = _has_numeric_value(row.get("Impairment %", ""))

        if has_lump:
            lump_sum_counts[case_type] += 1
        if has_impairment:
            impairment_counts[case_type] += 1
        if has_lump and has_impairment:
            both_counts[case_type] += 1

        inj = row.get("Injury Date", "").strip()
        if inj and inj != "Unknown":
            injury_dates[case_type].append(inj)

        dec = row.get("Decision Date", "").strip()
        if dec and dec != "Unknown":
            decision_dates[case_type].append(dec)

    case_types = sorted(set(
        list(lump_sum_counts) + list(impairment_counts) + list(both_counts)
        + list(injury_dates) + list(decision_dates)
    ))

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    for ct in case_types:
        print(f"\n--- {ct} ---")
        print(f"  Rows with Lump Sum:              {lump_sum_counts.get(ct, 0)}")
        print(f"  Rows with Impairment %:          {impairment_counts.get(ct, 0)}")
        print(f"  Rows with both:                  {both_counts.get(ct, 0)}")

        inj = injury_dates.get(ct, [])
        if inj:
            print(f"  Injury dates:                    {min(inj)} to {max(inj)}")
        else:
            print(f"  Injury dates:                    N/A")

        dec = decision_dates.get(ct, [])
        if dec:
            print(f"  Decision dates:                  {min(dec)} to {max(dec)}")
        else:
            print(f"  Decision dates:                  N/A")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
