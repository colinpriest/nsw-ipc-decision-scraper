import os
import time
import csv
import logging
import json
import re
from typing import Optional, List
from enum import Enum
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def extract_data(self, text):
        if not text:
            return self._get_empty_schema()
            
        highlights = self._get_financial_highlights(text)
        # logging.info("Running AI extraction (GPT-4o)...") 
        data = self._call_llm(text, highlights)
        
        # Retry logic
        if data.claimant_outcome == ClaimantOutcomeEnum.FOR_CLAIMANT and \
           not data.lump_sum_amount and \
           not data.weekly_benefit_amount and \
           data.medical_costs_awarded == MedicalCostsEnum.NO:
             
             if "$" in data.case_description or "compensation" in data.decision_nature.lower():
                # logging.warning(f"Detected inconsistencies. Retrying...")
                data = self._call_llm(text, highlights, retry_mode=True)
            
        return data.model_dump()

    def _call_llm(self, text, highlights, retry_mode=False):
        if len(text) > self.SINGLE_PASS_LIMIT_CHARS:
            processed_text = text[:50000] + "\n...[SNIP]...\n" + text[-50000:]
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
            return completion.choices[0].message.parsed
            
        except Exception as e:
            logging.error(f"LLM Error: {e}")
            return DecisionSchema(
                applicant_name="Error", respondent_name="Error", 
                claimant_outcome=ClaimantOutcomeEnum.AGAINST_CLAIMANT, 
                case_type=CaseCategoryEnum.OTHER,
                impairment_percentage="", lump_sum_amount="", weekly_benefit_amount="",
                medical_costs_awarded=MedicalCostsEnum.NA,
                decision_nature="Error", decision_result="Error", case_description="Failed extraction", 
                date_of_injury="", date_of_decision="", jurisdiction=JurisdictionEnum.NSW
            )

    def _get_empty_schema(self):
        return {
            "applicant_name": "", "respondent_name": "", 
            "claimant_outcome": ClaimantOutcomeEnum.AGAINST_CLAIMANT, 
            "case_type": CaseCategoryEnum.OTHER,
            "impairment_percentage": "", "lump_sum_amount": "", "weekly_benefit_amount": "",
            "medical_costs_awarded": MedicalCostsEnum.NA,
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
        self.cache = self._load_cache()
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        }
        
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2, default=str)

    def get_decision_links(self, index_url):
        logging.info(f"Fetching index: {index_url}")
        try:
            response = requests.get(index_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            decision_url_pattern = re.compile(r"\/NSWPIC\/20\d\d\/\d+\.html")

            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(self.base_url, href)
                
                if decision_url_pattern.search(full_url):
                    title = a.get_text(" ", strip=True)
                    if 5 < len(title) < 250:
                        links.append((title, full_url))
            
            unique_links = {}
            for title, url in links:
                if url not in unique_links:
                    unique_links[url] = title
            
            return [(title, url) for url, title in unique_links.items()]
            
        except Exception as e:
            logging.error(f"Error fetching index: {e}")
            return []

    def process_decision(self, title, url):
        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_', '.')])
        safe_title = re.sub(r'[\s]+', '_', safe_title).strip('_')
        safe_title = safe_title[:100]
        
        log_title = (title[:75] + '...') if len(title) > 75 else title

        if url in self.cache:
            return self.cache[url]

        logging.info(f"Processing: {log_title}")
        
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code in [403, 429]:
                logging.warning(f"Rate limited on {log_title}. Skipping.")
                return None
            
            response.raise_for_status()
            
            saved_filename = f"{safe_title}.html"
            full_path = os.path.join(self.output_folder, saved_filename)
            
            with open(full_path, 'wb') as f:
                f.write(response.content)
            
            decision_text = self.extractor.extract_text_from_html(response.content)
            
            if len(decision_text) < 500:
                return None

            result_data = None
            if self.extractor:
                llm_data = self.extractor.extract_data(decision_text)
                
                result_data = {
                    "Case Name": title,
                    "URL": url,
                    "File Saved": saved_filename,
                    "Jurisdiction": llm_data.get("jurisdiction"),
                    "Case Type": llm_data.get("case_type"),
                    "Decision Date": llm_data.get("date_of_decision"),
                    "Injury Date": llm_data.get("date_of_injury"),
                    "Applicant": llm_data.get("applicant_name"),
                    "Respondent": llm_data.get("respondent_name"),
                    "Claimant Outcome": llm_data.get("claimant_outcome"),
                    "Impairment %": llm_data.get("impairment_percentage"),
                    "Lump Sum": llm_data.get("lump_sum_amount"),
                    "Weekly Benefit": llm_data.get("weekly_benefit_amount"),
                    "Medical Costs": llm_data.get("medical_costs_awarded"),
                    "Nature": llm_data.get("decision_nature"),
                    "Result": llm_data.get("decision_result"),
                    "Description": llm_data.get("case_description")
                }
                
                self.cache[url] = result_data
                
            return result_data

        except Exception as e:
            logging.error(f"Failed to process {log_title}: {e}")
            return None

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logging.warning("⚠️ OPENAI_API_KEY not found in .env file.")
        return

    TARGET_INDEX_URL = "http://www.austlii.edu.au/cgi-bin/viewdb/au/cases/nsw/NSWPIC/2025/"
    BASE_DOMAIN = "http://www.austlii.edu.au"
    OUTPUT_DIR = "nsw_pic_decisions"
    CSV_REPORT = "detailed_payout_summary.csv"
    
    scraper = DecisionScraper(BASE_DOMAIN, OUTPUT_DIR, api_key)
    
    links = scraper.get_decision_links(TARGET_INDEX_URL)
    logging.info(f"Found {len(links)} potential decisions.")
    
    # UPDATED: Remove slicing to process ALL links found
    target_links = links 
    
    logging.info(f"Starting parallel processing of {len(target_links)} decisions (10 threads)...")
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scraper.process_decision, title, url): url for title, url in target_links}
        
        for future in as_completed(future_to_url):
            data = future.result()
            if data:
                results.append(data)
    
    scraper._save_cache()

    results.sort(key=lambda x: x.get("Decision Date") or "0000-00-00", reverse=True)

    all_data = list(scraper.cache.values())
    
    if all_data:
        keys = [
            "Case Name", "URL", "File Saved", "Jurisdiction", "Case Type", "Decision Date", 
            "Injury Date", "Applicant", "Respondent", "Claimant Outcome", "Impairment %", 
            "Lump Sum", "Weekly Benefit", "Medical Costs", "Nature", "Result", "Description"
        ]
        with open(CSV_REPORT, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_data)
        logging.info(f"Summary report saved to {CSV_REPORT}")

if __name__ == "__main__":
    main()