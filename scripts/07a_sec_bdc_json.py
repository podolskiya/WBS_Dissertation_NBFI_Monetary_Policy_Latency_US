import json
import time
import pandas as pd
import requests

# 1. Configuration & Setup
# The SEC strictly requires a professional User-Agent header to avoid 403 blocks.
HEADERS = {
    "User-Agent": "Institutional Asset Research Group contact@yourfirm.com",
    "Accept-Encoding": "gzip, deflate"
}

# The target credit facility breakout tags we want to collect
TARGET_TAGS = [
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "LineOfCreditFacilityAmountOutstanding",
    "DebtInstrumentCarryingAmount",
    "DebtInstrumentFaceAmount"
]

# Raw ticker list provided from your CapitalIQ file
raw_tickers = [
    "MutualFund:RCII.X", "MutualFund:CRDI.X", "NasdaqGS:ARCC", "NYSE:BCSF", 
    "NYSE:BBDC", "NYSE:MPV", "NasdaqGS:BCIC", "NasdaqGS:TCPC", "NYSE:BXSL", 
    "NYSE:OBDC", "NYSE:OTF", "NasdaqGS:CSWC", "NasdaqGS:CGBD", "MutualFund:TAKA.X", 
    "NasdaqGM:LIEN", "MutualFund:CADE.X", "NYSE:CION", "NasdaqGM:CCAP", 
    "MutualFund:CEDI.X", "NYSE:DBL", "NYSE:EQS", "NasdaqGS:FDUS", "MutualFund:FTPC.X", 
    "OTCPK:SVVC", "OTCPK:FRBP", "MutualFund:FCRI.X", "NYSE:FSCO", "NYSE:FSK", 
    "NasdaqGS:GLAD", "NasdaqGS:GAIN", "NYSE:GSBD", "NasdaqGS:GBDC", "NasdaqGM:GECC", 
    "NYSE:HTGC", "NasdaqGS:HRZN", "MutualFund:XCRT.X", "NasdaqGS:ICMB", "NYSE:KBDC", 
    "MutualFund:KABT.X", "NYSE:KIO", "MutualFund:LCRD.X", "NYSE:MAIN", "NasdaqGS:MFIC", 
    "NYSE:MSDL", "NYSE:MSIF", "NasdaqGS:NSLR", "NasdaqGS:NMFC", "NYSE:NCDL", 
    "NasdaqGS:OCSL", "NasdaqGS:OFS", "NasdaqGS:OXSQ", "NYSE:PSBD", "MutualFund:PSOI.X", 
    "NYSE:PFLT", "NYSE:PNNT", "NasdaqGM:PFX", "MutualFund:PFLE.X", "OTCPK:PIAC", 
    "NasdaqGS:PSEC", "NasdaqCM:RAND", "NasdaqGS:RWAY", "NYSE:SAR", "NYSE:TSLX", 
    "NasdaqGS:SLRC", "NYSE:SCM", "NasdaqGS:TRIN", "NYSE:TPVG"
]

# 2. Clean Tickers
# Strip the CapitalIQ market prefixes to get pure root tickers for the SEC
clean_to_raw = {}
for ticker in raw_tickers:
    clean_tk = ticker.split(":")[-1].replace(".X", "").strip()
    clean_to_raw[clean_tk] = ticker

print(f"Cleaned {len(clean_to_raw)} potential corporate tickers for evaluation.")

# 3. Generate the SEC Master Ticker-to-CIK Mapping Dictionary
print("Downloading SEC master company ticker mapping dataset...")
ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
map_res = requests.get(ticker_map_url, headers=HEADERS)

ticker_to_cik = {}
if map_res.status_code == 200:
    sec_data = map_res.json()
    for item in sec_data.values():
        sec_ticker = item["ticker"].upper()
        # SEC stores CIK as an integer; pad it to 10 digits with leading zeros
        ticker_to_cik[sec_ticker] = str(item["cik_str"]).zfill(10)
else:
    print("CRITICAL: Failed to download the SEC ticker mapping lookup database.")
    exit()

# Filter our list to tickers natively registered on the SEC EDGAR interface
final_processing_list = {tk: ticker_to_cik[tk] for tk in clean_to_raw if tk in ticker_to_cik}
print(f"Successfully matched {len(final_processing_list)} BDCs to valid SEC corporate CIK codes.\n")

# 4. Core Loop: Fetch and Parse data for each BDC
master_records = []

for ticker, cik in final_processing_list.items():
    raw_name = clean_to_raw[ticker]
    print(f"Processing: {ticker} (CIK: {cik}) -> Originating CapIQ field: {raw_name}")
    
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    
    try:
        response = requests.get(url, headers=HEADERS)
        # Note: The SEC dynamically throttles aggressive requests. 
        # We respect the SEC rate limit boundary (max 10 requests per second)
        time.sleep(0.15) 
        
        if response.status_code != 200:
            print(f"   [-] Skipping {ticker}: Received Status Code {response.status_code}")
            continue
            
        company_facts = response.json()
        us_gaap = company_facts.get("facts", {}).get("us-gaap", {})
        
        # Scrape through our prioritized accounting tags
        for tag in TARGET_TAGS:
            if tag in us_gaap:
                usd_entries = us_gaap[tag].get("units", {}).get("USD", [])
                
                for entry in usd_entries:
                    # Filter specifically for baseline regulated disclosures
                    if entry.get("form") in ["10-K", "10-Q"]:
                        record = {
                            "CapIQ_Ticker": raw_name,
                            "Pure_Ticker": ticker,
                            "CIK": cik,
                            "Tag": tag,
                            "Period": entry.get("frame"), # e.g., 'CY2024Q3I' or 'CY2024Q4'
                            "Value": entry.get("val"),
                            "Form": entry.get("form"),
                            "Filed_Date": entry.get("filed")
                        }
                        master_records.append(record)
                        
    except Exception as e:
        print(f"   [!] Error processing data cluster for ticker {ticker}: {str(e)}")

# 5. Export Compiled Matrix
if master_records:
    master_df = pd.DataFrame(master_records)
    # Drop records that lack a structured calendar frame index
    master_df = master_df.dropna(subset=["Period"])
    
    output_filename = "Master_BDC_Layer3_Breakouts.csv"
    master_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*50)
    print(f"AUTOMATION COMPLETE: Consolidated matrix saved as '{output_filename}'")
    print(f"Total Rows Extracted: {len(master_df)}")
    print(f"Unique BDCs Covered: {master_df['Pure_Ticker'].nunique()}")
    print("="*50)
else:
    print("\n[!] Loop concluded with no records successfully parsed.")