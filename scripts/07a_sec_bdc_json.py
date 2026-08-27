"""
07a_sec_bdc_json.py

Pulls SEC EDGAR XBRL company-facts data for the BDCs in the Capital IQ industry pull, filtering to the credit-facility breakout tags used as
the cross-check against Capital IQ (see 07b_bdc_supplement.py). Tickers are matched to CIKs via SEC's own company_tickers.json.

OUTPUT  Master_BDC_Layer3_Breakouts.csv
"""
import sys
import time
import pandas as pd
import requests

# SEC requires a valid contact in the User-Agent or requests get 403. #

HEADERS = {
    "User-Agent": "Institutional Asset Research Group Alex.Podolskiy@warwick.ac.uk",
    "Accept-Encoding": "gzip, deflate",
}

TARGET_TAGS = [
    "LineOfCreditFacilityMaximumBorrowingCapacity",
    "LineOfCreditFacilityAmountOutstanding",
    "DebtInstrumentCarryingAmount",
    "DebtInstrumentFaceAmount",
]

# Capital IQ BDC industry classification pull #

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
    "NasdaqGS:SLRC", "NYSE:SCM", "NasdaqGS:TRIN", "NYSE:TPVG",
]

# Strip the CapitalIQ market prefix and mutual-fund suffix to get bare tickers for matching against SEC's own ticker list. #

clean_to_raw = {}
for ticker in raw_tickers:
    clean_tk = ticker.split(":")[-1].replace(".X", "").strip()
    clean_to_raw[clean_tk] = ticker

ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
map_res = requests.get(ticker_map_url, headers=HEADERS)
ticker_to_cik = {}
if map_res.status_code == 200:
    sec_data = map_res.json()
    for item in sec_data.values():
        sec_ticker = item["ticker"].upper()
        ticker_to_cik[sec_ticker] = str(item["cik_str"]).zfill(10)
else:
    print(f"Failed to download SEC ticker mapping (status {map_res.status_code})")
    sys.exit(1)

matched_ciks = {tk: ticker_to_cik[tk] for tk in clean_to_raw if tk in ticker_to_cik}

master_records = []
for ticker, cik in matched_ciks.items():
    raw_name = clean_to_raw[ticker]
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        response = requests.get(url, headers=HEADERS)
        time.sleep(0.15) 
        if response.status_code != 200:
            print(f"Skipping {ticker}: status {response.status_code}")
            continue

        company_facts = response.json()
        us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

        for tag in TARGET_TAGS:
            if tag in us_gaap:
                usd_entries = us_gaap[tag].get("units", {}).get("USD", [])
                for entry in usd_entries:
                    if entry.get("form") in ["10-K", "10-Q"]:
                        master_records.append({
                            "CapIQ_Ticker": raw_name,
                            "Pure_Ticker": ticker,
                            "CIK": cik,
                            "Tag": tag,
                            "Period": entry.get("frame"),  
                            "Value": entry.get("val"),
                            "Form": entry.get("form"),
                            "Filed_Date": entry.get("filed"),
                        })
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

if master_records:
    master_df = pd.DataFrame(master_records)
    master_df = master_df.dropna(subset=["Period"])  
    master_df.to_csv("Master_BDC_Layer3_Breakouts.csv", index=False)
else:
    print("No records parsed.")