import io
import zipfile
import pandas as pd
import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

STATCAN_RETAIL_URL = "https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/20100056/en"
CACHED_DATA = None

def load_and_cache_data():
    global CACHED_DATA
    try:
        print("-> [Startup] Requesting download link for Retail Trade table from StatCan API...")
        response = requests.get(STATCAN_RETAIL_URL, timeout=30)
        response.raise_for_status()
        api_result = response.json()

        csv_zip_url = api_result.get("object")
        if not csv_zip_url:
            print(f"[ERROR] StatCan API response missing 'object': {api_result}")
            return

        print("-> [Startup] Downloading bulk Retail CSV zip from StatCan...")
        zip_resp = requests.get(csv_zip_url, timeout=120)
        zip_resp.raise_for_status()

        print("-> [Startup] Extracting zip archive in memory...")
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
            csv_filename = [name for name in z.namelist() if name.endswith(".csv") and "sub" not in name][0]
            
            retail_chunks = []
            print("-> [Startup] Processing Retail CSV in low-memory chunks...")
            
            with z.open(csv_filename) as f:
                for chunk in pd.read_csv(f, dtype={"VALUE": "float32"}, chunksize=100000, low_memory=True):
                    chunk["REF_DATE"] = pd.to_datetime(chunk["REF_DATE"], errors='coerce')
                    
                    # Filter for Canada, 2025 onwards, Seasonally adjusted
                    mask = (
                        (chunk["GEO"] == "Canada") & 
                        (chunk["REF_DATE"] >= "2025-01-01") &
                        (chunk["Adjustments"].str.contains("Seasonally adjusted", case=False, na=False))
                    )
                    
                    sub = chunk[mask]
                    if not sub.empty:
                        retail_chunks.append(sub)

        if not retail_chunks:
            print("[CRITICAL ERROR] Retail filtering resulted in 0 rows.")
            return

        retail_df = pd.concat(retail_chunks, ignore_index=True)
        naics_col = "North American Industry Classification System (NAICS)"
        retail_df["NAICS_Text"] = retail_df[naics_col].fillna("").astype(str)

        # 1. Total Retail Trade
        total_mask = retail_df["NAICS_Text"].str.match(r"^Retail trade \[44-45\]$", case=False)
        total_df = retail_df[total_mask].groupby("REF_DATE")["VALUE"].sum().reset_index()
        total_df.rename(columns={"VALUE": "Retail Sales"}, inplace=True)

        # 2. Core Retail Sales: Sum all subsectors excluding Motor vehicle and parts dealers [441]
        core_mask = (
            retail_df["NAICS_Text"].str.contains("Retail trade [44-45]", case=False, na=False) &
            ~retail_df["NAICS_Text"].str.contains("motor vehicle and parts dealers", case=False, na=False)
        )
        
        subsector_mask = (
            ~retail_df["NAICS_Text"].str.match(r"^Retail trade \[44-45\]$", case=False) &
            ~retail_df["NAICS_Text"].str.contains("motor vehicle and parts dealers", case=False, na=False)
        )
        
        core_df = retail_df[subsector_mask].groupby("REF_DATE")["VALUE"].sum().reset_index()
        
        if core_df["VALUE"].sum() == 0:
            core_df = retail_df[core_mask].groupby("REF_DATE")["VALUE"].sum().reset_index()

        core_df.rename(columns={"VALUE": "Core Retail Sales"}, inplace=True)

        merged_df = pd.merge(total_df, core_df, on="REF_DATE", how="inner")
        merged_df["Month"] = merged_df["REF_DATE"].dt.strftime("%b %Y")
        merged_df = merged_df.sort_values("REF_DATE")

        CACHED_DATA = {
            "months": merged_df["Month"].tolist(),
            "retail_sales": merged_df["Retail Sales"].dropna().tolist(),
            "core_retail_sales": merged_df["Core Retail Sales"].dropna().tolist(),
        }
        print("-> [Startup] Retail & Core Retail data successfully processed and cached!")
    except Exception as e:
        print(f"[CRITICAL ERROR during startup cache]: {e}")
        import traceback
        traceback.print_exc()

load_and_cache_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/labor-data")
def get_labor_data():
    if CACHED_DATA is None:
        load_and_cache_data()
    if CACHED_DATA is None:
        return jsonify({"error": "Retail data failed to initialize."}), 500
    return jsonify(CACHED_DATA)

if __name__ == "__main__":
    app.run(debug=True, port=6060)