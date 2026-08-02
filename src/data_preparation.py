# ---------- IMPORTS ----------
import os
import pandas as pd

# ---------- PROJECT DIR ----------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ---------- DEFINE FUNCTION ----------
def load_and_merge_data():
    """Load raw CSVs, merge, and save cleaned data"""
    
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    
    # 1) Load processed CSVs
    demo = pd.read_csv(os.path.join(processed_dir, "demo_clean.csv"))
    bpx = pd.read_csv(os.path.join(processed_dir, "bpx_clean.csv"))
    bmx = pd.read_csv(os.path.join(processed_dir, "bmx_clean.csv"))

    # 2) Merge
    dfs = [demo, bpx, bmx]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="SEQN", how="inner")
        
    # 3) Clean
    merged = merged.dropna(subset=["BPXOSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "BMXBMI"])
    merged = merged[["SEQN", "BPXOSY1", "BPXODI1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "BMXBMI"]]
    
    # 4) Save
    merged.to_csv(os.path.join(processed_dir, "cleaned_merged.csv"), index=False)
    print(f"Saved {len(merged)} rows to cleaned_merged.csv")
    
    return merged

# ---------- CALL FUNCTION ----------
if __name__ == "__main__":
    load_and_merge_data()