import os
import pandas as pd
import numpy as np

# Adjusting paths to fit the OULAD dataset in your data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INFO_PATH = r"C:\Users\srila\OneDrive\Documents\ML sem 21\ML-P-A\data\studentInfo.csv"
ASSESSMENT_PATH = r"C:\Users\srila\OneDrive\Documents\ML sem 21\ML-P-A\data\studentAssessment.csv"

def load_raw_data(info_path: str = INFO_PATH, nrows: int = None) -> pd.DataFrame:
    """Loads the original raw studentInfo dataset without any transformations or cleaning."""
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"File not found: {info_path}")
    df_raw = pd.read_csv(info_path, nrows=nrows)
    return df_raw

def load_data(info_path: str = INFO_PATH, assessment_path: str = ASSESSMENT_PATH, nrows: int = None) -> pd.DataFrame:
    """Loads raw data and cleans it (replaces '?' with NaN, merges average assessment score)."""
    df_info = load_raw_data(info_path=info_path, nrows=nrows)
    df_clean = df_info.copy()
    df_clean['imd_band'] = df_clean['imd_band'].replace('?', np.nan)
    
    if os.path.exists(assessment_path):
        df_assess = pd.read_csv(assessment_path)
        # Convert score to numeric, replacing non-numeric strings with NaN
        df_assess['score'] = pd.to_numeric(df_assess['score'], errors='coerce')
        avg_scores = df_assess.groupby('id_student')['score'].mean().reset_index()
        avg_scores.rename(columns={'score': 'avg_score'}, inplace=True)
        df_clean = pd.merge(df_clean, avg_scores, on='id_student', how='left')
    else:
        df_clean['avg_score'] = np.nan
        
    return df_clean

def get_data_summary(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_raw_data()
    
    # Replace NaN with None so it is valid JSON (null) in template
    preview_df = df.head(10).replace({np.nan: None})
    
    col_metadata = []
    missing_counts = {}
    q_mark_counts = {}
    for col in df.columns:
        null_c = int(df[col].isnull().sum())
        q_c = int((df[col] == '?').sum()) if df[col].dtype == object else 0
        missing_counts[col] = null_c
        q_mark_counts[col] = q_c
        col_metadata.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": null_c,
            "q_mark_count": q_c,
            "total_missing": null_c + q_c
        })

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": missing_counts,
        "q_mark_counts": q_mark_counts,
        "col_metadata": col_metadata,
        "preview": preview_df.to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = load_data()
    print("Raw Data Summary:", get_data_summary(raw_df))
    print("Clean Data Summary:", get_data_summary(clean_df))

