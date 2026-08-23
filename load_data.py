import os
import pandas as pd
import numpy as np

# Adjusting paths to fit the OULAD dataset in your data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INFO_PATH = r"C:\Users\srila\OneDrive\Documents\ML sem 21\ML-P-A\data\studentInfo.csv"
ASSESSMENT_PATH = r"C:\Users\srila\OneDrive\Documents\ML sem 21\ML-P-A\data\studentAssessment.csv"

def load_data(info_path: str = INFO_PATH, assessment_path: str = ASSESSMENT_PATH, nrows: int = None) -> pd.DataFrame:
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"File not found: {info_path}")
        
    df_info = pd.read_csv(info_path, nrows=nrows)
    df_info['imd_band'] = df_info['imd_band'].replace('?', np.nan)
    
    if os.path.exists(assessment_path):
        df_assess = pd.read_csv(assessment_path)
        # Convert score to numeric, replacing non-numeric strings with NaN
        df_assess['score'] = pd.to_numeric(df_assess['score'], errors='coerce')
        avg_scores = df_assess.groupby('id_student')['score'].mean().reset_index()
        avg_scores.rename(columns={'score': 'avg_score'}, inplace=True)
        df_clean = pd.merge(df_info, avg_scores, on='id_student', how='left')
    else:
        df_clean = df_info.copy()
        df_clean['avg_score'] = np.nan
        
    return df_clean

def get_data_summary(df: pd.DataFrame = None) -> dict:
    if df is None:
        df = load_data()
    
    # Replace NaN with None so it is valid JSON (null)
    preview_df = df.head(10).replace({np.nan: None})
    
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isnull().sum()) for col in df.columns},
        "preview": preview_df.to_dict("records"),
    }
    return summary

if __name__ == "__main__":
    print(get_data_summary())
