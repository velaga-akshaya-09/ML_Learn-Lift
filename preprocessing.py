import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from load_data import load_data

# =====================================================
# OULAD Dataset Specific Numeric Columns & Ordinal Maps
# =====================================================

NUMERIC_COLUMNS = [
    "studied_credits",
    "avg_score",
    "num_of_prev_attempts"
]

AGE_BAND_MAP = {
    "0-35": 0,
    "35-55": 1,
    "55<=": 2
}

HIGHEST_EDU_MAP = {
    "No Formal quals": 0,
    "Lower Than A Level": 1,
    "A Level or Equivalent": 2,
    "HE Qualification": 3,
    "Post Graduate Qualification": 4
}

IMD_BAND_MAP = {
    "0-10%": 0,
    "10-20%": 1,
    "20-30%": 2,
    "30-40%": 3,
    "40-50%": 4,
    "50-60%": 5,
    "60-70%": 6,
    "70-80%": 7,
    "80-90%": 8,
    "90-100%": 9
}

GENDER_MAP = {"F": 0, "M": 1}
DISABILITY_MAP = {"N": 0, "Y": 1}
FINAL_RESULT_MAP = {
    "Withdrawn": 0,
    "Fail": 1,
    "Pass": 2,
    "Distinction": 3
}

# =====================================================
# Task 0 : Missing Value Imputation
# =====================================================

def impute_missing(data):
    """Imputes missing values in OULAD dataset (avg_score with median, imd_band with mode)."""
    data = data.copy()
    imputed_counts = {}

    if 'avg_score' in data.columns:
        null_count = int(data['avg_score'].isnull().sum())
        imputed_counts['avg_score'] = null_count
        if null_count > 0:
            median_score = data['avg_score'].median()
            data['avg_score'] = data['avg_score'].fillna(median_score)

    if 'imd_band' in data.columns:
        null_count = int(data['imd_band'].isnull().sum())
        imputed_counts['imd_band'] = null_count
        if null_count > 0:
            mode_band = data['imd_band'].mode()[0] if not data['imd_band'].mode().empty else "10-20%"
            data['imd_band'] = data['imd_band'].fillna(mode_band)

    return data, imputed_counts

# =====================================================
# Task 1 : Outlier Fix (1.5 x IQR Clipping)
# =====================================================

def fix_outliers(data):
    """Clips outliers in numeric features using 1.5 x IQR limits."""
    data = data.copy()
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in data.columns]
    outlier_counts = {}

    for col in numeric_cols:
        col_data = data[col].dropna()
        if col_data.empty:
            outlier_counts[col] = 0
            continue

        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = (col_data < lower_limit) | (col_data > upper_limit)
        outlier_counts[col] = int(outliers.sum())

        # Fix outliers using clipping
        data[col] = data[col].clip(
            lower=lower_limit,
            upper=upper_limit
        )

    return data, outlier_counts

# =====================================================
# Task 2 : Min-Max Scaling
# =====================================================

def minmax_scaling(data):
    """Scales numeric features to range [0, 1]."""
    data = data.copy()
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in data.columns]
    if not numeric_cols:
        return data

    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# =====================================================
# Task 3 : Standard Scaling
# =====================================================

def standard_scaling(data):
    """Standardizes numeric features (Z-score: Mean=0, Std=1)."""
    data = data.copy()
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in data.columns]
    if not numeric_cols:
        return data

    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# =====================================================
# Task 4 : Categorical & Ordinal Encoding
# =====================================================

def encode_categorical(data):
    """Encodes categorical and ordinal columns into numerical values for OULAD dataset."""
    data = data.copy()
    encoded_cols = []

    if 'age_band' in data.columns:
        data['age_band_encoded'] = data['age_band'].map(AGE_BAND_MAP).fillna(0).astype(int)
        encoded_cols.append('age_band_encoded')

    if 'highest_education' in data.columns:
        data['highest_education_encoded'] = data['highest_education'].map(HIGHEST_EDU_MAP).fillna(0).astype(int)
        encoded_cols.append('highest_education_encoded')

    if 'imd_band' in data.columns:
        data['imd_band_encoded'] = data['imd_band'].map(IMD_BAND_MAP).fillna(1).astype(int)
        encoded_cols.append('imd_band_encoded')

    if 'gender' in data.columns:
        data['gender_encoded'] = data['gender'].map(GENDER_MAP).fillna(0).astype(int)
        encoded_cols.append('gender_encoded')

    if 'disability' in data.columns:
        data['disability_encoded'] = data['disability'].map(DISABILITY_MAP).fillna(0).astype(int)
        encoded_cols.append('disability_encoded')

    if 'final_result' in data.columns:
        data['final_result_encoded'] = data['final_result'].map(FINAL_RESULT_MAP).fillna(1).astype(int)
        encoded_cols.append('final_result_encoded')

    return data, encoded_cols

# =====================================================
# Main Preprocessing Pipeline Function
# =====================================================

def run_preprocessing():
    """Runs the full project-specific OULAD preprocessing pipeline."""
    raw_data = load_data()
    original_rows = len(raw_data)
    original_columns = len(raw_data.columns)

    # 1. Impute missing values
    imputed_data, imputation_summary = impute_missing(raw_data)

    # 2. Outlier Fix (Clipping)
    outlier_fixed_data, outlier_counts = fix_outliers(imputed_data)

    # 3. Min-Max Scaling
    minmax_data = minmax_scaling(outlier_fixed_data)

    # 4. Standard Scaling
    standard_data = standard_scaling(outlier_fixed_data)

    # 5. Categorical Encoding
    encoded_data, encoded_cols = encode_categorical(imputed_data)

    # Preview numeric columns
    numeric_cols = [col for col in NUMERIC_COLUMNS if col in raw_data.columns]

    orig_prev = raw_data[numeric_cols].head(10).round(3).replace({np.nan: None}).to_dict(orient="records")
    minmax_prev = minmax_data[numeric_cols].head(10).round(3).replace({np.nan: None}).to_dict(orient="records")
    standard_prev = standard_data[numeric_cols].head(10).round(3).replace({np.nan: None}).to_dict(orient="records")
    encoded_prev = encoded_data[encoded_cols].head(10).to_dict(orient="records")

    return {
        "n_rows": original_rows,
        "n_cols": original_columns,
        "numeric_columns": numeric_cols,
        "encoded_columns": encoded_cols,
        "imputation_summary": imputation_summary,
        "outlier_counts": outlier_counts,
        "total_outliers": sum(outlier_counts.values()),
        "original_preview": orig_prev,
        "minmax_preview": minmax_prev,
        "standard_preview": standard_prev,
        "encoded_preview": encoded_prev
    }

if __name__ == "__main__":
    res = run_preprocessing()
    print("OULAD Preprocessing Pipeline Test:")
    print("Shape:", res["n_rows"], "x", res["n_cols"])
    print("Imputation Summary:", res["imputation_summary"])
    print("Outlier Counts:", res["outlier_counts"])
    print("Encoded Columns:", res["encoded_columns"])
