import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_base64_image():
    """Helper to convert the current matplotlib figure to a base64 string."""
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return img_base64

def generate_eda_charts(df_clean, df_sample):
    """Generates all 15 EDA charts using Seaborn/Matplotlib and returns base64 images as a dict."""
    charts = {}
    
    # Apply a dark theme to match the frontend glassmorphism style
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#00000000",
        "figure.facecolor": "#00000000",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
        "grid.color": "#ffffff1a"
    })
    
    # 2, 3, 4. Missing & Duplicates
    missing = (df_clean.isnull().sum() / len(df_clean) * 100).to_dict()
    charts['stats'] = {
        "missing": missing,
        "duplicates": int(df_clean.duplicated().sum())
    }
    
    # 5. Target Variable Distribution
    plt.figure(figsize=(8, 5))
    if 'final_result' in df_clean.columns:
        sns.countplot(x="final_result", data=df_clean, order=["Distinction", "Pass", "Fail", "Withdrawn"], palette="viridis")
        plt.title("Distribution of Final Results")
    charts['chart_target_dist'] = get_base64_image()
    
    # 6. Numeric Feature Distributions
    if 'studied_credits' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_clean['studied_credits'], bins=30, kde=False, color="skyblue")
        plt.title("Studied Credits Distribution")
        charts['chart_num_dist_1'] = get_base64_image()
        
    if 'avg_score' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_clean['avg_score'].dropna(), bins=30, kde=True, color="salmon")
        plt.axvline(x=np.nanmean(df_clean['avg_score']), color="white", linestyle="--", label="Mean")
        plt.legend()
        plt.title("Average Assessment Scores Distribution")
        charts['chart_num_dist_2'] = get_base64_image()
    
    # 7. Outlier Detection (Boxplots)
    plt.figure(figsize=(10, 5))
    cols_to_plot = [c for c in ['studied_credits', 'avg_score'] if c in df_clean.columns]
    if cols_to_plot:
        sns.boxplot(data=df_clean[cols_to_plot], orient="h", palette="Set2")
        plt.title("Outlier Detection (Boxplots)")
    charts['chart_boxplots'] = get_base64_image()
    
    # 8. Correlation Analysis
    numeric_df = df_clean.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.round(corr, 2), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        charts['chart_corr'] = get_base64_image()
    
    # 9. Relationship Plots
    plt.figure(figsize=(8, 5))
    if 'studied_credits' in df_sample.columns and 'avg_score' in df_sample.columns and 'final_result' in df_sample.columns:
        sns.scatterplot(x="studied_credits", y="avg_score", hue="final_result", data=df_sample, alpha=0.6, palette="viridis")
        plt.title("Studied Credits vs Avg Score (Sampled)")
    charts['chart_scatter'] = get_base64_image()
    
    # 10. Categorical Feature Counts
    if 'highest_education' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(y="highest_education", data=df_clean, palette="mako")
        plt.title("Education Levels")
        charts['chart_cat_1'] = get_base64_image()
        
    if 'age_band' in df_clean.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(y="age_band", data=df_clean, palette="mako")
        plt.title("Age Bands")
        charts['chart_cat_2'] = get_base64_image()
    
    # 11. Gender vs Target
    plt.figure(figsize=(8, 5))
    if 'gender' in df_clean.columns and 'final_result' in df_clean.columns:
        sns.countplot(x="gender", hue="final_result", data=df_clean, palette="viridis")
        plt.title("Final Result by Gender")
    charts['chart_gender_target'] = get_base64_image()
    
    # 12. Education vs Target
    plt.figure(figsize=(10, 6))
    if 'highest_education' in df_clean.columns and 'final_result' in df_clean.columns:
        sns.countplot(y="highest_education", hue="final_result", data=df_clean, palette="viridis")
        plt.title("Final Result by Education")
    charts['chart_edu_target'] = get_base64_image()
    
    # 13. Trend Analysis
    plt.figure(figsize=(8, 5))
    if 'age_band' in df_clean.columns and 'avg_score' in df_clean.columns:
        trend_df = df_clean.groupby("age_band")["avg_score"].mean().reset_index()
        sns.lineplot(x="age_band", y="avg_score", data=trend_df, marker="o", color="cyan")
        plt.title("Average Score Trend by Age Band")
    charts['chart_trend'] = get_base64_image()
    
    # 14. Salary/Score Analysis
    plt.figure(figsize=(10, 6))
    if 'highest_education' in df_sample.columns and 'avg_score' in df_sample.columns:
        sns.boxplot(x="highest_education", y="avg_score", data=df_sample, palette="Set3")
        plt.xticks(rotation=45, ha="right")
        plt.title("Avg Score by Education")
    charts['chart_score_edu'] = get_base64_image()
    
    # 15. Pairplot
    pair_cols = ["num_of_prev_attempts", "studied_credits", "avg_score"]
    if all(col in df_clean.columns for col in pair_cols) and 'final_result' in df_clean.columns:
        # Pairplot creates its own figure
        sns.pairplot(df_sample, vars=pair_cols, hue="final_result", palette="viridis")
        plt.suptitle("Pairplot (Sampled)", y=1.02)
        charts['chart_pairplot'] = get_base64_image()
        
    return charts
