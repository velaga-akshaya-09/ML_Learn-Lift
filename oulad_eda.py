import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def get_base64_image(fig=None):
    """Helper to convert a matplotlib figure to a base64 string cleanly."""
    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(buf, format='png', facecolor='#ffffff', edgecolor='none', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return img_base64

def set_light_theme():
    """Applies a clean, modern light theme matching the professional blue-and-white visual identity."""
    plt.style.use("default")
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "text.color": "#0f172a",
        "axes.labelcolor": "#1e293b",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.edgecolor": "#e2e8f0",
        "grid.color": "#f1f5f9",
        "font.family": "sans-serif"
    })

def generate_eda_charts(df_clean, df_sample):
    """
    Generates EDA charts organized into 5 explicit tasks using a crisp blue/white color palette:
      • Task 1: Data Integrity & Missing Value Profiling
      • Task 2: Target Variable Distribution (Final Result)
      • Task 3: Numeric Feature Distributions & Outliers (Boxplots)
      • Task 4: Correlation Matrix & Feature Relationships
      • Task 5: Demographics & Categorical Breakdown
    """
    charts = {}
    set_light_theme()
    
    total_len = len(df_clean)
    null_missing = df_clean.isnull().sum()
    missing_pct = (null_missing / total_len * 100).to_dict()
    
    charts['stats'] = {
        "missing": missing_pct,
        "duplicates": int(df_clean.duplicated().sum()),
        "total_rows": total_len,
        "total_cols": df_clean.shape[1]
    }
    
    # ------------------------------------------------------------
    # TASK 1: Data Integrity & Missing Profiling
    # ------------------------------------------------------------
    missing_cols = {col: pct for col, pct in missing_pct.items() if pct > 0}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if missing_cols:
        cols = list(missing_cols.keys())
        vals = list(missing_cols.values())
        sns.barplot(x=cols, y=vals, hue=cols, palette="Blues_r", ax=ax, legend=False)
        ax.set_title("Task 1: Missing Values Profiling per Feature (%)", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
        ax.set_ylabel("Missing Percentage (%)", fontsize=10, color='#1e293b')
        ax.set_ylim(0, max(vals) * 1.30 if vals else 100)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha='center', va='bottom', color='#1e3a8a', fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha='center', va='center', color='#059669', fontsize=14, fontweight='bold')
        ax.set_title("Task 1: Missing Values Check", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
    charts['chart_missing'] = get_base64_image(fig)
    
    # ------------------------------------------------------------
    # TASK 2: Target Variable Distribution
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'final_result' in df_clean.columns:
        order = ["Distinction", "Pass", "Fail", "Withdrawn"]
        colors = ["#2563eb", "#3b82f6", "#f43f5e", "#64748b"]
        sns.countplot(x="final_result", hue="final_result", data=df_clean, order=order, palette=colors, ax=ax, legend=False)
        ax.set_title("Task 2: Target Variable Distribution (Final Result)", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
        ax.set_xlabel("Final Result", fontsize=10, color='#1e293b')
        ax.set_ylabel("Student Count", fontsize=10, color='#1e293b')
        
        max_height = df_clean['final_result'].value_counts().max()
        ax.set_ylim(0, max_height * 1.18)
        
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                pct = (height / total_len) * 100
                ax.annotate(f'{int(height):,}\n({pct:.1f}%)',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, color='#1e293b', fontweight='bold',
                            xytext=(0, 4), textcoords='offset points')
    charts['chart_target_dist'] = get_base64_image(fig)
    
    # ------------------------------------------------------------
    # TASK 3: Numeric Feature Distributions & Outliers
    # ------------------------------------------------------------
    if 'studied_credits' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df_clean['studied_credits'], bins=30, kde=True, color="#2563eb", ax=ax)
        mean_val = df_clean['studied_credits'].mean()
        median_val = df_clean['studied_credits'].median()
        ax.axvline(mean_val, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean ({mean_val:.1f})")
        ax.axvline(median_val, color="#059669", linestyle=":", linewidth=1.5, label=f"Median ({median_val:.1f})")
        ax.legend()
        ax.set_title("Task 3a: Studied Credits Distribution", fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        ax.set_xlabel("Studied Credits", fontsize=10, color='#1e293b')
        charts['chart_num_dist_1'] = get_base64_image(fig)
        
    if 'avg_score' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        valid_scores = df_clean['avg_score'].dropna()
        sns.histplot(valid_scores, bins=30, kde=True, color="#0284c7", ax=ax)
        mean_score = valid_scores.mean()
        median_score = valid_scores.median()
        ax.axvline(mean_score, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean ({mean_score:.1f})")
        ax.axvline(median_score, color="#2563eb", linestyle=":", linewidth=1.5, label=f"Median ({median_score:.1f})")
        ax.legend()
        ax.set_title("Task 3b: Average Assessment Score Distribution", fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        ax.set_xlabel("Average Score", fontsize=10, color='#1e293b')
        charts['chart_num_dist_2'] = get_base64_image(fig)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
    if 'studied_credits' in df_clean.columns:
        sns.boxplot(y=df_clean['studied_credits'], color="#bfdbfe", ax=ax1)
        ax1.set_title("Studied Credits Outliers", fontsize=10, fontweight='bold', color='#0f172a')
    if 'avg_score' in df_clean.columns:
        sns.boxplot(y=df_clean['avg_score'].dropna(), color="#bae6fd", ax=ax2)
        ax2.set_title("Average Score Outliers", fontsize=10, fontweight='bold', color='#0f172a')
    if 'num_of_prev_attempts' in df_clean.columns:
        sns.boxplot(y=df_clean['num_of_prev_attempts'], color="#ddd6fe", ax=ax3)
        ax3.set_title("Prev Attempts Outliers", fontsize=10, fontweight='bold', color='#0f172a')
    charts['chart_boxplots'] = get_base64_image(fig)
    
    # ------------------------------------------------------------
    # TASK 4: Correlation & Pairwise Relationship Analysis
    # ------------------------------------------------------------
    ignore_cols = ['id_student', 'id_assessment', 'id_site']
    numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        corr = df_clean[numeric_cols].corr()
        sns.heatmap(np.round(corr, 2), annot=True, cmap="Blues", fmt=".2f", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title("Task 4a: Correlation Heatmap", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
        charts['chart_corr'] = get_base64_image(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient numeric features for correlation", ha='center', va='center')
        charts['chart_corr'] = get_base64_image(fig)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'studied_credits' in df_sample.columns and 'avg_score' in df_sample.columns and 'final_result' in df_sample.columns:
        clean_sample = df_sample.dropna(subset=['avg_score', 'studied_credits', 'final_result'])
        sns.scatterplot(x="studied_credits", y="avg_score", hue="final_result", data=clean_sample, alpha=0.7, palette="tab10", ax=ax)
        ax.set_title("Task 4b: Studied Credits vs Average Score", fontsize=12, fontweight='bold', color='#0f172a', pad=12)
    charts['chart_scatter'] = get_base64_image(fig)
    
    # ------------------------------------------------------------
    # TASK 5: Demographics & Categorical Breakdown
    # ------------------------------------------------------------
    if 'highest_education' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        edu_order = df_clean['highest_education'].value_counts().index
        sns.countplot(y="highest_education", hue="highest_education", data=df_clean, order=edu_order, palette="Blues_r", ax=ax, legend=False)
        ax.set_title("Task 5a: Education Level Breakdown", fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        charts['chart_cat_1'] = get_base64_image(fig)
        
    if 'gender' in df_clean.columns and 'final_result' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.countplot(x="gender", hue="final_result", data=df_clean, palette="Set2", ax=ax)
        ax.set_title("Task 5b: Gender vs Final Result Outcome", fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        charts['chart_gender_target'] = get_base64_image(fig)
        
    return charts
