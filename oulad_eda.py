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
    fig.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return img_base64

def generate_eda_charts(df_clean, df_sample):
    """Generates all EDA charts using Seaborn/Matplotlib and returns base64 images as a dict."""
    charts = {}
    
    # Apply a dark theme matching the frontend glassmorphism style
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#0f172a",
        "figure.facecolor": "#00000000",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "#cbd5e1",
        "ytick.color": "#cbd5e1",
        "axes.edgecolor": "#334155",
        "grid.color": "#1e293b"
    })
    
    # 2, 3, 4. Data Integrity (Missing & Duplicates)
    null_missing = df_clean.isnull().sum()
    total_len = len(df_clean)
    missing_pct = (null_missing / total_len * 100).to_dict()
    
    charts['stats'] = {
        "missing": missing_pct,
        "duplicates": int(df_clean.duplicated().sum()),
        "total_rows": total_len,
        "total_cols": df_clean.shape[1]
    }
    
    # Missing Values Visualization Chart
    missing_cols = {col: pct for col, pct in missing_pct.items() if pct > 0}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if missing_cols:
        cols = list(missing_cols.keys())
        vals = list(missing_cols.values())
        sns.barplot(x=cols, y=vals, hue=cols, palette="magma", ax=ax, legend=False)
        ax.set_title("Percentage of Missing Values per Feature", fontsize=12, fontweight='bold')
        ax.set_ylabel("Missing Percentage (%)")
        ax.set_ylim(0, max(vals) * 1.30 if vals else 100)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.8, f"{v:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')
    else:
        ax.text(0.5, 0.5, "No Missing Values Found", ha='center', va='center', color='white', fontsize=14)
        ax.set_title("Missing Values Check", fontsize=12, fontweight='bold')
    charts['chart_missing'] = get_base64_image(fig)
    
    # 5. Target Variable Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'final_result' in df_clean.columns:
        order = ["Distinction", "Pass", "Fail", "Withdrawn"]
        sns.countplot(x="final_result", hue="final_result", data=df_clean, order=order, palette="viridis", ax=ax, legend=False)
        ax.set_title("Target Variable Distribution (Final Result)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Final Result")
        ax.set_ylabel("Student Count")
        
        # Determine maximum height for proper y-limit scaling
        max_height = df_clean['final_result'].value_counts().max()
        ax.set_ylim(0, max_height * 1.18)
        
        # Add value and percentage annotations on top of bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                pct = (height / total_len) * 100
                ax.annotate(f'{int(height):,}\n({pct:.1f}%)',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=9, color='white',
                            xytext=(0, 4), textcoords='offset points')
    charts['chart_target_dist'] = get_base64_image(fig)
    
    # 6. Numeric Feature Distributions
    if 'studied_credits' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_clean['studied_credits'], bins=30, kde=True, color="#38bdf8", ax=ax)
        mean_val = df_clean['studied_credits'].mean()
        median_val = df_clean['studied_credits'].median()
        ax.axvline(mean_val, color="#f43f5e", linestyle="--", linewidth=1.5, label=f"Mean ({mean_val:.1f})")
        ax.axvline(median_val, color="#a855f7", linestyle=":", linewidth=1.5, label=f"Median ({median_val:.1f})")
        ax.legend()
        ax.set_title("Studied Credits Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Studied Credits")
        charts['chart_num_dist_1'] = get_base64_image(fig)
        
    if 'avg_score' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_scores = df_clean['avg_score'].dropna()
        sns.histplot(valid_scores, bins=30, kde=True, color="#fb7185", ax=ax)
        mean_score = valid_scores.mean()
        median_score = valid_scores.median()
        ax.axvline(mean_score, color="white", linestyle="--", linewidth=1.5, label=f"Mean ({mean_score:.1f})")
        ax.axvline(median_score, color="#38bdf8", linestyle=":", linewidth=1.5, label=f"Median ({median_score:.1f})")
        ax.legend()
        ax.set_title("Average Assessment Score Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Average Assessment Score")
        charts['chart_num_dist_2'] = get_base64_image(fig)
    
    # 7. Outlier Detection (Separate Subplots for all numeric features)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
    if 'studied_credits' in df_clean.columns:
        sns.boxplot(y=df_clean['studied_credits'], color="#38bdf8", ax=ax1)
        ax1.set_title("Studied Credits", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Studied Credits")
    if 'avg_score' in df_clean.columns:
        sns.boxplot(y=df_clean['avg_score'].dropna(), color="#fb7185", ax=ax2)
        ax2.set_title("Average Score", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Average Score")
    if 'num_of_prev_attempts' in df_clean.columns:
        sns.boxplot(y=df_clean['num_of_prev_attempts'], color="#a78bfa", ax=ax3)
        ax3.set_title("Prev Attempts", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Number of Attempts")
    charts['chart_boxplots'] = get_base64_image(fig)
    
    # 8. Correlation Analysis (Exclude ID non-feature columns)
    ignore_cols = ['id_student', 'id_assessment', 'id_site']
    numeric_cols = [c for c in df_clean.select_dtypes(include=[np.number]).columns if c not in ignore_cols]
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df_clean[numeric_cols].corr()
        sns.heatmap(np.round(corr, 2), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=12, fontweight='bold')
        charts['chart_corr'] = get_base64_image(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient numeric features for correlation", ha='center', va='center')
        charts['chart_corr'] = get_base64_image(fig)
    
    # 9. Relationship Plots
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'studied_credits' in df_sample.columns and 'avg_score' in df_sample.columns and 'final_result' in df_sample.columns:
        clean_sample = df_sample.dropna(subset=['avg_score', 'studied_credits', 'final_result'])
        sns.scatterplot(x="studied_credits", y="avg_score", hue="final_result", data=clean_sample, alpha=0.6, palette="viridis", ax=ax)
        ax.set_title("Studied Credits vs Avg Score (Sampled)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Studied Credits")
        ax.set_ylabel("Average Assessment Score")
    charts['chart_scatter'] = get_base64_image(fig)
    
    # 10. Categorical Feature Counts
    if 'highest_education' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        edu_order = df_clean['highest_education'].value_counts().index
        sns.countplot(y="highest_education", hue="highest_education", data=df_clean, order=edu_order, palette="mako", ax=ax, legend=False)
        ax.set_title("Highest Education Breakdown", fontsize=12, fontweight='bold')
        ax.set_xlabel("Student Count")
        ax.set_ylabel("Education Level")
        charts['chart_cat_1'] = get_base64_image(fig)
        
    if 'age_band' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        age_order = ['0-35', '35-55', '55<=']
        present_age = [a for a in age_order if a in df_clean['age_band'].values] or df_clean['age_band'].value_counts().index
        sns.countplot(y="age_band", hue="age_band", data=df_clean, order=present_age, palette="mako", ax=ax, legend=False)
        ax.set_title("Age Band Breakdown", fontsize=12, fontweight='bold')
        ax.set_xlabel("Student Count")
        ax.set_ylabel("Age Band")
        charts['chart_cat_2'] = get_base64_image(fig)
    
    # 11. Gender vs Target
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'gender' in df_clean.columns and 'final_result' in df_clean.columns:
        sns.countplot(x="gender", hue="final_result", data=df_clean, palette="viridis", ax=ax)
        ax.set_title("Final Result Distribution by Gender", fontsize=12, fontweight='bold')
        ax.set_xlabel("Gender")
        ax.set_ylabel("Student Count")
    charts['chart_gender_target'] = get_base64_image(fig)
    
    # 12. Education vs Target
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'highest_education' in df_clean.columns and 'final_result' in df_clean.columns:
        sns.countplot(y="highest_education", hue="final_result", data=df_clean, palette="viridis", ax=ax)
        ax.set_title("Final Result Distribution by Education Level", fontsize=12, fontweight='bold')
        ax.set_xlabel("Student Count")
        ax.set_ylabel("Highest Education Level")
    charts['chart_edu_target'] = get_base64_image(fig)
    
    # 13. Trend Analysis (Logical age band ordering: ['0-35', '35-55', '55<='])
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'age_band' in df_clean.columns and 'avg_score' in df_clean.columns:
        age_order = ['0-35', '35-55', '55<=']
        trend_df = df_clean.groupby("age_band")["avg_score"].mean().reset_index()
        # Sort trend_df according to logical age_order
        trend_df['age_order'] = trend_df['age_band'].map(lambda x: age_order.index(x) if x in age_order else 99)
        trend_df = trend_df.sort_values('age_order')
        
        sns.lineplot(x="age_band", y="avg_score", data=trend_df, marker="o", markersize=8, color="#38bdf8", linewidth=2.5, ax=ax)
        for _, row in trend_df.iterrows():
            ax.annotate(f"{row['avg_score']:.1f}", (row['age_band'], row['avg_score']),
                        xytext=(0, 8), textcoords='offset points', ha='center', color='white', fontweight='bold')
        ax.set_title("Average Score Trend Across Age Bands", fontsize=12, fontweight='bold')
        ax.set_xlabel("Age Band")
        ax.set_ylabel("Mean Average Score")
    charts['chart_trend'] = get_base64_image(fig)
    
    # 14. Average Score by Education
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'highest_education' in df_sample.columns and 'avg_score' in df_sample.columns:
        edu_levels = ['No Formal quals', 'Lower Than A Level', 'A Level or Equivalent', 'HE Qualification', 'Post Graduate Qualification']
        order = [e for e in edu_levels if e in df_sample['highest_education'].values] or list(df_sample['highest_education'].value_counts().index)
        sns.boxplot(x="highest_education", y="avg_score", hue="highest_education", data=df_sample, order=order, palette="Set3", ax=ax, legend=False)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title("Average Assessment Score Distribution by Education Level", fontsize=12, fontweight='bold')
        ax.set_xlabel("Highest Education Level")
        ax.set_ylabel("Average Score")
    charts['chart_score_edu'] = get_base64_image(fig)
    
    # 15. Pairplot
    pair_cols = ["num_of_prev_attempts", "studied_credits", "avg_score"]
    valid_pair_cols = [c for c in pair_cols if c in df_sample.columns]
    if len(valid_pair_cols) >= 2 and 'final_result' in df_sample.columns:
        clean_sample = df_sample.dropna(subset=valid_pair_cols + ['final_result'])
        g = sns.pairplot(clean_sample, vars=valid_pair_cols, hue="final_result", palette="viridis", corner=True)
        g.fig.suptitle("Feature Pairplot Matrix (Sampled)", y=1.02, fontsize=14, fontweight='bold')
        charts['chart_pairplot'] = get_base64_image(g.fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Insufficient numeric variables for pairplot", ha='center', va='center')
        charts['chart_pairplot'] = get_base64_image(fig)
        
    return charts
