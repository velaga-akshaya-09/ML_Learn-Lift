import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

from load_data import load_data

def get_base64_image(fig=None):
    """Converts a matplotlib figure to a base64 string cleanly."""
    buf = io.BytesIO()
    if fig is None:
        fig = plt.gcf()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(buf, format='png', facecolor='#ffffff', edgecolor='none', bbox_inches='tight', dpi=120)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return img_base64

def set_light_theme():
    """Applies modern light theme matching the professional blue-and-white visual identity."""
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

def prepare_clustering_data(max_samples=None):
    """Loads clean student dataset and prepares numeric features for distance-based clustering."""
    df = load_data()
    
    # Impute avg_score missing values with median
    median_score = df['avg_score'].median()
    df['avg_score_clean'] = df['avg_score'].fillna(median_score)
    
    # Numeric features for clustering
    num_cols = ["studied_credits", "avg_score_clean", "num_of_prev_attempts"]
    
    df_clean = df.copy()
    df_clean['studied_credits'] = pd.to_numeric(df_clean['studied_credits'], errors='coerce').fillna(df_clean['studied_credits'].median())
    df_clean['num_of_prev_attempts'] = pd.to_numeric(df_clean['num_of_prev_attempts'], errors='coerce').fillna(0)
    
    if max_samples and len(df_clean) > max_samples:
        df_sample = df_clean.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_clean.reset_index(drop=True)
        
    X_num = df_sample[num_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    return X_scaled, df_sample, scaler

# ============================================================
# 1. K-MEANS CLUSTERING ROUTINE
# ============================================================

def run_kmeans_clustering(k=4):
    set_light_theme()
    k = max(2, min(8, int(k)))
    
    X_scaled, df_sample, scaler = prepare_clustering_data()
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_sample['cluster'] = cluster_labels
    
    inertia = float(kmeans.inertia_)
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    db_score = float(davies_bouldin_score(X_scaled, cluster_labels))
    
    # Calculate Elbow Curve values for k in 2..8
    elbow_ks = list(range(2, 9))
    elbow_inertias = []
    for test_k in elbow_ks:
        km_test = KMeans(n_clusters=test_k, random_state=42, n_init=10)
        km_test.fit(X_scaled)
        elbow_inertias.append(float(km_test.inertia_))
        
    # Chart 1: 2D Cluster Scatter Plot (studied_credits vs avg_score_clean)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = sns.color_palette("tab10", k)
    
    sns.scatterplot(
        data=df_sample,
        x='studied_credits',
        y='avg_score_clean',
        hue='cluster',
        palette=palette,
        alpha=0.6,
        s=30,
        ax=ax
    )
    
    # Transform centroids back to original feature scale for plotting
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids_orig[:, 0],
        centroids_orig[:, 1],
        c='red',
        marker='X',
        s=140,
        linewidths=2,
        edgecolor='white',
        label='Centroids',
        zorder=5
    )
    
    ax.set_title(f'2D Cluster Scatter Visualization (k={k})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Studied Credits', fontsize=10, color='#1e293b')
    ax.set_ylabel('Average Assessment Score (%)', fontsize=10, color='#1e293b')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    chart_scatter = get_base64_image(fig)
    
    # Chart 2: Elbow Method Optimization Curve
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(elbow_ks, elbow_inertias, 'o-', color='#0284c7', linewidth=2.5, markersize=7, label='Inertia (WCSS)')
    ax.axvline(x=k, color='#dc2626', linestyle='--', linewidth=1.8, label=f'Selected k = {k}')
    ax.plot(k, elbow_inertias[k-2], 'ro', markersize=9)
    
    ax.set_title('Elbow Method Optimization Curve', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Number of Clusters (k)', fontsize=10, color='#1e293b')
    ax.set_ylabel('Inertia (Sum of Squared Distances)', fontsize=10, color='#1e293b')
    ax.legend(loc='upper right')
    chart_elbow = get_base64_image(fig)
    
    # Cluster Size Distribution Table
    total_pts = len(df_sample)
    size_dist = []
    for c in range(k):
        cnt = int((cluster_labels == c).sum())
        prop = round((cnt / total_pts) * 100, 2)
        size_dist.append({
            "cluster": f"Cluster {c + 1}",
            "count": cnt,
            "proportion": f"{prop}%"
        })
        
    # Cluster Centroid Feature Values Table (Original Scale)
    centroids_table = []
    for c in range(k):
        sub = df_sample[df_sample['cluster'] == c]
        centroids_table.append({
            "cluster": f"Cluster {c + 1}",
            "avg_score": round(float(sub['avg_score_clean'].mean()), 2),
            "studied_credits": round(float(sub['studied_credits'].mean()), 2),
            "prev_attempts": round(float(sub['num_of_prev_attempts'].mean()), 2)
        })
        
    return {
        "metrics": {
            "Cluster Count (k)": k,
            "Inertia (WCSS)": round(inertia, 2),
            "Silhouette Score": round(sil_score, 4),
            "Davies-Bouldin Index": round(db_score, 4)
        },
        "chart_scatter": chart_scatter,
        "chart_elbow": chart_elbow,
        "size_distribution": size_dist,
        "centroids_table": centroids_table
    }

# ============================================================
# 2. DBSCAN (DENSITY + AUTO-KNEE) ROUTINE
# ============================================================

def run_dbscan_clustering(min_samples=5, eps=None, auto_eps=True):
    set_light_theme()
    min_samples = max(2, int(min_samples))
    
    # Sample 5,000 records for responsiveness
    sample_limit = 5000
    X_scaled, df_sample, scaler = prepare_clustering_data(max_samples=sample_limit)
    notice = f"Notice: DBSCAN executed on a representative sample of {len(df_sample):,} records for responsiveness."
    
    # Calculate k-NN distance curve
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, min_samples - 1])
    
    # Detect knee automatically if requested or eps is empty
    if auto_eps or eps is None or eps == "" or float(eps) <= 0:
        # Detect elbow point via distance difference acceleration
        diffs = np.diff(k_distances)
        knee_idx = int(np.argmax(diffs > np.percentile(diffs, 95)))
        if knee_idx <= 0 or knee_idx >= len(k_distances) - 1:
            knee_idx = int(len(k_distances) * 0.90)
        detected_eps = float(k_distances[knee_idx])
        eps_val = round(max(0.1, detected_eps), 3)
    else:
        eps_val = float(eps)
        # Find closest index for plotting knee
        knee_idx = int(np.searchsorted(k_distances, eps_val))
        knee_idx = min(knee_idx, len(k_distances) - 1)
        
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    df_sample['cluster'] = cluster_labels
    
    unique_clusters = set(cluster_labels) - {-1}
    clusters_found = len(unique_clusters)
    core_points = len(dbscan.core_sample_indices_)
    noise_points = int(np.sum(cluster_labels == -1))
    
    if clusters_found > 1 and len(df_sample[cluster_labels != -1]) > 0:
        non_noise_mask = cluster_labels != -1
        sil_score = float(silhouette_score(X_scaled[non_noise_mask], cluster_labels[non_noise_mask]))
    else:
        sil_score = 0.0
        
    # Chart 1: DBSCAN Density Scatter (Noise vs Clusters)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    
    # Plot noise points first in grey
    noise_mask = (cluster_labels == -1)
    if np.any(noise_mask):
        ax.scatter(
            df_sample.loc[noise_mask, 'studied_credits'],
            df_sample.loc[noise_mask, 'avg_score_clean'],
            c='#94a3b8',
            label='Noise / Outliers (-1)',
            alpha=0.4,
            s=20
        )
        
    # Plot cluster points
    cluster_mask = ~noise_mask
    if np.any(cluster_mask):
        sns.scatterplot(
            data=df_sample[cluster_mask],
            x='studied_credits',
            y='avg_score_clean',
            hue='cluster',
            palette='tab10',
            alpha=0.7,
            s=35,
            ax=ax
        )
        
    ax.set_title(f'DBSCAN Density Clusters & Noise Scatter (eps={eps_val}, min_samples={min_samples})', fontsize=10, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Studied Credits', fontsize=10, color='#1e293b')
    ax.set_ylabel('Average Assessment Score (%)', fontsize=10, color='#1e293b')
    ax.legend(title='Density Clusters', bbox_to_anchor=(1.02, 1), loc='upper left')
    chart_scatter = get_base64_image(fig)
    
    # Chart 2: Automated Knee k-Distance Graph
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(k_distances, color='#0284c7', linewidth=2, label=f'{min_samples}-NN Distance Curve')
    ax.axhline(y=eps_val, color='#059669', linestyle='--', linewidth=1.5, label=f'Recommended eps = {eps_val}')
    ax.axvline(x=knee_idx, color='#dc2626', linestyle=':', linewidth=1.5, label=f'Detected Knee (Index={knee_idx})')
    ax.plot(knee_idx, k_distances[knee_idx], 'ro', markersize=8)
    
    ax.set_title(f'Automated Knee k-Distance Graph (k={min_samples})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Points Sorted by Distance', fontsize=10, color='#1e293b')
    ax.set_ylabel(f'{min_samples}-NN Distance (Standardized)', fontsize=10, color='#1e293b')
    ax.legend(loc='upper left')
    chart_knee = get_base64_image(fig)
    
    # DBSCAN Partition Breakdown Table
    total_pts = len(df_sample)
    breakdown = []
    
    # Noise row
    noise_prop = round((noise_points / total_pts) * 100, 2)
    breakdown.append({
        "partition": "Noise / Outliers",
        "count": noise_points,
        "proportion": f"{noise_prop}%"
    })
    
    for c in sorted(list(unique_clusters)):
        cnt = int((cluster_labels == c).sum())
        prop = round((cnt / total_pts) * 100, 2)
        breakdown.append({
            "partition": f"Cluster {c + 1} (Concentration Zone)",
            "count": cnt,
            "proportion": f"{prop}%"
        })
        
    return {
        "notice": notice,
        "metrics": {
            "Epsilon (eps)": eps_val,
            "Clusters Found": clusters_found,
            "Core Points": core_points,
            "Noise / Outliers": noise_points,
            "Silhouette Score": round(sil_score, 4)
        },
        "chart_scatter": chart_scatter,
        "chart_knee": chart_knee,
        "breakdown": breakdown
    }

# ============================================================
# 3. HIERARCHICAL CLUSTERING (DENDROGRAM) ROUTINE
# ============================================================

def run_hierarchical_clustering(k=3, linkage_criterion='ward'):
    set_light_theme()
    k = max(2, min(8, int(k)))
    valid_linkages = ['ward', 'complete', 'average', 'single']
    if linkage_criterion not in valid_linkages:
        linkage_criterion = 'ward'
        
    sample_limit = 3000
    X_scaled, df_sample, scaler = prepare_clustering_data(max_samples=sample_limit)
    notice = f"Notice: Hierarchical clustering fitted on a representative sample of {len(df_sample):,} records."
    
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_criterion)
    cluster_labels = model.fit_predict(X_scaled)
    df_sample['cluster'] = cluster_labels
    
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    db_score = float(davies_bouldin_score(X_scaled, cluster_labels))
    
    # Dendrogram Plot (Sample 150 points for visual clarity)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sample_sub_idx = np.random.choice(len(X_scaled), size=min(150, len(X_scaled)), replace=False)
    X_sub = X_scaled[sample_sub_idx]
    
    Z = linkage(X_sub, method=linkage_criterion)
    dendrogram(Z, p=30, truncate_mode='level', ax=ax, color_threshold=Z[-k+1, 2] if k > 1 else 0)
    
    ax.set_title(f'Hierarchical Tree Dendrogram (Sample: 150 pts, Linkage: {linkage_criterion.capitalize()})', fontsize=10, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Sample Cluster Subtrees (Condensed Branches)', fontsize=9, color='#1e293b')
    ax.set_ylabel('Euclidean Distance / Merge Height', fontsize=9, color='#1e293b')
    chart_dendrogram = get_base64_image(fig)
    
    # 2D Cluster Scatter Plot
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(
        data=df_sample,
        x='studied_credits',
        y='avg_score_clean',
        hue='cluster',
        palette='tab10',
        alpha=0.6,
        s=30,
        ax=ax
    )
    ax.set_title(f'Agglomerative Cluster Scatter Plot (k={k}, Linkage={linkage_criterion.capitalize()})', fontsize=10, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Studied Credits', fontsize=10, color='#1e293b')
    ax.set_ylabel('Average Assessment Score (%)', fontsize=10, color='#1e293b')
    ax.legend(title='Cluster', bbox_to_anchor=(1.02, 1), loc='upper left')
    chart_scatter = get_base64_image(fig)
    
    # Breakdown Table
    total_pts = len(df_sample)
    breakdown = []
    for c in range(k):
        sub = df_sample[df_sample['cluster'] == c]
        cnt = len(sub)
        prop = round((cnt / total_pts) * 100, 2)
        breakdown.append({
            "cluster": f"Cluster {c + 1}",
            "count": cnt,
            "proportion": f"{prop}%",
            "avg_score": round(float(sub['avg_score_clean'].mean()), 2),
            "avg_credits": round(float(sub['studied_credits'].mean()), 2)
        })
        
    return {
        "notice": notice,
        "metrics": {
            "Cluster Count (k)": k,
            "Linkage Method": linkage_criterion.capitalize(),
            "Silhouette Score": round(sil_score, 4),
            "Davies-Bouldin Index": round(db_score, 4)
        },
        "chart_dendrogram": chart_dendrogram,
        "chart_scatter": chart_scatter,
        "breakdown": breakdown
    }

# ============================================================
# 4. STUDENT DENSITY & DEMOGRAPHIC STUDIO ROUTINE
# ============================================================

def run_density_analysis(feature_x='studied_credits', feature_y='avg_score'):
    set_light_theme()
    
    sample_limit = 5000
    X_scaled, df_sample, scaler = prepare_clustering_data(max_samples=sample_limit)
    notice = f"Notice: Student Density maps rendered on a representative sample of {len(df_sample):,} student records."
    
    # Chart 1: 2D KDE & Contour Density Map of Studied Credits vs Average Score
    fig, ax = plt.subplots(figsize=(7, 4.8))
    
    sns.kdeplot(
        data=df_sample,
        x='studied_credits',
        y='avg_score_clean',
        cmap='Blues',
        fill=True,
        thresh=0.05,
        levels=10,
        alpha=0.7,
        ax=ax
    )
    
    sns.scatterplot(
        data=df_sample.sample(n=min(800, len(df_sample)), random_state=42),
        x='studied_credits',
        y='avg_score_clean',
        hue='final_result',
        alpha=0.5,
        s=20,
        ax=ax
    )
    
    ax.set_title('Student Score & Credit Density Distribution (KDE Contours)', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Studied Credits', fontsize=10, color='#1e293b')
    ax.set_ylabel('Average Assessment Score (%)', fontsize=10, color='#1e293b')
    ax.legend(title='Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
    chart_density = get_base64_image(fig)
    
    # Chart 2: Regional Student Concentration & Density Breakdown
    fig, ax = plt.subplots(figsize=(7, 4.8))
    
    reg_summary = df_sample.groupby('region')['avg_score_clean'].mean().sort_values(ascending=False).head(10).reset_index()
    sns.barplot(
        data=reg_summary,
        y='region',
        x='avg_score_clean',
        palette='Blues_r',
        ax=ax,
        hue='region',
        legend=False
    )
    ax.set_title('Top Geographic Region Concentration (Average Score)', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    ax.set_xlabel('Average Score (%)', fontsize=10, color='#1e293b')
    ax.set_ylabel('Student Region', fontsize=10, color='#1e293b')
    chart_demographics = get_base64_image(fig)
    
    # Bounding Box / Scope Metrics
    min_score = round(float(df_sample['avg_score_clean'].min()), 2)
    max_score = round(float(df_sample['avg_score_clean'].max()), 2)
    min_credits = round(float(df_sample['studied_credits'].min()), 2)
    max_credits = round(float(df_sample['studied_credits'].max()), 2)
    
    scope_text = f"Empirical Scope: Concentrations reflect historical student academic records within the OULAD dataset observation window. Score Range: {min_score}% to {max_score}% | Studied Credits Range: {min_credits} to {max_credits}."
    
    return {
        "notice": notice,
        "scope_text": scope_text,
        "chart_density": chart_density,
        "chart_demographics": chart_demographics
    }
