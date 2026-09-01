import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

from load_data import load_data

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

def set_dark_theme():
    """Applies a dark theme matching the frontend styling."""
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

def prepare_model_data():
    """Loads OULAD dataset and prepares dynamic features and targets for ML training."""
    df = load_data()
    
    # 1. Targets
    # For Regression: Predict avg_score (impute missing with median)
    median_score = df['avg_score'].median()
    y_reg = df['avg_score'].fillna(median_score).values
    
    # For Classification: Predict binary success (Pass/Distinction = 1, Fail/Withdrawn = 0)
    y_clf = df['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0).values
    
    # 2. Features
    # Numeric features
    num_cols = ["studied_credits", "num_of_prev_attempts"]
    X_num = df[num_cols].copy()
    X_num["studied_credits"] = pd.to_numeric(X_num["studied_credits"], errors="coerce").fillna(X_num["studied_credits"].median())
    X_num["num_of_prev_attempts"] = pd.to_numeric(X_num["num_of_prev_attempts"], errors="coerce").fillna(0)
    
    # Categorical features
    cat_cols = ["code_module", "code_presentation", "gender", "region", "highest_education", "age_band", "imd_band", "disability"]
    X_cat = df[cat_cols].copy()
    X_cat = X_cat.fillna("Missing")
    X_cat = X_cat.astype(str)
    
    # Encode categorical features via pandas get_dummies
    X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
    
    # Combine numerical and categorical features
    X_combined = pd.concat([X_num, X_cat_encoded], axis=1)
    feature_names = X_combined.columns.tolist()
    X = X_combined.values
    
    # 3. Train/Test Split (70/30)
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.3, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names

# ============================================================
# REGRESSION AND CLASSIFICATION ROUTINES
# ============================================================

def train_regression_model(model_type='linear', fit_intercept=True, c_val=1.0, solver='lbfgs'):
    """Trains Linear or Logistic Regression model and returns metrics + base64 plots."""
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    set_dark_theme()
    
    response = {}
    
    if model_type == 'linear':
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)
        mae = mean_absolute_error(y_test_reg, preds)
        
        response['metrics'] = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAE': round(mae, 4)
        }
        
        # Chart 1: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(x=y_test_reg, y=preds, alpha=0.3, color='#60a5fa', ax=ax)
        # Perfect prediction line
        min_val = min(y_test_reg.min(), preds.min())
        max_val = max(y_test_reg.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Fit')
        ax.set_xlabel('Actual average score')
        ax.set_ylabel('Predicted average score')
        ax.set_title('Actual vs Predicted Average Assessment Scores', fontsize=12, fontweight='bold')
        ax.legend()
        response['chart_main'] = get_base64_image(fig)
        
        # Chart 2: Residuals Plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        residuals = y_test_reg - preds
        sns.histplot(residuals, kde=True, color='#f43f5e', ax=ax, bins=30)
        ax.axvline(x=0, color='white', linestyle='--')
        ax.set_xlabel('Residual Error')
        ax.set_title('Residuals Error Distribution (Linear Regression)', fontsize=12, fontweight='bold')
        response['chart_secondary'] = get_base64_image(fig)
        
        # Table Coefficients
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_
        })
        coefs['Abs_Coef'] = coefs['Coefficient'].abs()
        coefs = coefs.sort_values(by='Abs_Coef', ascending=False).head(10)
        response['coefficients'] = coefs[['Feature', 'Coefficient']].round(4).to_dict('records')
        response['intercept'] = round(float(model.intercept_), 4)
        
    else:  # Logistic Regression
        model = LogisticRegression(C=c_val, solver=solver, max_iter=1000, random_state=42)
        model.fit(X_train, y_train_clf)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test_clf, preds)
        prec = precision_score(y_test_clf, preds, zero_division=0)
        rec = recall_score(y_test_clf, preds, zero_division=0)
        f1 = f1_score(y_test_clf, preds, zero_division=0)
        
        response['metrics'] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4)
        }
        
        # Chart 1: Confusion Matrix heatmap
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = confusion_matrix(y_test_clf, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Fail/Withdrawn', 'Pass/Distinction'],
                    yticklabels=['Fail/Withdrawn', 'Pass/Distinction'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix (Logistic Regression)', fontsize=12, fontweight='bold')
        response['chart_main'] = get_base64_image(fig)
        
        # Chart 2: ROC Curve
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fpr, tpr, _ = roc_curve(y_test_clf, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='#a78bfa', label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right")
        response['chart_secondary'] = get_base64_image(fig)
        
        # Table Coefficients
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        })
        coefs['Abs_Coef'] = coefs['Coefficient'].abs()
        coefs = coefs.sort_values(by='Abs_Coef', ascending=False).head(10)
        response['coefficients'] = coefs[['Feature', 'Coefficient']].round(4).to_dict('records')
        response['intercept'] = round(float(model.intercept_[0]), 4)
        
    return response

# ============================================================
# REGULARIZATION ROUTINES
# ============================================================

def train_regularized_model(task_type='regression', reg_type='lasso', alpha=1.0, l1_ratio=0.5):
    """Trains Lasso, Ridge, or ElasticNet models for continuous prediction or binary classification."""
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    set_dark_theme()
    
    response = {}
    
    if task_type == 'regression':
        if reg_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=2000, random_state=42)
        elif reg_type == 'ridge':
            model = Ridge(alpha=alpha, random_state=42)
        else:  # elasticnet
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
            
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)
        mae = mean_absolute_error(y_test_reg, preds)
        
        response['metrics'] = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'MAE': round(mae, 4)
        }
        coefficients = model.coef_
        intercept = model.intercept_
        
        # Chart 1: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.scatterplot(x=y_test_reg, y=preds, alpha=0.3, color='#60a5fa', ax=ax)
        min_val = min(y_test_reg.min(), preds.min())
        max_val = max(y_test_reg.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Fit')
        ax.set_xlabel('Actual average score')
        ax.set_ylabel('Predicted average score')
        ax.set_title(f'Actual vs Predicted ({reg_type.upper()} Regression)', fontsize=12, fontweight='bold')
        ax.legend()
        response['chart_main'] = get_base64_image(fig)
        
    else:  # classification (Logistic Regression with regularization)
        # For LogisticRegression in sklearn, C is inverse of regularization strength
        # Convert alpha to C (e.g. C = 1 / alpha)
        c_val = 1.0 / alpha if alpha > 0 else 9999.0
        
        if reg_type == 'lasso':
            model = LogisticRegression(penalty='l1', C=c_val, solver='saga', max_iter=2000, random_state=42)
        elif reg_type == 'ridge':
            model = LogisticRegression(penalty='l2', C=c_val, solver='lbfgs', max_iter=1000, random_state=42)
        else:  # elasticnet
            model = LogisticRegression(penalty='elasticnet', C=c_val, solver='saga', l1_ratio=l1_ratio, max_iter=2000, random_state=42)
            
        model.fit(X_train, y_train_clf)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test_clf, preds)
        prec = precision_score(y_test_clf, preds, zero_division=0)
        rec = recall_score(y_test_clf, preds, zero_division=0)
        f1 = f1_score(y_test_clf, preds, zero_division=0)
        
        response['metrics'] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4)
        }
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Chart 1: Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = confusion_matrix(y_test_clf, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax,
                    xticklabels=['Fail/Withdrawn', 'Pass/Distinction'],
                    yticklabels=['Fail/Withdrawn', 'Pass/Distinction'])
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix (Regularized {reg_type.upper()})', fontsize=12, fontweight='bold')
        response['chart_main'] = get_base64_image(fig)
        
    # Coefficient Plot (Shows shrinkage effects!)
    fig, ax = plt.subplots(figsize=(7, 5))
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': coefficients
    })
    # Remove coefficients that are exactly 0 to show what got dropped (essential for Lasso!)
    coef_df['Abs_Value'] = coef_df['Value'].abs()
    coef_df = coef_df[coef_df['Abs_Value'] > 1e-5].sort_values(by='Abs_Value', ascending=False).head(15)
    
    if len(coef_df) > 0:
        sns.barplot(data=coef_df, y='Feature', x='Value', hue='Feature', palette='coolwarm', ax=ax, legend=False)
        ax.set_title(f'Top Non-Zero Coefficients ({reg_type.title()} Penalty)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Coefficient Impact')
        ax.set_ylabel('Features')
    else:
        ax.text(0.5, 0.5, 'All features shrunk to 0\n(Try lowering alpha)', ha='center', va='center', fontsize=12)
        ax.set_title('Coefficients Plot', fontsize=12, fontweight='bold')
    response['chart_coef'] = get_base64_image(fig)
    
    # Store list of active features vs total features
    total_feats = len(feature_names)
    non_zero_count = int(np.sum(np.abs(coefficients) > 1e-5))
    response['shrunk_stats'] = {
        'total_features': total_feats,
        'active_features': non_zero_count,
        'eliminated_features': total_feats - non_zero_count
    }
    
    # Table details
    top_coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    top_coefs['Abs_Val'] = top_coefs['Coefficient'].abs()
    top_coefs = top_coefs.sort_values(by='Abs_Val', ascending=False).head(10)
    response['coefficients'] = top_coefs[['Feature', 'Coefficient']].round(4).to_dict('records')
    response['intercept'] = round(float(intercept), 4)
    
    return response

# ============================================================
# DECISION TREE ROUTINES
# ============================================================

def train_decision_tree_model(criterion='gini', max_depth=4, min_samples_split=2):
    """Trains Decision Tree Classifier and returns metrics + custom visualizations."""
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    set_dark_theme()
    
    # Handle depth input
    depth = None if max_depth == 0 else int(max_depth)
    
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=depth,
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    model.fit(X_train, y_train_clf)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    response = {}
    
    # Calculate metrics
    acc = accuracy_score(y_test_clf, preds)
    prec = precision_score(y_test_clf, preds, zero_division=0)
    rec = recall_score(y_test_clf, preds, zero_division=0)
    f1 = f1_score(y_test_clf, preds, zero_division=0)
    
    response['metrics'] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4)
    }
    
    # Chart 1: Feature Importances
    fig, ax = plt.subplots(figsize=(6, 4.5))
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)
    
    # Filter out 0 importance items
    imp_df = imp_df[imp_df['Importance'] > 0.0]
    
    if len(imp_df) > 0:
        sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette='viridis', ax=ax, legend=False)
        ax.set_title('Feature Importances Breakdown', fontsize=12, fontweight='bold')
        ax.set_xlabel('Gini Importance Ratio')
    else:
        ax.text(0.5, 0.5, 'No features have importance > 0', ha='center', va='center')
        ax.set_title('Feature Importances', fontsize=12, fontweight='bold')
    response['chart_importance'] = get_base64_image(fig)
    
    # Chart 2: Tree Visualization Plot
    # Since deep trees are unreadable when plotted, restrict plot_tree max_depth dynamically to 3
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_tree(
        model, 
        max_depth=min(3, depth) if depth else 3,
        feature_names=feature_names,
        class_names=['Fail/Withdrawn', 'Pass/Distinction'],
        filled=True,
        rounded=True,
        fontsize=7,
        ax=ax
    )
    # Style tree nodes text colors
    for text in ax.texts:
        text.set_color('black')
        
    ax.set_title(f'Decision Tree Split Logic (Max Depth Shown: 3)', fontsize=14, fontweight='bold', pad=10, color='white')
    response['chart_tree'] = get_base64_image(fig)
    
    # Generate clean textual decision rules (top 4 levels)
    rules_text = generate_tree_rules_text(model, feature_names)
    response['rules_text'] = rules_text
    
    return response

def generate_tree_rules_text(tree_model, feature_names, max_print_depth=3):
    """Recursively walks decision tree and generates nested HTML tree representations."""
    tree = tree_model.tree_
    class_names = ['Fail/Withdrawn', 'Pass/Distinction']
    
    def recurse(node, depth):
        indent = "  " * depth
        # Leaf node check
        if tree.feature[node] == -2: # TREE_UNDEFINED
            val = tree.value[node][0]
            class_idx = np.argmax(val)
            samples = tree.n_node_samples[node]
            pct = round(val[class_idx] / np.sum(val) * 100, 1)
            color = "#fda4af" if class_idx == 0 else "#6ee7b7" # Light red vs light green
            return f"{indent}<span style='color: var(--text-muted);'>↳</span> <strong style='color: {color};'>{class_names[class_idx]}</strong> <span style='font-size: 0.8rem; color: var(--text-muted);'>(n={samples}, conf={pct}%)</span>\n"
            
        if depth >= max_print_depth:
            # Reached printing limit
            val = tree.value[node][0]
            class_idx = np.argmax(val)
            samples = tree.n_node_samples[node]
            return f"{indent}<span style='color: var(--text-muted);'>↳</span> [Deep Splits] predicting <strong style='color: #cbd5e1;'>{class_names[class_idx]}</strong> <span style='font-size: 0.8rem; color: var(--text-muted);'>(n={samples})</span>\n"
            
        name = feature_names[tree.feature[node]]
        threshold = round(tree.threshold[node], 4)
        
        # Left (True: <= threshold) & Right (False: > threshold) branches
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]
        
        rule_str = f"{indent}<strong style='color: #60a5fa;'>IF {name} &le; {threshold}</strong>\n"
        rule_str += recurse(left_child, depth + 1)
        rule_str += f"{indent}<strong style='color: #a78bfa;'>ELSE (IF {name} &gt; {threshold})</strong>\n"
        rule_str += recurse(right_child, depth + 1)
        
        return rule_str

    return recurse(0, 0)

if __name__ == "__main__":
    print("Testing ML Data prep...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    print("Prepared Data shapes:")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("Features count:", len(feature_names))
    
    print("\nTesting linear regression training...")
    res = train_regression_model('linear')
    print("Linear Metrics:", res['metrics'])
    print("Linear Coef count:", len(res['coefficients']))
    
    print("\nTesting decision tree training...")
    res_dt = train_decision_tree_model(max_depth=3)
    print("DT Metrics:", res_dt['metrics'])
    print("DT Rules preview:\n", res_dt['rules_text'][:200])
