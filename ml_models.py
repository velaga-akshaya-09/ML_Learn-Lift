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
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# Try importing specialized gradient boosting frameworks
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from load_data import load_data

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

def prepare_model_data():
    """Loads OULAD dataset and prepares features and targets for ML training."""
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
# LINEAR & LOGISTIC REGRESSION ROUTINES (WITH / WITHOUT REGULARIZATION)
# ============================================================

def train_regression_model(model_type='linear', use_regularization=False, reg_type='lasso', alpha=1.0, fit_intercept=True, c_val=1.0, solver='lbfgs', l1_ratio=0.5):
    """
    Trains Linear or Logistic Regression models.
    Supports explicit dropdown choice for 'with regularization' vs 'without regularization'.
    Returns metrics, actual vs predicted / confusion matrix, residuals / ROC curve, AND coefficient shrinkage plot + statistics.
    """
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    set_light_theme()
    
    response = {}
    
    if model_type == 'linear':
        if not use_regularization:
            model = LinearRegression(fit_intercept=fit_intercept)
            model_title = "Linear Regression (Without Regularization)"
        else:
            if reg_type == 'lasso':
                model = Lasso(alpha=alpha, max_iter=2000, random_state=42)
                model_title = f"Linear Regression with Lasso Regularization (alpha={alpha})"
            elif reg_type == 'ridge':
                model = Ridge(alpha=alpha, random_state=42)
                model_title = f"Linear Regression with Ridge Regularization (alpha={alpha})"
            else:  # elasticnet
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000, random_state=42)
                model_title = f"Linear Regression with ElasticNet Regularization (alpha={alpha})"
                
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)
        mae = mean_absolute_error(y_test_reg, preds)
        
        response['metrics'] = {
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'R2 Score': round(r2, 4),
            'MAE': round(mae, 4)
        }
        
        # Chart 1: Actual vs Predicted
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(x=y_test_reg, y=preds, alpha=0.4, color='#2563eb', ax=ax)
        min_val = min(y_test_reg.min(), preds.min())
        max_val = max(y_test_reg.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Perfect Fit (Y = X)')
        ax.set_xlabel('Actual Average Score', fontsize=10, color='#1e293b')
        ax.set_ylabel('Predicted Average Score', fontsize=10, color='#1e293b')
        ax.set_title(model_title, fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        ax.legend()
        response['chart_main'] = get_base64_image(fig)
        
        # Chart 2: Residuals Plot
        fig, ax = plt.subplots(figsize=(7, 4.5))
        residuals = y_test_reg - preds
        sns.histplot(residuals, kde=True, color='#0284c7', ax=ax, bins=30)
        ax.axvline(x=0, color='#dc2626', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Residual Error', fontsize=10, color='#1e293b')
        ax.set_title(f'Residuals Error Distribution ({model_title})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        response['chart_secondary'] = get_base64_image(fig)
        
        # Coefficients Table
        raw_coefs = model.coef_
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': raw_coefs
        })
        coefs['Abs_Coef'] = coefs['Coefficient'].abs()
        coefs = coefs.sort_values(by='Abs_Coef', ascending=False).head(10)
        response['coefficients'] = coefs[['Feature', 'Coefficient']].round(4).to_dict('records')
        response['intercept'] = round(float(model.intercept_), 4)
        
    else:  # Logistic Regression
        if not use_regularization:
            model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42)
            model_title = "Logistic Regression (Without Regularization)"
        else:
            if reg_type == 'lasso':
                model = LogisticRegression(penalty='l1', C=c_val, solver='saga', max_iter=2000, random_state=42)
                model_title = f"Logistic Regression with L1 (Lasso) Regularization (C={c_val})"
            elif reg_type == 'ridge':
                model = LogisticRegression(penalty='l2', C=c_val, solver=solver, max_iter=1000, random_state=42)
                model_title = f"Logistic Regression with L2 (Ridge) Regularization (C={c_val})"
            else:  # elasticnet
                model = LogisticRegression(penalty='elasticnet', C=c_val, solver='saga', l1_ratio=l1_ratio, max_iter=2000, random_state=42)
                model_title = f"Logistic Regression with ElasticNet Regularization (C={c_val})"
                
        model.fit(X_train, y_train_clf)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        # Calculate classification metrics
        acc = accuracy_score(y_test_clf, preds)
        prec = precision_score(y_test_clf, preds, zero_division=0)
        rec = recall_score(y_test_clf, preds, zero_division=0)
        f1 = f1_score(y_test_clf, preds, zero_division=0)
        
        response['metrics'] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1 Score': round(f1, 4)
        }
        
        # Chart 1: Confusion Matrix heatmap
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = confusion_matrix(y_test_clf, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Fail/Withdrawn', 'Pass/Distinction'],
                    yticklabels=['Fail/Withdrawn', 'Pass/Distinction'])
        ax.set_xlabel('Predicted Label', fontsize=10, color='#1e293b')
        ax.set_ylabel('True Label', fontsize=10, color='#1e293b')
        ax.set_title(f'Confusion Matrix ({model_title})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        response['chart_main'] = get_base64_image(fig)
        
        # Chart 2: ROC Curve
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fpr, tpr, _ = roc_curve(y_test_clf, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='#2563eb', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--')
        ax.set_xlabel('False Positive Rate', fontsize=10, color='#1e293b')
        ax.set_ylabel('True Positive Rate', fontsize=10, color='#1e293b')
        ax.set_title('ROC Curve (Receiver Operating Characteristic)', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        ax.legend(loc="lower right")
        response['chart_secondary'] = get_base64_image(fig)
        
        # Coefficients Table
        raw_coefs = model.coef_[0]
        coefs = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': raw_coefs
        })
        coefs['Abs_Coef'] = coefs['Coefficient'].abs()
        coefs = coefs.sort_values(by='Abs_Coef', ascending=False).head(10)
        response['coefficients'] = coefs[['Feature', 'Coefficient']].round(4).to_dict('records')
        response['intercept'] = round(float(model.intercept_[0]), 4)
        
    # Chart 3: Coefficient Shrinkage Bar Plot (chart_coef) & Shrunk Stats (for Regularization Page)
    fig, ax = plt.subplots(figsize=(7, 5))
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': raw_coefs
    })
    coef_df['Abs_Value'] = coef_df['Value'].abs()
    coef_df = coef_df[coef_df['Abs_Value'] > 1e-5].sort_values(by='Abs_Value', ascending=False).head(15)
    
    if len(coef_df) > 0:
        sns.barplot(data=coef_df, y='Feature', x='Value', hue='Feature', palette='Blues_r', ax=ax, legend=False)
        ax.set_title(f'Top Non-Zero Coefficients ({reg_type.upper()} Penalty)', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        ax.set_xlabel('Coefficient Value / Weight Impact', fontsize=10, color='#1e293b')
        ax.set_ylabel('Feature Name', fontsize=10, color='#1e293b')
    else:
        ax.text(0.5, 0.5, 'All features shrunk to 0\n(Try lowering alpha penalty strength)', ha='center', va='center', fontsize=11, color='#64748b')
        ax.set_title('Coefficients Plot (All Shrunk to 0)', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        
    response['chart_coef'] = get_base64_image(fig)
    
    total_feats = len(feature_names)
    non_zero_count = int(np.sum(np.abs(raw_coefs) > 1e-5))
    response['shrunk_stats'] = {
        'total_features': total_feats,
        'active_features': non_zero_count,
        'eliminated_features': total_feats - non_zero_count
    }
    
    return response

# Backward compatibility alias
def train_regularized_model(task_type='regression', reg_type='lasso', alpha=1.0, l1_ratio=0.5):
    if task_type == 'regression':
        return train_regression_model(model_type='linear', use_regularization=True, reg_type=reg_type, alpha=alpha, l1_ratio=l1_ratio)
    else:
        c_val = 1.0 / alpha if alpha > 0 else 9999.0
        return train_regression_model(model_type='logistic', use_regularization=True, reg_type=reg_type, c_val=c_val, l1_ratio=l1_ratio)

# ============================================================
# TREE BASED ALGORITHMS ROUTINES (6 ALGORITHMS TOTAL)
# ============================================================

def train_tree_based_model(
    algorithm='decision_tree',
    criterion='gini',
    max_depth=4,
    min_samples_split=2,
    n_estimators=100,
    learning_rate=0.1
):
    """
    Trains any of the 6 Tree-Based Algorithms specified in handwritten notes:
    1. Decision Tree
    2. Random Forest
    3. AdaBoost
    4. Gradient Boosting
    5. XGBoost
    6. LightBoost (LightGBM)
    """
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf, feature_names = prepare_model_data()
    set_light_theme()
    
    depth = None if max_depth == 0 else int(max_depth)
    min_split = int(min_samples_split)
    n_est = int(n_estimators)
    lr = float(learning_rate)
    
    algorithm = algorithm.lower()
    algo_name = "Decision Tree"
    
    clean_feature_names = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in feature_names]
    
    if algorithm == 'decision_tree':
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )
        algo_name = "1. Decision Tree"
        
    elif algorithm == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=n_est,
            criterion=criterion,
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )
        algo_name = "2. Random Forest"
        
    elif algorithm == 'adaboost':
        model = AdaBoostClassifier(
            n_estimators=n_est,
            learning_rate=lr,
            random_state=42
        )
        algo_name = "3. AdaBoost"
        
    elif algorithm == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=n_est,
            max_depth=depth or 3,
            learning_rate=lr,
            random_state=42
        )
        algo_name = "4. Gradient Boosting"
        
    elif algorithm == 'xgboost':
        algo_name = "5. XGBoost"
        if HAS_XGBOOST:
            model = xgb.XGBClassifier(
                n_estimators=n_est,
                max_depth=depth or 3,
                learning_rate=lr,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42
            )
            algo_name += " (HistGradientBoosting Fallback)"
            
    elif algorithm in ['lightboost', 'lightgbm']:
        algo_name = "6. LightBoost (LightGBM)"
        if HAS_LIGHTGBM:
            model = lgb.LGBMClassifier(
                n_estimators=n_est,
                max_depth=depth or -1,
                learning_rate=lr,
                random_state=42,
                verbosity=-1
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42
            )
            algo_name += " (HistGradientBoosting Fallback)"
            
    else:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_split=min_split, random_state=42)
        algo_name = "1. Decision Tree"
        
    model.fit(X_train, y_train_clf)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    
    response = {}
    
    acc = accuracy_score(y_test_clf, preds)
    prec = precision_score(y_test_clf, preds, zero_division=0)
    rec = recall_score(y_test_clf, preds, zero_division=0)
    f1 = f1_score(y_test_clf, preds, zero_division=0)
    
    response['algorithm'] = algo_name
    response['metrics'] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4)
    }
    
    # Chart 1: Feature Importances
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({
            'Feature': clean_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)
        
        imp_df = imp_df[imp_df['Importance'] > 0.0]
        
        if len(imp_df) > 0:
            sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette='Blues_r', ax=ax, legend=False)
            ax.set_title(f'Feature Importances ({algo_name})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
            ax.set_xlabel('Importance Score', fontsize=10, color='#1e293b')
        else:
            ax.text(0.5, 0.5, 'No features have importance > 0', ha='center', va='center')
            ax.set_title(f'Feature Importances ({algo_name})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    else:
        ax.text(0.5, 0.5, f'Feature importances not directly computed for {algo_name}', ha='center', va='center')
        ax.set_title(f'{algo_name} Feature Analysis', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
    response['chart_importance'] = get_base64_image(fig)
    
    # Chart 2: Tree Visualizer / Confusion Matrix
    if algorithm == 'decision_tree':
        fig, ax = plt.subplots(figsize=(12, 6.5))
        plot_tree(
            model, 
            max_depth=min(3, depth) if depth else 3,
            feature_names=clean_feature_names,
            class_names=['Fail/Withdrawn', 'Pass/Distinction'],
            filled=True,
            rounded=True,
            fontsize=7,
            ax=ax
        )
        for text in ax.texts:
            text.set_color('#0f172a')
        ax.set_title('Decision Tree Split Logic (Max Depth Shown: 3)', fontsize=12, fontweight='bold', pad=12, color='#0f172a')
        response['chart_tree'] = get_base64_image(fig)
        response['rules_text'] = generate_tree_rules_text(model, clean_feature_names)
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        cm = confusion_matrix(y_test_clf, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Fail/Withdrawn', 'Pass/Distinction'],
                    yticklabels=['Fail/Withdrawn', 'Pass/Distinction'])
        ax.set_xlabel('Predicted Label', fontsize=10, color='#1e293b')
        ax.set_ylabel('True Label', fontsize=10, color='#1e293b')
        ax.set_title(f'Confusion Matrix ({algo_name})', fontsize=11, fontweight='bold', color='#0f172a', pad=12)
        response['chart_tree'] = get_base64_image(fig)
        response['rules_text'] = f"<div style='padding: 10px; color: #1e293b;'><strong>{algo_name} Model Training Complete!</strong><br>Ensemble contains <strong>{n_est if algorithm != 'decision_tree' else 1}</strong> estimators split across feature subsets.<br>Evaluated on test set of <strong>{len(y_test_clf):,}</strong> students.</div>"
        
    return response

def train_decision_tree_model(criterion='gini', max_depth=4, min_samples_split=2):
    return train_tree_based_model('decision_tree', criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)

def generate_tree_rules_text(tree_model, feature_names, max_print_depth=3):
    """Recursively walks decision tree and generates nested HTML tree representations."""
    tree = tree_model.tree_
    class_names = ['Fail/Withdrawn', 'Pass/Distinction']
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] == -2:
            val = tree.value[node][0]
            class_idx = np.argmax(val)
            samples = tree.n_node_samples[node]
            pct = round(val[class_idx] / np.sum(val) * 100, 1)
            color = "#dc2626" if class_idx == 0 else "#059669"
            return f"{indent}<span style='color: #64748b;'>↳</span> <strong style='color: {color};'>{class_names[class_idx]}</strong> <span style='font-size: 0.8rem; color: #64748b;'>(n={samples}, conf={pct}%)</span>\n"
            
        if depth >= max_print_depth:
            val = tree.value[node][0]
            class_idx = np.argmax(val)
            samples = tree.n_node_samples[node]
            return f"{indent}<span style='color: #64748b;'>↳</span> [Deep Splits] predicting <strong style='color: #1e293b;'>{class_names[class_idx]}</strong> <span style='font-size: 0.8rem; color: #64748b;'>(n={samples})</span>\n"
            
        name = feature_names[tree.feature[node]]
        threshold = round(tree.threshold[node], 4)
        
        left_child = tree.children_left[node]
        right_child = tree.children_right[node]
        
        rule_str = f"{indent}<strong style='color: #1d4ed8;'>IF {name} &le; {threshold}</strong>\n"
        rule_str += recurse(left_child, depth + 1)
        rule_str += f"{indent}<strong style='color: #6366f1;'>ELSE (IF {name} &gt; {threshold})</strong>\n"
        rule_str += recurse(right_child, depth + 1)
        
        return rule_str

    return recurse(0, 0)
