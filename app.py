from flask import Flask, render_template, redirect, url_for, request, jsonify
from load_data import load_raw_data, load_data, get_data_summary
from oulad_eda import generate_eda_charts
from preprocessing import run_preprocessing
from ml_models import train_regression_model, train_regularized_model, train_tree_based_model

app = Flask(__name__)

# Cache to prevent reloading data and re-generating charts on every request
cache = {
    "df_raw": None,
    "summary_raw": None,
    "df_clean": None,
    "summary_clean": None,
    "charts": None,
    "preprocessing": None
}

@app.route("/")
def index():
    return redirect(url_for("data_loading"))

@app.route("/data-loading")
def data_loading():
    error = None

    try:
        if cache["df_raw"] is None:
            cache["df_raw"] = load_raw_data()
        if cache["summary_raw"] is None:
            cache["summary_raw"] = get_data_summary(cache["df_raw"])

        if cache["df_clean"] is None:
            cache["df_clean"] = load_data()
        if cache["summary_clean"] is None:
            cache["summary_clean"] = get_data_summary(cache["df_clean"])
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary_raw=cache["summary_raw"],
        summary_clean=cache["summary_clean"],
        error=error
    )

@app.route("/eda")
def eda():
    error = None

    try:
        if cache["df_clean"] is None:
            cache["df_clean"] = load_data()
            
        if cache["df_clean"].empty:
            error = "Data not found or is empty."
        elif cache["charts"] is None:
            df_sample = cache["df_clean"].sample(n=min(2000, len(cache["df_clean"])), random_state=42)
            cache["charts"] = generate_eda_charts(cache["df_clean"], df_sample)
            
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="eda",
        charts=cache["charts"],
        error=error
    )

@app.route("/preprocessing")
def preprocessing():
    error = None

    try:
        if cache["preprocessing"] is None:
            cache["preprocessing"] = run_preprocessing()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="preprocessing",
        preprocessing=cache["preprocessing"],
        error=error
    )

@app.route("/regression")
def regression():
    return render_template(
        "index.html",
        active="regression",
        error=None
    )

@app.route("/regularization")
def regularization():
    return render_template(
        "index.html",
        active="regularization",
        error=None
    )

@app.route("/tree-based")

@app.route("/decision-tree")
def tree_based():
    return render_template(
        "index.html",
        active="tree-based",
        error=None
    )

@app.route("/api/train/regression", methods=["POST"])
def api_train_regression():
    try:
        data = request.json or {}
        model_type = data.get("model_type", "linear")
        use_regularization = bool(data.get("use_regularization", False))
        reg_type = data.get("reg_type", "lasso")
        alpha = float(data.get("alpha", 1.0))
        c_val = float(data.get("c_val", 1.0))
        l1_ratio = float(data.get("l1_ratio", 0.5))
        fit_intercept = bool(data.get("fit_intercept", True))
        solver = data.get("solver", "lbfgs")
        
        result = train_regression_model(
            model_type=model_type,
            use_regularization=use_regularization,
            reg_type=reg_type,
            alpha=alpha,
            c_val=c_val,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            solver=solver
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/train/regularization", methods=["POST"])
def api_train_regularization():
    try:
        data = request.json or {}
        task_type = data.get("task_type", "regression")
        reg_type = data.get("reg_type", "lasso")
        alpha = float(data.get("alpha", 1.0))
        l1_ratio = float(data.get("l1_ratio", 0.5))
        
        result = train_regularized_model(
            task_type=task_type,
            reg_type=reg_type,
            alpha=alpha,
            l1_ratio=l1_ratio
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/train/tree-based", methods=["POST"])
@app.route("/api/train/decision-tree", methods=["POST"])
def api_train_tree_based():
    try:
        data = request.json or {}
        algorithm = data.get("algorithm", "decision_tree")
        criterion = data.get("criterion", "gini")
        max_depth = int(data.get("max_depth", 4))
        min_samples_split = int(data.get("min_samples_split", 2))
        n_estimators = int(data.get("n_estimators", 100))
        learning_rate = float(data.get("learning_rate", 0.1))
        
        result = train_tree_based_model(
            algorithm=algorithm,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
