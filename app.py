from flask import Flask, render_template
from load_data import load_data, get_data_summary
from oulad_eda import generate_eda_charts

app = Flask(__name__)

# Cache to prevent reloading data and re-generating charts on every request
cache = {
    "df_clean": None,
    "summary": None,
    "charts": None
}

@app.route("/")
def index():
    return render_template("index.html", active="none")

@app.route("/data-loading")
def data_loading():
    error = None

    try:
        if cache["df_clean"] is None:
            cache["df_clean"] = load_data()
        if cache["summary"] is None:
            cache["summary"] = get_data_summary(cache["df_clean"])
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=cache["summary"],
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
            # Generate charts only the first time the page is loaded
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
