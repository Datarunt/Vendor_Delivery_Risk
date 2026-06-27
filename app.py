import pandas as pd
import numpy as np
from scipy.stats import norm

from flask import Flask, render_template, request, send_file
from io import BytesIO
import webbrowser
from threading import Timer
from xgboost import XGBRegressor
import warnings
from model import xgboost_forecast_with_uncertainty
from data_processing import load_historical_file
from forecast_accuracy_services import calculate_forecast_accuracy
from risk_engine import apply_risk_model
from forecast_build_services import build_forecast
from combine_services import build_combined_output
from upload_services import process_forecast_uploads

warnings.filterwarnings('ignore')

app = Flask(__name__)

# --------------------------
# GLOBALS
forecast_csv = None
commits_csv = None
owner_matrix_csv = None
new_otd_file = None  
forecast_start_date = None
forecast_horizon = 14
accuracy_metrics = None

# --------------------------
# HELPERS
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ["xlsx", "xls", "csv"]

# --------------------------
# COLUMN DETECTION
@app.route("/get_columns", methods=["POST"])
def get_columns():
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        return {"columns": []}, 400
    
    try:
        if file.filename.lower().endswith(".csv"):
            for enc in ("utf-8", "cp1252", "latin1"):
                try:
                    df = pd.read_csv(file, encoding=enc, nrows=0)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            df = pd.read_excel(file, nrows=0)
        
        columns = df.columns.str.strip().tolist()
        return {"columns": columns}
    
    except Exception as e:
        return {"error": str(e)}, 500


# --------------------------
# ROUTES
@app.route("/")
def index():
    return render_template(
        "index.html",
        forecast_ready=forecast_csv is not None,
        commits_csv_ready=commits_csv is not None,
        owner_matrix_ready=owner_matrix_csv is not None
        
    )

@app.route("/run_forecast", methods=["POST"])
def run_forecast():
    global forecast_csv, commits_csv
    global owner_matrix_csv
    global new_otd_file_csv
    global forecast_start_date
    global forecast_horizon
    global accuracy_metrics

    hist_file = request.files.get("hist_file")
    commit_file = request.files.get("commit_file")
    owner_file = request.files.get("owner_file")
    new_otd_upload = request.files.get("new_otd_file")

    # --------------------------
    # READ FIELD MAPPINGS FROM FORM
    def get_mapping(prefix):
        mapping = {}
        for key, value in request.form.items():
            if key.startswith(f"map_{prefix}__") and value:
                required_field = key.replace(f"map_{prefix}__", "")
                mapping[value] = required_field
        return mapping

    hist_mapping = get_mapping("hist_mapping")
    otd_mapping = get_mapping("otd_mapping")
    commit_mapping = get_mapping("commit_mapping")
    owner_mapping = get_mapping("owner_mapping")

    # --------------------------
    # APPLY MAPPINGS TO FILES
    def remap_file(file, mapping):
        if file.filename.lower().endswith(".csv"):
            for enc in ("utf-8", "cp1252", "latin1"):
                try:
                    df = pd.read_csv(file, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
        else:
            df = pd.read_excel(file)
        df.columns = df.columns.str.strip()
        df = df.rename(columns=mapping)
        buf = BytesIO()
        buf.write(df.to_csv(index=False).encode("utf-8"))
        buf.seek(0)
        buf.filename = "remapped.csv"
        buf.name = "remapped.csv"
        return buf

    from werkzeug.datastructures import FileStorage

    def remap_to_filestorage(file, mapping):
        buf = remap_file(file, mapping)
        return FileStorage(
            stream=buf,
            filename="remapped.csv",
            content_type="text/csv"
        )

    hist_file = remap_to_filestorage(hist_file, hist_mapping)
    new_otd_upload = remap_to_filestorage(new_otd_upload, otd_mapping)
    commit_file = remap_to_filestorage(commit_file, commit_mapping)
    owner_file = remap_to_filestorage(owner_file, owner_mapping)

    forecast_start_date = pd.to_datetime(
        request.form.get("start_date")
    )

    forecast_horizon = int(
        request.form.get("horizon", 14)
    )

    if not hist_file or not allowed_file(hist_file.filename):
        return "Invalid historical file", 400

    if not commit_file or not allowed_file(commit_file.filename):
        return "Invalid commit file", 400

    if not owner_file or not allowed_file(owner_file.filename):
        return "Invalid owner matrix file", 400
    
    if not new_otd_upload or not allowed_file(new_otd_upload.filename):
        return "Invalid New OTD Data file", 400


    (
        forecast_csv,
        commits_csv,
        owner_matrix_csv,
        accuracy_metrics
    ) = process_forecast_uploads(
        hist_file,
        commit_file,
        owner_file,
        new_otd_upload,
        forecast_start_date,
        forecast_horizon
    )

    return render_template(
        "index.html",
        forecast_ready=True,
        commits_csv_ready=True,
        owner_matrix_ready=True,
        accuracy=accuracy_metrics
    )
    

@app.route("/download_forecast.csv")
def download_forecast():
    return send_file(BytesIO(forecast_csv), mimetype="text/csv",
                     as_attachment=True, download_name="forecast.csv")

@app.route("/combine", methods=["POST"])
def combine():
    global forecast_csv, commits_csv, owner_matrix_csv,new_otd_file

    forecast_df = pd.read_csv(BytesIO(forecast_csv))
    commits_df = pd.read_csv(BytesIO(commits_csv))
    owner_df = pd.read_csv(BytesIO(owner_matrix_csv))
    #new_otd_df = pd.read_csv(BytesIO(new_otd_file_csv))

    final = build_combined_output(
        forecast_df=forecast_df,
        commits_df=commits_df,
        owner_df=owner_df,
        #new_otd_df = new_otd_df,
        forecast_start_date=forecast_start_date,
        forecast_horizon=forecast_horizon
    )

    # ---------------------------
    # OUTPUT EXCEL WITH ACCURACY TAB
    out_excel = BytesIO()

    with pd.ExcelWriter(out_excel, engine='xlsxwriter') as writer:
        # Main output
        final.to_excel(writer, index=False, sheet_name="Risk Output")

        # Accuracy tab
        if accuracy_metrics is not None:

            if isinstance(accuracy_metrics, dict):
                accuracy_df = pd.DataFrame([accuracy_metrics])
            else:
                accuracy_df = accuracy_metrics

            accuracy_df.to_excel(
                writer,
                index=False,
                sheet_name="Accuracy"
            )

    out_excel.seek(0)

    return send_file(
        out_excel,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="vendor_commit_risk.xlsx"
    )

# --------------------------
# RUN
if __name__ == "__main__":
    port = 5000
    Timer(1, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)