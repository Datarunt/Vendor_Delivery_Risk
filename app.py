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
commit_aggregation = "day"

# Fields the user must map in the UI. Vendor Code and Vendor Name on the
# Current Commits file are optional (ticked by the user), and the Owner Matrix
# Supplier # / Supplier Name columns are only needed when the matching one is ticked.
REQUIRED_FIELDS = {
    "hist_mapping": [
        "Material", "Vendor", "Commit Date", "Commit Qty",
        "Actual Delivery Date", "Actual Delivery Qty"
    ],
    "commit_mapping": ["Material", "Commit Date", "Commit Qty"],
    "owner_mapping": ["Assigned Buyer"],
}

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
        owner_matrix_ready=owner_matrix_csv is not None,
        selected_aggregation=commit_aggregation
    )

@app.route("/run_forecast", methods=["POST"])
def run_forecast():
    global forecast_csv, commits_csv
    global owner_matrix_csv
    global new_otd_file_csv
    global forecast_start_date
    global forecast_horizon
    global accuracy_metrics
    global commit_aggregation

    hist_file = request.files.get("hist_file")
    commit_file = request.files.get("commit_file")
    owner_file = request.files.get("owner_file")
    new_otd_upload = request.files.get("new_otd_file")

    # --------------------------
    # CHECK THE UPLOADS
    if not hist_file or not hist_file.filename or not allowed_file(hist_file.filename):
        return "Invalid or missing Historical Commits file", 400

    if not commit_file or not commit_file.filename or not allowed_file(commit_file.filename):
        return "Invalid or missing Current Commits file", 400

    if not owner_file or not owner_file.filename or not allowed_file(owner_file.filename):
        return "Invalid or missing Owner Matrix file", 400

    if new_otd_upload and new_otd_upload.filename and not allowed_file(new_otd_upload.filename):
        return "Invalid New OTD Data file", 400

    # --------------------------
    # READ FIELD MAPPINGS FROM FORM
    # Each mapping is a list of (column in the user's file, field the model needs).
    # Optional fields the user left unticked are not submitted, so they are
    # simply absent here.
    def get_mapping(prefix):
        pairs = []
        for key, value in request.form.items():
            if key.startswith(f"map_{prefix}__") and value:
                required_field = key.replace(f"map_{prefix}__", "")
                pairs.append((value, required_field))
        return pairs

    hist_mapping = get_mapping("hist_mapping")
    otd_mapping = get_mapping("otd_mapping")
    commit_mapping = get_mapping("commit_mapping")
    owner_mapping = get_mapping("owner_mapping")

    # --------------------------
    # MAKE SURE EVERY REQUIRED FIELD WAS MAPPED
    def missing_fields(pairs, required):
        mapped = {field for _, field in pairs}
        return [f for f in required if f not in mapped]

    problems = []

    for label, pairs, required in (
        ("Historical Commits", hist_mapping, REQUIRED_FIELDS["hist_mapping"]),
        ("Current Commits", commit_mapping, REQUIRED_FIELDS["commit_mapping"]),
        ("Owner Matrix", owner_mapping, REQUIRED_FIELDS["owner_mapping"]),
    ):
        missing = missing_fields(pairs, required)
        if missing:
            problems.append(f"{label}: select a column for {', '.join(missing)}")

    commit_fields = {field for _, field in commit_mapping}
    owner_fields = {field for _, field in owner_mapping}

    if "Vendor Code" in commit_fields and "Supplier #" not in owner_fields:
        problems.append(
            "Owner Matrix: select the Vendor Code (Supplier #) column "
            "to match on Vendor Code"
        )

    if "Vendor Name" in commit_fields and "Supplier Name" not in owner_fields:
        problems.append(
            "Owner Matrix: select the Vendor Name (Supplier Name) column "
            "to match on Vendor Name"
        )

    if problems:
        return "<br>".join(problems), 400

    # --------------------------
    # APPLY MAPPINGS TO FILES
    # The remapped file holds ONLY the mapped fields, named the way the
    # rest of the app expects them.
    def remap_file(file, pairs):
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

        remapped = pd.DataFrame({
            field: df[source]
            for source, field in pairs
            if source in df.columns
        })

        buf = BytesIO()
        buf.write(remapped.to_csv(index=False).encode("utf-8"))
        buf.seek(0)
        buf.filename = "remapped.csv"
        buf.name = "remapped.csv"
        return buf

    from werkzeug.datastructures import FileStorage

    def remap_to_filestorage(file, pairs):
        buf = remap_file(file, pairs)
        return FileStorage(
            stream=buf,
            filename="remapped.csv",
            content_type="text/csv"
        )

    hist_file = remap_to_filestorage(hist_file, hist_mapping)
    if new_otd_upload and new_otd_upload.filename:
        new_otd_upload = remap_to_filestorage(new_otd_upload, otd_mapping)
    commit_file = remap_to_filestorage(commit_file, commit_mapping)
    owner_file = remap_to_filestorage(owner_file, owner_mapping)

    forecast_start_date = pd.to_datetime(
        request.form.get("start_date")
    )

    forecast_horizon = int(
        request.form.get("horizon", 14)
    )

    commit_aggregation = request.form.get("aggregation", "day")

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
        accuracy=accuracy_metrics,
        selected_aggregation=commit_aggregation
    )
    

@app.route("/download_forecast.csv")
def download_forecast():
    return send_file(BytesIO(forecast_csv), mimetype="text/csv",
                     as_attachment=True, download_name="forecast.csv")

@app.route("/combine", methods=["POST"])
def combine():
    global forecast_csv, commits_csv, owner_matrix_csv,new_otd_file, commit_aggregation

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
        forecast_horizon=forecast_horizon,
        aggregation=commit_aggregation
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