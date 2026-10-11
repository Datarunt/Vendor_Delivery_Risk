import pandas as pd
import numpy as np
from scipy.stats import norm
 
from flask import Flask, render_template, request, send_file
from io import BytesIO
import webbrowser
from threading import Timer
 
from xgboost import XGBRegressor
import warnings
 
from werkzeug.datastructures import FileStorage
 
from model import xgboost_forecast_with_uncertainty
from data_processing import load_historical_file
from forecast_accuracy_services import calculate_forecast_accuracy
from risk_engine import apply_risk_model
from forecast_build_services import build_forecast
from combine_services import build_combined_output
from upload_services import process_forecast_uploads
 
warnings.filterwarnings("ignore")
 
app = Flask(__name__)
 
# ============================================================
# GLOBALS
# ============================================================
 
forecast_csv = None
commits_csv = None
owner_matrix_csv = None
new_otd_file = None
 
forecast_start_date = None
forecast_horizon = 14
accuracy_metrics = None
commit_aggregation = "day"
 
# ============================================================
# REQUIRED FIELD MAPPINGS
# ============================================================
 
REQUIRED_FIELDS = {
    "hist_mapping": [
        "Material",
        "Vendor",
        "Commit Date",
        "Commit Qty",
        "Actual Delivery Date",
        "Actual Delivery Qty"
    ],
 
    "commit_mapping": [
        "Material",
        "Commit Date",
        "Commit Qty"
    ],
 
    "owner_mapping": [
        "Assigned Buyer"
    ],
}
 
# ============================================================
# HELPER FUNCTIONS
# ============================================================
 
def allowed_file(filename):
    """
    Check whether the uploaded file has an allowed extension.
    """
    if not filename:
        return False
 
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ["xlsx", "xls", "csv"]
    )
 
def clean_column_names(df):
    """
    Standardise dataframe column names.
 
    - Convert column names to strings
    - Strip leading/trailing spaces
    - Detect duplicate column names
    """
 
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )
 
    # Detect duplicate columns
    duplicated = df.columns[df.columns.duplicated()].tolist()
 
    if duplicated:
        raise ValueError(
            f"Duplicate column names found: {duplicated}. "
            f"Please rename the duplicate columns in the source file."
        )
 
    return df
 
def read_uploaded_file(file):
    """
    Read an uploaded CSV/XLS/XLSX file into a pandas DataFrame.
    """
 
    if file is None:
        raise ValueError("No file supplied.")
 
    # Reset stream position
    try:
        file.stream.seek(0)
    except Exception:
        pass
 
    filename = file.filename.lower()
 
    if filename.endswith(".csv"):
 
        last_error = None
 
        for enc in ("utf-8", "cp1252", "latin1"):
            try:
                file.stream.seek(0)
 
                df = pd.read_csv(
                    file.stream,
                    encoding=enc
                )
 
                return clean_column_names(df)
 
            except UnicodeDecodeError as e:
                last_error = e
                continue
 
        raise ValueError(
            f"Unable to decode CSV file '{file.filename}'. "
            f"Last error: {last_error}"
        )
 
    elif filename.endswith((".xlsx", ".xls")):
 
        file.stream.seek(0)
 
        df = pd.read_excel(
            file.stream
        )
 
        return clean_column_names(df)
 
    else:
        raise ValueError(
            f"Unsupported file type: {file.filename}"
        )
 
def get_mapping(prefix):
    """
    Read mapping fields submitted by the HTML form.
 
    Expected HTML names look like:
 
        map_hist_mapping__Material
        map_hist_mapping__Vendor
 
    Returns:
 
        [
            ("Material Number", "Material"),
            ("Supplier", "Vendor"),
            ...
        ]
    """
 
    pairs = []
 
    for key, value in request.form.items():
 
        if key.startswith(f"map_{prefix}__") and value:
 
            required_field = key.replace(
                f"map_{prefix}__",
                "",
                1
            ).strip()
 
            source_column = str(value).strip()
 
            if source_column:
 
                pairs.append(
                    (
                        source_column,
                        required_field
                    )
                )
 
    return pairs
 
def missing_fields(pairs, required):
    """
    Determine which required destination fields
    have not been mapped.
    """
 
    mapped = {
        field.strip()
        for _, field in pairs
    }
 
    return [
        field
        for field in required
        if field not in mapped
    ]
 
def validate_mapping_sources(df, pairs, filename):
    """
    Ensure every source column selected by the user
    actually exists in the uploaded dataframe.
    """
 
    available_columns = set(df.columns)
 
    missing = []
 
    for source, field in pairs:
 
        source = source.strip()
 
        if source not in available_columns:
 
            missing.append(
                f"'{source}' -> '{field}'"
            )
 
    if missing:
 
        raise ValueError(
            f"Mapping error in '{filename}'. "
            f"The following selected source columns were not found: "
            f"{', '.join(missing)}. "
            f"Available columns are: "
            f"{df.columns.tolist()}"
        )
 
def validate_destination_fields(df, required_fields, filename):
    """
    Validate that the remapped dataframe contains
    the fields expected by the application.
    """
 
    missing = [
        field
        for field in required_fields
        if field not in df.columns
    ]
 
    if missing:
 
        raise ValueError(
            f"After remapping '{filename}', the following required "
            f"fields are missing: {missing}. "
            f"Available columns are: {df.columns.tolist()}"
        )
 
def remap_file(file, pairs, required_fields=None):
    """
    Read the source file and create a new CSV containing
    only the mapped fields.
 
    IMPORTANT:
    Previously this code used:
 
        if source in df.columns
 
    which silently removed missing columns.
 
    This version explicitly validates every mapping instead.
    """
 
    df = read_uploaded_file(file)
 
    # --------------------------------------------------------
    # Validate selected source columns
    # --------------------------------------------------------
 
    validate_mapping_sources(
        df,
        pairs,
        file.filename
    )
 
    # --------------------------------------------------------
    # Build remapped dataframe
    # --------------------------------------------------------
 
    remapped = pd.DataFrame()
 
    for source, field in pairs:
 
        source = source.strip()
        field = field.strip()
 
        remapped[field] = df[source]
 
    # --------------------------------------------------------
    # Validate output fields
    # --------------------------------------------------------
 
    if required_fields:
 
        validate_destination_fields(
            remapped,
            required_fields,
            file.filename
        )
 
    # --------------------------------------------------------
    # Special validation for Material
    # --------------------------------------------------------
 
    if "Material" in remapped.columns:
 
        if remapped["Material"].isna().all():
 
            raise ValueError(
                f"The 'Material' column was mapped successfully "
                f"in '{file.filename}', but every Material value is blank."
            )
 
    # --------------------------------------------------------
    # Create CSV in memory
    # --------------------------------------------------------
 
    buf = BytesIO()
 
    remapped.to_csv(
        buf,
        index=False
    )
 
    buf.seek(0)
 
    buf.filename = "remapped.csv"
    buf.name = "remapped.csv"
 
    return buf
 
def remap_to_filestorage(
    file,
    pairs,
    required_fields=None
):
    """
    Convert the remapped BytesIO object back into
    a FileStorage object so existing services can consume it.
    """
 
    buf = remap_file(
        file,
        pairs,
        required_fields=required_fields
    )
 
    return FileStorage(
        stream=buf,
        filename="remapped.csv",
        content_type="text/csv"
    )
 
def validate_remapped_file(
    file,
    required_columns,
    name
):
    """
    Final validation immediately before the file
    is passed to the processing service.
    """
 
    try:
        file.stream.seek(0)
 
        df = pd.read_csv(
            file.stream
        )
 
    except Exception as e:
 
        raise ValueError(
            f"Could not read remapped {name} file: {e}"
        )
 
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )
 
    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]
 
    if missing:
 
        raise ValueError(
            f"{name} file is missing required columns: "
            f"{missing}. "
            f"Actual columns are: "
            f"{df.columns.tolist()}"
        )
 
    # Explicit Material validation
    if "Material" in required_columns:
 
        if "Material" not in df.columns:
 
            raise ValueError(
                f"{name} file does not contain a 'Material' column "
                f"after remapping. Actual columns: "
                f"{df.columns.tolist()}"
            )
 
        if df["Material"].isna().all():
 
            raise ValueError(
                f"{name} file contains a 'Material' column, "
                f"but all Material values are blank."
            )
 
    # Reset stream for downstream processing
    file.stream.seek(0)
 
    return df
 
# ============================================================
# COLUMN DETECTION
# ============================================================
 
@app.route("/get_columns", methods=["POST"])
def get_columns():
 
    file = request.files.get("file")
 
    if not file or not allowed_file(file.filename):
 
        return {
            "columns": []
        }, 400
 
    try:
 
        df = read_uploaded_file(file)
 
        return {
            "columns": df.columns.tolist()
        }
 
    except Exception as e:
 
        return {
            "error": str(e),
            "columns": []
        }, 500
 
# ============================================================
# MAIN PAGE
# ============================================================
 
@app.route("/")
def index():
 
    return render_template(
        "index.html",
 
        forecast_ready=(
            forecast_csv is not None
        ),
 
        commits_csv_ready=(
            commits_csv is not None
        ),
 
        owner_matrix_ready=(
            owner_matrix_csv is not None
        ),
 
        selected_aggregation=commit_aggregation
    )
 
# ============================================================
# RUN FORECAST
# ============================================================
 
@app.route("/run_forecast", methods=["POST"])
def run_forecast():
 
    global forecast_csv
    global commits_csv
    global owner_matrix_csv
    global new_otd_file
    global forecast_start_date
    global forecast_horizon
    global accuracy_metrics
    global commit_aggregation
 
    # --------------------------------------------------------
    # Get uploaded files
    # --------------------------------------------------------
 
    hist_file = request.files.get(
        "hist_file"
    )
 
    commit_file = request.files.get(
        "commit_file"
    )
 
    owner_file = request.files.get(
        "owner_file"
    )
 
    new_otd_upload = request.files.get(
        "new_otd_file"
    )
 
    # --------------------------------------------------------
    # Validate uploads
    # --------------------------------------------------------
 
    if (
        not hist_file
        or not hist_file.filename
        or not allowed_file(hist_file.filename)
    ):
 
        return (
            "Invalid or missing Historical Commits file",
            400
        )
 
    if (
        not commit_file
        or not commit_file.filename
        or not allowed_file(commit_file.filename)
    ):
 
        return (
            "Invalid or missing Current Commits file",
            400
        )
 
    if (
        not owner_file
        or not owner_file.filename
        or not allowed_file(owner_file.filename)
    ):
 
        return (
            "Invalid or missing Owner Matrix file",
            400
        )
 
    if (
        new_otd_upload
        and new_otd_upload.filename
        and not allowed_file(new_otd_upload.filename)
    ):
 
        return (
            "Invalid New OTD Data file",
            400
        )
 
    # --------------------------------------------------------
    # Read mappings
    # --------------------------------------------------------
 
    hist_mapping = get_mapping(
        "hist_mapping"
    )
 
    otd_mapping = get_mapping(
        "otd_mapping"
    )
 
    commit_mapping = get_mapping(
        "commit_mapping"
    )
 
    owner_mapping = get_mapping(
        "owner_mapping"
    )
 
    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------
 
    print("\n========================================")
    print("FIELD MAPPINGS")
    print("========================================")
 
    print("Historical:", hist_mapping)
    print("OTD:", otd_mapping)
    print("Current Commits:", commit_mapping)
    print("Owner Matrix:", owner_mapping)
 
    # --------------------------------------------------------
    # Validate required mappings
    # --------------------------------------------------------
 
    problems = []
 
    for label, pairs, required in (
 
        (
            "Historical Commits",
            hist_mapping,
            REQUIRED_FIELDS["hist_mapping"]
        ),
 
        (
            "Current Commits",
            commit_mapping,
            REQUIRED_FIELDS["commit_mapping"]
        ),
 
        (
            "Owner Matrix",
            owner_mapping,
            REQUIRED_FIELDS["owner_mapping"]
        ),
 
    ):
 
        missing = missing_fields(
            pairs,
            required
        )
 
        if missing:
 
            problems.append(
                f"{label}: select a column for "
                f"{', '.join(missing)}"
            )
 
    # --------------------------------------------------------
    # Validate optional vendor mappings
    # --------------------------------------------------------
 
    commit_fields = {
        field.strip()
        for _, field in commit_mapping
    }
 
    owner_fields = {
        field.strip()
        for _, field in owner_mapping
    }
 
    if (
        "Vendor Code" in commit_fields
        and "Supplier #" not in owner_fields
    ):
 
        problems.append(
            "Owner Matrix: select the Vendor Code "
            "(Supplier #) column to match on Vendor Code"
        )
 
    if (
        "Vendor Name" in commit_fields
        and "Supplier Name" not in owner_fields
    ):
 
        problems.append(
            "Owner Matrix: select the Vendor Name "
            "(Supplier Name) column to match on Vendor Name"
        )
 
    if problems:
 
        return (
            "<br>".join(problems),
            400
        )
 
    # --------------------------------------------------------
    # Validate actual source files BEFORE remapping
    # --------------------------------------------------------
 
    try:
 
        hist_original_df = read_uploaded_file(
            hist_file
        )
 
        commit_original_df = read_uploaded_file(
            commit_file
        )
 
        owner_original_df = read_uploaded_file(
            owner_file
        )
 
        print("\n========================================")
        print("SOURCE FILE COLUMNS")
        print("========================================")
 
        print(
            "Historical:",
            hist_original_df.columns.tolist()
        )
 
        print(
            "Current Commits:",
            commit_original_df.columns.tolist()
        )
 
        print(
            "Owner Matrix:",
            owner_original_df.columns.tolist()
        )
 
        # Validate mappings against original files
        validate_mapping_sources(
            hist_original_df,
            hist_mapping,
            hist_file.filename
        )
 
        validate_mapping_sources(
            commit_original_df,
            commit_mapping,
            commit_file.filename
        )
 
        validate_mapping_sources(
            owner_original_df,
            owner_mapping,
            owner_file.filename
        )
 
        # Optional OTD file
        if (
            new_otd_upload
            and new_otd_upload.filename
        ):
 
            otd_original_df = read_uploaded_file(
                new_otd_upload
            )
 
            validate_mapping_sources(
                otd_original_df,
                otd_mapping,
                new_otd_upload.filename
            )
 
    except Exception as e:
 
        return (
            f"File/mapping validation error:<br><br>{str(e)}",
            400
        )
 
    # --------------------------------------------------------
    # Apply mappings
    # --------------------------------------------------------
 
    try:
 
        hist_file = remap_to_filestorage(
            hist_file,
            hist_mapping,
            required_fields=REQUIRED_FIELDS[
                "hist_mapping"
            ]
        )
 
        commit_file = remap_to_filestorage(
            commit_file,
            commit_mapping,
            required_fields=REQUIRED_FIELDS[
                "commit_mapping"
            ]
        )
 
        owner_file = remap_to_filestorage(
            owner_file,
            owner_mapping,
            required_fields=REQUIRED_FIELDS[
                "owner_mapping"
            ]
        )
 
        if (
            new_otd_upload
            and new_otd_upload.filename
        ):
 
            new_otd_upload = remap_to_filestorage(
                new_otd_upload,
                otd_mapping
            )
 
    except Exception as e:
 
        return (
            f"Column remapping error:<br><br>{str(e)}",
            400
        )
 
    # --------------------------------------------------------
    # Validate remapped files
    # --------------------------------------------------------
 
    try:
 
        validate_remapped_file(
            hist_file,
            REQUIRED_FIELDS["hist_mapping"],
            "Historical Commits"
        )
 
        validate_remapped_file(
            commit_file,
            REQUIRED_FIELDS["commit_mapping"],
            "Current Commits"
        )
 
        validate_remapped_file(
            owner_file,
            REQUIRED_FIELDS["owner_mapping"],
            "Owner Matrix"
        )
 
        if (
            new_otd_upload
            and new_otd_upload.filename
        ):
 
            # No fixed required OTD fields are defined
            # in the original application, so only make
            # sure the file can be read.
            new_otd_upload.stream.seek(0)
 
            pd.read_csv(
                new_otd_upload.stream
            )
 
            new_otd_upload.stream.seek(0)
 
    except Exception as e:
 
        return (
            f"Remapped file validation error:<br><br>{str(e)}",
            400
        )
 
    # --------------------------------------------------------
    # Debug remapped files
    # --------------------------------------------------------
 
    print("\n========================================")
    print("REMAPPED FILE COLUMNS")
    print("========================================")
 
    try:
 
        hist_file.stream.seek(0)
        hist_debug = pd.read_csv(
            hist_file.stream
        )
 
        commit_file.stream.seek(0)
        commit_debug = pd.read_csv(
            commit_file.stream
        )
 
        owner_file.stream.seek(0)
        owner_debug = pd.read_csv(
            owner_file.stream
        )
 
        print(
            "Historical:",
            hist_debug.columns.tolist()
        )
 
        print(
            "Current Commits:",
            commit_debug.columns.tolist()
        )
 
        print(
            "Owner Matrix:",
            owner_debug.columns.tolist()
        )
 
        print(
            "Historical Material:",
            "Material" in hist_debug.columns
        )
 
        print(
            "Current Commits Material:",
            "Material" in commit_debug.columns
        )
 
        # Reset streams
        hist_file.stream.seek(0)
        commit_file.stream.seek(0)
        owner_file.stream.seek(0)
 
    except Exception as e:
 
        return (
            f"Could not inspect remapped files:<br><br>{str(e)}",
            400
        )
 
    # --------------------------------------------------------
    # Forecast parameters
    # --------------------------------------------------------
 
    try:
 
        start_date_value = request.form.get(
            "start_date"
        )
 
        if not start_date_value:
 
            return (
                "Forecast start date is required.",
                400
            )
 
        forecast_start_date = pd.to_datetime(
            start_date_value
        )
 
        horizon_value = request.form.get(
            "horizon",
            14
        )
 
        forecast_horizon = int(
            horizon_value
        )
 
        if forecast_horizon <= 0:
 
            return (
                "Forecast horizon must be greater than zero.",
                400
            )
 
        commit_aggregation = request.form.get(
            "aggregation",
            "day"
        )
 
    except Exception as e:
 
        return (
            f"Invalid forecast settings:<br><br>{str(e)}",
            400
        )
 
    # --------------------------------------------------------
    # Run forecast processing
    # --------------------------------------------------------
 
    try:
 
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
 
    except KeyError as e:
 
        # This gives a much clearer error if a downstream
        # function tries to access a missing column.
        missing_column = str(e)
 
        return (
            f"""
            <strong>Forecast processing failed.</strong><br><br>
 
            Missing column: <strong>{missing_column}</strong><br><br>
 
            The files were successfully uploaded and remapped,
            but the processing service attempted to access a
            column that does not exist in the dataframe.<br><br>
 
            Please check the terminal/console for the full
            traceback. In particular, check the
            process_forecast_uploads() function and any functions
            it calls.
            """,
            400
        )
 
    except Exception as e:
 
        return (
            f"""
            <strong>Forecast processing failed.</strong><br><br>
            {str(e)}
            """,
            500
        )
 
    # --------------------------------------------------------
    # Validate returned forecast data
    # --------------------------------------------------------
 
    try:
 
        if forecast_csv is None:
 
            raise ValueError(
                "Forecast processing returned no forecast data."
            )
 
        if commits_csv is None:
 
            raise ValueError(
                "Forecast processing returned no commits data."
            )
 
        if owner_matrix_csv is None:
 
            raise ValueError(
                "Forecast processing returned no owner matrix data."
            )
 
    except Exception as e:
 
        return (
            f"Forecast output validation failed:<br><br>{str(e)}",
            500
        )
 
    # --------------------------------------------------------
    # Render result
    # --------------------------------------------------------
 
    return render_template(
 
        "index.html",
 
        forecast_ready=True,
 
        commits_csv_ready=True,
 
        owner_matrix_ready=True,
 
        accuracy=accuracy_metrics,
 
        selected_aggregation=commit_aggregation
    )
 
# ============================================================
# DOWNLOAD FORECAST
# ============================================================
 
@app.route("/download_forecast.csv")
def download_forecast():
 
    if forecast_csv is None:
 
        return (
            "No forecast is available. "
            "Please run the forecast first.",
            400
        )
 
    return send_file(
 
        BytesIO(forecast_csv),
 
        mimetype="text/csv",
 
        as_attachment=True,
 
        download_name="forecast.csv"
    )
 
# ============================================================
# COMBINE
# ============================================================
 
@app.route("/combine", methods=["POST"])
def combine():
 
    global forecast_csv
    global commits_csv
    global owner_matrix_csv
    global new_otd_file
    global commit_aggregation
 
    # --------------------------------------------------------
    # Make sure forecast has been run
    # --------------------------------------------------------
 
    if forecast_csv is None:
 
        return (
            "Forecast has not been generated yet.",
            400
        )
 
    if commits_csv is None:
 
        return (
            "Current commits data is not available.",
            400
        )
 
    if owner_matrix_csv is None:
 
        return (
            "Owner matrix data is not available.",
            400
        )
 
    # --------------------------------------------------------
    # Read output CSVs
    # --------------------------------------------------------
 
    try:
 
        forecast_df = pd.read_csv(
            BytesIO(forecast_csv)
        )
 
        commits_df = pd.read_csv(
            BytesIO(commits_csv)
        )
 
        owner_df = pd.read_csv(
            BytesIO(owner_matrix_csv)
        )
 
    except Exception as e:
 
        return (
            f"Could not read forecast output files:<br><br>{str(e)}",
            500
        )
 
    # --------------------------------------------------------
    # Validate combined inputs
    # --------------------------------------------------------
 
    print("\n========================================")
    print("COMBINE INPUT COLUMNS")
    print("========================================")
 
    print(
        "Forecast:",
        forecast_df.columns.tolist()
    )
 
    print(
        "Commits:",
        commits_df.columns.tolist()
    )
 
    print(
        "Owner:",
        owner_df.columns.tolist()
    )
 
    # --------------------------------------------------------
    # Build combined output
    # --------------------------------------------------------
 
    try:
 
        final = build_combined_output(
 
            forecast_df=forecast_df,
 
            commits_df=commits_df,
 
            owner_df=owner_df,
 
            forecast_start_date=forecast_start_date,
 
            forecast_horizon=forecast_horizon,
 
            aggregation=commit_aggregation
        )
 
    except KeyError as e:
 
        return (
            f"""
            <strong>Combine failed.</strong><br><br>
            Missing column: <strong>{str(e)}</strong><br><br>
            Check the columns printed in the application console.
            """,
            400
        )
 
    except Exception as e:
 
        return (
            f"Combine failed:<br><br>{str(e)}",
            500
        )
 
    # --------------------------------------------------------
    # Create Excel output
    # --------------------------------------------------------
 
    out_excel = BytesIO()
 
    try:
 
        with pd.ExcelWriter(
            out_excel,
            engine="xlsxwriter"
        ) as writer:
 
            # Main output
            final.to_excel(
                writer,
                index=False,
                sheet_name="Risk Output"
            )
 
            # Accuracy tab
            if accuracy_metrics is not None:
 
                if isinstance(
                    accuracy_metrics,
                    dict
                ):
 
                    accuracy_df = pd.DataFrame(
                        [accuracy_metrics]
                    )
 
                else:
 
                    accuracy_df = accuracy_metrics
 
                accuracy_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Accuracy"
                )
 
    except Exception as e:
 
        return (
            f"Could not create Excel output:<br><br>{str(e)}",
            500
        )
 
    out_excel.seek(0)
 
    return send_file(
 
        out_excel,
 
        mimetype=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
 
        as_attachment=True,
 
        download_name="vendor_commit_risk.xlsx"
    )
 
# ============================================================
# ERROR HANDLERS
# ============================================================
 
@app.errorhandler(400)
def bad_request(error):
 
    return (
        f"""
        <h3>Bad Request</h3>
        <p>{error}</p>
        """,
        400
    )
 
@app.errorhandler(404)
def page_not_found(error):
 
    return (
        """
        <h3>Page not found</h3>
        """,
        404
    )
 
@app.errorhandler(500)
def internal_server_error(error):
 
    return (
        f"""
        <h3>Internal Server Error</h3>
        <p>{error}</p>
        """,
        500
    )
 
# ============================================================
# RUN
# ============================================================
 
if __name__ == "__main__":
 
    port = 5000
 
    Timer(
        1,
        lambda: webbrowser.open(
            f"http://127.0.0.1:{port}"
        )
    ).start()
 
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=False
    )

