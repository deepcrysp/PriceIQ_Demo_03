
from pathlib import Path
from zoneinfo import ZoneInfo
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# PRICEIQ CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PriceIQ | Pricing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "PriceIQ"
APP_SUBTITLE = "Pricing Intelligence & Margin Governance"

INDIA_TZ = ZoneInfo("Asia/Kolkata")


# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: #F7F9FC;
        border: 1px solid #E5EAF1;
        padding: 14px;
        border-radius: 10px;
    }

    .section-note {
        color: #64748B;
        font-size: 0.88rem;
    }

    .priceiq-header {
        font-size: 1.65rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .priceiq-subheader {
        color: #64748B;
        margin-top: 0;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA DISCOVERY AND LOADING
# ============================================================

def find_data_file():
    """
    Search common deployment locations for customer_agreements.csv.
    """

    current_dir = Path(__file__).resolve().parent

    possible_paths = [
        current_dir / "data" / "customer_agreements.csv",
        current_dir / "customer_agreements.csv",
        current_dir / "PriceIQ_Customer_Adapted" / "data" / "customer_agreements.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Recursive fallback
    matches = list(current_dir.rglob("customer_agreements.csv"))

    if matches:
        return matches[0]

    return None


def normalize_column_name(column):
    """
    Convert SAP and source column names into normalized names.
    """

    column = str(column).strip().lower()
    column = re.sub(r"[^a-z0-9]+", "_", column)
    column = re.sub(r"_+", "_", column)

    return column.strip("_")


COLUMN_ALIASES = {
    "customer": [
        "customer",
        "customer_id",
        "customer_number",
        "customer_no",
        "customer_code",
        "sold_to",
        "sold_to_party",
        "kunnr",
        "debtor",
    ],

    "material": [
        "material",
        "material_id",
        "material_number",
        "material_no",
        "product",
        "product_id",
        "product_code",
        "matnr",
    ],

    "unit": [
        "unit",
        "uom",
        "sales_unit",
        "base_unit",
        "unit_of_measure",
        "meins",
        "vrkme",
    ],

    "agreement_value": [
        "agreement_value",
        "agreement_price",
        "price",
        "pricing_value",
        "condition_value",
        "condition_rate",
        "rate",
        "value",
        "kbetr",
        "net_price",
    ],

    "valid_from": [
        "valid_from",
        "validity_start",
        "start_date",
        "effective_from",
        "date_from",
        "datab",
    ],

    "valid_to": [
        "valid_to",
        "validity_end",
        "end_date",
        "effective_to",
        "date_to",
        "datbi",
    ],

    "region": [
        "region",
        "sales_region",
        "region_name",
    ],

    "district": [
        "district",
        "sales_district",
        "district_name",
    ],

    "agreement_type": [
        "agreement_type",
        "condition_type",
        "condition",
        "agreement_category",
        "kschl",
    ],

    "agreement_id": [
        "agreement_id",
        "agreement_number",
        "contract_number",
        "condition_record",
        "condition_record_number",
        "knumh",
    ],

    "currency": [
        "currency",
        "currency_code",
        "waers",
    ],

    "description": [
        "description",
        "material_description",
        "product_description",
        "maktx",
    ],
}


def normalize_dataframe(df):
    """
    Normalize SAP columns and create standard fields.
    """

    df = df.copy()

    df.columns = [
        normalize_column_name(c)
        for c in df.columns
    ]

    # Resolve aliases
    for standard_name, aliases in COLUMN_ALIASES.items():

        if standard_name in df.columns:
            continue

        for alias in aliases:
            normalized_alias = normalize_column_name(alias)

            if normalized_alias in df.columns:
                df.rename(
                    columns={normalized_alias: standard_name},
                    inplace=True,
                )
                break

    # Required fields
    required_defaults = {
        "customer": "Unknown",
        "material": "Unknown",
        "unit": "Unknown",
        "agreement_value": np.nan,
        "valid_from": pd.NaT,
        "valid_to": pd.NaT,
        "region": "Unknown",
        "district": "Unknown",
        "agreement_type": "Unknown",
        "agreement_id": "Not Available",
        "currency": "Not Provided",
        "description": "",
    }

    for column, default_value in required_defaults.items():

        if column not in df.columns:
            df[column] = default_value

    # Standardize IDs and categories
    for column in [
        "customer",
        "material",
        "unit",
        "region",
        "district",
        "agreement_type",
        "agreement_id",
        "currency",
    ]:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    df["agreement_value"] = pd.to_numeric(
        df["agreement_value"],
        errors="coerce",
    )

    df["valid_from"] = pd.to_datetime(
        df["valid_from"],
        errors="coerce",
    )

    df["valid_to"] = pd.to_datetime(
        df["valid_to"],
        errors="coerce",
    )

    # Stable internal row identifier
    df = df.reset_index(drop=True)
    df["_row_id"] = np.arange(len(df))

    return df


@st.cache_data(show_spinner="Loading SAP agreement data...")
def load_data(file_path, modified_time):
    """
    Load the customer agreement CSV.
    modified_time helps Streamlit refresh its cache when the file changes.
    """

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    return normalize_dataframe(df)


# ============================================================
# DATA SOURCE: MULTI-FILE UPLOAD OR REPOSITORY DEFAULT
# ============================================================

st.sidebar.markdown("### Data source")
uploaded_files = st.sidebar.file_uploader(
    "Upload agreement data (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="Upload one or more SAP agreement extracts. Uploaded files are used for this session only.",
)

def read_uploaded_agreement(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        frame = pd.read_csv(uploaded_file, low_memory=False)
    else:
        frame = pd.read_excel(uploaded_file)
    if frame.empty:
        raise ValueError(f"{uploaded_file.name} contains no rows.")
    normalized = normalize_dataframe(frame)
    normalized["source_file"] = uploaded_file.name
    return normalized

try:
    if uploaded_files:
        frames = []
        for uploaded_file in uploaded_files:
            try:
                frames.append(read_uploaded_agreement(uploaded_file))
            except Exception as upload_error:
                st.sidebar.error(f"{uploaded_file.name}: {upload_error}")
        if not frames:
            st.error("None of the uploaded files could be read. Check the file format and column headers.")
            st.stop()
        raw_df = pd.concat(frames, ignore_index=True)
        raw_df = raw_df.reset_index(drop=True)
        raw_df["_row_id"] = np.arange(len(raw_df))
        st.sidebar.success(
            f"Using {len(uploaded_files)} uploaded file(s), {len(raw_df):,} rows."
        )
        st.sidebar.caption("Uploads are session-only and are not saved to the repository.")
    else:
        data_file = find_data_file()
        if data_file is None:
            st.info(
                "Upload one or more SAP agreement extracts in the sidebar to get started. "
                "You can also add data/customer_agreements.csv to the repository."
            )
            st.stop()
        raw_df = load_data(str(data_file), data_file.stat().st_mtime)
        raw_df["source_file"] = data_file.name
        st.sidebar.caption(f"Using repository data: {data_file.name}")
except Exception as error:
    st.error(f"Unable to load agreement data: {error}")
    st.stop()

if raw_df.empty:
    st.warning("The agreement dataset is empty.")
    st.stop()

# Basic data-quality context. Agreement values are source-maintained values,
# not verified realized transaction prices or profitability measures.
missing_values = int(raw_df["agreement_value"].isna().sum())
if missing_values:
    st.sidebar.warning(f"{missing_values:,} rows have no numeric agreement value.")



# ============================================================
# COMMON HELPERS
# ============================================================

def format_number(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:,.2f}"


def format_integer(value):
    if pd.isna(value):
        return "0"

    return f"{int(value):,}"


def get_today():
    return (
        pd.Timestamp.now(INDIA_TZ)
        .tz_localize(None)
        .normalize()
    )


def get_expiry_days(df):
    return (df["valid_to"] - get_today()).dt.days


def safe_percentage(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return 0

    return numerator / denominator * 100


def show_page_header(title, subtitle=None):
    st.markdown(
        f'<p class="priceiq-header">{title}</p>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<p class="priceiq-subheader">{subtitle}</p>',
            unsafe_allow_html=True,
        )


def display_download_button(df, filename, label="Download CSV"):
    if df.empty:
        return

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


def get_material_unit_dispersion(df, minimum_records=5):
    """
    Calculate descriptive agreement-value variation by material and unit.
    """

    valid = df[
        df["agreement_value"].notna()
        & (df["agreement_value"] > 0)
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    result = (
        valid.groupby(["material", "unit"])
        ["agreement_value"]
        .agg(
            agreement_count="count",
            median_value="median",
            minimum_value="min",
            maximum_value="max",
            average_value="mean",
            standard_deviation="std",
        )
        .reset_index()
    )

    result = result[
        result["agreement_count"] >= minimum_records
    ].copy()

    result["range_vs_median_pct"] = (
        (
            result["maximum_value"]
            - result["minimum_value"]
        )
        / result["median_value"].replace(0, np.nan)
    ) * 100

    result["coefficient_of_variation_pct"] = (
        result["standard_deviation"]
        / result["average_value"].replace(0, np.nan)
    ) * 100

    return result.sort_values(
        "range_vs_median_pct",
        ascending=False,
    )


# ============================================================
# ANOMALY DETECTION ENGINE
# ============================================================

@st.cache_data(show_spinner="Running pricing anomaly detection...")
def detect_anomalies(source_df):
    """
    Detect unusual agreement records using:
    1. Material/unit price dispersion
    2. Isolation Forest on agreement values
    3. Repeated agreement patterns
    4. Overlapping validity periods

    These are investigation signals, not confirmed pricing violations.
    """

    df = source_df.copy().reset_index(drop=True)

    df["anomaly_score"] = 0.0
    df["anomaly_type"] = ""
    df["anomaly_reason"] = ""
    df["severity"] = "Normal"
    df["is_anomaly"] = False

    flag_reasons = [[] for _ in range(len(df))]
    flag_types = [[] for _ in range(len(df))]
    scores = np.zeros(len(df), dtype=float)

    def add_flag(indexes, category, reason, score):
        for idx in indexes:
            if 0 <= idx < len(df):
                flag_types[idx].append(category)
                flag_reasons[idx].append(reason)
                scores[idx] += score

    # --------------------------------------------------------
    # 1. Material/unit price dispersion
    # --------------------------------------------------------

    valid_prices = df[
        df["agreement_value"].notna()
        & (df["agreement_value"] > 0)
    ]

    for (material, unit), group in valid_prices.groupby(
        ["material", "unit"],
        sort=False,
    ):

        if len(group) < 5:
            continue

        median_value = group["agreement_value"].median()

        if pd.isna(median_value) or median_value <= 0:
            continue

        deviation = (
            (group["agreement_value"] - median_value)
            / median_value
        ).abs()

        unusual = group[deviation >= 0.50]

        if not unusual.empty:
            add_flag(
                unusual.index.tolist(),
                "Price dispersion",
                "Agreement value differs by at least 50% "
                "from the material/unit median.",
                2.0,
            )

    # --------------------------------------------------------
    # 2. Isolation Forest
    # --------------------------------------------------------

    model_data = df[
        df["agreement_value"].notna()
        & (df["agreement_value"] > 0)
    ].copy()

    if len(model_data) >= 30:

        log_values = np.log1p(
            model_data["agreement_value"].values
        ).reshape(-1, 1)

        # Fit on a sample for scalability with large SAP extracts.
        if len(log_values) > 50000:
            rng = np.random.RandomState(42)
            sample_positions = rng.choice(
                len(log_values),
                size=50000,
                replace=False,
            )
            training_values = log_values[sample_positions]
        else:
            training_values = log_values

        model = IsolationForest(
            n_estimators=100,
            contamination=0.03,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(training_values)

        predictions = model.predict(log_values)

        anomaly_positions = np.where(predictions == -1)[0]

        if len(anomaly_positions):
            anomaly_indexes = (
                model_data.index[anomaly_positions].tolist()
            )

            add_flag(
                anomaly_indexes,
                "Statistical outlier",
                "Agreement value was identified as unusual "
                "relative to the overall value distribution.",
                1.0,
            )

    # --------------------------------------------------------
    # 3. Repeated agreement patterns
    # --------------------------------------------------------

    duplicate_columns = [
        "customer",
        "material",
        "unit",
        "agreement_value",
    ]

    duplicate_mask = df.duplicated(
        subset=duplicate_columns,
        keep=False,
    )

    duplicate_indexes = df.index[duplicate_mask].tolist()

    if duplicate_indexes:
        add_flag(
            duplicate_indexes,
            "Repeated pattern",
            "Repeated customer, material, unit, and agreement value.",
            1.0,
        )

    # --------------------------------------------------------
    # 4. Overlapping validity periods
    # --------------------------------------------------------

    overlap_df = df[
        df["valid_from"].notna()
        & df["valid_to"].notna()
    ].copy()

    overlap_indexes = []

    grouping_columns = [
        "customer",
        "material",
        "unit",
    ]

    for _, group in overlap_df.groupby(
        grouping_columns,
        sort=False,
    ):

        group = group.sort_values("valid_from")

        maximum_end_date = None

        for row in group.itertuples():

            current_start = row.valid_from
            current_end = row.valid_to

            if (
                maximum_end_date is not None
                and current_start <= maximum_end_date
            ):
                overlap_indexes.append(row.Index)

            if (
                maximum_end_date is None
                or current_end > maximum_end_date
            ):
                maximum_end_date = current_end

    if overlap_indexes:
        add_flag(
            overlap_indexes,
            "Validity overlap",
            "Agreement validity period overlaps with another "
            "agreement for the same customer/material/unit.",
            2.0,
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    df["anomaly_score"] = scores

    df["anomaly_type"] = [
        "; ".join(sorted(set(items)))
        for items in flag_types
    ]

    df["anomaly_reason"] = [
        " ".join(dict.fromkeys(items))
        for items in flag_reasons
    ]

    df["is_anomaly"] = df["anomaly_score"] > 0

    def determine_severity(row):
        if not row["is_anomaly"]:
            return "Normal"

        if (
            row["anomaly_score"] >= 3
            or "Price dispersion" in row["anomaly_type"]
            and row["anomaly_score"] >= 4
        ):
            return "High"

        return "Medium"

    df["severity"] = df.apply(
        determine_severity,
        axis=1,
    )

    return df


# ============================================================
# SIDEBAR AND GLOBAL FILTERS
# ============================================================

st.sidebar.markdown(
    f"""
    <div>
        <h2 style="margin-bottom:0;">{APP_TITLE}</h2>
        <p style="color:#64748B;margin-top:0;">
            {APP_SUBTITLE}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption(
    f"Loaded records: {len(raw_df):,}"
)

pages = [
    "Executive Overview",
    "Agreement Explorer",
    "AI Anomaly Detection",
    "Pricing Intelligence",
    "Pricing Opportunity Finder",
    "Expiry & Renewals",
    "Price Benchmark",
    "Customer Profile",
    "Margin Guardrail Sandbox",
    "Data Readiness",
    "Uploaded Data Preview",
]

page = st.sidebar.radio(
    "Navigation",
    pages,
    index=0,
)

st.sidebar.divider()
st.sidebar.subheader("Data Filters")

region_options = sorted(
    raw_df["region"].dropna().astype(str).unique().tolist()
)

selected_regions = st.sidebar.multiselect(
    "Region",
    region_options,
    default=region_options,
)

filtered_by_region = raw_df[
    raw_df["region"].astype(str).isin(selected_regions)
].copy()

district_options = sorted(
    filtered_by_region["district"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_districts = st.sidebar.multiselect(
    "District",
    district_options,
    default=district_options,
)

filtered_df = filtered_by_region[
    filtered_by_region["district"].astype(str).isin(
        selected_districts
    )
].copy()

st.sidebar.divider()

st.sidebar.caption(
    "PriceIQ POC | SAP Agreement Data"
)

st.sidebar.caption(
    "Agreement values are not necessarily realized transaction prices."
)


# ============================================================
# EXECUTIVE OVERVIEW 2.0
# ============================================================

if page == "Executive Overview":

    show_page_header(
        "Executive Overview",
        "Pricing visibility, agreement exposure, and areas requiring commercial review.",
    )

    data = filtered_df.copy()

    if data.empty:
        st.warning("No records available for the selected filters.")
        st.stop()

    anomalies = detect_anomalies(data)

    total_agreements = len(data)
    total_customers = data["customer"].nunique()
    total_materials = data["material"].nunique()

    anomaly_data = anomalies[
        anomalies["is_anomaly"]
    ].copy()

    anomaly_count = len(anomaly_data)

    high_count = int(
        (anomaly_data["severity"] == "High").sum()
    )

    medium_count = int(
        (anomaly_data["severity"] == "Medium").sum()
    )

    expiry_days = get_expiry_days(data)

    expired_count = int((expiry_days < 0).sum())
    expiry_30 = int(expiry_days.between(0, 30).sum())
    expiry_60 = int(expiry_days.between(31, 60).sum())
    expiry_90 = int(expiry_days.between(61, 90).sum())

    st.subheader("Agreement Portfolio")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Agreement Records",
        f"{total_agreements:,}",
    )

    c2.metric(
        "Customers",
        f"{total_customers:,}",
    )

    c3.metric(
        "Materials",
        f"{total_materials:,}",
    )

    c4.metric(
        "Anomaly Flags",
        f"{anomaly_count:,}",
    )

    st.divider()

    st.subheader("Where Attention Is Needed")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("High-Severity Flags", f"{high_count:,}")
    c2.metric("Medium-Severity Flags", f"{medium_count:,}")
    c3.metric("Expired Agreements", f"{expired_count:,}")
    c4.metric("Expiring in 30 Days", f"{expiry_30:,}")

    st.divider()

    st.subheader("Agreement Expiry Outlook")

    expiry_summary = pd.DataFrame({
        "Expiry Window": [
            "Already expired",
            "Next 30 days",
            "31–60 days",
            "61–90 days",
        ],
        "Agreements": [
            expired_count,
            expiry_30,
            expiry_60,
            expiry_90,
        ],
    })

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            expiry_summary,
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.bar_chart(
            expiry_summary.set_index("Expiry Window")
        )

    st.divider()

    st.subheader("Materials with High Agreement-Value Variation")

    dispersion = get_material_unit_dispersion(data)

    if not dispersion.empty:

        display = dispersion.head(10).copy()

        display = display.rename(columns={
            "material": "Material",
            "unit": "Unit",
            "agreement_count": "Agreement Count",
            "median_value": "Median Agreement Value",
            "minimum_value": "Minimum Value",
            "maximum_value": "Maximum Value",
            "range_vs_median_pct": "Range vs Median (%)",
        })

        display["Range vs Median (%)"] = (
            display["range_vs_median_pct"].round(1)
            if "range_vs_median_pct" in display.columns
            else display["Range vs Median (%)"].round(1)
        )

        st.dataframe(
            display[
                [
                    "Material",
                    "Unit",
                    "Agreement Count",
                    "Median Agreement Value",
                    "Minimum Value",
                    "Maximum Value",
                    "Range vs Median (%)",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Large variation is a review signal. It may reflect "
            "customer-specific terms, contract timing, or other "
            "commercial differences."
        )

    else:
        st.info(
            "Not enough comparable records to calculate price variation."
        )

    st.divider()

    st.subheader("Customers Requiring Agreement Review")

    if not anomaly_data.empty:

        customer_review = (
            anomaly_data.groupby("customer")
            .agg(
                flagged_agreements=("customer", "size"),
            )
            .reset_index()
            .sort_values(
                "flagged_agreements",
                ascending=False,
            )
            .head(10)
        )

        st.dataframe(
            customer_review.rename(columns={
                "customer": "Customer",
                "flagged_agreements": "Flagged Agreements",
            }),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info("No anomaly flags found for the selected filters.")

    st.divider()

    st.subheader("Suggested Management Actions")

    a, b, c = st.columns(3)

    with a:
        st.markdown("**1. Review pricing variation**")
        st.write(
            "Investigate materials with a wide spread in maintained "
            "agreement values."
        )

    with b:
        st.markdown("**2. Prioritize renewals**")
        st.write(
            "Review expired and soon-to-expire agreements with "
            "commercial teams."
        )

    with c:
        st.markdown("**3. Investigate unusual records**")
        st.write(
            "Review outliers, repeated agreement patterns, and "
            "overlapping validity periods."
        )

    st.caption(
        "Data limitation: Currency codes, invoice quantities, realized "
        "prices, costs, rebates, and freight are not available in the "
        "current extracts. No revenue or margin impact is inferred."
    )


# ============================================================
# AGREEMENT EXPLORER
# ============================================================

elif page == "Agreement Explorer":

    show_page_header(
        "Agreement Explorer",
        "Search, filter, and inspect SAP pricing agreement records.",
    )

    data = filtered_df.copy()

    search_text = st.text_input(
        "Search customer, material, or agreement ID",
        "",
    )

    if search_text.strip():

        search = search_text.strip().lower()

        mask = (
            data["customer"].str.lower().str.contains(
                search,
                na=False,
            )
            | data["material"].str.lower().str.contains(
                search,
                na=False,
            )
            | data["agreement_id"].str.lower().str.contains(
                search,
                na=False,
            )
        )

        data = data[mask]

    col1, col2, col3 = st.columns(3)

    with col1:
        agreement_types = sorted(
            data["agreement_type"].unique().tolist()
        )

        selected_types = st.multiselect(
            "Agreement Type",
            agreement_types,
            default=agreement_types,
        )

    with col2:
        units = sorted(data["unit"].unique().tolist())

        selected_units = st.multiselect(
            "Unit",
            units,
            default=units,
        )

    with col3:
        date_filter = st.selectbox(
            "Validity",
            [
                "All Agreements",
                "Currently Valid",
                "Expired",
                "Upcoming",
            ],
        )

    data = data[
        data["agreement_type"].isin(selected_types)
        & data["unit"].isin(selected_units)
    ]

    today = get_today()

    if date_filter == "Currently Valid":
        data = data[
            (data["valid_from"].isna() | (data["valid_from"] <= today))
            & (data["valid_to"].isna() | (data["valid_to"] >= today))
        ]

    elif date_filter == "Expired":
        data = data[data["valid_to"] < today]

    elif date_filter == "Upcoming":
        data = data[data["valid_from"] > today]

    st.metric("Matching Agreement Records", f"{len(data):,}")

    display_columns = [
        "agreement_id",
        "agreement_type",
        "customer",
        "material",
        "description",
        "unit",
        "agreement_value",
        "valid_from",
        "valid_to",
        "region",
        "district",
        "currency",
    ]

    st.dataframe(
        data[display_columns],
        hide_index=True,
        use_container_width=True,
    )

    display_download_button(
        data[display_columns],
        "priceiq_agreement_records.csv",
    )


# ============================================================
# AI ANOMALY DETECTION
# ============================================================

elif page == "AI Anomaly Detection":

    show_page_header(
        "AI Anomaly Detection",
        "Identify unusual agreement patterns and investigate the underlying records.",
    )

    anomaly_df = detect_anomalies(filtered_df)

    flagged = anomaly_df[
        anomaly_df["is_anomaly"]
    ].copy()

    if flagged.empty:
        st.success(
            "No anomaly signals were detected for the selected filters."
        )
        st.stop()

    # Summary metrics
    high_count = int(
        (flagged["severity"] == "High").sum()
    )

    medium_count = int(
        (flagged["severity"] == "Medium").sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Flagged Agreements", f"{len(flagged):,}")
    c2.metric("High Severity", f"{high_count:,}")
    c3.metric("Medium Severity", f"{medium_count:,}")

    st.divider()

    st.subheader("Filter Anomaly Signals")

    f1, f2 = st.columns(2)

    with f1:
        severity_options = ["High", "Medium"]

        selected_severity = st.multiselect(
            "Severity",
            severity_options,
            default=severity_options,
        )

    with f2:
        anomaly_categories = sorted(
            set(
                category
                for value in flagged["anomaly_type"].fillna("")
                for category in value.split("; ")
                if category
            )
        )

        selected_categories = st.multiselect(
            "Anomaly Category",
            anomaly_categories,
            default=anomaly_categories,
        )

    display = flagged[
        flagged["severity"].isin(selected_severity)
        & flagged["anomaly_type"].apply(
            lambda value: any(
                category in value.split("; ")
                for category in selected_categories
            )
        )
    ].copy()

    if display.empty:
        st.info("No records match the selected anomaly filters.")
        st.stop()

    display["source_row"] = display["_row_id"].astype(str)

    table_columns = [
        "source_row",
        "severity",
        "anomaly_type",
        "customer",
        "material",
        "unit",
        "agreement_value",
        "valid_from",
        "valid_to",
        "anomaly_score",
        "anomaly_reason",
    ]

    st.dataframe(
        display[table_columns],
        hide_index=True,
        use_container_width=True,
    )

    download_df = display[table_columns].drop(
        columns=["source_row"]
    )

    display_download_button(
        download_df,
        "priceiq_anomaly_review.csv",
        "Download Flagged Agreements",
    )

    st.divider()

    st.subheader("Investigate an Agreement")

    selected_row_id = st.selectbox(
        "Select agreement to investigate",
        display["source_row"].tolist(),
        format_func=lambda row_id: (
            display.loc[
                display["source_row"] == row_id,
                "customer",
            ].iloc[0]
            + " | "
            + display.loc[
                display["source_row"] == row_id,
                "material",
            ].iloc[0]
            + " | Row "
            + row_id
        ),
    )

    selected = display[
        display["source_row"] == selected_row_id
    ].iloc[0]

    st.markdown("**Agreement details**")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Customer", str(selected["customer"]))
    d2.metric("Material", str(selected["material"]))
    d3.metric("Agreement Value", format_number(selected["agreement_value"]))
    d4.metric("Severity", str(selected["severity"]))

    st.markdown("**Why was this agreement flagged?**")

    st.write(selected["anomaly_reason"])

    st.caption(
        "An anomaly is a signal for investigation, not proof of an "
        "incorrect agreement or commercial policy violation."
    )

    st.divider()

    # Comparable agreements
    st.subheader("Comparable Agreement Analysis")

    peers = anomaly_df[
        (anomaly_df["material"] == selected["material"])
        & (anomaly_df["unit"] == selected["unit"])
        & (anomaly_df["_row_id"] != int(selected["_row_id"]))
        & anomaly_df["agreement_value"].notna()
        & (anomaly_df["agreement_value"] > 0)
    ].copy()

    if not peers.empty:

        peer_median = peers["agreement_value"].median()
        peer_min = peers["agreement_value"].min()
        peer_max = peers["agreement_value"].max()

        selected_value = selected["agreement_value"]

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("Comparable Records", f"{len(peers):,}")
        p2.metric("Peer Median", format_number(peer_median))
        p3.metric("Peer Minimum", format_number(peer_min))
        p4.metric("Peer Maximum", format_number(peer_max))

        if pd.notna(selected_value) and peer_median > 0:

            difference = (
                (selected_value - peer_median)
                / peer_median
            ) * 100

            st.write(
                f"Selected agreement is **{difference:+.1f}%** "
                "relative to the comparable agreement median."
            )

        peer_display = peers[
            [
                "customer",
                "material",
                "unit",
                "agreement_value",
                "valid_from",
                "valid_to",
                "anomaly_type",
            ]
        ].sort_values(
            "agreement_value",
            ascending=True,
        )

        st.dataframe(
            peer_display.head(100),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info("No comparable agreements are available.")

    st.divider()

    st.subheader("Customer and Material Agreement History")

    history = anomaly_df[
        (anomaly_df["customer"] == selected["customer"])
        & (anomaly_df["material"] == selected["material"])
        & (anomaly_df["unit"] == selected["unit"])
    ].copy()

    if not history.empty:

        history = history.sort_values("valid_from")

        st.dataframe(
            history[
                [
                    "agreement_id",
                    "agreement_value",
                    "valid_from",
                    "valid_to",
                    "severity",
                    "anomaly_type",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Suggested Review Guidance")

    if "Price dispersion" in selected["anomaly_type"]:
        st.write(
            "Review the agreement against other customer agreements "
            "for the same material and unit. Check customer-specific "
            "terms, volume commitments, and validity dates."
        )

    if "Validity overlap" in selected["anomaly_type"]:
        st.write(
            "Check whether overlapping validity periods are intentional "
            "or represent multiple active conditions."
        )

    if "Repeated pattern" in selected["anomaly_type"]:
        st.write(
            "Check whether repeated agreement patterns represent "
            "separate conditions, versions, or duplicate records."
        )

    if "Statistical outlier" in selected["anomaly_type"]:
        st.write(
            "Review the value against the original SAP condition record "
            "and comparable agreements. Statistical outliers are not "
            "automatically incorrect."
        )

    with st.expander("Anomaly Detection Methodology"):

        st.markdown("""
        **Price dispersion**

        Flags agreement values that differ by at least 50% from the
        median for the same material and unit, where at least five
        comparable records exist.

        **Isolation Forest**

        Identifies unusual agreement values relative to the overall
        agreement-value distribution. The model uses log-transformed
        agreement values. Its output is an anomaly signal, not a
        probability of error.

        **Repeated patterns**

        Identifies repeated customer, material, unit, and agreement-value
        combinations. Repetition may be valid.

        **Validity overlaps**

        Identifies overlapping validity periods for the same customer,
        material, and unit.

        **Severity**

        Severity is a rule-based prioritization aid. It is not a
        confirmed business-risk classification.
        """)


# ============================================================
# PRICING INTELLIGENCE
# ============================================================

elif page == "Pricing Intelligence":

    show_page_header(
        "Pricing Intelligence",
        "Understand agreement-value distribution and variation across materials and customers.",
    )

    data = filtered_df.copy()

    valid = data[
        data["agreement_value"].notna()
        & (data["agreement_value"] > 0)
    ].copy()

    if valid.empty:
        st.warning("No valid agreement values available.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Records with Values",
        f"{len(valid):,}",
    )

    c2.metric(
        "Median Agreement Value",
        format_number(valid["agreement_value"].median()),
    )

    c3.metric(
        "Average Agreement Value",
        format_number(valid["agreement_value"].mean()),
    )

    c4.metric(
        "Distinct Materials",
        f"{valid['material'].nunique():,}",
    )

    st.divider()

    st.subheader("Agreement Value Distribution")

    st.bar_chart(
        valid["agreement_value"]
        .clip(upper=valid["agreement_value"].quantile(0.99))
        .value_counts(bins=20)
        .sort_index()
    )

    st.caption(
        "The chart caps values at the 99th percentile for readability. "
        "This does not change the underlying source data."
    )

    st.divider()

    st.subheader("Material-Level Price Variation")

    dispersion = get_material_unit_dispersion(valid)

    if not dispersion.empty:

        st.dataframe(
            dispersion.rename(columns={
                "material": "Material",
                "unit": "Unit",
                "agreement_count": "Agreement Count",
                "median_value": "Median Value",
                "minimum_value": "Minimum Value",
                "maximum_value": "Maximum Value",
                "range_vs_median_pct": "Range vs Median (%)",
                "coefficient_of_variation_pct": "Coefficient of Variation (%)",
            }),
            hide_index=True,
            use_container_width=True,
        )

    else:
        st.info("Insufficient records for material-level comparison.")

    st.divider()

    st.subheader("Agreement Value by Unit")

    unit_summary = (
        valid.groupby("unit")["agreement_value"]
        .agg(
            agreement_count="count",
            median_value="median",
            average_value="mean",
            minimum_value="min",
            maximum_value="max",
        )
        .reset_index()
        .sort_values("agreement_count", ascending=False)
    )

    st.dataframe(
        unit_summary,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Different units should not be compared directly. "
        "The analysis is descriptive and does not account for "
        "contract-specific commercial terms."
    )


# ============================================================
# PRICING OPPORTUNITY FINDER
# ============================================================

elif page == "Pricing Opportunity Finder":

    show_page_header(
        "Pricing Opportunity Finder",
        "Identify customer-material agreements that may warrant commercial pricing review.",
    )

    data = filtered_df.copy()

    data = data[
        data["material"].notna()
        & data["unit"].notna()
        & data["customer"].notna()
        & data["agreement_value"].notna()
        & (data["agreement_value"] > 0)
    ].copy()

    if data.empty:
        st.warning(
            "No valid positive agreement values are available for comparison."
        )
        st.stop()

    customer_material = (
        data.groupby(
            ["material", "unit", "customer"],
            dropna=False,
        )
        .agg(
            customer_median_value=("agreement_value", "median"),
            agreement_count=("agreement_value", "size"),
            latest_valid_to=("valid_to", "max"),
        )
        .reset_index()
    )

    result_rows = []

    for (material, unit), group in customer_material.groupby(
        ["material", "unit"],
        sort=False,
    ):

        if len(group) < 6:
            continue

        values = group["customer_median_value"].to_numpy(
            dtype=float
        )

        customers = group["customer"].tolist()

        for i, row in enumerate(group.itertuples(index=False)):

            peer_values = np.delete(values, i)

            peer_values = peer_values[
                np.isfinite(peer_values)
                & (peer_values > 0)
            ]

            if len(peer_values) < 5:
                continue

            peer_median = float(np.median(peer_values))

            if peer_median <= 0:
                continue

            customer_value = float(row.customer_median_value)

            gap_pct = (
                (peer_median - customer_value)
                / peer_median
            ) * 100

            result_rows.append({
                "material": material,
                "unit": unit,
                "customer": row.customer,
                "customer_value": customer_value,
                "peer_median": peer_median,
                "gap_pct": gap_pct,
                "peer_customer_count": len(peer_values),
                "agreement_count": int(row.agreement_count),
                "latest_valid_to": row.latest_valid_to,
            })

    opportunity_df = pd.DataFrame(result_rows)

    if opportunity_df.empty:
        st.info(
            "There are not enough material/unit combinations with "
            "at least six distinct customers to create peer comparisons."
        )
        st.stop()

    st.subheader("Review Criteria")

    threshold = st.slider(
        "Flag agreements below peer median by at least (%)",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    material_options = [
        "All Materials"
    ] + sorted(
        opportunity_df["material"].astype(str).unique().tolist()
    )

    selected_material = st.selectbox(
        "Material",
        material_options,
    )

    candidates = opportunity_df[
        opportunity_df["gap_pct"] >= threshold
    ].copy()

    if selected_material != "All Materials":
        candidates = candidates[
            candidates["material"].astype(str)
            == selected_material
        ]

    candidates = candidates.sort_values(
        "gap_pct",
        ascending=False,
    )

    total_candidates = len(candidates)

    affected_customers = (
        candidates["customer"].nunique()
        if total_candidates else 0
    )

    affected_materials = (
        candidates["material"].nunique()
        if total_candidates else 0
    )

    average_gap = (
        candidates["gap_pct"].mean()
        if total_candidates else 0
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Review Candidates", f"{total_candidates:,}")
    k2.metric("Customers Affected", f"{affected_customers:,}")
    k3.metric("Materials Affected", f"{affected_materials:,}")
    k4.metric("Average Gap", f"{average_gap:.1f}%")

    st.divider()

    st.subheader("Customer-Material Review Candidates")

    if candidates.empty:
        st.success(
            "No customer-material combinations meet the selected threshold."
        )

    else:

        display = candidates.rename(columns={
            "material": "Material",
            "unit": "Unit",
            "customer": "Customer",
            "customer_value": "Customer Median Agreement Value",
            "peer_median": "Other-Customer Median Value",
            "gap_pct": "Below Peer Median (%)",
            "peer_customer_count": "Peer Customer Count",
            "agreement_count": "Agreement Records",
            "latest_valid_to": "Latest Valid-To Date",
        }).copy()

        display["Below Peer Median (%)"] = (
            display["Below Peer Median (%)"].round(1)
        )

        display["Customer Median Agreement Value"] = (
            display["Customer Median Agreement Value"].round(2)
        )

        display["Other-Customer Median Value"] = (
            display["Other-Customer Median Value"].round(2)
        )

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
        )

        display_download_button(
            display,
            "priceiq_pricing_review_candidates.csv",
            "Download Review Candidates",
        )

    st.divider()

    st.subheader("Investigate a Candidate")

    if not candidates.empty:

        candidate_labels = candidates.apply(
            lambda row: (
                f"Customer {row['customer']} | "
                f"Material {row['material']} | "
                f"{row['gap_pct']:.1f}% below peer median"
            ),
            axis=1,
        ).tolist()

        selected_label = st.selectbox(
            "Select a customer-material combination",
            candidate_labels,
        )

        selected_position = candidate_labels.index(
            selected_label
        )

        selected = candidates.iloc[selected_position]

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Customer Median Value",
            format_number(selected["customer_value"]),
        )

        d2.metric(
            "Other-Customer Median",
            format_number(selected["peer_median"]),
        )

        d3.metric(
            "Below Peer Median",
            f"{selected['gap_pct']:.1f}%",
        )

        st.markdown("**Why this candidate was flagged**")

        st.write(
            f"For material **{selected['material']}**, unit "
            f"**{selected['unit']}**, customer **{selected['customer']}** "
            f"has a median maintained agreement value of "
            f"**{format_number(selected['customer_value'])}**."
        )

        st.write(
            f"The median value across "
            f"**{int(selected['peer_customer_count'])} other customers** "
            f"is **{format_number(selected['peer_median'])}**."
        )

        st.info(
            "This is a pricing review signal, not a confirmed underpricing "
            "issue or a quantified revenue opportunity. Differences may "
            "reflect volume commitments, contract terms, geography, product "
            "configuration, or other commercial conditions."
        )

    with st.expander("View Methodology and Limitations"):

        st.markdown("""
        **Comparison method**

        1. Groups agreements by material, unit, and customer.
        2. Calculates the median agreement value for each customer.
        3. Compares that value with the median of other customers for the same material and unit.
        4. Requires at least five other customers in the peer group.
        5. Flags candidates when the customer median is below the selected threshold.

        **Limitations**

        - Currency codes are not available in the current extracts.
        - Agreement values are not necessarily realized invoice prices.
        - Quantities, costs, rebates, freight, and cost-to-serve are unavailable.
        - Comparisons do not account for customer-specific contract terms or volume tiers.
        - No revenue uplift or margin improvement is calculated.
        """)


# ============================================================
# EXPIRY AND RENEWALS
# ============================================================

elif page == "Expiry & Renewals":

    show_page_header(
        "Expiry & Renewals",
        "Monitor agreement validity and prepare renewal actions.",
    )

    data = filtered_df.copy()

    today = get_today()

    data["days_to_expiry"] = (
        data["valid_to"] - today
    ).dt.days

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Expired",
        f"{int((data['days_to_expiry'] < 0).sum()):,}",
    )

    c2.metric(
        "Next 30 Days",
        f"{int(data['days_to_expiry'].between(0, 30).sum()):,}",
    )

    c3.metric(
        "31–60 Days",
        f"{int(data['days_to_expiry'].between(31, 60).sum()):,}",
    )

    c4.metric(
        "61–90 Days",
        f"{int(data['days_to_expiry'].between(61, 90).sum()):,}",
    )

    st.divider()

    expiry_filter = st.selectbox(
        "Select expiry window",
        [
            "All",
            "Expired",
            "Next 30 Days",
            "31–60 Days",
            "61–90 Days",
            "Beyond 90 Days",
        ],
    )

    if expiry_filter == "Expired":
        selected = data[data["days_to_expiry"] < 0]

    elif expiry_filter == "Next 30 Days":
        selected = data[
            data["days_to_expiry"].between(0, 30)
        ]

    elif expiry_filter == "31–60 Days":
        selected = data[
            data["days_to_expiry"].between(31, 60)
        ]

    elif expiry_filter == "61–90 Days":
        selected = data[
            data["days_to_expiry"].between(61, 90)
        ]

    elif expiry_filter == "Beyond 90 Days":
        selected = data[data["days_to_expiry"] > 90]

    else:
        selected = data

    selected = selected.sort_values(
        "valid_to",
        ascending=True,
        na_position="last",
    )

    display_columns = [
        "agreement_id",
        "customer",
        "material",
        "unit",
        "agreement_value",
        "valid_from",
        "valid_to",
        "days_to_expiry",
        "region",
        "district",
    ]

    st.metric(
        "Agreements in Selected Window",
        f"{len(selected):,}",
    )

    st.dataframe(
        selected[display_columns],
        hide_index=True,
        use_container_width=True,
    )

    display_download_button(
        selected[display_columns],
        "priceiq_expiry_renewal_list.csv",
        "Download Renewal Action List",
    )

    st.divider()

    st.subheader("Renewal Workload by Customer")

    renewal_workload = (
        selected.groupby("customer")
        .size()
        .reset_index(name="Agreements Requiring Review")
        .sort_values(
            "Agreements Requiring Review",
            ascending=False,
        )
        .head(15)
    )

    st.dataframe(
        renewal_workload,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# PRICE BENCHMARK
# ============================================================

elif page == "Price Benchmark":

    show_page_header(
        "Price Benchmark",
        "Compare maintained agreement values for the same material and unit.",
    )

    data = filtered_df.copy()

    valid = data[
        data["agreement_value"].notna()
        & (data["agreement_value"] > 0)
    ].copy()

    if valid.empty:
        st.warning("No valid agreement values available.")
        st.stop()

    materials = sorted(
        valid["material"].unique().tolist()
    )

    selected_material = st.selectbox(
        "Select Material",
        materials,
    )

    material_data = valid[
        valid["material"] == selected_material
    ].copy()

    units = sorted(
        material_data["unit"].unique().tolist()
    )

    selected_unit = st.selectbox(
        "Select Unit",
        units,
    )

    benchmark_data = material_data[
        material_data["unit"] == selected_unit
    ].copy()

    if benchmark_data.empty:
        st.info("No comparable records available.")
        st.stop()

    median_value = benchmark_data["agreement_value"].median()
    average_value = benchmark_data["agreement_value"].mean()
    minimum_value = benchmark_data["agreement_value"].min()
    maximum_value = benchmark_data["agreement_value"].max()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Agreement Records", f"{len(benchmark_data):,}")
    c2.metric("Median Value", format_number(median_value))
    c3.metric("Minimum Value", format_number(minimum_value))
    c4.metric("Maximum Value", format_number(maximum_value))

    st.divider()

    st.subheader("Agreement Values by Customer")

    customer_benchmark = (
        benchmark_data.groupby("customer")
        ["agreement_value"]
        .agg(
            agreement_count="count",
            median_value="median",
            minimum_value="min",
            maximum_value="max",
        )
        .reset_index()
        .sort_values("median_value")
    )

    st.dataframe(
        customer_benchmark,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Agreement-Level Comparison")

    benchmark_display = benchmark_data[
        [
            "agreement_id",
            "customer",
            "material",
            "unit",
            "agreement_value",
            "valid_from",
            "valid_to",
        ]
    ].sort_values(
        "agreement_value",
        ascending=True,
    )

    st.dataframe(
        benchmark_display,
        hide_index=True,
        use_container_width=True,
    )

    display_download_button(
        benchmark_display,
        "priceiq_material_benchmark.csv",
    )

    st.caption(
        "Benchmarking is restricted to the same material and unit. "
        "It does not normalize for customer terms, volumes, geography, "
        "currency, or other commercial conditions."
    )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

elif page == "Customer Profile":

    show_page_header(
        "Customer Profile",
        "View agreement coverage and pricing patterns for an individual customer.",
    )

    data = filtered_df.copy()

    customers = sorted(
        data["customer"].unique().tolist()
    )

    if not customers:
        st.warning("No customers available.")
        st.stop()

    selected_customer = st.selectbox(
        "Select Customer",
        customers,
    )

    customer_data = data[
        data["customer"] == selected_customer
    ].copy()

    customer_data["days_to_expiry"] = (
        customer_data["valid_to"] - get_today()
    ).dt.days

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Agreement Records",
        f"{len(customer_data):,}",
    )

    c2.metric(
        "Materials",
        f"{customer_data['material'].nunique():,}",
    )

    c3.metric(
        "Expired Agreements",
        f"{int((customer_data['days_to_expiry'] < 0).sum()):,}",
    )

    c4.metric(
        "Expiring in 30 Days",
        f"{int(customer_data['days_to_expiry'].between(0, 30).sum()):,}",
    )

    st.divider()

    st.subheader("Customer Agreement History")

    st.dataframe(
        customer_data[
            [
                "agreement_id",
                "agreement_type",
                "material",
                "description",
                "unit",
                "agreement_value",
                "valid_from",
                "valid_to",
                "region",
                "district",
            ]
        ].sort_values(
            "valid_from",
            ascending=False,
        ),
        hide_index=True,
        use_container_width=True,
    )

    display_download_button(
        customer_data,
        f"priceiq_customer_{selected_customer}.csv",
        "Download Customer Agreements",
    )

    st.divider()

    st.subheader("Agreement Records by Material")

    material_summary = (
        customer_data.groupby("material")
        .agg(
            agreement_count=("agreement_value", "size"),
            median_value=("agreement_value", "median"),
            minimum_value=("agreement_value", "min"),
            maximum_value=("agreement_value", "max"),
        )
        .reset_index()
        .sort_values(
            "agreement_count",
            ascending=False,
        )
    )

    st.dataframe(
        material_summary,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Customer Agreement Value Distribution")

    customer_values = customer_data[
        customer_data["agreement_value"].notna()
        & (customer_data["agreement_value"] > 0)
    ]

    if not customer_values.empty:

        st.bar_chart(
            customer_values["agreement_value"]
            .value_counts(bins=15)
            .sort_index()
        )

    else:
        st.info("No valid agreement values available for this customer.")


# ============================================================
# MARGIN GUARDRAIL SANDBOX
# ============================================================

elif page == "Margin Guardrail Sandbox":

    show_page_header(
        "Margin Guardrail Sandbox",
        "Test hypothetical pricing guardrails using user-supplied commercial assumptions.",
    )

    st.info(
        "This is a scenario calculator. The current SAP extracts do not "
        "contain product costs or list prices, so these values must be "
        "entered by the user. Results are hypothetical, not actual margins."
    )

    st.subheader("Scenario Inputs")

    col1, col2 = st.columns(2)

    with col1:

        product_cost = st.number_input(
            "Assumed Product Cost",
            min_value=0.0,
            value=50.0,
            step=1.0,
        )

        list_price = st.number_input(
            "Assumed List Price",
            min_value=0.0,
            value=100.0,
            step=1.0,
        )

        proposed_price = st.number_input(
            "Proposed Agreement Price",
            min_value=0.0,
            value=85.0,
            step=1.0,
        )

    with col2:

        minimum_margin_pct = st.slider(
            "Minimum Margin (%)",
            min_value=0,
            max_value=90,
            value=25,
        )

        maximum_discount_pct = st.slider(
            "Maximum Discount (%)",
            min_value=0,
            max_value=90,
            value=20,
        )

    st.divider()

    if list_price <= 0:
        st.error("List price must be greater than zero.")

    elif product_cost < 0:
        st.error("Product cost cannot be negative.")

    else:

        minimum_margin = minimum_margin_pct / 100
        maximum_discount = maximum_discount_pct / 100

        margin_floor = (
            product_cost / (1 - minimum_margin)
            if minimum_margin < 1
            else float("inf")
        )

        discount_floor = (
            list_price * (1 - maximum_discount)
        )

        recommended_floor = max(
            margin_floor,
            discount_floor,
        )

        proposed_discount = (
            (list_price - proposed_price) / list_price
        ) * 100

        proposed_margin = (
            (proposed_price - product_cost) / proposed_price
            * 100
            if proposed_price > 0
            else np.nan
        )

        st.subheader("Scenario Results")

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Margin-Based Floor",
            format_number(margin_floor),
        )

        r2.metric(
            "Discount-Based Floor",
            format_number(discount_floor),
        )

        r3.metric(
            "Recommended Price Floor",
            format_number(recommended_floor),
        )

        st.divider()

        result1, result2, result3 = st.columns(3)

        result1.metric(
            "Proposed Discount",
            f"{proposed_discount:.1f}%",
        )

        result2.metric(
            "Hypothetical Margin",
            (
                f"{proposed_margin:.1f}%"
                if pd.notna(proposed_margin)
                else "N/A"
            ),
        )

        result3.metric(
            "Price vs Guardrail",
            (
                f"{proposed_price - recommended_floor:,.2f}"
                if np.isfinite(recommended_floor)
                else "N/A"
            ),
        )

        if proposed_price < recommended_floor:

            st.error(
                "The proposed price is below the calculated guardrail. "
                "Review the scenario before approval."
            )

        else:

            st.success(
                "The proposed price meets the entered hypothetical guardrails."
            )

        scenario = pd.DataFrame({
            "Metric": [
                "Product Cost",
                "List Price",
                "Proposed Price",
                "Minimum Margin (%)",
                "Maximum Discount (%)",
                "Margin-Based Floor",
                "Discount-Based Floor",
                "Recommended Price Floor",
                "Proposed Discount (%)",
                "Hypothetical Margin (%)",
            ],
            "Value": [
                product_cost,
                list_price,
                proposed_price,
                minimum_margin_pct,
                maximum_discount_pct,
                margin_floor,
                discount_floor,
                recommended_floor,
                proposed_discount,
                proposed_margin,
            ],
        })

        st.dataframe(
            scenario,
            hide_index=True,
            use_container_width=True,
        )

        display_download_button(
            scenario,
            "priceiq_margin_scenario.csv",
            "Download Scenario",
        )

    st.caption(
        "Guardrail formula: maximum of cost divided by one minus "
        "minimum margin, and list price multiplied by one minus "
        "maximum discount. Assumptions are supplied by the user."
    )


# ============================================================
# DATA READINESS
# ============================================================

elif page == "Data Readiness":

    show_page_header(
        "Data Readiness",
        "Assess the available data and identify what is needed for advanced pricing analytics.",
    )

    data = filtered_df.copy()

    total_records = len(data)

    if total_records == 0:
        st.warning("No data available for the selected filters.")
        st.stop()

    st.subheader("Current Data Coverage")

    coverage_fields = [
        "customer",
        "material",
        "unit",
        "agreement_value",
        "valid_from",
        "valid_to",
        "region",
        "district",
        "agreement_type",
        "agreement_id",
        "currency",
    ]

    coverage_rows = []

    for column in coverage_fields:

        if column not in data.columns:
            populated = 0
        else:
            if column == "agreement_value":
                populated = int(data[column].notna().sum())

            elif column in ["valid_from", "valid_to"]:
                populated = int(data[column].notna().sum())

            else:
                populated = int(
                    data[column]
                    .notna()
                    .astype(int)
                    .sum()
                )

        missing = total_records - populated

        coverage_rows.append({
            "Field": column,
            "Populated Records": populated,
            "Missing Records": missing,
            "Coverage (%)": round(
                safe_percentage(populated, total_records),
                1,
            ),
        })

    coverage_df = pd.DataFrame(coverage_rows)

    st.dataframe(
        coverage_df,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Data Quality Checks")

    quality_checks = []

    quality_checks.append({
        "Check": "Missing Customer ID",
        "Records": int(
            data["customer"].isin(["Unknown", "", "nan"]).sum()
        ),
    })

    quality_checks.append({
        "Check": "Missing Material ID",
        "Records": int(
            data["material"].isin(["Unknown", "", "nan"]).sum()
        ),
    })

    quality_checks.append({
        "Check": "Missing Agreement Value",
        "Records": int(
            data["agreement_value"].isna().sum()
        ),
    })

    quality_checks.append({
        "Check": "Missing Unit",
        "Records": int(
            data["unit"].isin(["Unknown", "", "nan"]).sum()
        ),
    })

    quality_checks.append({
        "Check": "Missing Valid-From Date",
        "Records": int(data["valid_from"].isna().sum()),
    })

    quality_checks.append({
        "Check": "Missing Valid-To Date",
        "Records": int(data["valid_to"].isna().sum()),
    })

    quality_checks.append({
        "Check": "Invalid Validity Period",
        "Records": int(
            (
                data["valid_from"].notna()
                & data["valid_to"].notna()
                & (data["valid_from"] > data["valid_to"])
            ).sum()
        ),
    })

    duplicate_count = int(
        data.duplicated(
            subset=[
                "customer",
                "material",
                "unit",
                "agreement_value",
            ],
            keep=False,
        ).sum()
    )

    quality_checks.append({
        "Check": "Repeated Customer-Material-Unit-Value Patterns",
        "Records": duplicate_count,
    })

    quality_df = pd.DataFrame(quality_checks)

    st.dataframe(
        quality_df,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Data Needed for Advanced Pricing Capabilities")

    readiness = pd.DataFrame([
        {
            "Capability": "Agreement Lookup and Visibility",
            "Additional Data Needed": "Not required for basic POC",
            "Current Readiness": "Available",
        },
        {
            "Capability": "Agreement Expiry and Renewal Monitoring",
            "Additional Data Needed": "Accurate validity dates and agreement status",
            "Current Readiness": "Partially available",
        },
        {
            "Capability": "Descriptive Price Benchmarking",
            "Additional Data Needed": "Currency, unit normalization, and commercial terms",
            "Current Readiness": "Partially available",
        },
        {
            "Capability": "True Price Elasticity Modelling",
            "Additional Data Needed": "Historical realized prices, quantities, and demand",
            "Current Readiness": "Not available",
        },
        {
            "Capability": "Demand Forecasting",
            "Additional Data Needed": "Historical sales, volume, seasonality, and demand drivers",
            "Current Readiness": "Not available",
        },
        {
            "Capability": "Actual Gross or Pocket Margin",
            "Additional Data Needed": "Product costs, rebates, freight, and cost-to-serve",
            "Current Readiness": "Not available",
        },
        {
            "Capability": "Competitor Price Intelligence",
            "Additional Data Needed": "Competitor pricing and market reference data",
            "Current Readiness": "Not available",
        },
        {
            "Capability": "Quote Win/Loss Analysis",
            "Additional Data Needed": "Quote history, quoted price, outcome, and competitor context",
            "Current Readiness": "Not available",
        },
    ])

    st.dataframe(
        readiness,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("Recommended Data Enrichment")

    st.markdown("""
    **Priority 1: Transaction and sales data**

    - Invoice-level realized selling prices
    - Sales quantities
    - Invoice dates
    - Customer and material identifiers
    - Currency and unit of measure

    **Priority 2: Margin data**

    - Standard and actual product cost
    - Rebates and discounts
    - Freight and logistics
    - Cost-to-serve

    **Priority 3: Commercial intelligence**

    - Quote history
    - Quote win/loss outcomes
    - Competitor pricing
    - Contract and volume-tier terms
    - Customer segmentation

    These datasets will enable more reliable elasticity modelling,
    margin analysis, pricing simulation, and pricing recommendations.
    """)

    st.caption(
        "Readiness assessments reflect the fields available in the "
        "current agreement extracts, not the full SAP environment."
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "PriceIQ | Customer Proof of Concept"
)

st.sidebar.caption(
    "Pricing Intelligence & Margin Governance"
)


# ============================================================
# UPLOADED DATA PREVIEW
# ============================================================

elif page == "Uploaded Data Preview":

    show_page_header(
        "Uploaded Data Preview",
        "Inspect the normalized agreement records currently loaded in this session.",
    )

    st.caption(
        "Uploaded files are held in Streamlit session memory only. "
        "They are not written to GitHub or persisted by this application."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Loaded rows", f"{len(raw_df):,}")
    c2.metric("Customers", f"{raw_df['customer'].nunique():,}")
    c3.metric("Materials", f"{raw_df['material'].nunique():,}")

    if "source_file" in raw_df.columns:
        st.subheader("Source files")
        source_summary = (
            raw_df.groupby("source_file", dropna=False)
            .size()
            .reset_index(name="Rows")
            .sort_values("Rows", ascending=False)
        )
        st.dataframe(source_summary, hide_index=True, use_container_width=True)

    st.subheader("Normalized data")
    st.dataframe(raw_df.head(500), hide_index=True, use_container_width=True)

    st.subheader("Column coverage")
    coverage = pd.DataFrame({
        "Column": raw_df.columns,
        "Non-null rows": [int(raw_df[col].notna().sum()) for col in raw_df.columns],
        "Missing rows": [int(raw_df[col].isna().sum()) for col in raw_df.columns],
        "Data type": [str(raw_df[col].dtype) for col in raw_df.columns],
    })
    st.dataframe(coverage, hide_index=True, use_container_width=True)

    display_download_button(
        raw_df,
        "priceiq_normalized_agreements.csv",
        "Download normalized data",
    )
