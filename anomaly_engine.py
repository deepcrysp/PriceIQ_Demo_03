
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def _robust_z_score(values):
    """Calculate robust z-scores using median absolute deviation."""
    values = pd.to_numeric(values, errors="coerce")
    median = values.median()
    mad = (values - median).abs().median()

    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=values.index)

    return 0.6745 * (values - median) / mad


def detect_anomalies(
    df,
    value_col="agreement_value",
    material_col="material",
    unit_col="unit",
    customer_col="customer",
    date_from_col="valid_from",
    date_to_col="valid_to",
    contamination=0.03
):
    """
    Detect unusual SAP agreement patterns.

    Results are review flags, not confirmed pricing violations.
    """
    result = df.copy()

    if value_col not in result.columns:
        result["anomaly_score"] = 0.0
        result["anomaly_type"] = "Not evaluated"
        result["severity"] = "Not evaluated"
        result["anomaly_reason"] = "Agreement value field unavailable."
        return result

    result[value_col] = pd.to_numeric(
        result[value_col], errors="coerce"
    )

    result["anomaly_score"] = 0.0
    result["anomaly_type"] = ""
    result["severity"] = "Normal"
    result["anomaly_reason"] = ""

    valid = result[value_col].notna() & (result[value_col] > 0)

    # 1. Material and unit level price dispersion
    group_cols = [
        col for col in [material_col, unit_col]
        if col in result.columns
    ]

    if group_cols:
        grouped = result.loc[valid].groupby(
            group_cols, dropna=False
        )[value_col]

        group_stats = grouped.agg(
            group_median="median",
            group_count="count"
        ).reset_index()

        result = result.merge(
            group_stats,
            on=group_cols,
            how="left"
        )

        result["relative_to_median_pct"] = np.where(
            result["group_median"].notna()
            & (result["group_median"] != 0),
            (
                result[value_col] / result["group_median"] - 1
            ) * 100,
            np.nan
        )

        # Flag only groups with enough comparison records
        dispersion_flag = (
            result["group_count"].fillna(0).ge(5)
            & result["relative_to_median_pct"].abs().ge(50)
            & valid
        )

        result.loc[
            dispersion_flag, "anomaly_type"
        ] = "Material price dispersion"

        result.loc[
            dispersion_flag, "anomaly_reason"
        ] = (
            "Agreement value differs by at least 50% "
            "from the median for the same material and unit."
        )

    else:
        result["relative_to_median_pct"] = np.nan
        result["group_count"] = 0

    # 2. Isolation Forest on agreement value and log value
    model_data = result.loc[valid, [value_col]].copy()

    if len(model_data) >= 30:
        model_data["log_value"] = np.log1p(
            model_data[value_col].clip(lower=0)
        )

        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42
        )

        predictions = model.fit_predict(
            model_data[["log_value"]]
        )

        scores = -model.decision_function(
            model_data[["log_value"]]
        )

        result.loc[
            model_data.index, "anomaly_score"
        ] = scores

        isolation_flag = pd.Series(
            predictions == -1,
            index=model_data.index
        )

        isolation_flag = isolation_flag.reindex(
            result.index, fill_value=False
        )

        # Avoid treating every statistical outlier as a violation.
        result.loc[
            isolation_flag & (result["anomaly_type"] == ""),
            "anomaly_type"
        ] = "Unusual agreement value"

        result.loc[
            isolation_flag & (result["anomaly_reason"] == ""),
            "anomaly_reason"
        ] = (
            "Machine-learning model identified an unusual "
            "agreement value relative to the overall dataset."
        )

    # 3. Duplicate agreement patterns
    duplicate_cols = [
        col for col in [
            customer_col, material_col, unit_col, value_col
        ]
        if col in result.columns
    ]

    if len(duplicate_cols) >= 3:
        duplicate_flag = result.duplicated(
            subset=duplicate_cols, keep=False
        )

        result.loc[
            duplicate_flag, "anomaly_type"
        ] = result.loc[
            duplicate_flag, "anomaly_type"
        ].apply(
            lambda x: (
                x + "; Duplicate pattern"
                if x else "Duplicate pattern"
            )
        )

        result.loc[
            duplicate_flag, "anomaly_reason"
        ] = result.loc[
            duplicate_flag, "anomaly_reason"
        ].apply(
            lambda x: (
                x + " Similar customer, material, unit and value "
                "records exist."
                if x else
                "Similar customer, material, unit and value records exist."
            )
        )

    # 4. Overlapping agreement validity periods
    if (
        customer_col in result.columns
        and material_col in result.columns
        and date_from_col in result.columns
        and date_to_col in result.columns
    ):
        result[date_from_col] = pd.to_datetime(
            result[date_from_col], errors="coerce"
        )
        result[date_to_col] = pd.to_datetime(
            result[date_to_col], errors="coerce"
        )

        overlap_flag = pd.Series(
            False, index=result.index
        )

        overlap_groups = [
            customer_col, material_col
        ]

        if unit_col in result.columns:
            overlap_groups.append(unit_col)

        usable = result.dropna(
            subset=[date_from_col, date_to_col]
        ).sort_values(date_from_col)

        for _, group in usable.groupby(
            overlap_groups, dropna=False
        ):
            previous_end = None

            for idx, row in group.iterrows():
                start = row[date_from_col]
                end = row[date_to_col]

                if previous_end is not None and start <= previous_end:
                    overlap_flag.loc[idx] = True

                if previous_end is None or end > previous_end:
                    previous_end = end

        result.loc[
            overlap_flag, "anomaly_type"
        ] = result.loc[
            overlap_flag, "anomaly_type"
        ].apply(
            lambda x: (
                x + "; Overlapping validity"
                if x else "Overlapping validity"
            )
        )

        result.loc[
            overlap_flag, "anomaly_reason"
        ] = result.loc[
            overlap_flag, "anomaly_reason"
        ].apply(
            lambda x: (
                x + " Another agreement for the same customer "
                "and material has an overlapping validity period."
                if x else
                "Another agreement for the same customer and material "
                "has an overlapping validity period."
            )
        )

    # 5. Severity classification
    flagged = result["anomaly_type"].ne("")

    result.loc[flagged, "severity"] = "Medium"

    high_flag = (
        result["relative_to_median_pct"].abs().ge(100)
        if "relative_to_median_pct" in result.columns
        else pd.Series(False, index=result.index)
    )

    high_flag = high_flag.fillna(False)

    result.loc[flagged & high_flag, "severity"] = "High"

    # Multiple independent flags increase review priority
    multiple_flags = (
        result["anomaly_type"].str.count(";") >= 1
    )

    result.loc[
        flagged & multiple_flags, "severity"
    ] = "High"

    result.loc[
        ~flagged, "anomaly_reason"
    ] = "No anomaly detected by the configured checks."

    result["is_anomaly"] = flagged

    return result
