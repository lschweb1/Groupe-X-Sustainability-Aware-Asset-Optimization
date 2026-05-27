from __future__ import annotations
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MONTHLY_DATA_FILE = "B_EM_Monthly_Data.xlsx"
ANNUAL_DATA_FILE = "C_EM_Annual_Data.xlsx"
BASE_INVESTMENT_SET_FILE = "D_EM_Base_Investment_Set.xlsx"

MIN_MONTHLY_OBSERVATIONS = 36
ESTIMATION_WINDOW_YEARS = 10
FIRST_FORMATION_YEAR = 2013
LAST_FORMATION_YEAR = 2024

OUTPUT_FILES = {
    "investment_set": "F_MinVar_2_1_Investment_Set.xlsx",
    "expected_returns": "G_MinVar_2_1_Expected_Returns.xlsx",
    "covariances": "H_MinVar_2_1_Covariance_Matrices.xlsx",
    "summary": "I_MinVar_2_1_Summary.xlsx"}

def log_step(message: str):
    """Print a progress message to the terminal."""
    print(message, flush=True)

def write_excel_with_fallback(df: pd.DataFrame, file_name: str):
    """Write an Excel file, or a _new version if the file is already open."""
    target_path = PROCESSED_DIR / file_name

    try:
        df.to_excel(target_path, index=False)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        df.to_excel(fallback_path, index=False)
        return fallback_path

def write_covariance_workbook_with_fallback(
    covariance_matrices: dict[int, pd.DataFrame],
    file_name: str):
    """Write covariance matrices to a single Excel file with one sheet per formation year."""
    target_path = PROCESSED_DIR / file_name

    def write_workbook(path: Path):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for formation_year, covariance_matrix in covariance_matrices.items():
                covariance_matrix.to_excel(writer, sheet_name=f"Y_{formation_year}")

    try:
        write_workbook(target_path)
        return target_path
    except PermissionError:
        fallback_path = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        write_workbook(fallback_path)
        return fallback_path


def load_part1_outputs():
    """
    Load the three Part I outputs required for Section 2.1.

    File B contains cleaned monthly returns, file C contains annual carbon data,
    and file D already includes the price and stale-price filters.
    """
    monthly_data = pd.read_excel(
        PROCESSED_DIR / MONTHLY_DATA_FILE,
        parse_dates=["Date", "Delisting Date"],
    )
    annual_data = pd.read_excel(
        PROCESSED_DIR / ANNUAL_DATA_FILE,
        parse_dates=["Delisting Date"],
    )
    base_investment_set = pd.read_excel(
        PROCESSED_DIR / BASE_INVESTMENT_SET_FILE,
        parse_dates=["Delisting Date"],
    )

    monthly_data = monthly_data.rename(columns={
        "ISIN": "isin",
        "Company Name": "company_name",
        "Country": "country",
        "Region": "region",
        "Delisting Date": "delisting_date",
        "Date": "date",
        "Market Value MUSD": "market_value_musd",
        "Return Index": "return_index",
        "Monthly Return": "monthly_return",
        "Is Delisting Month": "is_delisting_month",
    })

    annual_data = annual_data.rename(columns={
        "ISIN": "isin",
        "Company Name": "company_name",
        "Country": "country",
        "Region": "region",
        "Delisting Date": "delisting_date",
        "Year": "year",
        "Scope 1 CO2": "scope1_co2",
        "Revenue Thousand USD": "revenue_thousand_usd",
        "Year End Market Value MUSD": "year_end_market_value_musd",
        "Year End Return Index": "year_end_return_index",
        "Price Available End Of Year": "price_available_eoy",
    })

    base_investment_set = base_investment_set.rename(columns={
        "ISIN": "isin",
        "Company Name": "company_name",
        "Country": "country",
        "Region": "region",
        "Delisting Date": "delisting_date",
        "Formation Year": "formation_year",
        "Investment Year": "investment_year",
        "Year End Market Value MUSD": "year_end_market_value_musd",
        "Year End Return Index": "year_end_return_index",
        "Price Available End Of Year": "price_available_eoy",
        "Valid Return Count 10Y": "valid_return_count_10y",
        "Zero Return Count 10Y": "zero_return_count_10y",
        "Zero Return Ratio 10Y": "zero_return_ratio_10y",
        "Stale Price Flag": "stale_price_flag",
        "Base Investable Next Year": "base_investable_next_year",
    })

    for df in [monthly_data, annual_data, base_investment_set]:
        df["isin"] = df["isin"].astype(str).str.strip()

    return monthly_data, annual_data, base_investment_set


def build_min_var_investment_set(
    annual_data: pd.DataFrame,
    base_investment_set: pd.DataFrame):
    """
    Build the investment set for Section 2.1.

    At the end of year Y, a firm is kept if it passes the Part I filters,
    has at least 36 monthly return observations over the previous 10 years,
    and has usable carbon data at the end of year Y.
    """
    carbon_data = annual_data[
        ["isin", "year", "scope1_co2", "revenue_thousand_usd"]
    ].copy()

    carbon_data["has_carbon_data"] = (
        carbon_data["scope1_co2"].notna()
        & carbon_data["revenue_thousand_usd"].notna())

    investment_set = base_investment_set.merge(
        carbon_data,
        left_on=["isin", "formation_year"],
        right_on=["isin", "year"],
        how="left")

    investment_set["enough_return_observations"] = (
        investment_set["valid_return_count_10y"] >= MIN_MONTHLY_OBSERVATIONS)

    investment_set["min_var_eligible"] = (
        investment_set["base_investable_next_year"]
        & investment_set["enough_return_observations"]
        & investment_set["has_carbon_data"])

    investment_set = investment_set.loc[
        investment_set["formation_year"].between(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR)].copy()

    investment_set = investment_set.sort_values(
        ["formation_year", "isin"]).reset_index(drop=True)

    selected_columns = [
        "isin",
        "company_name",
        "country",
        "region",
        "delisting_date",
        "formation_year",
        "investment_year",
        "year_end_market_value_musd",
        "year_end_return_index",
        "price_available_eoy",
        "valid_return_count_10y",
        "zero_return_count_10y",
        "zero_return_ratio_10y",
        "stale_price_flag",
        "base_investable_next_year",
        "scope1_co2",
        "revenue_thousand_usd",
        "has_carbon_data",
        "enough_return_observations",
        "min_var_eligible"]

    return investment_set[selected_columns]


def compute_expected_returns(
    monthly_data: pd.DataFrame,
    investment_set: pd.DataFrame,
):
    """
    Compute expected monthly returns for each formation year.

    The estimation window runs from January Y-9 to December Y.
    Only eligible firms are kept, and expected returns are computed
    as the mean of available monthly returns.
    """
    expected_returns_tables: list[pd.DataFrame] = []

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Minimum Variance 2.1 - Processing formation year {formation_year}...")
        eligible_isins = investment_set.loc[
            (investment_set["formation_year"] == formation_year)
            & (investment_set["min_var_eligible"]),
            "isin",
        ].tolist()

        if not eligible_isins:
            continue

        window_start = pd.Timestamp(formation_year - ESTIMATION_WINDOW_YEARS + 1, 1, 1)
        window_end = pd.Timestamp(formation_year, 12, 31)

        window_data = monthly_data.loc[
            (monthly_data["date"] >= window_start)
            & (monthly_data["date"] <= window_end)
            & (monthly_data["isin"].isin(eligible_isins)),
            ["isin", "monthly_return"],
        ].copy()

        mean_returns = (
            window_data.groupby("isin")["monthly_return"]
            .agg(
                used_monthly_observations=lambda values: int(values.notna().sum()),
                mean_monthly_return="mean",
            )
            .reset_index()
        )
        mean_returns["formation_year"] = formation_year
        mean_returns["investment_year"] = formation_year + 1
        expected_returns_tables.append(mean_returns)

    expected_returns = pd.concat(expected_returns_tables, ignore_index=True)
    expected_returns = expected_returns.merge(
        investment_set[
            ["isin", "company_name", "country", "formation_year", "investment_year"]
        ].drop_duplicates(),
        on=["isin", "formation_year", "investment_year"],
        how="left",
    )

    expected_returns = expected_returns[
        [
            "isin",
            "company_name",
            "country",
            "formation_year",
            "investment_year",
            "used_monthly_observations",
            "mean_monthly_return",
        ]
    ]
    expected_returns = expected_returns.sort_values(["formation_year", "isin"]).reset_index(drop=True)
    return expected_returns


def compute_covariance_matrices(
    monthly_data: pd.DataFrame,
    investment_set: pd.DataFrame,
):
    """
    Compute the monthly return covariance matrix for each formation year.
    The same 10-year estimation window is used as for expected returns.
    """
    covariance_matrices: dict[int, pd.DataFrame] = {}

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Minimum Variance 2.1 - Computing covariance for formation year {formation_year}...")
        eligible_isins = investment_set.loc[
            (investment_set["formation_year"] == formation_year)
            & (investment_set["min_var_eligible"]),
            "isin",
        ].tolist()

        if not eligible_isins:
            continue

        window_start = pd.Timestamp(formation_year - ESTIMATION_WINDOW_YEARS + 1, 1, 1)
        window_end = pd.Timestamp(formation_year, 12, 31)

        return_matrix = monthly_data.loc[
            (monthly_data["date"] >= window_start)
            & (monthly_data["date"] <= window_end)
            & (monthly_data["isin"].isin(eligible_isins)),
            ["date", "isin", "monthly_return"],
        ].pivot(index="date", columns="isin", values="monthly_return")

        return_matrix = return_matrix.reindex(columns=eligible_isins)
        covariance_matrices[formation_year] = return_matrix.cov(min_periods=MIN_MONTHLY_OBSERVATIONS)

    return covariance_matrices


def build_summary_table(
    investment_set: pd.DataFrame,
    expected_returns: pd.DataFrame,
    covariance_matrices: dict[int, pd.DataFrame],
):
    """Build a summary table to check the Section 2.1 outputs"""
    summary_rows: list[dict[str, object]] = []

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        eligible_firms = int(
            investment_set.loc[
                (investment_set["formation_year"] == formation_year)
                & (investment_set["min_var_eligible"]),
                "isin",
            ].nunique()
        )
        expected_return_vectors = int(
            expected_returns.loc[expected_returns["formation_year"] == formation_year, "isin"].nunique()
        )
        covariance_matrix_size = 0
        if formation_year in covariance_matrices:
            covariance_matrix_size = covariance_matrices[formation_year].shape[0]

        summary_rows.append(
            {
                "formation_year": formation_year,
                "investment_year": formation_year + 1,
                "eligible_firms": eligible_firms,
                "expected_return_vectors": expected_return_vectors,
                "covariance_matrix_size": covariance_matrix_size,
            }
        )

    return pd.DataFrame(summary_rows)

def save_outputs(
    investment_set: pd.DataFrame,
    expected_returns: pd.DataFrame,
    covariance_matrices: dict[int, pd.DataFrame],
    summary_table: pd.DataFrame):
    """Save the final outputs of Section 2.1"""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written_files = {
        "F": str(write_excel_with_fallback(investment_set, OUTPUT_FILES["investment_set"])),
        "G": str(write_excel_with_fallback(expected_returns, OUTPUT_FILES["expected_returns"])),
        "H": str(write_covariance_workbook_with_fallback(covariance_matrices, OUTPUT_FILES["covariances"])),
        "I": str(write_excel_with_fallback(summary_table, OUTPUT_FILES["summary"]))}
    
    return written_files


def main():
    log_step("  Minimum Variance 2.1 1/4 - Loading the required Part I outputs...")
    monthly_data, annual_data, base_investment_set = load_part1_outputs()

    log_step("  Minimum Variance 2.1 2/4 - Building the minimum-variance investment set...")
    investment_set = build_min_var_investment_set(
        annual_data=annual_data,
        base_investment_set=base_investment_set)

    log_step("  Minimum Variance 2.1 3/4 - Computing expected returns and covariance matrices...")
    expected_returns = compute_expected_returns(
        monthly_data=monthly_data,
        investment_set=investment_set,)
    
    covariance_matrices = compute_covariance_matrices(
        monthly_data=monthly_data,
        investment_set=investment_set)

    log_step("  Minimum Variance 2.1 4/4 - Saving the Excel outputs...")
    summary_table = build_summary_table(
        investment_set=investment_set,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices)
    
    written_files = save_outputs(
        investment_set=investment_set,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
        summary_table=summary_table)

    log_step("  Part 2.1 completed.")
    log_step(f"  Final investment set rows: {len(investment_set)}")
    log_step(f"  Expected return rows: {len(expected_returns)}")
    log_step("  Files written:")
    for label, path in written_files.items():
        log_step(f"{label} : {path}")

if __name__ == "__main__":
    main()
