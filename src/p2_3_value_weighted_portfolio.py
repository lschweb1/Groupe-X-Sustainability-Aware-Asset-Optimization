from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "Raw"

MONTHLY_DATA_FILE = "B_EM_Monthly_Data.xlsx"
INVESTMENT_SET_FILE = "F_MinVar_2_1_Investment_Set.xlsx"
RISK_FREE_FILE = "Risk_Free_Rate_2025.xlsx"
MONTHLY_MARKET_CAP_FILE = "DS_MV_T_USD_M_2025.xlsx"

FIRST_FORMATION_YEAR = 2013
LAST_FORMATION_YEAR = 2024

OUTPUT_FILES = {
    "weights": "M_ValueWeighted_2_3_Monthly_Weights.xlsx",
    "monthly_returns": "N_ValueWeighted_2_3_Monthly_Performance.xlsx",
    "summary": "O_ValueWeighted_2_3_Summary.xlsx"}


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


def load_monthly_market_cap_long():
    """
    Load the raw monthly market capitalization file and convert it to long format.
    """
    market_cap_raw = pd.read_excel(RAW_DIR / MONTHLY_MARKET_CAP_FILE)
    market_cap_raw = market_cap_raw.loc[market_cap_raw["ISIN"].notna()].copy()

    id_columns = ["NAME", "ISIN"]
    date_columns = [column for column in market_cap_raw.columns if column not in id_columns]

    market_cap_long = market_cap_raw.melt(
        id_vars=id_columns,
        value_vars=date_columns,
        var_name="date",
        value_name="market_cap",
    )

    market_cap_long = market_cap_long.rename(columns={"ISIN": "isin"})
    market_cap_long["isin"] = market_cap_long["isin"].astype(str).str.strip()
    market_cap_long["date"] = pd.to_datetime(market_cap_long["date"], errors="coerce")
    market_cap_long["date"] = market_cap_long["date"] + pd.offsets.MonthEnd(0)
    market_cap_long["market_cap"] = pd.to_numeric(market_cap_long["market_cap"], errors="coerce")
    market_cap_long.loc[market_cap_long["market_cap"] <= 0, "market_cap"] = np.nan

    return market_cap_long[["isin", "date", "market_cap"]].drop_duplicates()


def merge_market_cap_into_monthly_data(monthly_data: pd.DataFrame):
    """
    Merge monthly market capitalization into the cleaned monthly dataset.
    """
    market_cap_long = load_monthly_market_cap_long()

    monthly_data = monthly_data.copy()
    monthly_data["isin"] = monthly_data["isin"].astype(str).str.strip()
    monthly_data["date"] = pd.to_datetime(monthly_data["date"], errors="coerce")
    monthly_data["date"] = monthly_data["date"] + pd.offsets.MonthEnd(0)

    return monthly_data.merge(
        market_cap_long,
        on=["isin", "date"],
        how="left",
    )


def load_inputs():
    monthly_data = pd.read_excel(
        PROCESSED_DIR / MONTHLY_DATA_FILE,
        parse_dates=["Date", "Delisting Date"],
    )
    investment_set = pd.read_excel(
        PROCESSED_DIR / INVESTMENT_SET_FILE
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

    monthly_data = merge_market_cap_into_monthly_data(monthly_data)
    for df in [monthly_data, investment_set]:
        df["isin"] = df["isin"].astype(str).str.strip()

    risk_free_rate = pd.read_excel(RAW_DIR / RISK_FREE_FILE)
    first_column = risk_free_rate.columns[0]
    risk_free_rate = risk_free_rate.rename(columns={first_column: "yyyymm", "RF": "rf_percent"})
    risk_free_rate["yyyymm"] = risk_free_rate["yyyymm"].astype(str).str.strip()
    risk_free_rate["date"] = pd.to_datetime(risk_free_rate["yyyymm"] + "01", format="%Y%m%d")
    risk_free_rate["date"] = risk_free_rate["date"] + pd.offsets.MonthEnd(0)
    risk_free_rate["rf_decimal"] = pd.to_numeric(risk_free_rate["rf_percent"], errors="coerce") / 100
    risk_free_rate = risk_free_rate[["date", "rf_decimal"]].copy()

    return monthly_data, investment_set, risk_free_rate


def build_monthly_return_matrix(monthly_data: pd.DataFrame):
    return_matrix = monthly_data.pivot(index="date", columns="isin", values="monthly_return")
    return_matrix.columns = return_matrix.columns.astype(str)
    return return_matrix.sort_index()


def build_monthly_market_cap_matrix(monthly_data: pd.DataFrame):
    if "market_cap" not in monthly_data.columns:
        raise KeyError(
            "The 'market_cap' column is missing from B_EM_Monthly_Data.xlsx. "
            "DS_MV_T_USD_M_2025.xlsx must be merged into monthly_data first."
        )

    market_cap_matrix = monthly_data.pivot(index="date", columns="isin", values="market_cap")
    market_cap_matrix.columns = market_cap_matrix.columns.astype(str)
    market_cap_matrix = market_cap_matrix.sort_index()
    return market_cap_matrix.where(market_cap_matrix > 0)


def compute_value_weighted_performance(
    return_matrix: pd.DataFrame,
    market_cap_matrix: pd.DataFrame,
    investment_set: pd.DataFrame,
):
    """
    Poids au mois t = market cap relative à la fin du mois t
    Rendement du portefeuille au mois t+1 = somme(w_t * R_{t+1})
    """
    portfolio_rows: list[dict[str, object]] = []
    weight_rows: list[pd.DataFrame] = []
    previous_date_map = pd.Series(
        return_matrix.index[:-1],
        index=return_matrix.index[1:],
    )

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Value-Weighted 2.3 - Processing formation year {formation_year}...")
        year_investment_set = investment_set.loc[
            (investment_set["formation_year"] == formation_year)
            & (investment_set["min_var_eligible"]),
        ].copy()

        if year_investment_set.empty:
            continue

        eligible_isins = year_investment_set["isin"].astype(str).tolist()
        eligible_isins = [
            isin for isin in eligible_isins
            if isin in return_matrix.columns and isin in market_cap_matrix.columns
        ]

        if not eligible_isins:
            continue

        company_info = year_investment_set[
            ["isin", "company_name", "country", "region"]
        ].drop_duplicates()

        investment_year = formation_year + 1
        investment_dates = pd.date_range(
            start=pd.Timestamp(investment_year, 1, 31),
            end=pd.Timestamp(investment_year, 12, 31),
            freq="ME",
        )

        for actual_date in investment_dates:
            if actual_date not in return_matrix.index:
                continue

            previous_date = previous_date_map.get(actual_date)
            if pd.isna(previous_date):
                continue

            month_returns = return_matrix.loc[actual_date, eligible_isins].copy()
            lagged_caps = market_cap_matrix.loc[previous_date, eligible_isins].copy()

            valid_assets = lagged_caps.notna() & (lagged_caps > 0)
            lagged_caps = lagged_caps.loc[valid_assets]
            month_returns = month_returns.loc[lagged_caps.index]

            if lagged_caps.empty:
                continue

            month_returns = month_returns.fillna(0.0)

            weights = lagged_caps / lagged_caps.sum()
            portfolio_return = float((weights * month_returns).sum())

            portfolio_rows.append(
                {
                    "date": actual_date,
                    "formation_year": formation_year,
                    "investment_year": investment_year,
                    "rebalance_reference_date": previous_date,
                    "portfolio_return": portfolio_return,
                }
            )

            weights_df = weights.reset_index()
            weights_df.columns = ["isin", "weight"]
            weights_df["date"] = actual_date
            weights_df["rebalance_reference_date"] = previous_date
            weights_df["formation_year"] = formation_year
            weights_df["investment_year"] = investment_year

            weights_df = weights_df.merge(company_info, on="isin", how="left")

            weights_df = weights_df[
                [
                    "date",
                    "rebalance_reference_date",
                    "isin",
                    "company_name",
                    "country",
                    "region",
                    "formation_year",
                    "investment_year",
                    "weight",
                ]
            ]

            weight_rows.append(weights_df)

    portfolio_returns = pd.DataFrame(portfolio_rows).sort_values("date").reset_index(drop=True)
    portfolio_returns["cumulative_growth"] = (1 + portfolio_returns["portfolio_return"]).cumprod()

    monthly_weights = pd.concat(weight_rows, ignore_index=True)
    monthly_weights = monthly_weights.sort_values(["date", "weight"], ascending=[True, False]).reset_index(drop=True)

    return portfolio_returns, monthly_weights


def compute_summary_statistics(
    portfolio_returns: pd.DataFrame,
    risk_free_rate: pd.DataFrame,
):
    merged = portfolio_returns.merge(risk_free_rate, on="date", how="left")
    merged["excess_return"] = merged["portfolio_return"] - merged["rf_decimal"]

    annualized_average_return = merged["portfolio_return"].mean() * 12
    annualized_volatility = merged["portfolio_return"].std(ddof=1) * np.sqrt(12)

    if annualized_volatility == 0 or pd.isna(annualized_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (merged["excess_return"].mean() * 12) / annualized_volatility

    return pd.DataFrame(
        [
            {"metric": "monthly_observations", "value": len(merged)},
            {"metric": "annualized_average_return", "value": annualized_average_return},
            {"metric": "annualized_volatility", "value": annualized_volatility},
            {"metric": "sharpe_ratio", "value": sharpe_ratio},
            {"metric": "minimum_monthly_return", "value": merged["portfolio_return"].min()},
            {"metric": "maximum_monthly_return", "value": merged["portfolio_return"].max()},
            {
                "metric": "final_cumulative_growth",
                "value": merged["cumulative_growth"].iloc[-1] if not merged.empty else np.nan,
            },
        ]
    )


def save_outputs(
    monthly_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    summary_table: pd.DataFrame,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written_files = {
        "M": str(write_excel_with_fallback(monthly_weights, OUTPUT_FILES["weights"])),
        "N": str(write_excel_with_fallback(portfolio_returns, OUTPUT_FILES["monthly_returns"])),
        "O": str(write_excel_with_fallback(summary_table, OUTPUT_FILES["summary"])),
    }
    return written_files


def main():
    log_step("  Value-Weighted 2.3 1/4 - Loading the data and adding monthly market capitalization...")
    monthly_data, investment_set, risk_free_rate = load_inputs()

    log_step("  Value-Weighted 2.3 2/4 - Building the monthly matrices...")
    return_matrix = build_monthly_return_matrix(monthly_data)
    market_cap_matrix = build_monthly_market_cap_matrix(monthly_data)

    log_step("  Value-Weighted 2.3 3/4 - Computing the value-weighted portfolio...")
    portfolio_returns, monthly_weights = compute_value_weighted_performance(
        return_matrix=return_matrix,
        market_cap_matrix=market_cap_matrix,
        investment_set=investment_set,
    )

    log_step("  Value-Weighted 2.3 4/4 - Computing summary statistics and saving outputs...")
    summary_table = compute_summary_statistics(
        portfolio_returns=portfolio_returns,
        risk_free_rate=risk_free_rate,
    )
    written_files = save_outputs(
        monthly_weights=monthly_weights,
        portfolio_returns=portfolio_returns,
        summary_table=summary_table,
    )

    log_step("  Part 2.3 completed.")
    log_step(f"  Monthly VW return rows: {len(portfolio_returns)}")
    log_step(f"  Monthly VW weight rows: {len(monthly_weights)}")
    for label, path in written_files.items():
        log_step(f"{label} : {path}")


if __name__ == "__main__":
    main()
