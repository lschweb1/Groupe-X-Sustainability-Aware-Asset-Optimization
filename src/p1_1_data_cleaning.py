from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "Raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

STATIC_FILE = "Static_2025.xlsx"
SCOPE1_FILE = "DS_CO2_SCOPE_1_Y_2025.xlsx"
REVENUE_FILE = "DS_REV_Y_2025.xlsx"
MARKET_VALUE_MONTHLY_FILE = "DS_MV_T_USD_M_2025.xlsx"
RETURN_INDEX_MONTHLY_FILE = "DS_RI_T_USD_M_2025.xlsx"

# Rules
LOW_PRICE_THRESHOLD = 0.5
STALE_PRICE_THRESHOLD = 0.5
ESTIMATION_WINDOW_YEARS = 10
FIRST_FORMATION_YEAR = 2013
LAST_FORMATION_YEAR = 2024
 
OUTPUT_FILES = {
    "companies": "A_EM_Companies.xlsx",
    "monthly_data": "B_EM_Monthly_Data.xlsx",
    "annual_data": "C_EM_Annual_Data.xlsx",
    "base_investment_set": "D_EM_Base_Investment_Set.xlsx"}

EXPORT_COLUMN_NAMES = {
    "isin": "ISIN",
    "company_name": "Company Name",
    "country": "Country",
    "region": "Region",
    "delisting_date": "Delisting Date",
    "date": "Date",
    "market_value_musd": "Market Value MUSD",
    "return_index": "Return Index",
    "monthly_return": "Monthly Return",
    "is_delisting_month": "Is Delisting Month",
    "year": "Year",
    "scope1_co2": "Scope 1 CO2",
    "revenue_thousand_usd": "Revenue Thousand USD",
    "year_end_market_value_musd": "Year End Market Value MUSD",
    "year_end_return_index": "Year End Return Index",
    "price_available_eoy": "Price Available End Of Year",
    "formation_year": "Formation Year",
    "investment_year": "Investment Year",
    "valid_return_count_10y": "Valid Return Count 10Y",
    "zero_return_count_10y": "Zero Return Count 10Y",
    "zero_return_ratio_10y": "Zero Return Ratio 10Y",
    "stale_price_flag": "Stale Price Flag",
    "base_investable_next_year": "Base Investable Next Year"}

def log_step(message: str):
    """Print a progress message to the terminal"""
    print(message, flush=True)

def write_excel(df: pd.DataFrame, file_name: str):
    """Write a DataFrame to the processed folder"""
    target_path = PROCESSED_DIR / file_name
    df.to_excel(target_path, index=False)
    return target_path


def rename_columns_for_export(df: pd.DataFrame):
    renamed_df = df.copy()  
    renamed_df = renamed_df.rename(columns=EXPORT_COLUMN_NAMES)
    return renamed_df


def extract_delisting_date(company_name: str):
    """
    Extract a delisting date from the Datastream company name.

    Datastream sometimes appends labels such as:
    DEAD - DELIST.23/09/25
    """
    if not isinstance(company_name, str):
        return pd.NaT 

    marker = "DEAD - DELIST."

    if marker not in company_name:  
        return pd.NaT  

    date_text = company_name.split(marker, 1)[1][:8]
    return pd.to_datetime(date_text, format="%d/%m/%y", errors="coerce")  


def load_datastream_file(file_name: str):
    """
    Load a raw Datastream export and apply basic cleaning.

    Useful columns are renamed, rows without ISIN are removed,
    and value columns are converted to numeric.
    """
    df = pd.read_excel(RAW_DIR / file_name)
    df = df.rename(columns={"NAME": "company_name_raw", "ISIN": "isin"})
    df = df.dropna(subset=["isin"]).copy()
    df["isin"] = df["isin"].astype(str).str.strip()

    value_columns = [col for col in df.columns if col not in ["company_name_raw", "isin"]]
    for column in value_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df



def load_em_companies():
    static_df = pd.read_excel(RAW_DIR / STATIC_FILE)
    static_df = static_df.rename(
        columns={
            "ISIN": "isin",
            "NAME": "company_name",
            "Country": "country",
            "Region": "region"})

    static_df = static_df.dropna(subset=["isin"]).drop_duplicates(subset="isin")
    static_df["isin"] = static_df["isin"].astype(str).str.strip()
    static_df["region"] = static_df["region"].astype(str).str.strip() 
    static_df["delisting_date"] = static_df["company_name"].apply(extract_delisting_date)

    em_companies = static_df.loc[static_df["region"] == "EM"].copy()
    em_companies = em_companies.sort_values("isin").reset_index(drop=True)
    return em_companies


def keep_only_common_isins(em_companies: pd.DataFrame):
    """Keep only ISINs that are present in all required source tables."""
    common_isins = set(em_companies["isin"])

    for file_name in [SCOPE1_FILE, REVENUE_FILE, MARKET_VALUE_MONTHLY_FILE, RETURN_INDEX_MONTHLY_FILE]:
        file_df = load_datastream_file(file_name)
        common_isins &= set(file_df["isin"])

    filtered_companies = em_companies.loc[em_companies["isin"].isin(common_isins)].copy()
    filtered_companies = filtered_companies.sort_values("isin").reset_index(drop=True)

    return filtered_companies


def fill_annual_missing_with_previous_value(
    wide_df: pd.DataFrame,
    year_columns: list[int]):
    """
    Forward-fill missing annual values after the first observed value.
    Missing values at the beginning of the series are left unchanged.
    """
    cleaned_df = wide_df.copy() 
    filled_cells = 0  

    for row_index in cleaned_df.index:  
        previous_value = pd.NA  
        seen_first_value = False  

        for year in year_columns: 
            current_value = cleaned_df.at[row_index, year] 

            if pd.notna(current_value):  
                previous_value = current_value  
                seen_first_value = True  
                continue 

            if seen_first_value and pd.notna(previous_value): 
                cleaned_df.at[row_index, year] = previous_value  
                filled_cells += 1  

    return cleaned_df, filled_cells  


def find_matching_month_column(
    date_columns: list[pd.Timestamp | datetime],
    delisting_date: pd.Timestamp):
    """Return the last available market date in the delisting month"""

    matching_columns = [
        pd.Timestamp(column)
        for column in date_columns
        if pd.Timestamp(column).year == delisting_date.year
        and pd.Timestamp(column).month == delisting_date.month]

    if not matching_columns:
        return None

    return max(matching_columns)


def build_monthly_data(em_companies: pd.DataFrame):
    """
    Clean monthly price data and compute monthly returns.

    The function keeps the selected EM firms, treats return indexes below 0.5
    as missing values, sets the delisting month to zero, removes later months,
    and computes simple monthly returns from the return index.
    """
    market_value_wide = load_datastream_file(MARKET_VALUE_MONTHLY_FILE) 
    return_index_wide = load_datastream_file(RETURN_INDEX_MONTHLY_FILE)

    market_value_wide = em_companies.merge(market_value_wide, on="isin", how="left") 
    return_index_wide = em_companies.merge(return_index_wide, on="isin", how="left")

    market_value_columns = sorted( 
        [column for column in market_value_wide.columns if isinstance(column, (datetime, pd.Timestamp))]) 
    return_index_columns = sorted(
        [column for column in return_index_wide.columns if isinstance(column, (datetime, pd.Timestamp))])

    low_price_mask = return_index_wide[return_index_columns].lt(LOW_PRICE_THRESHOLD) 
    low_price_mask = low_price_mask & return_index_wide[return_index_columns].notna() 
    return_index_wide.loc[:, return_index_columns] = return_index_wide[return_index_columns].mask(low_price_mask) 
    
    for row_index in em_companies.index: 
        delisting_date = em_companies.at[row_index, "delisting_date"]
        if pd.isna(delisting_date):
            continue

        market_value_month = find_matching_month_column(market_value_columns, pd.Timestamp(delisting_date))
        return_index_month = find_matching_month_column(return_index_columns, pd.Timestamp(delisting_date))

        if market_value_month is None or return_index_month is None:
            continue

        later_market_value_columns = [column for column in market_value_columns if pd.Timestamp(column) > market_value_month]
        later_return_index_columns = [column for column in return_index_columns if pd.Timestamp(column) > return_index_month]

        market_value_wide.loc[row_index, later_market_value_columns] = pd.NA
        return_index_wide.loc[row_index, later_return_index_columns] = pd.NA

        market_value_wide.at[row_index, market_value_month] = 0.0
        return_index_wide.at[row_index, return_index_month] = 0.0

    id_columns = ["isin", "company_name", "country", "region", "delisting_date"]


    market_value_long = market_value_wide.melt(
        id_vars=id_columns, value_vars=market_value_columns, var_name="date", value_name="market_value_musd")
    
    return_index_long = return_index_wide.melt(
        id_vars=id_columns, value_vars=return_index_columns, var_name="date", value_name="return_index")

    monthly_data = market_value_long.merge(
        return_index_long[["isin", "date", "return_index"]], on=["isin", "date"], how="outer")


    monthly_data["date"] = pd.to_datetime(monthly_data["date"])
    monthly_data = monthly_data.sort_values(["isin", "date"]).reset_index(drop=True)

    monthly_data["return_index_lag"] = monthly_data.groupby("isin")["return_index"].shift(1)
    monthly_data["monthly_return"] = (monthly_data["return_index"] / monthly_data["return_index_lag"] - 1)

    invalid_return_mask = monthly_data["return_index"].isna() | monthly_data["return_index_lag"].isna()
    monthly_data.loc[invalid_return_mask, "monthly_return"] = pd.NA

    monthly_data["is_delisting_month"] = (
        monthly_data["delisting_date"].notna()
        & (monthly_data["date"].dt.to_period("M") == monthly_data["delisting_date"].dt.to_period("M"))
        & monthly_data["return_index"].eq(0))

    monthly_data = monthly_data[
        [
            "isin",
            "company_name",
            "country",
            "region",
            "delisting_date",
            "date",
        "market_value_musd",
            "return_index",
            "monthly_return",
            "is_delisting_month"]]

    return monthly_data


def build_annual_data(em_companies: pd.DataFrame, monthly_data: pd.DataFrame):
    """
    Build the annual dataset used for the investment universe and carbon analysis.
    The output includes Scope 1 emissions, revenue, and cleaned year-end prices.
    """
    scope1_wide = load_datastream_file(SCOPE1_FILE)
    revenue_wide = load_datastream_file(REVENUE_FILE)

    scope1_wide = scope1_wide.loc[scope1_wide["isin"].isin(em_companies["isin"])].copy()
    revenue_wide = revenue_wide.loc[revenue_wide["isin"].isin(em_companies["isin"])].copy()

    scope1_year_columns = sorted([column for column in scope1_wide.columns if isinstance(column, int)])
    revenue_year_columns = sorted([column for column in revenue_wide.columns if isinstance(column, int)])

    scope1_wide, _ = fill_annual_missing_with_previous_value(scope1_wide, scope1_year_columns)
    revenue_wide, _ = fill_annual_missing_with_previous_value(revenue_wide, revenue_year_columns)

    scope1_long = scope1_wide.melt(
        id_vars=["isin"], value_vars=scope1_year_columns, var_name="year", value_name="scope1_co2")
    
    revenue_long = revenue_wide.melt(
        id_vars=["isin"], value_vars=revenue_year_columns, var_name="year", value_name="revenue_thousand_usd")
    

    scope1_long["year"] = scope1_long["year"].astype(int)
    revenue_long["year"] = revenue_long["year"].astype(int)

    annual_data = em_companies.merge(scope1_long, on="isin", how="left")
    annual_data = annual_data.merge(revenue_long, on=["isin", "year"], how="left")

    year_end_prices = monthly_data.loc[monthly_data["date"].dt.month == 12].copy()
    year_end_prices["year"] = year_end_prices["date"].dt.year
    year_end_prices = year_end_prices.rename(
        columns={
        "market_value_musd": "year_end_market_value_musd",
            "return_index": "year_end_return_index"})
    
    year_end_prices["price_available_eoy"] = year_end_prices["year_end_return_index"].notna()

    annual_data = annual_data.merge(
        year_end_prices[
            [
                "isin",
                "year",
        "year_end_market_value_musd",
                "year_end_return_index",
                "price_available_eoy",
            ]
        ],
        on=["isin", "year"],
        how="left")

    annual_data = annual_data.sort_values(["isin", "year"]).reset_index(drop=True)

    return annual_data


def build_base_investment_set(monthly_data: pd.DataFrame):
    """
    Build the base investment universe for Part 2.1.

    At the end of year Y, a firm is kept if a year-end price is available
    and the share of zero monthly returns over the previous 10 years does not exceed 50%.

    Formation years are limited to 2013 through 2024.
    """ 
    # Keep December observations as year-end snapshots
    year_end_rows = monthly_data.loc[monthly_data["date"].dt.month == 12].copy()

    year_end_rows["formation_year"] = year_end_rows["date"].dt.year
    year_end_rows["investment_year"] = year_end_rows["formation_year"] + 1

    year_end_rows["price_available_eoy"] = year_end_rows["return_index"].notna()

    yearly_results = []

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        # Compute stale-price statistics over the 10-year rolling window.
        window_start = pd.Timestamp(formation_year - ESTIMATION_WINDOW_YEARS + 1, 1, 1)
        window_end = pd.Timestamp(formation_year, 12, 31)

        window_data = monthly_data.loc[
            (monthly_data["date"] >= window_start) & (monthly_data["date"] <= window_end),
            ["isin", "monthly_return"]].copy()

        valid_return_count = (
            window_data.groupby("isin")["monthly_return"]
            .apply(lambda values: int(values.notna().sum()))
            .rename("valid_return_count_10y")
            .reset_index())

        zero_return_count = (
            window_data.groupby("isin")["monthly_return"]
            .apply(lambda values: int((values.eq(0) & values.notna()).sum()))
            .rename("zero_return_count_10y")
            .reset_index())

        stale_stats = valid_return_count.merge(zero_return_count, on="isin", how="left")

        stale_stats["zero_return_ratio_10y"] = (stale_stats["zero_return_count_10y"] / stale_stats["valid_return_count_10y"])

        stale_stats.loc[stale_stats["valid_return_count_10y"] == 0, "zero_return_ratio_10y"] = pd.NA

        stale_stats["stale_price_flag"] = stale_stats["zero_return_ratio_10y"] > STALE_PRICE_THRESHOLD
        stale_stats["stale_price_flag"] = stale_stats["stale_price_flag"].fillna(False)

        year_slice = year_end_rows.loc[year_end_rows["formation_year"] == formation_year].copy()

        year_slice = year_slice.merge(stale_stats, on="isin", how="left")

        year_slice["valid_return_count_10y"] = year_slice["valid_return_count_10y"].fillna(0).astype(int)
        year_slice["zero_return_count_10y"] = year_slice["zero_return_count_10y"].fillna(0).astype(int)
        year_slice["stale_price_flag"] = year_slice["stale_price_flag"].fillna(False)

        year_slice["base_investable_next_year"] = (year_slice["price_available_eoy"] & year_slice["stale_price_flag"].eq(False))

        # Keep only the columns needed for the next steps.
        year_slice = year_slice[
            [
                "isin",
                "company_name",
                "country",
                "region",
                "delisting_date",
                "formation_year",
                "investment_year",
                "market_value_musd",
                "return_index",
                "price_available_eoy",
                "valid_return_count_10y",
                "zero_return_count_10y",
                "zero_return_ratio_10y",
                "stale_price_flag",
                "base_investable_next_year"]].copy()

        year_slice = year_slice.rename(
            columns={
                "market_value_musd": "year_end_market_value_musd",
                "return_index": "year_end_return_index"})

        yearly_results.append(year_slice)

    base_investment_set = pd.concat(yearly_results, ignore_index=True)
    base_investment_set = base_investment_set.sort_values(["formation_year", "isin"]).reset_index(drop=True)

    return base_investment_set


def save_outputs(
    em_companies: pd.DataFrame,
    monthly_data: pd.DataFrame,
    annual_data: pd.DataFrame,
    base_investment_set: pd.DataFrame):
    
    """Save the Part I outputs to the processed folder."""
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    written_files = {
        "A": str(write_excel(rename_columns_for_export(em_companies), OUTPUT_FILES["companies"])),
        "B": str(write_excel(rename_columns_for_export(monthly_data), OUTPUT_FILES["monthly_data"])),
        "C": str(write_excel(rename_columns_for_export(annual_data), OUTPUT_FILES["annual_data"])),
        "D": str(write_excel(rename_columns_for_export(base_investment_set), OUTPUT_FILES["base_investment_set"]))}
    
    return written_files

def main():
    log_step("  Data Cleaning 1/5 - Loading Emerging Markets companies...")
    em_companies = load_em_companies()

    log_step("  Data Cleaning 2/5 - Removing ISINs missing from required source tables...")
    em_companies = keep_only_common_isins(em_companies)

    log_step("  Data Cleaning 3/5 - Cleaning monthly prices and computing monthly returns...")
    monthly_data = build_monthly_data(em_companies)

    log_step("  Data Cleaning 4/5 - Cleaning Scope 1, revenue, and year-end prices...")
    annual_data = build_annual_data(em_companies, monthly_data)

    log_step("  Data Cleaning 5/5 - Building the base investment universe and saving outputs...")
    base_investment_set = build_base_investment_set(monthly_data)
    written_files = save_outputs(
        em_companies=em_companies,
        monthly_data=monthly_data,
        annual_data=annual_data,
        base_investment_set=base_investment_set)

    log_step("  Part I completed.")
    log_step(f"  Selected EM companies: {len(em_companies)}")
    log_step(f"  Monthly dataset rows: {len(monthly_data)}")
    log_step(f"  Annual dataset rows: {len(annual_data)}")
    log_step(f"  Base investment set rows: {len(base_investment_set)}")
    log_step("  Files written:")
    for label, path in written_files.items():
        log_step(f"{label} : {path}")

    print("  Data cleaning completed.", flush=True)

if __name__ == "__main__":
    main()
