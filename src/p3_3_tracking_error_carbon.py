from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from p3_0_carbon_utils import (
    PROCESSED_DIR,
    FIRST_FORMATION_YEAR,
    LAST_FORMATION_YEAR,
    align_covariance_universe,
    build_return_matrix,
    build_caption_table,
    build_drifted_performance,
    build_note_table,
    build_year_end_vw_benchmark_weights,
    compare_weight_structures,
    compute_portfolio_annual_carbon_metrics,
    compute_summary_stats,
    load_carbon_inputs,
    log_step,
    prepare_eligible_annual_panel,
    solve_quadratic_portfolio,
    warn,
    write_workbook,
)

WEIGHTS_FILE = "T_TrackingError_Carbon_3_3_Weights.xlsx"
PERFORMANCE_FILE = "U_TrackingError_Carbon_3_3_Monthly_Performance.xlsx"
SUMMARY_FILE = "V_TrackingError_Carbon_3_3_Summary.xlsx"
CUMULATIVE_FIGURE = "V_TrackingError_Carbon_3_3_Cumulative_Returns.png"
WACI_FIGURE = "V_TrackingError_Carbon_3_3_WACI.png"
CF_FIGURE = "V_TrackingError_Carbon_3_3_CF.png"


def save_cumulative_figure(base_performance: pd.DataFrame,
                           carbon_performance: pd.DataFrame):
    """Save the cumulative return figure."""

    figure_path = PROCESSED_DIR / CUMULATIVE_FIGURE

    # Local copies avoid modifying the original inputs.
    base = base_performance.copy()
    carbon = carbon_performance.copy()

    # Recompute cumulative returns directly for plotting.
    base["cumulative_growth_plot"] = (1 + base["portfolio_return"]).cumprod()

    carbon["cumulative_growth_plot"] = (1 + carbon["portfolio_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.plot(
        base["date"],
        base["cumulative_growth_plot"],
        color="steelblue",
        label="P(vw)_oos"
    )

    ax.plot(
        carbon["date"],
        carbon["cumulative_growth_plot"],
        color="seagreen",
        label="P(vw)_oos(0.5)"
    )

    ax.set_title("Cumulative Returns: P(vw)_oos vs P(vw)_oos(0.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def save_waci_figure(base_carbon: pd.DataFrame, carbon_portfolio: pd.DataFrame):
    """Save the WACI figure for Section 3.3."""
    figure_path = PROCESSED_DIR / WACI_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(base_carbon["formation_year"], base_carbon["waci"], color="steelblue", label="P(vw)_oos")
    ax.plot(carbon_portfolio["formation_year"], carbon_portfolio["waci"], color="seagreen", label="P(vw)_oos(0.5)")
    ax.set_title("WACI: P(vw)_oos vs P(vw)_oos(0.5)")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD of Revenue")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def save_cf_figure(base_carbon: pd.DataFrame, carbon_portfolio: pd.DataFrame):
    """Save the carbon footprint figure for Section 3.3."""
    figure_path = PROCESSED_DIR / CF_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(base_carbon["formation_year"], base_carbon["cf"], color="steelblue", label="P(vw)_oos")
    ax.plot(carbon_portfolio["formation_year"], carbon_portfolio["cf"], color="seagreen", label="P(vw)_oos(0.5)")
    ax.set_title("Carbon Footprint: P(vw)_oos vs P(vw)_oos(0.5)")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD Invested")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def compute_vw_reference_cf(year_universe: pd.DataFrame):
    """
    Compute the VW benchmark carbon footprint from year-end market capitalizations.
    """
    valid_carbon = year_universe.loc[year_universe["valid_carbon_inputs"]].copy()
    total_market_cap = float(valid_carbon["year_end_market_value_musd"].sum())
    if valid_carbon.empty or total_market_cap <= 0:
        return float("nan")
    return float(valid_carbon["scope1_co2"].sum() / total_market_cap)


def main():
    """Build the minimum tracking-error portfolio with a 50% reduction in VW carbon footprint."""
    log_step("  Tracking Error Carbon 3.3 1/5 - Loading the required data...")
    data = load_carbon_inputs()
    eligible_annual = prepare_eligible_annual_panel(data["annual_data"], data["investment_set"])
    return_matrix = build_return_matrix(data["monthly_data"])
    vw_benchmark_weights = build_year_end_vw_benchmark_weights(eligible_annual, require_valid_carbon=False)

    base_vw_carbon = pd.read_excel(PROCESSED_DIR / "P_Carbon_3_1_WACI_CF.xlsx", sheet_name="Annual Metrics")
    base_vw_carbon = base_vw_carbon.loc[base_vw_carbon["portfolio"] == "vw"].copy()

    log_step("  Tracking Error Carbon 3.3 2/5 - Solving the annual optimizations...")
    benchmark_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    optimization_logs: list[dict[str, object]] = []

    eligible_by_year = {
        year: group.copy()
        for year, group in eligible_annual.groupby("formation_year")
    }

    benchmark_by_year = {
        year: group.copy()
        for year, group in vw_benchmark_weights.groupby("formation_year")
    }

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Tracking Error Carbon 3.3 - Processing formation year {formation_year}...")
        year_universe = eligible_by_year.get(
            formation_year,
            pd.DataFrame()
        ).copy()
        optimization_universe = year_universe.loc[year_universe["valid_carbon_inputs"]].copy()
        covariance_matrix = data["covariance_matrices"][formation_year].copy()
        covariance_matrix, optimization_universe = align_covariance_universe(covariance_matrix, optimization_universe)

        if covariance_matrix.empty or optimization_universe.empty:
            warn(f"Tracking Error Carbon 3.3 - empty optimization universe for year {formation_year}.")
            continue

        benchmark_year = benchmark_by_year.get(
            formation_year,
            pd.DataFrame()
        ).copy()
        benchmark_year = benchmark_year.loc[benchmark_year["isin"].isin(optimization_universe["isin"])].copy()
        if benchmark_year.empty:
            warn(f"Tracking Error Carbon 3.3 - empty benchmark universe for year {formation_year}.")
            continue

        benchmark_year["weight"] = benchmark_year["weight"] / benchmark_year["weight"].sum()
        benchmark_series = benchmark_year.set_index("isin")["weight"].astype(float)
        benchmark_series = benchmark_series.reindex(covariance_matrix.columns).fillna(0.0)
        benchmark_series = benchmark_series / benchmark_series.sum()

        company_info = optimization_universe.set_index("isin")[["company_name", "country", "region"]]
        carbon_vector = optimization_universe.set_index("isin")["e_over_cap"].astype(float)
        carbon_vector = carbon_vector.reindex(covariance_matrix.columns).fillna(0.0)
        benchmark_cf = compute_vw_reference_cf(year_universe)
        carbon_target = 0.5 * benchmark_cf

        benchmark_rows.append(
            benchmark_series.rename_axis("isin").reset_index(name="weight").assign(
                company_name=lambda df: df["isin"].map(company_info["company_name"]),
                country=lambda df: df["isin"].map(company_info["country"]),
                region=lambda df: df["isin"].map(company_info["region"]),
                formation_year=formation_year,
                investment_year=formation_year + 1,
            )[["isin", "company_name", "country", "region", "formation_year", "investment_year", "weight"]]
        )

        constrained_weights, constrained_info = solve_quadratic_portfolio(
            covariance_matrix=covariance_matrix,
            objective="tracking_error",
            benchmark_weights=benchmark_series,
            carbon_vector=carbon_vector,
            carbon_target=carbon_target,
        )

        achieved_cf = float(constrained_info["achieved_cf"])
        fallback_used = False
        final_weights = constrained_weights

        if not constrained_info["success"]:
            warn(
                f"Section 3.3 - year {formation_year}: constrained problem failed or was infeasible. "
                f"Target={carbon_target:.6f}, benchmark_CF={benchmark_cf:.6f}. "
                f"The annual VW benchmark weights are used as fallback."
            )
            final_weights = benchmark_series.copy()
            achieved_cf = float((final_weights * carbon_vector.loc[final_weights.index]).sum())
            fallback_used = True

        year_weights = final_weights.rename_axis("isin").reset_index(name="weight")
        year_weights["company_name"] = year_weights["isin"].map(company_info["company_name"])
        year_weights["country"] = year_weights["isin"].map(company_info["country"])
        year_weights["region"] = year_weights["isin"].map(company_info["region"])
        year_weights["formation_year"] = formation_year
        year_weights["investment_year"] = formation_year + 1
        year_weights = year_weights[["isin", "company_name", "country", "region", "formation_year", "investment_year", "weight"]]
        weight_rows.append(year_weights)

        optimization_logs.append(
            {
                "formation_year": formation_year,
                "carbon_target": carbon_target,
                "benchmark_cf": benchmark_cf,
                "achieved_cf": achieved_cf,
                "success": constrained_info["success"],
                "scipy_success": constrained_info["scipy_success"],
                "optimizer_status": constrained_info["status"],
                "optimizer_message": constrained_info["message"],
                "fallback_used": fallback_used,
                "asset_count": len(final_weights),
            }
        )

    vw_benchmark_annual = pd.concat(benchmark_rows, ignore_index=True)
    vw_carbon_weights = pd.concat(weight_rows, ignore_index=True)
    optimization_log = pd.DataFrame(optimization_logs)

    log_step("  Tracking Error Carbon 3.3 3/5 - Computing the ex post monthly performance...")
    vw_carbon_performance = build_drifted_performance(return_matrix, vw_carbon_weights)
    risk_free_series = data["risk_free"].set_index("date")["rf_decimal"]
    summary_stats = compute_summary_stats(vw_carbon_performance.set_index("date")["portfolio_return"], risk_free_series)
    summary_table = pd.DataFrame([summary_stats])

    log_step("  Tracking Error Carbon 3.3 4/5 - Computing the carbon metrics and weight distortions...")
    _, vw_carbon_annual, vw_carbon_top10 = compute_portfolio_annual_carbon_metrics(
        vw_carbon_weights,
        eligible_annual,
        "vw_carbon",
    )
    _, excluded_firms, overweighted_firms, country_shift = compare_weight_structures(
        vw_benchmark_annual,
        vw_carbon_weights,
        "benchmark",
        "carbon",
    )
    sector_note = build_note_table(
        "Sector breakdown cannot be computed from the current processed project files because no sector variable is available."
    )
    captions = build_caption_table(
        [
            {
                "item": "Summary Stats",
                "caption": "This table reports annualized return, annualized volatility, Sharpe ratio, and monthly extremes for P(vw)_oos(0.5), computed from ex-post monthly returns over January 2014 to December 2025.",
            },
            {
                "item": "Annual Carbon",
                "caption": "This table reports annual WACI and annual carbon footprint for the tracking-error portfolio with a 50% carbon-footprint target, measured at each formation year from 2013 to 2024.",
            },
            {
                "item": "Excluded Firms / Overweighted Firms / Country Shift",
                "caption": "These tables compare the constrained tracking-error solution with the annual value-weighted benchmark built from year-end market capitalizations over the min-var-eligible universe.",
            },
        ]
    )

    log_step("  Tracking Error Carbon 3.3 5/5 - Saving the outputs...")
    weights_path = write_workbook(
        WEIGHTS_FILE,
        {
            "Weights": vw_carbon_weights,
            "Benchmark Weights": vw_benchmark_annual,
            "Optimization Log": optimization_log,
        },
    )
    performance_path = write_workbook(
        PERFORMANCE_FILE,
        {
            "Monthly Performance": vw_carbon_performance,
        },
    )
    summary_path = write_workbook(
        SUMMARY_FILE,
        {
            "Summary Stats": summary_table,
            "Annual Carbon": vw_carbon_annual,
            "Excluded Firms": excluded_firms,
            "Overweighted Firms": overweighted_firms,
            "Country Shift": country_shift,
            "Top 10 WACI": vw_carbon_top10,
            "Sector Note": sector_note,
            "Captions": captions,
        },
    )

    save_cumulative_figure(data["vw_performance"], vw_carbon_performance)
    save_waci_figure(base_vw_carbon, vw_carbon_annual)
    save_cf_figure(base_vw_carbon, vw_carbon_annual)

    log_step(f"  Weights file written: {weights_path}")
    log_step(f"  Performance file written: {performance_path}")
    log_step(f"  Summary file written: {summary_path}")
    print("  Section 3.3 completed.", flush=True)


if __name__ == "__main__":
    main()
