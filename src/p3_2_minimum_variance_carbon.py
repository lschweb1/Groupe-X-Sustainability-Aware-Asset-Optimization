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


WEIGHTS_FILE = "Q_MinVar_Carbon_3_2_Weights.xlsx"
PERFORMANCE_FILE = "R_MinVar_Carbon_3_2_Monthly_Performance.xlsx"
SUMMARY_FILE = "S_MinVar_Carbon_3_2_Summary.xlsx"
CUMULATIVE_FIGURE = "S_MinVar_Carbon_3_2_Cumulative_Returns.png"
WACI_FIGURE = "S_MinVar_Carbon_3_2_WACI.png"
CF_FIGURE = "S_MinVar_Carbon_3_2_CF.png"
SECTION_31_FILE = "P_Carbon_3_1_WACI_CF.xlsx"


def load_mv_reference_cf():
    """Load the annual CF reference of the minimum-variance portfolio."""
    annual_metrics = pd.read_excel(PROCESSED_DIR / SECTION_31_FILE, sheet_name="Annual Metrics")
    mv_metrics = annual_metrics.loc[annual_metrics["portfolio"] == "mv_oos", ["formation_year", "cf"]].copy()
    return mv_metrics.set_index("formation_year")["cf"]


def save_cumulative_figure(base_performance: pd.DataFrame, carbon_performance: pd.DataFrame):
    """Save the cumulative return figure."""
    figure_path = PROCESSED_DIR / CUMULATIVE_FIGURE

    # Local copies avoid modifying the original inputs.
    base = base_performance.copy()
    carbon = carbon_performance.copy()

    base["cumulative_growth_plot"] = (1 + base["portfolio_return"]).cumprod()
    carbon["cumulative_growth_plot"] = (1 + carbon["portfolio_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.plot(
        base["date"],
        base["cumulative_growth_plot"],
        color="darkorange",
        label="P(mv)_oos"
    )

    ax.plot(
        carbon["date"],
        carbon["cumulative_growth_plot"],
        color="firebrick",
        label="P(mv)_oos(0.5)"
    )

    ax.set_title("Cumulative Returns: P(mv)_oos vs P(mv)_oos(0.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def save_waci_figure(base_carbon: pd.DataFrame, carbon_portfolio: pd.DataFrame):
    """Save the WACI figure for Section 3.2."""
    figure_path = PROCESSED_DIR / WACI_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(base_carbon["formation_year"], base_carbon["waci"], color="darkorange", label="P(mv)_oos")
    ax.plot(carbon_portfolio["formation_year"], carbon_portfolio["waci"], color="firebrick", label="P(mv)_oos(0.5)")
    ax.set_title("WACI: P(mv)_oos vs P(mv)_oos(0.5)")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD of Revenue")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def save_cf_figure(base_carbon: pd.DataFrame, carbon_portfolio: pd.DataFrame):
    """Save the CF figure for Section 3.2."""
    figure_path = PROCESSED_DIR / CF_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(base_carbon["formation_year"], base_carbon["cf"], color="darkorange", label="P(mv)_oos")
    ax.plot(carbon_portfolio["formation_year"], carbon_portfolio["cf"], color="firebrick", label="P(mv)_oos(0.5)")
    ax.set_title("Carbon Footprint: P(mv)_oos vs P(mv)_oos(0.5)")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD Invested")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Build the minimum-variance portfolio with a 50% carbon constraint."""
    log_step("  Minimum Variance Carbon 3.2 1/5 - Loading the required data and the CF reference...")
    data = load_carbon_inputs()
    eligible_annual = prepare_eligible_annual_panel(data["annual_data"], data["investment_set"])
    return_matrix = build_return_matrix(data["monthly_data"])
    mv_reference_cf = load_mv_reference_cf()

    base_carbon = pd.read_excel(PROCESSED_DIR / SECTION_31_FILE, sheet_name="Annual Metrics")
    base_carbon = base_carbon.loc[base_carbon["portfolio"] == "mv_oos"].copy()

    log_step("  Minimum Variance Carbon 3.2 2/5 - Solving the annual optimizations...")
    weight_rows: list[pd.DataFrame] = []
    optimization_logs: list[dict[str, object]] = []

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Minimum Variance Carbon 3.2 - Processing formation year {formation_year}...")
        year_universe = eligible_annual.loc[
            (eligible_annual["formation_year"] == formation_year) & (eligible_annual["valid_carbon_inputs"])
        ].copy()
        covariance_matrix = data["covariance_matrices"][formation_year].copy()
        covariance_matrix, year_universe = align_covariance_universe(covariance_matrix, year_universe)

        if covariance_matrix.empty or year_universe.empty:
            warn(f"Section 3.2 - empty optimization universe for year {formation_year}.")
            continue

        carbon_vector = year_universe.set_index("isin")["e_over_cap"]
        carbon_target = 0.5 * float(mv_reference_cf.loc[formation_year])

        constrained_weights, constrained_info = solve_quadratic_portfolio(
            covariance_matrix=covariance_matrix,
            objective="min_var",
            carbon_vector=carbon_vector,
            carbon_target=carbon_target,
        )

        if constrained_info["success"]:
            final_weights = constrained_weights
            fallback_used = False
            unconstrained_cf = float("nan")
            achieved_cf = float(constrained_info["achieved_cf"])
        else:
            unconstrained_weights, _ = solve_quadratic_portfolio(
                covariance_matrix=covariance_matrix,
                objective="min_var",
            )
            unconstrained_cf = float((unconstrained_weights * carbon_vector.loc[unconstrained_weights.index]).sum())
            warn(
                f"Section 3.2 - year {formation_year}: constrained problem failed or was infeasible. "
                f"Target={carbon_target:.6f}, unconstrained_CF={unconstrained_cf:.6f}. "
                f"I fall back to the unconstrained minimum-variance solution."
            )
            final_weights = unconstrained_weights
            fallback_used = True
            achieved_cf = unconstrained_cf

        year_weights = final_weights.reset_index()
        year_weights.columns = ["isin", "weight"]
        year_weights["formation_year"] = formation_year
        year_weights["investment_year"] = formation_year + 1
        company_info = year_universe[["isin", "company_name", "country", "region"]].drop_duplicates()
        year_weights = year_weights.merge(
            company_info,
            on="isin",
            how="left",
        )
        year_weights = year_weights[["isin", "company_name", "country", "region", "formation_year", "investment_year", "weight"]]
        weight_rows.append(year_weights)

        optimization_logs.append(
            {
                "formation_year": formation_year,
                "carbon_target": carbon_target,
                "unconstrained_cf": unconstrained_cf,
                "achieved_cf": achieved_cf,
                "success": constrained_info["success"],
                "scipy_success": constrained_info["scipy_success"],
                "optimizer_status": constrained_info["status"],
                "optimizer_message": constrained_info["message"],
                "fallback_used": fallback_used,
                "asset_count": len(final_weights),
            }
        )

    mv_carbon_weights = pd.concat(weight_rows, ignore_index=True)
    optimization_log = pd.DataFrame(optimization_logs)

    log_step("  Minimum Variance Carbon 3.2 3/5 - Computing the ex post monthly performance...")
    mv_carbon_performance = build_drifted_performance(return_matrix, mv_carbon_weights)
    risk_free_series = data["risk_free"].set_index("date")["rf_decimal"]
    summary_stats = compute_summary_stats(mv_carbon_performance.set_index("date")["portfolio_return"], risk_free_series)
    summary_table = pd.DataFrame([summary_stats])

    log_step("  Minimum Variance Carbon 3.2 4/5 - Computing the carbon metrics and portfolio changes...")
    _, mv_carbon_annual, mv_carbon_top10 = compute_portfolio_annual_carbon_metrics(
        mv_carbon_weights,
        eligible_annual,
        "mv_carbon",
    )
    _, excluded_firms, overweighted_firms, country_shift = compare_weight_structures(
        data["mv_weights"],
        mv_carbon_weights,
        "base",
        "carbon",
    )
    sector_note = build_note_table(
        "Sector breakdown cannot be computed from the current processed project files because no sector variable is available."
    )
    captions = build_caption_table(
        [
            {
                "item": "Summary Stats",
                "caption": "This table reports annualized return, annualized volatility, Sharpe ratio, and monthly extremes for P(mv)_oos(0.5), computed from ex-post monthly returns over January 2014 to December 2025.",
            },
            {
                "item": "Annual Carbon",
                "caption": "This table reports annual WACI and annual carbon footprint for the constrained minimum-variance portfolio, measured at each formation year from 2013 to 2024.",
            },
            {
                "item": "Excluded Firms / Overweighted Firms / Country Shift",
                "caption": "These tables compare the constrained solution with the unconstrained minimum-variance benchmark from Part I and describe how the carbon cap changes the portfolio composition.",
            },
        ]
    )

    log_step("  Minimum Variance Carbon 3.2 5/5 - Saving the outputs...")
    weights_path = write_workbook(
        WEIGHTS_FILE,
        {
            "Weights": mv_carbon_weights,
            "Optimization Log": optimization_log,
        },
    )
    performance_path = write_workbook(
        PERFORMANCE_FILE,
        {
            "Monthly Performance": mv_carbon_performance,
        },
    )
    summary_path = write_workbook(
        SUMMARY_FILE,
        {
            "Summary Stats": summary_table,
            "Annual Carbon": mv_carbon_annual,
            "Excluded Firms": excluded_firms,
            "Overweighted Firms": overweighted_firms,
            "Country Shift": country_shift,
            "Top 10 WACI": mv_carbon_top10,
            "Sector Note": sector_note,
            "Captions": captions,
        },
    )

    save_cumulative_figure(data["mv_performance"], mv_carbon_performance)
    save_waci_figure(base_carbon, mv_carbon_annual)
    save_cf_figure(base_carbon, mv_carbon_annual)

    log_step(f"  Weights file written: {weights_path}")
    log_step(f"  Performance file written: {performance_path}")
    log_step(f"  Summary file written: {summary_path}")
    print("Section 3.2 complete.", flush=True)


if __name__ == "__main__":
    main()
