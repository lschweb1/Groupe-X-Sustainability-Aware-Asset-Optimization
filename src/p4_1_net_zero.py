from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from p3_0_carbon_utils import (
    PROCESSED_DIR,
    FIRST_FORMATION_YEAR,
    LAST_FORMATION_YEAR,
    align_covariance_universe,
    build_caption_table,
    build_drifted_performance,
    build_note_table,
    build_return_matrix,
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


WEIGHTS_FILE = "W_NetZero_4_1_Weights.xlsx"
PERFORMANCE_FILE = "X_NetZero_4_1_Monthly_Performance.xlsx"
SUMMARY_FILE = "Y_NetZero_4_1_Summary.xlsx"
CUMULATIVE_FIGURE = "Y_NetZero_4_1_Cumulative_Returns.png"
CF_PATH_FIGURE = "Y_NetZero_4_1_CF_Path.png"


def compute_vw_reference_cf(year_universe: pd.DataFrame):
    """Compute the annual carbon footprint of the VW benchmark."""
    valid_carbon = year_universe.loc[year_universe["valid_carbon_inputs"]].copy()
    total_market_cap = float(valid_carbon["year_end_market_value_musd"].sum())
    if valid_carbon.empty or total_market_cap <= 0:
        return float("nan")
    return float(valid_carbon["scope1_co2"].sum() / total_market_cap)


def save_cumulative_figure(base_performance: pd.DataFrame,
                           nz_performance: pd.DataFrame):
    """Save the cumulative return comparison against the VW benchmark."""

    figure_path = PROCESSED_DIR / CUMULATIVE_FIGURE

    # Local copies avoid modifying the original inputs.
    base = base_performance.copy()
    nz = nz_performance.copy()

    # Recompute cumulative returns directly for plotting.
    base["cumulative_growth_plot"] = (
        1 + base["portfolio_return"]
    ).cumprod()

    nz["cumulative_growth_plot"] = (
        1 + nz["portfolio_return"]
    ).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    ax.plot(
        base["date"],
        base["cumulative_growth_plot"],
        color="steelblue",
        label="P(vw)_oos"
    )

    ax.plot(
        nz["date"],
        nz["cumulative_growth_plot"],
        color="purple",
        label="P(vw)_oos(NZ)"
    )

    ax.set_title("Cumulative Returns: P(vw)_oos vs P(vw)_oos(NZ)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def save_cf_path_figure(base_carbon: pd.DataFrame, nz_annual: pd.DataFrame, target_table: pd.DataFrame):
    """Save the carbon footprint figure and the net-zero target path."""
    figure_path = PROCESSED_DIR / CF_PATH_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(base_carbon["formation_year"], base_carbon["cf"], color="steelblue", label="P(vw)_oos")
    ax.plot(nz_annual["formation_year"], nz_annual["cf"], color="purple", label="P(vw)_oos(NZ)")
    ax.plot(target_table["formation_year"], target_table["cf_target"], color="black", linestyle="--", label="NZ Constraint Path")
    ax.set_title("Carbon Footprint: P(vw)_oos, P(vw)_oos(NZ), and Constraint Path")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD Invested")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """Build the passive net-zero portfolio with a fixed 2013 baseline target."""
    log_step("  Net Zero 4.1 1/5 - Loading the data and setting the 2013 baseline...")
    data = load_carbon_inputs()
    eligible_annual = prepare_eligible_annual_panel(data["annual_data"], data["investment_set"])
    return_matrix = build_return_matrix(data["monthly_data"])
    vw_benchmark_weights = build_year_end_vw_benchmark_weights(eligible_annual, require_valid_carbon=False)
    base_vw_carbon = pd.read_excel(PROCESSED_DIR / "P_Carbon_3_1_WACI_CF.xlsx", sheet_name="Annual Metrics")
    base_vw_carbon = base_vw_carbon.loc[base_vw_carbon["portfolio"] == "vw"].copy()

    base_2013_cf = compute_vw_reference_cf(eligible_annual.loc[eligible_annual["formation_year"] == FIRST_FORMATION_YEAR].copy())

    log_step("  Net Zero 4.1 2/5 - Solving the annual net-zero optimizations...")
    benchmark_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    optimization_logs: list[dict[str, object]] = []
    target_rows: list[dict[str, float]] = []

    for formation_year in range(FIRST_FORMATION_YEAR, LAST_FORMATION_YEAR + 1):
        log_step(f"  Net Zero 4.1 - Processing formation year {formation_year}...")
        year_universe = eligible_annual.loc[eligible_annual["formation_year"] == formation_year].copy()
        optimization_universe = year_universe.loc[year_universe["valid_carbon_inputs"]].copy()
        covariance_matrix = data["covariance_matrices"][formation_year].copy()
        covariance_matrix, optimization_universe = align_covariance_universe(covariance_matrix, optimization_universe)

        if covariance_matrix.empty or optimization_universe.empty:
            warn(f"Net Zero 4.1 - empty optimization universe for year {formation_year}.")
            continue

        benchmark_year = vw_benchmark_weights.loc[vw_benchmark_weights["formation_year"] == formation_year].copy()
        benchmark_year = benchmark_year.loc[benchmark_year["isin"].isin(optimization_universe["isin"])].copy()
        if benchmark_year.empty:
            warn(f"Net Zero 4.1 - empty benchmark universe for year {formation_year}.")
            continue

        benchmark_year["weight"] = benchmark_year["weight"] / benchmark_year["weight"].sum()
        benchmark_series = benchmark_year.set_index("isin")["weight"].astype(float)
        benchmark_series = benchmark_series.reindex(covariance_matrix.columns).fillna(0.0)
        benchmark_series = benchmark_series / benchmark_series.sum()

        company_info = optimization_universe.set_index("isin")[["company_name", "country", "region"]]
        carbon_vector = optimization_universe.set_index("isin")["e_over_cap"].astype(float)
        carbon_vector = carbon_vector.reindex(covariance_matrix.columns).fillna(0.0)
        target_multiplier = 0.9 ** (formation_year - FIRST_FORMATION_YEAR + 1)
        carbon_target = target_multiplier * base_2013_cf
        benchmark_cf = compute_vw_reference_cf(year_universe)

        target_rows.append(
            {
                "formation_year": formation_year,
                "investment_year": formation_year + 1,
                "cf_target": carbon_target,
                "target_multiplier": target_multiplier,
                "benchmark_cf": benchmark_cf,
            }
        )

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
                f"Section 4.1 - year {formation_year}: constrained problem failed or was infeasible. "
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

    vw_nz_weights = pd.concat(weight_rows, ignore_index=True)
    vw_benchmark_annual = pd.concat(benchmark_rows, ignore_index=True)
    optimization_log = pd.DataFrame(optimization_logs)
    target_table = pd.DataFrame(target_rows)

    log_step("  Net Zero 4.1 3/5 - Computing the ex post monthly performance...")
    vw_nz_performance = build_drifted_performance(return_matrix, vw_nz_weights)
    risk_free_series = data["risk_free"].set_index("date")["rf_decimal"]
    summary_stats = compute_summary_stats(vw_nz_performance.set_index("date")["portfolio_return"], risk_free_series)
    summary_table = pd.DataFrame([summary_stats])

    log_step("  Net Zero 4.1 4/5 - Computing the carbon metrics and benchmark distortions...")
    _, vw_nz_annual, vw_nz_top10 = compute_portfolio_annual_carbon_metrics(
        vw_nz_weights,
        eligible_annual,
        "vw_nz",
    )
    _, excluded_firms, overweighted_firms, country_shift = compare_weight_structures(
        vw_benchmark_annual,
        vw_nz_weights,
        "benchmark",
        "nz",
    )
    sector_note = build_note_table(
        "Sector breakdown cannot be computed from the current processed project files because no sector variable is available."
    )
    captions = build_caption_table(
        [
            {
                "item": "Summary Stats",
                "caption": "This table reports annualized return, annualized volatility, Sharpe ratio, and monthly extremes for P(vw)_oos(NZ), computed from ex-post monthly returns over January 2014 to December 2025.",
            },
            {
                "item": "Annual Carbon / Constraint Path",
                "caption": "These tables report the realized carbon footprint path of P(vw)_oos(NZ) and the fixed net-zero target path based on the 2013 benchmark carbon-footprint baseline.",
            },
            {
                "item": "Excluded Firms / Overweighted Firms / Country Shift",
                "caption": "These tables compare the net-zero tracking-error solution with the annual value-weighted benchmark built from year-end market capitalizations over the min-var-eligible universe.",
            },
        ]
    )

    log_step("  Net Zero 4.1 5/5 - Saving the outputs...")
    weights_path = write_workbook(
        WEIGHTS_FILE,
        {
            "Weights": vw_nz_weights,
            "Benchmark Weights": vw_benchmark_annual,
            "Optimization Log": optimization_log,
        },
    )
    performance_path = write_workbook(
        PERFORMANCE_FILE,
        {
            "Monthly Performance": vw_nz_performance,
        },
    )
    summary_path = write_workbook(
        SUMMARY_FILE,
        {
            "Summary Stats": summary_table,
            "Annual Carbon": vw_nz_annual,
            "Constraint Path": target_table,
            "Excluded Firms": excluded_firms,
            "Overweighted Firms": overweighted_firms,
            "Country Shift": country_shift,
            "Top 10 WACI": vw_nz_top10,
            "Sector Note": sector_note,
            "Captions": captions,
        },
    )

    save_cumulative_figure(data["vw_performance"], vw_nz_performance)
    save_cf_path_figure(base_vw_carbon, vw_nz_annual, target_table)

    log_step(f"  Weights file written: {weights_path}")
    log_step(f"  Performance file written: {performance_path}")
    log_step(f"  Summary file written: {summary_path}")
    print("  Section 4.1 completed.", flush=True)


if __name__ == "__main__":
    main()
