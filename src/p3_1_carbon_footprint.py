from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from p3_0_carbon_utils import (
    PROCESSED_DIR,
    build_caption_table,
    build_reference_summary_table,
    compute_annual_wealth_path,
    compute_portfolio_annual_carbon_metrics,
    ensure_part1_cross_check,
    get_vw_rebalancing_weights,
    load_carbon_inputs,
    log_step,
    prepare_eligible_annual_panel,
    write_workbook,
)


OUTPUT_FILE = "P_Carbon_3_1_WACI_CF.xlsx"
WACI_FIGURE = "P_Carbon_3_1_WACI.png"
CF_FIGURE = "P_Carbon_3_1_CF.png"
ANNUAL_METRIC_COLUMNS = [
    "portfolio",
    "portfolio_label",
    "formation_year",
    "investment_year",
    "waci",
    "cf",
    "covered_weight",
    "firm_count",
    "formation_wealth_usd",
]


def save_waci_figure(mv_annual_carbon, vw_annual_carbon):
    """Save the WACI figure for Section 3.1."""
    figure_path = PROCESSED_DIR / WACI_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(mv_annual_carbon["formation_year"], mv_annual_carbon["waci"], color="darkorange", label="P(mv)_oos")
    ax.plot(vw_annual_carbon["formation_year"], vw_annual_carbon["waci"], color="steelblue", label="P(vw)_oos")
    ax.set_title("Figure 1. Weighted-Average Carbon Intensity by Formation Year")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD of Revenue")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_cf_figure(mv_annual_carbon, vw_annual_carbon):
    """Save the carbon footprint figure for Section 3.1."""
    figure_path = PROCESSED_DIR / CF_FIGURE
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(mv_annual_carbon["formation_year"], mv_annual_carbon["cf"], color="darkorange", label="P(mv)_oos")
    ax.plot(vw_annual_carbon["formation_year"], vw_annual_carbon["cf"], color="steelblue", label="P(vw)_oos")
    ax.set_title("Figure 2. Carbon Footprint by Formation Year")
    ax.set_xlabel("Formation Year")
    ax.set_ylabel("Tonnes CO2 per Million USD Invested")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def prepare_annual_metrics_table(mv_annual_carbon, vw_annual_carbon, mv_wealth_path: pd.DataFrame):
    """Combine the annual MV and VW metrics into a single table."""
    mv_annual_metrics = (
        mv_annual_carbon.merge(
            mv_wealth_path,
            on=["formation_year", "investment_year"],
            how="left",
        )
        .assign(portfolio_label="P(mv)_oos")
        .loc[:, ANNUAL_METRIC_COLUMNS]
        .copy()
    )
    vw_annual_metrics = (
        vw_annual_carbon.assign(
            formation_wealth_usd=float("nan"),
            portfolio_label="P(vw)",
        )
        .loc[:, ANNUAL_METRIC_COLUMNS]
        .copy()
    )

    combined_annual_metrics = pd.concat([mv_annual_metrics, vw_annual_metrics], ignore_index=True)
    return combined_annual_metrics.sort_values(["formation_year", "portfolio"]).reset_index(drop=True)


def main():
    """Compute WACI and carbon footprint for the reference portfolios."""
    log_step("  Carbon Footprint 3.1 1/4 - Loading the Part I outputs...")
    data = load_carbon_inputs()

    log_step("  Carbon Footprint 3.1 2/4 - Checking the Part I reference values...")
    reference_table = build_reference_summary_table(
        data["mv_performance"],
        data["vw_performance"],
        data["risk_free"],
    )
    ensure_part1_cross_check(reference_table)

    log_step("  Carbon Footprint 3.1 3/4 - Computing the annual carbon metrics...")
    eligible_annual = prepare_eligible_annual_panel(data["annual_data"], data["investment_set"])

    mv_details, mv_annual_carbon, mv_top10 = compute_portfolio_annual_carbon_metrics(
        data["mv_weights"],
        eligible_annual,
        "mv_oos",
    )
    vw_rebalancing_weights = get_vw_rebalancing_weights(data["vw_monthly_weights"])
    vw_details, vw_annual_carbon, vw_top10 = compute_portfolio_annual_carbon_metrics(
        vw_rebalancing_weights,
        eligible_annual,
        "vw",
    )

    mv_wealth_path = compute_annual_wealth_path(data["mv_performance"])
    combined_annual_metrics = prepare_annual_metrics_table(
        mv_annual_carbon,
        vw_annual_carbon,
        mv_wealth_path,
    )
    captions = build_caption_table(
        [
            {
                "item": "Part1 Cross Check",
                "caption": "This table verifies that the Part I monthly performance files reproduce the validated summary statistics for P(mv)_oos and P(vw) within the required tolerance of plus or minus 0.001.",
            },
            {
                "item": "Annual Metrics",
                "caption": "This table reports annual WACI and annual carbon footprint for P(mv)_oos and P(vw) by formation year, using Scope 1 emissions and revenue converted into million USD.",
            },
            {
                "item": "Top 10 WACI",
                "caption": "These tables list the ten largest firm-level contributors to WACI in each portfolio-year, with contribution defined as portfolio weight times firm carbon intensity.",
            },
        ]
    )

    log_step("  Carbon Footprint 3.1 4/4 - Saving the tables and figures...")
    workbook_path = write_workbook(
        OUTPUT_FILE,
        {
            "Part1 Cross Check": reference_table,
            "Annual Metrics": combined_annual_metrics,
            "MV Top 10 WACI": mv_top10,
            "VW Top 10 WACI": vw_top10,
            "MV Wealth Path": mv_wealth_path,
            "MV Firm Details": mv_details,
            "VW Firm Details": vw_details,
            "Captions": captions,
        },
    )
    waci_figure_path = save_waci_figure(mv_annual_carbon, vw_annual_carbon)
    cf_figure_path = save_cf_figure(mv_annual_carbon, vw_annual_carbon)

    log_step(f"  Table written: {workbook_path}")
    log_step(f"  WACI figure written: {waci_figure_path}")
    log_step(f"  Carbon footprint figure written: {cf_figure_path}")
    print("Section 3.1 complete.", flush=True)


if __name__ == "__main__":
    main()
