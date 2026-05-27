from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from p3_0_carbon_utils import (
    PROCESSED_DIR,
    build_caption_table,
    compute_summary_stats,
    load_carbon_inputs,
    log_step,
    write_workbook,
)


OUTPUT_FILE = "AA_Passive_Comparison_4_2.xlsx"
FIGURE_FILE = "AA_Passive_Comparison_4_2_Cumulative_Returns.png"


def load_annual_carbon(file_name: str, sheet_name: str):
    """Load an annual carbon table that was already produced."""
    return pd.read_excel(PROCESSED_DIR / file_name, sheet_name=sheet_name)


def build_passive_comparison_table(data: dict[str, object]):
    """Compare the three passive strategies required in Section 4.2."""
    risk_free_series = data["risk_free"].set_index("date")["rf_decimal"]
    section_31_annual = load_annual_carbon("P_Carbon_3_1_WACI_CF.xlsx", "Annual Metrics")

    performance_map = {
        "vw": data["vw_performance"],
        "vw_carbon": pd.read_excel(PROCESSED_DIR / "U_TrackingError_Carbon_3_3_Monthly_Performance.xlsx", sheet_name="Monthly Performance"),
        "vw_nz": pd.read_excel(PROCESSED_DIR / "X_NetZero_4_1_Monthly_Performance.xlsx", sheet_name="Monthly Performance"),
    }

    annual_carbon_map = {
        "vw": section_31_annual.query("portfolio == 'vw'"),
        "vw_carbon": load_annual_carbon("V_TrackingError_Carbon_3_3_Summary.xlsx", "Annual Carbon"),
        "vw_nz": load_annual_carbon("Y_NetZero_4_1_Summary.xlsx", "Annual Carbon"),
    }

    reference_cf_total = annual_carbon_map["vw"]["cf"].sum()
    reference_final_cf = float(annual_carbon_map["vw"].sort_values("formation_year")["cf"].iloc[-1])

    rows: list[dict[str, object]] = []
    for portfolio_name, performance_df in performance_map.items():
        performance_df = performance_df.copy()
        performance_df["date"] = pd.to_datetime(performance_df["date"])
        stats = compute_summary_stats(performance_df.set_index("date")["portfolio_return"], risk_free_series)
        annual_carbon = annual_carbon_map[portfolio_name].copy().sort_values("formation_year")
        total_cf = float(annual_carbon["cf"].sum())
        final_cf = float(annual_carbon["cf"].iloc[-1])

        rows.append(
            {
                "portfolio": portfolio_name,
                "annualized_return": stats["annualized_return"],
                "annualized_volatility": stats["annualized_volatility"],
                "sharpe_ratio": stats["sharpe_ratio"],
                "average_annual_cf": annual_carbon["cf"].mean(),
                "final_year_cf": final_cf,
                "total_cumulative_carbon_reduction_pct_vs_vw": 100.0 * (1.0 - total_cf / reference_cf_total),
                "final_year_carbon_reduction_pct_vs_vw": 100.0 * (1.0 - final_cf / reference_final_cf),
            }
        )

    return pd.DataFrame(rows)


def build_comment_table(comparison_table: pd.DataFrame):
    """Build the interpretation notes for the final passive comparison."""
    comparison = comparison_table.set_index("portfolio")
    vw = comparison.loc["vw"]
    vw_carbon = comparison.loc["vw_carbon"]
    vw_nz = comparison.loc["vw_nz"]
    optimization_log = pd.read_excel(PROCESSED_DIR / "W_NetZero_4_1_Weights.xlsx", sheet_name="Optimization Log")
    fallback_years = optimization_log.loc[optimization_log["fallback_used"], "formation_year"].tolist()

    fallback_text = "none" if not fallback_years else ", ".join(str(year) for year in fallback_years)

    return pd.DataFrame(
        [
            {
                "topic": "Net-zero path achievement",
                "comment": (
                    "The realized net-zero portfolio should be compared with the fixed target path in Y_NetZero_4_1_Summary.xlsx. "
                    f"Its final-year carbon footprint reduction versus P(vw) is {vw_nz['final_year_carbon_reduction_pct_vs_vw']:.2f}%."
                ),
            },
            {
                "topic": "Financial cost of net zero",
                "comment": (
                    "Relative to P(vw), the net-zero strategy changes annualized return by "
                    f"{vw_nz['annualized_return'] - vw['annualized_return']:.4f}. "
                    "Relative to the static 50% strategy, it changes annualized return by "
                    f"{vw_nz['annualized_return'] - vw_carbon['annualized_return']:.4f}."
                ),
            },
            {
                "topic": "Feasibility trajectory",
                "comment": (
                    "The optimization log shows whether the tightening constraint became difficult to satisfy over time. "
                    f"Fallback years: {fallback_text}."
                ),
            },
        ]
    )


def save_cumulative_figure(vw_performance,
                           vw_carbon_performance,
                           vw_nz_performance):
    """Save the cumulative return figure for the three passive strategies."""

    figure_path = PROCESSED_DIR / FIGURE_FILE

    # Local copies avoid modifying the original inputs.
    base = vw_performance.copy()
    carbon = vw_carbon_performance.copy()
    nz = vw_nz_performance.copy()

    # Recompute cumulative returns directly for plotting.
    base["cumulative_growth_plot"] = (
        1 + base["portfolio_return"]
    ).cumprod()

    carbon["cumulative_growth_plot"] = (
        1 + carbon["portfolio_return"]
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
        carbon["date"],
        carbon["cumulative_growth_plot"],
        color="seagreen",
        label="P(vw)_oos(0.5)"
    )

    ax.plot(
        nz["date"],
        nz["cumulative_growth_plot"],
        color="purple",
        label="P(vw)_oos(NZ)"
    )

    ax.set_title("Cumulative Returns of the Passive Strategies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth")

    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

def main():
    """Compare the three passive strategies in Section 4.2."""
    log_step("  Passive Comparison 4.2 1/3 - Loading the required outputs...")
    data = load_carbon_inputs()
    vw_carbon_performance = pd.read_excel(PROCESSED_DIR / "U_TrackingError_Carbon_3_3_Monthly_Performance.xlsx", sheet_name="Monthly Performance", parse_dates=["date"])
    vw_nz_performance = pd.read_excel(PROCESSED_DIR / "X_NetZero_4_1_Monthly_Performance.xlsx", sheet_name="Monthly Performance", parse_dates=["date"])

    log_step("  Passive Comparison 4.2 2/3 - Building the final comparison...")
    comparison_table = build_passive_comparison_table(data)
    comments_table = build_comment_table(comparison_table)
    captions = build_caption_table(
        [
            {
                "item": "Passive Comparison",
                "caption": "This table compares the passive benchmark, the 50% carbon-reduction tracking-error strategy, and the net-zero strategy on financial performance and carbon outcomes over the common out-of-sample period.",
            },
            {
                "item": "Comments",
                "caption": "These comments interpret the realized net-zero path, the financial cost of tighter decarbonization, and the feasibility pattern observed in the annual optimization log.",
            },
            {
                "item": "Figure",
                "caption": "The cumulative return figure plots the three passive strategies from January 2014 to December 2025, all starting from 1.",
            },
        ]
    )

    log_step("  Passive Comparison 4.2 3/3 - Saving the outputs...")
    workbook_path = write_workbook(
        OUTPUT_FILE,
        {
            "Passive Comparison": comparison_table,
            "Comments": comments_table,
            "Captions": captions,
        },
    )
    save_cumulative_figure(data["vw_performance"], vw_carbon_performance, vw_nz_performance)

    log_step(f"  Comparison workbook written: {workbook_path}")
    print("  Section 4.2 completed.", flush=True)


if __name__ == "__main__":
    main()
