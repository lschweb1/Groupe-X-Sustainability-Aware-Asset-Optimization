from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

def log_step(message: str):
    """Print a clear progress message to the terminal."""
    print(message, flush=True)


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

def run_step(step_label: str, script_name: str):
    """Run a step in a subprocess and display its execution time"""

    start_time = datetime.now()
    log_step(f"{step_label} - Started")

    script_path = SRC_DIR / script_name  # Pointing to the exact script to be run.
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(BASE_DIR),
        check=True)  # Run each step in a separate Python process.

    end_time = datetime.now()
    duration = end_time - start_time  
    log_step(f"{step_label} - Completed in {duration}")


def main():
    log_step("Starting the project pipeline...")

    run_step("Pipeline Step 1/11 - Data Cleaning", "p1_1_data_cleaning.py")
    run_step("Pipeline Step 2/11 - Minimum Variance 2.1", "p2_1_investment_set.py")
    run_step("Pipeline Step 3/11 - Minimum Variance 2.2", "p2_2_minimum_variance_portfolio.py")
    run_step("Pipeline Step 4/11 - Value-Weighted 2.3", "p2_3_value_weighted_portfolio.py")
    run_step("Pipeline Step 5/11 - MinVar vs Value-Weighted Comparison 2.4", "p2_4_minvar_vs_valueweighted_comparison.py")
    run_step("Pipeline Step 6/11 - Carbon Footprint 3.1", "p3_1_carbon_footprint.py")
    run_step("Pipeline Step 7/11 - Minimum Variance Carbon 3.2", "p3_2_minimum_variance_carbon.py")
    run_step("Pipeline Step 8/11 - Tracking Error Carbon 3.3", "p3_3_tracking_error_carbon.py")
    run_step("Pipeline Step 9/11 - Carbon Comparison 3.4", "p3_4_carbon_comparison.py")
    run_step("Pipeline Step 10/11 - Net Zero 4.1", "p4_1_net_zero.py")
    run_step("Pipeline Step 11/11 - Passive Comparison 4.2", "p4_2_passive_comparison.py")

    log_step("Pipeline completed.")


if __name__ == "__main__":
    main()
