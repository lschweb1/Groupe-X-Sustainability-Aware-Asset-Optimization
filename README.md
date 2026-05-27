# Groupe X - Sustainability Aware Asset Management

Emerging Markets / Scope 1

## Project Overview

This repository contains the full workflow for the SAAM group project.
It starts with Datastream data cleaning, then builds the standard portfolio allocation of Part I, and finally extends the analysis to carbon-constrained and net-zero portfolio allocation for Parts III and IV.

The project focuses on:
- Emerging Markets firms only
- Scope 1 carbon emissions only
- Long-only portfolio construction
- Annual rebalancing with monthly weight drift

## Project Structure

```text
Groupe-X-Sustainability-Aware-Asset-main/
|-- data/
|   |-- Raw/
|   |   `-- source Datastream files and risk-free rate
|   `-- processed/
|       `-- generated outputs from Parts I, III, and IV
|-- src/
|   |-- p0_pipeline.py
|   |-- p1_1_data_cleaning.py
|   |-- p2_1_investment_set.py
|   |-- p2_2_minimum_variance_portfolio.py
|   |-- p2_3_value_weighted_portfolio.py
|   |-- p2_4_minvar_vs_valueweighted_comparison.py
|   |-- p3_0_carbon_utils.py
|   |-- p3_1_carbon_footprint.py
|   |-- p3_2_minimum_variance_carbon.py
|   |-- p3_3_tracking_error_carbon.py
|   |-- p3_4_carbon_comparison.py
|   |-- p4_1_net_zero.py
|   `-- p4_2_passive_comparison.py
|-- .gitignore
`-- README.md
```

## Main Scripts

### Project Order

- `p0_pipeline.py`
  Launches the full workflow from Part I to Part IV in one run.

- `p1_1_data_cleaning.py`
  Cleans the raw Datastream files and builds the processed Emerging Markets datasets.

- `p2_1_investment_set.py`
  Builds the investment set for Section 2.1 and computes expected returns and covariance matrices.

- `p2_2_minimum_variance_portfolio.py`
  Computes the long-only minimum-variance portfolio for Section 2.2.

- `p2_3_value_weighted_portfolio.py`
  Computes the value-weighted benchmark portfolio for Section 2.3.

- `p2_4_minvar_vs_valueweighted_comparison.py`
  Computes and saves the cumulative return comparison between the out-of-sample minimum-variance portfolio and the value-weighted benchmark.

- `p3_0_carbon_utils.py`
  Groups the shared carbon, optimization, and reporting helper functions used in Parts III and IV.

- `p3_1_carbon_footprint.py`
  Computes the annual WACI and carbon footprint of the reference minimum-variance and value-weighted portfolios.

- `p3_2_minimum_variance_carbon.py`
  Builds the long-only minimum-variance portfolio under a 50% carbon-footprint reduction constraint.

- `p3_3_tracking_error_carbon.py`
  Builds the tracking-error-minimizing passive portfolio under a 50% carbon-footprint reduction constraint.

- `p3_4_carbon_comparison.py`
  Compares the four portfolios of Part III on financial and carbon metrics.

- `p4_1_net_zero.py`
  Builds the passive net-zero portfolio with a tightening carbon-footprint path based on the 2013 baseline.

- `p4_2_passive_comparison.py`
  Compares the passive benchmark, the 50% carbon-reduction passive portfolio, and the net-zero passive portfolio.

## Pipeline

### Full Project Pipeline

Runs the whole project workflow from data cleaning to the end of Part IV:

```powershell
python src\p0_pipeline.py
```

## Tested Environment

The project was validated locally with:

- Python 3.11.4
- numpy 2.4.6
- pandas 2.2.3
- scipy 1.17.1
- openpyxl 3.1.5
- bottleneck 1.6.0

## Notes

- Scope 1 is the only emissions scope used in the analysis.
- The project uses annual portfolio formation years from 2013 to 2024 and monthly out-of-sample returns from 2014 to 2025.
- The code uses `freq="ME"` for month-end date ranges, so the tested pandas version matters for reproducibility.

## Results Snapshot

The main takeaways from a validated run of the project are the following:

### Part I

| Portfolio | Annualized Return | Annualized Volatility | Sharpe Ratio |
|--|--:|--:|--:|
| P(mv)_oos | 7.90% | 10.27% | 0.595 |
| P(vw) | 8.65% | 15.69% | 0.440 |

### Part III

| Portfolio | Annualized Return | Annualized Volatility | Sharpe Ratio | Average Annual CF |
|--|--:|--:|--:|--:|
| P(mv)_oos | 7.90% | 10.27% | 0.595 | 491.95 |
| P(mv)_oos(0.5) | 7.61% | 10.17% | 0.577 | 245.97 |
| P(vw) | 8.65% | 15.69% | 0.440 | 535.05 |
| P(vw)_oos(0.5) | 8.85% | 15.73% | 0.451 | 267.92 |

### Part IV

| Portfolio | Annualized Return | Annualized Volatility | Sharpe Ratio | Average Annual CF |
|--|--:|--:|--:|--:|
| P(vw) | 8.65% | 15.69% | 0.440 | 535.05 |
| P(vw)_oos(0.5) | 8.85% | 15.73% | 0.451 | 267.92 |
| P(vw)_oos(NZ) | 8.53% | 16.14% | 0.420 | 232.57 |

These values are intended as a compact reference point for the project report.

## Installation & Execution

### 1. Clone repository

```bash
git clone https://github.com/MattisD3/Groupe-X-Sustainability-Aware-Asset-Optimization.git
```

### 2. Open in PyCharm

Open the cloned folder in PyCharm.

### 3. Create a virtual environment

PyCharm:

File → Settings → Project → Python Interpreter → Add Interpreter → Virtual Environment

Recommended:

- Python 3.11.x

### 4. Install dependencies

Open terminal and run:

```bash
pip install -r requirements.txt
```

### 5. Run the full pipeline

```bash
python src/p0_pipeline.py
```

### Expected runtime

Validated locally:

- Full pipeline: ~24–26 minutes
- Tracking Error Carbon (3.3): ~7 minutes