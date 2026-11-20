# Experiment specification

- **Forecasting target:** consumption during the target hour `t+1`, using a prediction issued at forecast origin `t`.
- **Temporal split:** expanding training windows, validation in 2020, 2021 and 2022; independent 2023 holdout.
- **Predictors:** 16 calendar, 7 demand lag, 24 demand rolling, 9 price lag, 51 generation lag and 26 one-year-lagged OWID predictors, totaling 133 in the recorded run.
- **History windows:** demand lags `[1,2,3,24,48,72,168]`, rolling windows `[3,6,12,24,48,168]`, generation/price lags `[1,24,168]`.
- **Model development:** naive, linear, tree and sequence baselines; grid/random search, Optuna/TPE and differential evolution for selected tree-based models.
- **Model selection:** mean expanding-window validation RMSE, without independent test metrics in the selection objective.
- **Post-selection evaluation:** standard errors, subgroup error profiles, block-bootstrap intervals, Diebold–Mariano tests with Bartlett/Newey–West dependence estimates and Holm correction, conformal prediction intervals, SHAP, inference time and model size.

The ordered code in `steps/` contains the executable definitions and experimental procedures. Run it through `run_pipeline.py` to preserve shared state; this document provides a concise navigation guide, not an independent methodological specification.

**Operational-availability caveat:** Timestamp-based lagging alone is not equivalent to availability from a public market data feed at the forecast origin. Do not characterize these results as a validated public-data-only real-time deployment.
