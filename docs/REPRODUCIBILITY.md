# Verification and release checklist

## Completed checks

- [x] The supplied master experiment was partitioned into **32 ordered source steps** (31 Python scripts and one installation script).
- [x] All 31 Python scripts pass syntax compilation and match the ordered source manifest.
- [x] No `.ipynb`, source CSV, saved model, fitted results or notebook execution state is included in the public release.
- [x] The package includes a single shared-namespace runner and automated source-integrity tests.
- [x] No reviewer-response or manuscript-revision notes remain in the released experimental scripts.
- [x] The two supplied input CSVs were checked against full SHA-256 hashes, file sizes, schemas and coverage recorded in the previous completed run.
- [x] The recorded configuration uses three expanding validation folds (2020–2022) and a 2023 held-out test year.

## Not yet established by these checks

- [ ] Install and run the exact target software stack on a fresh machine or reproducible runtime.
- [ ] Execute **all** pipeline stages from preprocessing to final exports without errors.
- [ ] Reconcile every newly generated benchmark result, diagnostic, statistical test and figure with the published article and supplement.
- [ ] Verify Figure 9's period labels against the saved plotting data.
- [ ] Confirm the upstream data licenses and permitted redistribution terms if distributing raw CSVs separately.
- [ ] Choose and explicitly add a software license if reusable open-source rights are intended.
- [ ] Publish a tagged release and optionally archive it with a persistent DOI after numerical reproduction.

## Code execution model

Every numbered stage was extracted from a single sequential research workflow, with some shared global state and function definitions. `run_pipeline.py` is the supported execution interface; individual stage scripts are **not** guaranteed to run independently. Step 01 installs dependencies and is not executed automatically by the runner. Runtime flags are documented in `README.md`.

## Provenance

The metadata files `reported_run_summary.json` and `reported_run_configuration.json` are reference exports from a previously completed research execution. They are not a second validation of reported performance. `dataset_fingerprints.json` identifies the **exact** user-provided input snapshots and supports input-level provenance checks without redistributing the datasets.

## Scientific limitations

Timestamp-based input shifting does not establish real-time public availability. Forecast-model SHAP values should not be interpreted causally. Held-out comparisons must not be used for selecting the final model after the fact.
