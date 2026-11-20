# Data sources and provenance

**Hourly electricity-market dataset:** The manuscript cites the public dataset at https://www.kaggle.com/datasets/ahmetzamanis/energy-consumption-and-pricing-trkiye-2018-2023 and its originator's project at https://github.com/AhmetZamanis/DeepLearningEnergyForecasting . This originator repository is **not** the study-specific MS-XAI-Load-Opt code.

**Annual energy indicators:** OWID Energy data are available at https://github.com/owid/energy-data . The pipeline uses the Türkiye subset, retaining annual indicators complete for the required 2016–2022 years and applying lagged annual alignment. The exact downloaded snapshot/version must be recorded to reproduce the original result because OWID data may be revised.

**Required files:**

| File | Expected contents |
|---|---|
| `hourly_energy_data.csv` | Hourly consumption in `consumption_MWh`, price and generation columns, and a parseable timestamp; 2018–2023 coverage |
| `owid_energy_data.csv` | OWID country-year data with `country`, `year`, and required energy indicators |

The scripts audit input columns, timestamps, duplicate rows, missing data, OWID availability and file SHA-256 fingerprints. Expected final predictor count for the provided completed run is **133**. Reusing a different OWID snapshot may change retained indicators.

**Reproducibility requirement:** For a public release, provide a precise source version/DOI or dated archival snapshot plus the input SHA-256 values from the completed execution. The original CSV files were subsequently provided, and their **complete SHA-256 fingerprints and byte sizes were confirmed to match the input data embedded in the executed master notebook**. Exact fingerprints and dataset dimensions are recorded in `../metadata/dataset_fingerprints.json`. This verifies the input snapshot identity, not a fresh full model run.

To verify inputs on another computer, execute `python verify_inputs.py --data-dir data`. Do not publicly redistribute the raw CSV files without checking applicable source licenses and reuse terms.
