# Publishing this code on GitHub

1. Create a new **public** repository named `MS-XAI-Load-Opt` at https://github.com/new. Use the repository owner's choice of GitHub account.
2. **Do not initialize it with an additional README, license or .gitignore**; these already exist in the source package (except for the license, which needs an explicit author decision).
3. Extract the ZIP and upload **the contents of the extracted folder**, including hidden `.github/` and `.gitignore` files. Uploading the ZIP as a single GitHub file is not equivalent to publishing the source tree.
4. Confirm that `README.md`, `steps/`, `run_pipeline.py`, `verify_inputs.py`, `metadata/`, `docs/`, `tests/` and `requirements.txt` are visible on the repository homepage.
5. Check the **Actions** tab for a passing *Repository integrity* workflow. This performs static checks only, not model training.
6. Obtain the two CSV datasets from their original providers; check their fingerprints locally with `python verify_inputs.py --data-dir data`.
7. Execute the full pipeline, compare all reported outputs, and review the reproduction checklist before describing the project as end-to-end verified.
8. Once the result audit is complete, create a tagged release (for example `v1.0.0`) and link to the public repository in the article's Data Availability section. A tagged, archived snapshot can subsequently support a persistent DOI.

The software license has deliberately been left unspecified. Public visibility permits inspection; broad reuse rights require a license chosen by the copyright holders. No raw datasets or full notebook are included.
