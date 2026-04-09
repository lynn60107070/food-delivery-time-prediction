# Food delivery time prediction

Regression project (**DSAI4103-style business analytics / ML**) to predict **delivery time in minutes** from order, driver, route, and context features. The pipeline covers EDA → cleaning & feature engineering → feature exploration → AutoML-style modeling → explainability → **bias / disparity analysis** → **batch scoring** for deployment-style use and **Power BI**-friendly CSV exports.

**Target variable:** The label comes from the raw column `Time_taken(min)` (renamed to `delivery_time_min`). In the usual public dataset for this assignment, that measures **time to deliver after pickup** (restaurant → customer), not order-placed → delivered—confirm on your dataset page if you cite it in a report.

---

## Repository layout

| Path | Purpose |
|------|--------|
| `data/raw/` | Source CSVs (`train.csv`, `test.csv`, etc.) |
| `data/processed/` | **`cleaned_delivery_data.csv`** — cleaned + engineered training table from notebook 02 |
| `notebooks/` | Numbered notebooks **01–05** (run in order for a full rerun) |
| `src/` | Shared Python code (see **Source code** below) |
| `models/` | **`model_metadata.json`**, **`model_full.pkl`**, **`model_validation.pkl`**, optional **AutoGluon** folders (large; see `.gitignore`) |
| `reports/` | Figures under `figures/`, **`test_predictions.csv`**, **`validation_predictions.csv`** |
| `deployment/` | **`score.py`** (batch CSV scoring), **`simulate_scoring.py`** (append demo runs), **`append_test_prediction_date_range.py`** (custom date ranges), **`package_models.py`**, **`requirements-inference.txt`** |
| `deployment/dist/` | Optional output from `package_models.py` (bundled artifacts) |
| `bias_analysis/` | **`bias_report.ipynb`** — disparity metrics, Kruskal tests, plots on validation predictions |
| `dashboard/` | Power BI assets (e.g. `.pbix`); the published report is linked under **Reporting & Power BI** below |
| `shap/` | Optional SHAP figures from explainability work |

---

## Environment setup

**Python 3.10+** recommended. From the **repository root**:

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is **version-pinned** (including **`numpy` 2.x**, **`scipy`**, **`flaml[automl]`**, **`xgboost`**, **`shap`**, etc.). Recreate the venv when bumping Python or doing a clean install.

**Optional:**

- **Jupyter / IPython** — `pip install jupyter ipykernel` if you run notebooks outside VS Code / Cursor.
- **AutoGluon** — only for AutoGluon cells in `04_modeling.ipynb` (large install); see the comment at the top of `requirements.txt`.

**Inference only** (load `*.pkl` and score CSVs without full training stack):

```bash
pip install -r deployment/requirements-inference.txt
```

**Important:** Load `joblib` models with the **same scikit-learn version** (or very close) as when you trained in `04_modeling.ipynb`. Mismatched sklearn can break unpickling of `ColumnTransformer` / pipelines.

---

## Notebooks (run order)

| Notebook | Role |
|----------|------|
| **`01_eda.ipynb`** | Exploratory analysis on raw data |
| **`02_data_cleaning_feature_engineering.ipynb`** | Cleaning, feature engineering (distance, hour, peaks, etc.), **writes** `data/processed/cleaned_delivery_data.csv` |
| **`03_feature_selection.ipynb`** | Feature importance / selection views |
| **`04_modeling.ipynb`** | Baselines, tuned models, FLAML (and optional AutoGluon); saves **`models/model_metadata.json`**, **`model_validation.pkl`**, refits **`model_full.pkl`**, **`reports/test_predictions.csv`**, **`reports/validation_predictions.csv`** |
| **`05_explainability_bias.ipynb`** | SHAP / explainability and error-by-group plots (optional) |

Run Jupyter with the **working directory = project root** (or `sys.path` to root as the notebooks already do) so imports like `from src.data_preprocessing import ...` work.

---

## Source code (`src/`)

| Module | Role |
|--------|------|
| **`data_preprocessing.py`** | Rename/clean features, **`preprocess_pipeline`** (training), **`preprocess_for_scoring`** (no row-dropping outliers), **`load_clean_data`** |
| **`model_config.py`** | **`TARGET`**, **`DROP_COLS`** — aligned with `04_modeling.ipynb` |
| **`flaml_wrapper.py`** | **`FLAMLRegressorWrapper`** + **`register_notebook_pickles()`** so `joblib` models saved from Jupyter load outside the notebook |
| **`scoring.py`** | **`load_model`**, **`build_feature_matrix`**, **`predict_delivery_time`**, **`predict_delivery_time_preprocessed`**, **`add_predicted_sla_status`** (SLA label from predicted minutes) — shared by deployment scripts |

---

## Model artifacts

| File | Role |
|------|------|
| **`models/model_metadata.json`** | Best model name, feature lists, validation MAE/RMSE/R² |
| **`models/model_validation.pkl`** | Pipeline trained on **80% train** split — matches validation diagnostics and **`bias_report.ipynb`** |
| **`models/model_full.pkl`** | Pipeline refit on **full** labeled data — use for **`deployment/score.py`** default and final test predictions |

Large files (`*.pkl`, AutoGluon trees) are **gitignored** by default; commit `model_metadata.json` if you like, and keep pickles local or in artifact storage.

---

## Batch scoring (deployment)

**Scoring** here means **inference**: load the saved sklearn pipeline (`*.pkl`) and produce **`predicted_delivery_time_min`** (and optional **`predicted_SLA_status`**: On-Time if predicted minutes ≤ 30, else Delayed). Training happens in **`04_modeling.ipynb`**; deployment scripts only load artifacts and transform raw rows the same way as training.

### One-shot batch score (full file)

From the **repo root**, score a CSV whose columns match the **raw** training/test schema:

```bash
python deployment/score.py -i data/raw/test.csv -o reports/test_predictions.csv
```

Defaults: **`models/model_full.pkl`**, **`models/model_metadata.json`**. In a copied bundle where those files sit next to `score.py`, paths resolve automatically.

**Bundle** models + metadata + optional `src/` for offline use:

```bash
python deployment/package_models.py --out deployment/dist/model_bundle --include-src
```

### Simulated “live” scoring (demo / Power BI refresh)

For demos, **`deployment/simulate_scoring.py`** samples rows from raw test data, runs the same pipeline as **`score.py`**, and **appends** to **`reports/test_prediction.csv`** (singular — not `test_predictions.csv`). Each run adds **`simulation_batch_id`** and **`scored_at_utc`** (UTC ISO timestamps) so you can filter batches in Power BI.

This is **not** a live API; it is **batch simulation**: run the script → refresh the dataset in Power BI → visuals update. Rows include context columns such as **`distance_km`**, **`traffic_density`**, **`weather`**, **`order_hour`**, and **`num_deliveries`** where available.

```bash
python deployment/simulate_scoring.py --batch-size 30
```

Useful options:

| Flag | Purpose |
|------|--------|
| `--no-scenario-tweaks` | Score from real preprocessed rows (more natural On-Time vs Delayed mix). |
| (default tweaks on) | Stress-style inputs (longer distance, heavier traffic, peak hours) for dashboard demos. |
| `--demo-delay-bump` | Adds random 0 / 10 / 15 minutes to predictions after scoring (demo only). |
| `--seed` | Reproducible sampling and tweaks. |
| `--source`, `--output` | Input CSV and append target; override model paths like `score.py`. |

### Date-range append (custom `scored_at_utc` span)

**`deployment/append_test_prediction_date_range.py`** appends rows with timestamps spread across a **chosen inclusive date range** (default Jan 1–Apr 8, 2026). It guarantees **at least one row per calendar day** in that range; extra rows are random across those days. Use this when the dashboard needs a believable time axis.

```bash
python deployment/append_test_prediction_date_range.py --start-date 2026-01-01 --end-date 2026-04-08 --total-rows 2000 --no-scenario-tweaks --seed 42
```

| Flag | Purpose |
|------|--------|
| `--start-date`, `--end-date` | Inclusive `YYYY-MM-DD` bounds for **`scored_at_utc`**. |
| `--total-rows` | Rows per run; must be ≥ number of days in the range. |
| `--runs` | Run multiple disjoint batches (one model load); multiplies rows and varies samples. |
| `--no-scenario-tweaks` / `--mild-scenario-tweaks` | Natural or lightly varied inputs vs default stress tweaks. |

Requires enough rows in **`--source`** (default `data/raw/test.csv`): **`runs × total-rows`** for multi-run mode.

---

## Bias analysis

**`bias_analysis/bias_report.ipynb`** loads **`model_validation.pkl`**, reproduces the **same validation split** as `04_modeling.ipynb` (`test_size=0.2`, `random_state=42`), and reports **MAE disparity ratios**, **Kruskal–Wallis** tests on absolute error by group, and plots. Extended sections slice **all** input features (categoricals + binned numerics). Run it after the model exists and the env matches training.

---

## Reporting & Power BI

### Published dashboard

**[Open the Power BI report (cloud)](https://app.powerbi.com/links/b8FRH8WhHd?ctid=b30f4b44-46c6-4070-9997-f87b38d4771c&pbi_source=linkShare)** — interactive dashboard for this project (sign-in may be required for your organization).

The report has **four pages**:

| Page | Purpose |
|------|--------|
| **Overview** | High-level **performance** — how the solution is doing at a glance. |
| **Analysis** | **Factors that drive delay** — explore which inputs and conditions line up with longer delivery times / SLA risk. |
| **Model performance (historical)** | Quality on **historical** labeled data (actual vs predicted, errors, slices). |
| **Live model prediction** | **Deployable, operational-style** view: simulated scoring runs (**`test_prediction.csv`**) show predictions updating as new batches are scored — a realistic stand-in for a **live / real-time** monitoring workflow (refresh the dataset after each run; a production API would call the same **`deployment/score.py`** pipeline). |

Store local **`.pbix`** or supporting files under **`dashboard/`** in this repo; the link above is the **published** version for sharing and demos.

### Data connections

Power BI does **not** execute `*.pkl` models inside the service; it reads **tabular outputs** you produce in Python. Typical **CSV** sources:

| File | Use |
|------|-----|
| **`data/processed/cleaned_delivery_data.csv`** | EDA — actual `delivery_time_min`, full cleaned features |
| **`reports/validation_predictions.csv`** | Model quality — actual vs predicted, errors, slices (`city`, etc.) |
| **`reports/test_predictions.csv`** | One-shot batch scoring from **`deployment/score.py`** or export from **`04_modeling.ipynb`** — full test set predictions |
| **`reports/test_prediction.csv`** | **Append-only** simulated scoring (`simulate_scoring.py`, **`append_test_prediction_date_range.py`**) — includes **`scored_at_utc`**, **`simulation_batch_id`**, **`predicted_SLA_status`** for time-series and SLA views |

Use **Import** mode and **Refresh** the dataset after regenerating CSVs (or use DirectQuery if you load the same schema from a database).

---

## Requirements files

- **`requirements.txt`** — Pinned versions for training, notebooks, SHAP, and **`flaml[automl]`** (see file header for optional AutoGluon).
- **`deployment/requirements-inference.txt`** — Pinned minimal set to **load** the saved sklearn + FLAML pipeline and run **`deployment/score.py`**.

---

## Git / GitHub

**`.gitignore`** excludes virtualenvs, **`*.pkl`**, AutoGluon artifact trees, regenerated **`deployment/dist/`**, secrets (`.env`, keys), Python caches, test/coverage artifacts, IDE folders (`.idea/`, `.cursor/`), logs, and common ML experiment folders (`wandb/`, `mlruns/`). CSVs, notebooks, and `model_metadata.json` can stay tracked; large model binaries should not be pushed unless you use Git LFS.

---

## License / academic use

Course / academic project. 
