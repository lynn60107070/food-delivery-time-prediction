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
| `deployment/` | **`score.py`** (CLI), **`package_models.py`**, **`requirements-inference.txt`** |
| `deployment/dist/` | Optional output from `package_models.py` (bundled artifacts) |
| `bias_analysis/` | **`bias_report.ipynb`** — disparity metrics, Kruskal tests, plots on validation predictions |
| `dashboard/` | Place Power BI (`.pbix`) or other dashboard assets here |
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
| **`scoring.py`** | **`load_model`**, **`build_feature_matrix`**, **`predict_delivery_time`** for scripts and reuse |

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

From the **repo root**, score a CSV whose columns match **raw** training/test schema (same as Kaggle-style input):

```bash
python deployment/score.py -i data/raw/test.csv -o reports/test_predictions.csv
```

Defaults: **`models/model_full.pkl`**, **`models/model_metadata.json`**. In a copied bundle where those files sit next to `score.py`, paths resolve automatically.

**Bundle** models + metadata + optional `src/` for offline use:

```bash
python deployment/package_models.py --out deployment/dist/model_bundle --include-src
```

---

## Bias analysis

**`bias_analysis/bias_report.ipynb`** loads **`model_validation.pkl`**, reproduces the **same validation split** as `04_modeling.ipynb` (`test_size=0.2`, `random_state=42`), and reports **MAE disparity ratios**, **Kruskal–Wallis** tests on absolute error by group, and plots. Extended sections slice **all** input features (categoricals + binned numerics). Run it after the model exists and the env matches training.

---

## Reporting & Power BI

Typical **CSV** inputs:

| File | Use |
|------|-----|
| **`data/processed/cleaned_delivery_data.csv`** | EDA — actual `delivery_time_min`, full cleaned features |
| **`reports/validation_predictions.csv`** | Model quality — actual vs predicted, errors, slices (`city`, etc.) |
| **`reports/test_predictions.csv`** | Deployment — predictions on holdout test (usually **no** true delivery time) |

Power BI does **not** run `*.pkl` inside the report; connect to these CSVs (or to a database you populate with the same outputs).

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
