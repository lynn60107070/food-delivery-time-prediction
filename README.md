# Food delivery time prediction

Regression project to predict **delivery time (minutes)** from order, driver, route, and context features. It includes notebooks for EDA through modeling, optional explainability and bias checks, exported predictions for reporting, and a small batch **scoring** workflow for deployment-style inference.

## Project layout

| Path | Purpose |
|------|--------|
| `data/raw/` | Source CSVs (e.g. `train.csv`, `test.csv`) |
| `data/processed/` | Cleaned, feature-engineered training table (`cleaned_delivery_data.csv`) |
| `notebooks/` | Analysis and modeling (run in order below) |
| `src/` | Shared Python helpers (`data_preprocessing.py`, `scoring.py`, `model_config.py`, `flaml_wrapper.py`) |
| `models/` | Serialized models (`*.pkl`), `model_metadata.json`, optional AutoGluon runs (large; often gitignored) |
| `reports/` | Figures, `test_predictions.csv`, `validation_predictions.csv` |
| `deployment/` | Batch scorer CLI, packaging script, `requirements-inference.txt` |
| `bias_analysis/`, `shap/` | Optional analysis outputs |

## Environment

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Training and AutoML in `04_modeling.ipynb` may need extra packages (e.g. **FLAML**, **AutoGluon**); install as required when you run those cells.

**Inference only** (load saved pipeline and score CSVs):

```bash
pip install -r deployment/requirements-inference.txt
```

## Notebook pipeline

1. **`01_eda.ipynb`** — Exploratory analysis on raw data.  
2. **`02_data_cleaning_feature_engineering.ipynb`** — Cleaning, features (distance, time, peaks, etc.), saves `data/processed/cleaned_delivery_data.csv`.  
3. **`03_feature_selection.ipynb`** — Feature importance / selection context.  
4. **`04_modeling.ipynb`** — Baselines, tuned models, FLAML / optional AutoGluon; train/validation split; saves best pipeline(s), `models/model_metadata.json`, `reports/test_predictions.csv`, and **`reports/validation_predictions.csv`** (actual vs predicted on validation).  
5. **`05_explainability_bias.ipynb`** — SHAP-style explainability and error-by-group checks (optional).

## Model outputs

- **`models/model_metadata.json`** — Target name, feature lists, validation metrics, best model metadata.  
- **`models/model_full.pkl`** — Fitted pipeline for production-style scoring (after full-data refit in the notebook).  
- **`models/model_validation.pkl`** — Pipeline fit without the final full-data refit (optional).

Large binaries and AutoGluon trees are listed in `.gitignore`; keep them locally or store elsewhere if you share the repo without artifacts.

## Batch scoring (deployment prep)

From the repository root, score a CSV with the **same raw column names** as training/test:

```bash
python deployment/score.py -i data/raw/test.csv -o reports/test_predictions.csv
```

Defaults expect `models/model_full.pkl` and `models/model_metadata.json`. The script uses `src/scoring.py` and preprocessing aligned with training (`preprocess_for_scoring`).

**Package** model artifacts + optional `src/` copy:

```bash
python deployment/package_models.py --out deployment/dist/model_bundle --include-src
```

## Reporting / dashboards

Typical CSVs for analytics or Power BI:

- **`data/processed/cleaned_delivery_data.csv`** — EDA on historical training data.  
- **`reports/validation_predictions.csv`** — Validation actual vs predicted, errors, and a few slice columns.  
- **`reports/test_predictions.csv`** — Scored test rows (predictions; usually no labels).

## License / course use

Academic project (e.g. Business Analytics). Dataset and use follow your course and data provider terms.
