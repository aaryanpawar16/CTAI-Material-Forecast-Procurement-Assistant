# CTAI CTD Hackathon — Material Forecast & Procurement

## Overview
Predict MasterItemNo and QtyShipped for construction projects and build a procurement plan + vendor discovery.

## Repo structure
(Describe the folders — copy the folder structure included earlier.)

## Quickstart (run baseline prediction)

1. Create virtual env and install requirements:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
2.Preprocess (optional):
python -m src.data_preprocessing --input data/train.csv --output data/train_preprocessed.csv
3.Train classifier (baseline):
python -m src.train_model --train data/train.csv --out artifacts
4.Create submission using baseline trainer:
python -m src.make_submission --test data/test.csv --out submission/submission.csv
5.Run the app (optional):
cd app
streamlit run streamlit_app.py
Submission
Put submission.csv, README.md, and code files into a zip named submission.zip and upload per hackathon instructions.
Notes
This repo contains baseline implementations; tune models and features for better performance.
Ensure data/test.csv has id column before running predictions.

---

## notebooks/ (scaffolds)

Each notebook below is a simple scaffold you can paste into a Jupyter notebook. Replace the .ipynb placeholders with real notebooks in your repo.

### notebooks/01_data_exploration.ipynb (scaffold content as markdown)

```markdown
# 01_data_exploration

- Load train/test
- Basic value counts, missing value table
- Distribution of QtyShipped
- Top MasterItemNo
notebooks/02_feature_engineering.ipynb
# 02_feature_engineering

- Run src.data_preprocessing.preprocess
- Create features: project_duration_days, line_pct_of_invoice, invoice_month/dayofweek
- Encode text features using TF-IDF for ItemDescription
notebooks/03_model_training.ipynb
# 03_model_training

- Fit classifier baseline (LightGBM)
- Fit regressor baseline (LightGBM or median fallback)
- Cross-validation with GroupKFold
- Save artifacts
notebooks/04_submission_generation.ipynb
# 04_submission_generation

- Load artifacts
- Run inference on test set
- Save submission/submission.csv
```}