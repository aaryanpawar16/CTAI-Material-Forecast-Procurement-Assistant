CTAI — Material Forecast & Procurement Assistant
📌 Overview

CTAI is an AI-powered assistant for construction procurement planning.
It predicts required materials & equipment (Stage 1), hosts an interactive Streamlit app (Stage 2), integrates vendor lookup via web scraping (Stage 3), aligns procurement with the project schedule in Gantt charts (Stage 4), generates a procurement plan & summary (Stage 5), and includes a lightweight procurement request workflow + dashboard (Stage 6).

⚙️ Installation & Setup
1️⃣ Clone repository
git clone <your_github_repo_url>
cd CTAI

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the app locally
streamlit run app/streamlit_app.py

📊 Workflow

Train model

python -m src.train_model_full --train data/train.csv


Generate predictions

python -m src.predict


Evaluate submission (Stage 1 scoring)

python src/evaluate_submission.py data/val_ground_truth.csv submission/submission.csv


Generate procurement tasks

python procurement/generate_procurement_tasks_from_predictions.py


Generate procurement plan & summary

python procurement/generate_procurement_plan.py


Generate integrated Gantt

python procurement/generate_gantt_integrated.py


Link approved requests (Stage 6)

python procurement/link_approved_requests.py

🌐 Deployment

The app is deployed and publicly accessible here:
👉 [Live Demo Link](https://ctai-material-forecast-procurement-assistant.streamlit.app/)

📂 Repository

Public GitHub repository link:
👉 GitHub Repo

📑 Submission

submission.csv → Final predictions

src/ → ML model training, prediction, evaluation, vendor scraper

procurement/ → Procurement plan, Gantt integration, request workflow

app/streamlit_app.py → Streamlit interface

requirements.txt → Dependencies

README.md → This file

