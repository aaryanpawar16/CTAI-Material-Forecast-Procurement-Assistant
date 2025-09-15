#!/usr/bin/env python3
"""
Streamlit app for CTAI — Material Forecast & Procurement Assistant
Covers:
 - Stage 1: Predictions
 - Stage 3: Vendor lookup
 - Stage 4: Procurement outputs & integrated Gantt
 - Stage 5: Procurement plan view + summary
 - Stage 6: Procurement request workflow & dashboard

Enhancements in this version:
 - tries to ensure a classifier artifact is available by downloading from a user-provided URL
   (set env var MODEL_URL or add MODEL_URL to Streamlit secrets) or prompting the user to train/upload the model if missing.
 - more defensive imports and clearer diagnostics for missing pieces.
 - improved MODEL_URL resolution and clearer download diagnostics.
"""

import os
import sys
import subprocess
import json
import datetime
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# Optional download libs (used only if model URL provided)
try:
    import requests
except Exception:
    requests = None

try:
    import gdown
except Exception:
    gdown = None

# ---------------------------
# Setup project root
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CLASSIFIER_ARTIFACT = os.path.join(ARTIFACTS_DIR, "classifier_trainer.pkl")
LABEL_ENCODER_ARTIFACT = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------
# Try imports (deferred / defensive)
# ---------------------------
predict_on_test = None
predict_import_error = None
simple_search_scrape = None
scrape_indiamart = None
vendor_import_error = None


def _resolve_model_url_from_env_or_secrets():
    """Return a MODEL_URL string from environment variable or Streamlit secrets if present."""
    # Priority: ENV var > Streamlit secrets
    env_val = os.environ.get("MODEL_URL")
    if env_val and str(env_val).strip():
        return str(env_val).strip()
    # Streamlit secrets (when running on Streamlit Cloud / using secrets.toml)
    try:
        if hasattr(st, "secrets") and isinstance(st.secrets, dict) and st.secrets.get("MODEL_URL"):
            return str(st.secrets.get("MODEL_URL")).strip()
    except Exception:
        # silence any streamlit secrets access errors
        pass
    return None


def ensure_model_available():
    """
    Ensure classifier artifact is present. Strategy:
      1) If CLASSIFIER_ARTIFACT exists -> OK
      2) If MODEL_URL is set (env or Streamlit secrets) -> try to download (supports Google Drive via gdown or direct URL via requests)
      3) Else: instruct user to either (a) push artifact via Git LFS / vendor storage or (b) run training locally.
    Returns (bool_ok, message)
    """
    if os.path.exists(CLASSIFIER_ARTIFACT):
        return True, "Artifact found locally."

    model_url = _resolve_model_url_from_env_or_secrets()
    if not model_url:
        return False, (
            "Model artifact not found. Set env var MODEL_URL to a publicly downloadable URL (Google Drive/HF/S3) "
            "or place classifier_trainer.pkl into the artifacts/ directory. Or run training locally:\n\n"
            "`python -m src.train_model_full --train data/train.csv`"
        )

    # attempt download with diagnostics
    try:
        parsed = None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(model_url)
        except Exception:
            parsed = None

        # Google Drive share links are commonly used; prefer gdown when appropriate
        if "drive.google.com" in model_url and gdown:
            try:
                st.info("Downloading model via gdown from Google Drive...")
                gdown.download(model_url, CLASSIFIER_ARTIFACT, quiet=False)
                return os.path.exists(CLASSIFIER_ARTIFACT), "Downloaded via gdown."
            except Exception as e:
                return False, f"gdown download attempt failed: {e}\n{traceback.format_exc()}"

        # direct HTTP/HTTPS
        if requests and parsed and parsed.scheme in ("http", "https"):
            try:
                st.info("Downloading model via requests...")
                resp = requests.get(model_url, stream=True, timeout=60)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                # refuse to save HTML error pages
                if "text/html" in content_type.lower():
                    return False, f"Remote URL returned HTML content-type ({content_type}) — check URL/auth."
                os.makedirs(os.path.dirname(CLASSIFIER_ARTIFACT), exist_ok=True)
                with open(CLASSIFIER_ARTIFACT, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return os.path.exists(CLASSIFIER_ARTIFACT), "Downloaded via requests."
            except Exception as e:
                return False, f"HTTP download failed: {e}\n{traceback.format_exc()}"

        # fallback: try gdown even if not drive
        if gdown:
            try:
                st.info("Attempting download via gdown fallback...")
                gdown.download(model_url, CLASSIFIER_ARTIFACT, quiet=False)
                return os.path.exists(CLASSIFIER_ARTIFACT), "Downloaded via gdown fallback."
            except Exception as e:
                return False, f"gdown fallback failed: {e}\n{traceback.format_exc()}"

        return False, "Model URL provided but download libraries are unavailable in this environment."

    except Exception as e:
        return False, f"Download failed: {e}\n{traceback.format_exc()}"


# attempt to import predict (after ensuring artifact is available or at least giving clear info)
_model_ok, model_msg = True, "skipped"
try:
    # If artifact not present, try to ensure via download before importing predict
    model_ok_flag, msg = ensure_model_available()
    _model_ok, model_msg = model_ok_flag, msg
    # Now import predict module; predict will itself attempt to load trainer from artifacts
    from src.predict import predict_on_test  # type: ignore
except Exception as e:
    predict_on_test = None
    predict_import_error = f"{e}\n{traceback.format_exc()}"

# vendor scraper import
try:
    from src.vendor_scraper import simple_search_scrape, scrape_indiamart  # type: ignore
except Exception as e:
    simple_search_scrape = None
    scrape_indiamart = None
    vendor_import_error = f"{e}\n{traceback.format_exc()}"

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="CTAI Procurement Assistant", layout="wide")
st.title("CTAI — Material Forecast & Procurement Assistant")

# ---------------------------
# Diagnostics
# ---------------------------
with st.expander("Status / diagnostics", expanded=False):
    st.write(f"Project root: `{PROJECT_ROOT}`")
    try:
        st.write("Files in root:", os.listdir(PROJECT_ROOT)[:20])
    except Exception:
        st.write("Unable to list project root files.")

    # Model artifact status
    if os.path.exists(CLASSIFIER_ARTIFACT):
        st.success("Model artifact: classifier_trainer.pkl found in artifacts/")
    else:
        st.warning("Model artifact classifier_trainer.pkl not found.")
        st.info(model_msg)

    # Predict import
    if predict_on_test is None:
        st.error("Prediction import failed. See details below.")
        if predict_import_error:
            st.code(str(predict_import_error)[:800])
        st.info("If you want the app to auto-train the model on first run, run `src/train_model_full.py` locally and copy the artifact to artifacts/.")
    else:
        st.success("Prediction module OK.")

    # Vendor scraper status
    if simple_search_scrape is None:
        st.warning("Vendor scraper not available; falling back to vendor_data/vendors.json placeholders.")
        if vendor_import_error:
            st.code(str(vendor_import_error)[:800])
    else:
        st.success("Vendor scraper OK.")

st.markdown("---")

# ============================================================== #
# Stage 1 — Predictions
# ============================================================== #
with st.expander("Upload test CSV (optional)"):
    uploaded = st.file_uploader("Upload test.csv", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
            df.to_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"), index=False)
            st.success("Saved to data/test.csv")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to save uploaded CSV: {e}")

if st.button("Run baseline prediction"):
    if predict_on_test is None:
        st.error("Prediction function is unavailable. See diagnostics above.")
    else:
        try:
            st.info("Running predictions...")
            test_path = os.path.join(PROJECT_ROOT, "data", "test.csv")
            if not os.path.exists(test_path):
                st.warning("data/test.csv not found. Upload via the uploader above or run create_submission_placeholder.py")
            sub = predict_on_test(test_path)
            if not isinstance(sub, pd.DataFrame):
                st.error("predict_on_test did not return a DataFrame.")
            else:
                st.success("Prediction complete")
                st.dataframe(sub.head(50))
                csv_data = sub.to_csv(index=False)
                os.makedirs(os.path.join(PROJECT_ROOT, "submission"), exist_ok=True)
                sub.to_csv(os.path.join(PROJECT_ROOT, "submission", "submission.csv"), index=False)
                st.download_button("Download submission.csv", csv_data, file_name="submission.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)

st.markdown("---")

# ============================================================== #
# Stage 3 — Vendor Lookup
# ============================================================== #
st.header("Vendor Lookup (Stage 3)")

def _get_top_submission_items(path=os.path.join(PROJECT_ROOT,"submission","submission.csv"), topn=5):
    if not os.path.exists(path): return []
    try:
        sub = pd.read_csv(path)
        return sub["MasterItemNo"].value_counts().head(topn).index.astype(str).tolist()
    except Exception:
        return []

def _build_item_desc_map(path=os.path.join(PROJECT_ROOT,"data","train.csv")):
    if not os.path.exists(path): return {}
    try:
        df = pd.read_csv(path, usecols=["MasterItemNo", "ItemDescription"])
        df["MasterItemNo_str"] = df["MasterItemNo"].astype(str).str.strip()
        df["ItemDescription"] = df["ItemDescription"].fillna("").astype(str)
        grouped = df.groupby("MasterItemNo_str")["ItemDescription"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )
        return grouped.to_dict()
    except Exception:
        return {}

top_items = _get_top_submission_items()
item_desc_map = _build_item_desc_map()
quick_labels = [(it, " ".join(item_desc_map.get(it, f"Item {it}").split()[:5])) for it in top_items]

if quick_labels:
    st.markdown("**Quick Search** — click to lookup vendors")
    cols = st.columns(len(quick_labels))
    for i, (itm, lab) in enumerate(quick_labels):
        if cols[i].button(lab):
            keyword = lab
            vendors = []
            try:
                if scrape_indiamart:
                    vendors = scrape_indiamart(keyword, max_results=8)
                if not vendors and simple_search_scrape:
                    vendors = simple_search_scrape(keyword, max_results=8)
            except Exception:
                vendors = []
            if not vendors and os.path.exists(os.path.join(PROJECT_ROOT, "vendor_data", "vendors.json")):
                try:
                    with open(os.path.join(PROJECT_ROOT, "vendor_data", "vendors.json"), "r", encoding="utf-8") as f:
                        jd = json.load(f)
                        vendors = [v for entry in jd.get("items", []) if str(entry.get("MasterItemNo")) == itm for v in entry.get("vendors", [])]
                except Exception:
                    vendors = []
            if vendors:
                st.success(f"Found {len(vendors)} vendors for {keyword}")
                try:
                    st.dataframe(pd.DataFrame(vendors))
                except Exception:
                    st.json(vendors)
            else:
                st.warning(f"No vendors found for {keyword}")

st.markdown("Manual search:")
keyword = st.text_input("Enter item keyword")
if st.button("Search vendors (manual)") and keyword:
    vendors = []
    try:
        if scrape_indiamart:
            vendors = scrape_indiamart(keyword, max_results=8)
        if not vendors and simple_search_scrape:
            vendors = simple_search_scrape(keyword, max_results=8)
    except Exception:
        vendors = []
    if vendors:
        try:
            st.dataframe(pd.DataFrame(vendors))
        except Exception:
            st.json(vendors)
    else:
        st.warning("No vendor results found.")

st.markdown("---")

# ============================================================== #
# Stage 4 — Procurement Outputs & Gantt
# ============================================================== #
st.header("Procurement Outputs (Stage 4)")
cols = st.columns(3)

if os.path.exists(os.path.join(PROJECT_ROOT, "procurement", "procurement_tasks.csv")):
    try:
        cols[0].download_button(
            "Download procurement_tasks.csv",
            data=open(os.path.join(PROJECT_ROOT, "procurement", "procurement_tasks.csv"), "rb").read(),
            file_name="procurement_tasks.csv"
        )
    except Exception as e:
        cols[0].info("Could not attach procurement CSV for download.")
else:
    cols[0].warning("procurement_tasks.csv not found")

if os.path.exists(os.path.join(PROJECT_ROOT, "procurement", "gantt_chart_fast.png")):
    cols[1].image(os.path.join(PROJECT_ROOT, "procurement", "gantt_chart_fast.png"), caption="Gantt (static)", use_container_width=True)
else:
    cols[1].info("gantt_chart_fast.png not found")

if os.path.exists(os.path.join(PROJECT_ROOT, "procurement", "gantt_chart.html")):
    if cols[2].button("Embed Gantt chart"):
        try:
            import streamlit.components.v1 as components
            with open(os.path.join(PROJECT_ROOT, "procurement", "gantt_chart.html"), "r", encoding="utf-8") as f:
                components.html(f.read(), height=700, scrolling=True)
        except Exception as e:
            cols[2].error(f"Embed failed: {e}")
else:
    cols[2].info("procurement/gantt_chart.html not found")

st.markdown("---")

# ============================================================== #
# Stage 5 — Procurement Plan
# ============================================================== #
st.header("Procurement Plan (Stage 5)")

plan_csv = os.path.join(PROJECT_ROOT, "procurement", "procurement_plan.csv")
plan_md = os.path.join(PROJECT_ROOT, "procurement", "procurement_plan_summary.md")

if os.path.exists(plan_csv):
    try:
        df_plan = pd.read_csv(plan_csv)
        st.dataframe(df_plan.head(50))
        st.download_button("Download plan CSV", data=open(plan_csv, "rb").read(), file_name="procurement_plan.csv")
    except Exception as e:
        st.error(f"Failed to load procurement_plan.csv: {e}")
else:
    st.info("procurement_plan.csv not found")

if os.path.exists(plan_md):
    try:
        with open(plan_md, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except Exception as e:
        st.error(f"Failed to load plan summary: {e}")
else:
    st.info("procurement_plan_summary.md not found")

st.markdown("---")

# ============================================================== #
# Stage 6 — Procurement Requests
# ============================================================== #
st.header("Procurement Requests (Stage 6 prototype)")

REQUESTS_PATH = os.path.join(PROJECT_ROOT, "procurement", "requests.json")
os.makedirs(os.path.join(PROJECT_ROOT, "procurement"), exist_ok=True)

def load_requests():
    """Load requests JSON robustly and normalize to a list of dicts."""
    if not os.path.exists(REQUESTS_PATH):
        return []
    try:
        raw = json.load(open(REQUESTS_PATH, "r", encoding="utf-8"))
    except Exception:
        return []

    if isinstance(raw, dict):
        if 'requests' in raw and isinstance(raw['requests'], list):
            raw = raw['requests']
        elif 'data' in raw and isinstance(raw['data'], list):
            raw = raw['data']
        else:
            raw = [raw]

    normalized = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    normalized.append(parsed)
                else:
                    normalized.append({'id': parsed, 'status': 'Pending'})
            except Exception:
                normalized.append({'id': item, 'status': 'Pending'})
        else:
            normalized.append({'id': str(item), 'status': 'Pending'})

    # ensure every request has an integer id
    max_id = 0
    for r in normalized:
        if 'id' not in r:
            max_id += 1
            r['id'] = max_id
        else:
            try:
                r['id'] = int(r['id'])
                max_id = max(max_id, r['id'])
            except Exception:
                max_id += 1
                r['id'] = max_id
        if 'status' not in r:
            r['status'] = 'Pending'

    return normalized

def save_requests(reqs):
    with open(REQUESTS_PATH, "w", encoding="utf-8") as f:
        json.dump(reqs, f, indent=2, default=str)

with st.form("request_form"):
    item = st.text_input("Item")
    qty = st.number_input("Quantity", 1, value=1)
    needed_by = st.date_input("Needed by", value=datetime.date.today() + datetime.timedelta(days=30))
    notes = st.text_area("Notes")
    submitted = st.form_submit_button("Submit request")
    if submitted:
        reqs = load_requests()
        next_id = max([r.get('id', 0) for r in reqs] + [0]) + 1
        new = {"id": next_id, "item": item, "qty": int(qty), "needed_by": str(needed_by),
               "notes": notes, "status": "Pending", "created_at": str(datetime.datetime.now())}
        reqs.append(new)
        save_requests(reqs)
        st.success(f"Request #{new['id']} submitted")

reqs = load_requests()
if reqs:
    try:
        st.dataframe(pd.DataFrame(reqs))
    except Exception:
        pass

    for i, r in enumerate(reqs):
        c1, c2, c3 = st.columns([2,2,4])
        req_id = r.get('id', f'row_{i}')
        status = r.get('status', 'Pending')
        c1.write(f"Req #{req_id}")
        c2.write(status)

        approve_key = f"approve_{req_id}"
        reject_key = f"reject_{req_id}"

        def _set_status_and_save(reqs_list, target_id, new_status):
            # update in-memory list and save to disk
            for rr in reqs_list:
                if rr.get('id') == target_id:
                    rr['status'] = new_status
            save_requests(reqs_list)
            # try to reload immediately so the local 'reqs' reflects saved state
            try:
                updated = load_requests()
                reqs.clear()
                reqs.extend(updated)
            except Exception:
                pass

        if c3.button("Approve", key=approve_key):
            _set_status_and_save(reqs, req_id, "Approved")
            st.success(f"Request #{req_id} approved.")

        if c3.button("Reject", key=reject_key):
            _set_status_and_save(reqs, req_id, "Rejected")
            st.success(f"Request #{req_id} rejected.")

    if not reqs:
        st.info("No requests yet")

st.markdown("---")

# ============================================================== #
# Quick Actions
# ============================================================== #
with st.expander("Quick Actions"):
    if st.button("Generate procurement tasks"):
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "procurement", "generate_procurement_tasks_from_predictions.py")])
        st.success("Generated procurement_tasks.csv")
    if st.button("Generate procurement plan"):
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "procurement", "generate_procurement_plan.py")])
        st.success("Generated procurement_plan.csv & summary")
    if st.button("Generate Gantt"):
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "procurement", "generate_gantt_integrated.py")])
        st.success("Generated integrated Gantt")
