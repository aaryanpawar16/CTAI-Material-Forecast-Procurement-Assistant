#!/usr/bin/env python3
"""
Streamlit app for CTAI — Material Forecast & Procurement Assistant
Covers:
 - Stage 1: Predictions
 - Stage 3: Vendor lookup
 - Stage 4: Procurement outputs & integrated Gantt
 - Stage 5: Procurement plan view + summary
 - Stage 6: Procurement request workflow & dashboard
"""

import os, sys, subprocess, json, datetime
import pandas as pd
import streamlit as st

# ---------------------------
# Setup project root
# ---------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------
# Try imports
# ---------------------------
try:
    from src.predict import predict_on_test
except Exception as e:
    predict_on_test, predict_import_error = None, e
else:
    predict_import_error = None

try:
    from src.vendor_scraper import simple_search_scrape, scrape_indiamart
except Exception as e:
    simple_search_scrape, scrape_indiamart, vendor_import_error = None, None, e
else:
    vendor_import_error = None

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
        st.write("Files in root:", os.listdir(PROJECT_ROOT)[:12])
    except Exception:
        st.write("Unable to list project root files.")

    if predict_on_test is None:
        st.error(f"Prediction module error: {predict_import_error}")
    else:
        st.success("Prediction module OK")

    if simple_search_scrape is None:
        st.warning(f"Vendor scraper error: {vendor_import_error}")
        st.info("Will fallback to vendor_data/vendors.json")
    else:
        st.success("Vendor scraper OK")

st.markdown("---")

# ==============================================================
# Stage 1 — Predictions
# ==============================================================
with st.expander("Upload test CSV (optional)"):
    uploaded = st.file_uploader("Upload test.csv", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/test.csv", index=False)
            st.success("Saved to data/test.csv")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to save uploaded CSV: {e}")

if st.button("Run baseline prediction"):
    if predict_on_test is None:
        st.error("Prediction unavailable.")
    else:
        try:
            st.info("Running predictions...")
            sub = predict_on_test("data/test.csv")
            st.success("Prediction complete")
            st.dataframe(sub.head(50))
            csv_data = sub.to_csv(index=False)
            os.makedirs("submission", exist_ok=True)
            sub.to_csv("submission/submission.csv", index=False)
            st.download_button("Download submission.csv", csv_data, file_name="submission.csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")

# ==============================================================
# Stage 3 — Vendor Lookup
# ==============================================================
st.header("Vendor Lookup (Stage 3)")

def _get_top_submission_items(path="submission/submission.csv", topn=5):
    if not os.path.exists(path): return []
    try:
        sub = pd.read_csv(path)
        return sub["MasterItemNo"].value_counts().head(topn).index.astype(str).tolist()
    except Exception:
        return []


def _build_item_desc_map(path="data/train.csv"):
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
                if scrape_indiamart: vendors = scrape_indiamart(keyword, max_results=8)
                if not vendors and simple_search_scrape: vendors = simple_search_scrape(keyword, max_results=8)
            except Exception:
                pass
            if not vendors and os.path.exists("vendor_data/vendors.json"):
                try:
                    with open("vendor_data/vendors.json") as f:
                        jd = json.load(f)
                        vendors = [v for entry in jd.get("items",[]) if str(entry.get("MasterItemNo"))==itm for v in entry.get("vendors",[])]
                except Exception:
                    pass
            if vendors:
                st.success(f"Found {len(vendors)} vendors for {keyword}")
                st.dataframe(pd.DataFrame(vendors))
            else:
                st.warning(f"No vendors found for {keyword}")

st.markdown("Manual search:")
keyword = st.text_input("Enter item keyword")
if st.button("Search vendors (manual)") and keyword:
    vendors = []
    try:
        if scrape_indiamart: vendors = scrape_indiamart(keyword, max_results=8)
        if not vendors and simple_search_scrape: vendors = simple_search_scrape(keyword, max_results=8)
    except Exception:
        pass
    if vendors:
        st.dataframe(pd.DataFrame(vendors))
    else:
        st.warning("No vendor results found.")

st.markdown("---")

# ==============================================================
# Stage 4 — Procurement Outputs & Gantt
# ==============================================================
st.header("Procurement Outputs (Stage 4)")
cols = st.columns(3)

if os.path.exists("procurement/procurement_tasks.csv"):
    cols[0].download_button("Download procurement_tasks.csv", data=open("procurement/procurement_tasks.csv","rb").read(), file_name="procurement_tasks.csv")
else:
    cols[0].warning("procurement_tasks.csv not found")

if os.path.exists("procurement/gantt_chart_fast.png"):
    cols[1].image("procurement/gantt_chart_fast.png", caption="Gantt (static)", use_container_width=True)
else:
    cols[1].info("gantt_chart_fast.png not found")

if os.path.exists("procurement/gantt_chart.html"):
    if cols[2].button("Embed Gantt chart"):
        import streamlit.components.v1 as components
        with open("procurement/gantt_chart.html") as f: components.html(f.read(), height=700, scrolling=True)
else:
    cols[2].info("procurement/gantt_chart.html not found")

st.markdown("---")

# ==============================================================
# Stage 5 — Procurement Plan
# ==============================================================
st.header("Procurement Plan (Stage 5)")

if os.path.exists("procurement/procurement_plan.csv"):
    df_plan = pd.read_csv("procurement/procurement_plan.csv")
    st.dataframe(df_plan.head(50))
    st.download_button("Download plan CSV", data=open("procurement/procurement_plan.csv","rb").read(), file_name="procurement_plan.csv")
else:
    st.info("procurement_plan.csv not found")

if os.path.exists("procurement/procurement_plan_summary.md"):
    with open("procurement/procurement_plan_summary.md") as f:
        st.markdown(f.read())
else:
    st.info("procurement_plan_summary.md not found")

st.markdown("---")

# ==============================================================
# Stage 6 — Procurement Requests
# ==============================================================
st.header("Procurement Requests (Stage 6 prototype)")

REQUESTS_PATH = "procurement/requests.json"
os.makedirs("procurement", exist_ok=True)


def load_requests():
    """Load requests JSON robustly and normalize to a list of dicts.

    Accepts either:
      - a list of dicts
      - a dict like {"requests": [...]} or {"data": [...]} 
      - a list of strings/ids (will be wrapped)
      - mixed content
    """
    if not os.path.exists(REQUESTS_PATH):
        return []
    try:
        raw = json.load(open(REQUESTS_PATH))
    except Exception:
        return []

    # If top-level is a dict with a container key, extract it
    if isinstance(raw, dict):
        if 'requests' in raw and isinstance(raw['requests'], list):
            raw = raw['requests']
        elif 'data' in raw and isinstance(raw['data'], list):
            raw = raw['data']
        else:
            # unknown dict shape -> try to convert to single-item list
            raw = [raw]

    normalized = []
    for item in raw:
        if isinstance(item, dict):
            normalized.append(item)
        elif isinstance(item, str):
            # try parse JSON string
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    normalized.append(parsed)
                else:
                    normalized.append({'id': parsed, 'status': 'Pending'})
            except Exception:
                normalized.append({'id': item, 'status': 'Pending'})
        else:
            # e.g., a number
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
    # write as a plain list of dicts
    with open(REQUESTS_PATH, "w") as f:
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
    st.dataframe(pd.DataFrame(reqs))

    # render rows defensively and use unique button keys
    for i, r in enumerate(reqs):
        c1, c2, c3 = st.columns([2,2,4])
        req_id = r.get('id', f'row_{i}')
        status = r.get('status', 'Pending')
        c1.write(f"Req #{req_id}")
        c2.write(status)

        # unique keys for streamlit buttons
        approve_key = f"approve_{req_id}"
        reject_key = f"reject_{req_id}"

        if c3.button("Approve", key=approve_key):
            r['status'] = "Approved"
            save_requests(reqs)
            st.experimental_rerun()

        if c3.button("Reject", key=reject_key):
            r['status'] = "Rejected"
            save_requests(reqs)
            st.experimental_rerun()
else:
    st.info("No requests yet")

st.markdown("---")

# ==============================================================
# Quick Actions
# ==============================================================
with st.expander("Quick Actions"):
    if st.button("Generate procurement tasks"):
        subprocess.run([sys.executable, "procurement/generate_procurement_tasks_from_predictions.py"])
        st.success("Generated procurement_tasks.csv")
    if st.button("Generate procurement plan"):
        subprocess.run([sys.executable, "procurement/generate_procurement_plan.py"])
        st.success("Generated procurement_plan.csv & summary")
    if st.button("Generate Gantt"):
        subprocess.run([sys.executable, "procurement/generate_gantt_integrated.py"])
        st.success("Generated integrated Gantt")
