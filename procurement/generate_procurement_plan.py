#!/usr/bin/env python3
"""
Generate a synthetic procurement plan (Stage 5).

Outputs:
 - procurement/procurement_plan.csv         (detailed per-item plan)
 - procurement/procurement_plan_summary.md  (markdown summary + risks)

Inputs (preferred order):
 - submission/submission.csv
 - procurement/procurement_tasks.csv  (fallback)
 - vendor_data/vendors.json           (optional, for vendor mapping)
 - procurement/project_schedule.csv   (optional; else synthetic schedule used)
"""

import os
import json
from datetime import timedelta
import math
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
OUT_CSV = os.path.join(ROOT, "procurement_plan.csv")
OUT_MD = os.path.join(ROOT, "procurement_plan_summary.md")

SUBMISSION_PATH = os.path.join(os.path.dirname(ROOT), "submission", "submission.csv")
PROC_TASKS_PATH = os.path.join(ROOT, "procurement_tasks.csv")
VENDOR_JSON = os.path.join(os.path.dirname(ROOT), "vendor_data", "vendors.json")
PROJECT_SCHEDULE = os.path.join(ROOT, "project_schedule.csv")

DEFAULT_LEAD_DAYS = 21
DEFAULT_BUFFER_DAYS = 3
SAFETY_STOCK_PCT = 0.10  # 10% safety stock
MIN_ORDER_QTY = 1

def read_submission_or_tasks():
    # prefer submission
    if os.path.exists(SUBMISSION_PATH):
        try:
            df = pd.read_csv(SUBMISSION_PATH)
            # try to merge item description if present in a train mapping
            df.columns = [c.strip() for c in df.columns]
            if 'QtyShipped' not in df.columns:
                df.rename(columns={'Qty': 'QtyShipped'}, inplace=True)
            return df
        except Exception:
            pass
    # fallback
    if os.path.exists(PROC_TASKS_PATH):
        df = pd.read_csv(PROC_TASKS_PATH)
        return df
    raise FileNotFoundError("No submission/submission.csv or procurement/procurement_tasks.csv found. Run prediction/generator first.")

def load_vendor_db():
    if not os.path.exists(VENDOR_JSON):
        return {}
    try:
        with open(VENDOR_JSON, 'r', encoding='utf-8') as f:
            jd = json.load(f)
    except Exception:
        return {}
    mapping = {}
    for entry in jd.get('items', []):
        key = str(entry.get('MasterItemNo'))
        mapping[key] = entry.get('vendors', [])
    return mapping

def read_or_create_project_schedule():
    if os.path.exists(PROJECT_SCHEDULE):
        try:
            df = pd.read_csv(PROJECT_SCHEDULE, parse_dates=['Start','Finish'])
            return df
        except Exception:
            pass
    # synthetic (dates relative to today)
    today = pd.Timestamp.today().normalize()
    sched = [
        ("Design & Planning", today, today + pd.Timedelta(days=20)),
        ("Foundation & Civil Works", today + pd.Timedelta(days=21), today + pd.Timedelta(days=50)),
        ("Electrical & HVAC", today + pd.Timedelta(days=51), today + pd.Timedelta(days=80)),
        ("IT Infrastructure", today + pd.Timedelta(days=81), today + pd.Timedelta(days=95)),
        ("Commissioning & Handover", today + pd.Timedelta(days=96), today + pd.Timedelta(days=110)),
    ]
    df = pd.DataFrame(sched, columns=['Task','Start','Finish'])
    return df

def choose_vendor(master_itemno, vendor_db):
    key = str(master_itemno)
    vendors = vendor_db.get(key) or []
    if vendors:
        # prefer vendor with lead_time_days if present
        vendors_sorted = sorted(vendors, key=lambda v: v.get('lead_time_days') if v.get('lead_time_days') is not None else DEFAULT_LEAD_DAYS)
        return vendors_sorted[0]
    return None

def infer_item_phase(item_desc, project_df):
    """
    Heuristic: map common keywords to phases.
    """
    if pd.isna(item_desc):
        return None
    s = str(item_desc).lower()
    if any(k in s for k in ['cable','tray','transformer','breaker','panel','hvac','duct']):
        return 'Electrical & HVAC'
    if any(k in s for k in ['server','rack','network','fiber','switch']):
        return 'IT Infrastructure'
    if any(k in s for k in ['cement','concrete','steel','rebar','brick','block','mortar']):
        return 'Foundation & Civil Works'
    # fallback to Electrical if unknown
    # ensure the phase exists in project schedule
    if 'Electrical & HVAC' in project_df['Task'].values:
        return 'Electrical & HVAC'
    return project_df['Task'].iloc[0]

def coerce_numeric_qty(x):
    try:
        return float(x)
    except Exception:
        try:
            s = str(x)
            s = s.replace(',', '').strip()
            return float(s)
        except Exception:
            return np.nan

def generate_plan():
    items_df = read_submission_or_tasks()
    vendor_db = load_vendor_db()
    project_df = read_or_create_project_schedule()

    # ensure columns exist
    items_df = items_df.copy()
    # normalize column names
    items_df.columns = [c.strip() for c in items_df.columns]

    # prefer columns: id, MasterItemNo, QtyShipped, ItemDescription, needed_by_date
    if 'MasterItemNo' not in items_df.columns:
        raise ValueError("Input items missing MasterItemNo column.")

    # Qty numeric
    if 'QtyShipped' in items_df.columns:
        items_df['Qty_num'] = items_df['QtyShipped'].apply(coerce_numeric_qty)
    elif 'Qty' in items_df.columns:
        items_df['Qty_num'] = items_df['Qty'].apply(coerce_numeric_qty)
    else:
        items_df['Qty_num'] = 1.0

    # try to get ItemDescription from local train.csv if available
    train_map = {}
    train_csv = os.path.join(os.path.dirname(ROOT), 'data', 'train.csv')
    if os.path.exists(train_csv):
        try:
            df_train = pd.read_csv(train_csv, usecols=['MasterItemNo','ItemDescription'])
            df_train['MasterItemNo_str'] = df_train['MasterItemNo'].astype(str).str.strip()
            grouped = df_train.groupby('MasterItemNo_str')['ItemDescription'].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
            train_map = grouped.to_dict()
        except Exception:
            train_map = {}

    plan_rows = []
    for _, r in items_df.iterrows():
        master = r['MasterItemNo']
        master_str = str(master)
        qty = r.get('Qty_num', 1.0) if not pd.isna(r.get('Qty_num')) else 1.0
        qty = max(MIN_ORDER_QTY, int(round(qty)))

        desc = r.get('ItemDescription') or train_map.get(master_str) or r.get('Item') or ''
        phase = infer_item_phase(desc, project_df)
        # get phase finish date
        phase_row = project_df[project_df['Task'] == phase]
        if not phase_row.empty:
            needed_by = phase_row.iloc[0]['Finish']
        else:
            needed_by = project_df['Finish'].max()

        # vendor selection
        vendor = choose_vendor(master, vendor_db)
        if vendor:
            vendor_name = vendor.get('company') or vendor.get('name') or 'Vendor'
            vendor_contact = vendor.get('contact') or vendor.get('url') or None
            lead_days = vendor.get('lead_time_days') or vendor.get('lead_time') or vendor.get('lead') or DEFAULT_LEAD_DAYS
            try:
                lead_days = int(lead_days)
            except Exception:
                lead_days = DEFAULT_LEAD_DAYS
        else:
            vendor_name = None
            vendor_contact = None
            lead_days = DEFAULT_LEAD_DAYS

        # compute dates
        proc_finish = pd.to_datetime(needed_by)
        proc_start = proc_finish - pd.Timedelta(days=lead_days + DEFAULT_BUFFER_DAYS)
        # do not start before project start
        project_start = project_df['Start'].min()
        if proc_start < project_start:
            proc_start = project_start

        safety = max(1, int(math.ceil(qty * SAFETY_STOCK_PCT)))
        recommended_order = max(MIN_ORDER_QTY, qty + safety)

        # procurement cycles (simple rule)
        cycle = 'Single order'
        if recommended_order > 1000:
            cycle = 'Multiple orders (staged deliveries)'
        elif recommended_order > 100:
            cycle = 'Consider 2-stage procurement'
        else:
            cycle = 'Single order'

        # risks & mitigation
        risks = []
        mitigations = []
        if vendor is None:
            risks.append("No vendor in DB — sourcing risk")
            mitigations.append("Identify alternate vendors; consider higher safety stock; allow longer lead time")
        if lead_days >= 30:
            risks.append("Long supplier lead time")
            mitigations.append("Pre-book orders; negotiate faster lead time or local stocking")
        if qty > 500 and cycle == 'Single order':
            mitigations.append("Consider staged deliveries to reduce storage and cashflow impact")
        if 'import' in (vendor.get('source') or '').lower() if vendor else False:
            risks.append("Import dependency")
            mitigations.append("Account for customs and transport delays; increase safety stock")

        plan_rows.append({
            "MasterItemNo": master,
            "ItemDescription": desc,
            "PredictedQty": qty,
            "SafetyStock": safety,
            "RecommendedOrderQty": recommended_order,
            "Vendor": vendor_name,
            "VendorContact": vendor_contact,
            "VendorLeadDays": lead_days,
            "ProcurementStart": proc_start.date().isoformat(),
            "ProcurementFinish": proc_finish.date().isoformat(),
            "NeededBy": proc_finish.date().isoformat(),
            "ProjectPhase": phase,
            "ProcurementCycle": cycle,
            "RiskNotes": "; ".join(risks) if risks else "",
            "Mitigations": "; ".join(mitigations) if mitigations else ""
        })

    plan_df = pd.DataFrame(plan_rows)
    # write outputs
    plan_df.to_csv(OUT_CSV, index=False)
    # md summary
    total_items = len(plan_df)
    unique_vendors = plan_df['Vendor'].nunique()
    est_total_qty = plan_df['PredictedQty'].sum()
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Procurement Plan Summary\n\n")
        f.write(f"- Generated: {pd.Timestamp.now().isoformat()}\n")
        f.write(f"- Total line items: **{total_items}**\n")
        f.write(f"- Unique vendors: **{unique_vendors}**\n")
        f.write(f"- Estimated total quantity (sum of predicted): **{int(est_total_qty)}**\n\n")
        f.write("## Top risks & suggested mitigations\n\n")
        # aggregate risk counts
        risks_series = plan_df['RiskNotes'].value_counts().head(10)
        for idx, val in risks_series.items():
            if pd.isna(idx) or idx == "":
                continue
            f.write(f"- **{idx}** (examples: see Mitigations column)\n")
        f.write("\n## Sample Plan (first 20 rows)\n\n")
        f.write(plan_df.head(20).to_markdown(index=False))
        f.write("\n\n## Full plan CSV\n\n")
        f.write(f"Saved: `{OUT_CSV}`\n")

    print("Wrote procurement plan CSV:", OUT_CSV)
    print("Wrote procurement plan summary (markdown):", OUT_MD)
    return plan_df

if __name__ == '__main__':
    generate_plan()
