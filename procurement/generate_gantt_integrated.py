#!/usr/bin/env python3
"""
Generate an integrated Gantt chart combining:
 - project schedule (synthetic or procurement/project_schedule.csv)
 - procurement tasks (procurement/procurement_tasks.csv)
 - vendor info (vendor_data/vendors.json)

Outputs:
 - procurement/gantt_chart_integrated.html  (interactive plotly)
 - procurement/gantt_chart_integrated_fast.png (PNG fallback)
 - procurement/gantt_tasks_integrated.csv (enriched task table)
"""

from datetime import timedelta
import os
import json
import math
import logging
import pandas as pd
import numpy as np

# plotting
import plotly.express as px
import plotly.io as pio

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
os.makedirs(ROOT, exist_ok=True)

# input paths
PROC_TASKS = os.path.join(ROOT, 'procurement_tasks.csv')
PROJECT_SCHEDULE = os.path.join(ROOT, 'project_schedule.csv')
VENDOR_JSON = os.path.join(os.path.dirname(ROOT), 'vendor_data', 'vendors.json')

# outputs
OUT_HTML = os.path.join(ROOT, 'gantt_chart_integrated.html')
OUT_PNG = os.path.join(ROOT, 'gantt_chart_integrated_fast.png')
OUT_CSV = os.path.join(ROOT, 'gantt_tasks_integrated.csv')

# ----------------- small helpers -----------------

def _short_str(val, length=40):
    if pd.isna(val):
        return ''
    return str(val)[:length]


def safe_get(r, key, default=None):
    try:
        return r.get(key, default)
    except Exception:
        try:
            return r[key]
        except Exception:
            return default


def parse_date_maybe(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


# ----------------- reading helpers -----------------

def read_or_create_project_schedule():
    from pathlib import Path
    PROJECT_SCHEDULE_PATH = Path(__file__).parent / "project_schedule.csv"

    if PROJECT_SCHEDULE_PATH.exists():
        try:
            df = pd.read_csv(PROJECT_SCHEDULE_PATH, parse_dates=['Start', 'Finish'])
            if 'Start' in df.columns and 'Finish' in df.columns:
                return df
            else:
                logger.warning('project_schedule.csv missing required columns; falling back to synthetic schedule.')
        except Exception as e:
            logger.warning('Failed to read project_schedule.csv; falling back to synthetic schedule. Error: %s', e)

    today = pd.Timestamp.today().normalize()
    sched = [
        ("Design & Planning", today, today + pd.Timedelta(days=20)),
        ("Foundation & Civil Works", today + pd.Timedelta(days=21), today + pd.Timedelta(days=50)),
        ("Electrical & HVAC", today + pd.Timedelta(days=51), today + pd.Timedelta(days=80)),
        ("IT Infrastructure", today + pd.Timedelta(days=81), today + pd.Timedelta(days=95)),
        ("Commissioning & Handover", today + pd.Timedelta(days=96), today + pd.Timedelta(days=110)),
    ]
    df = pd.DataFrame(sched, columns=['Task','Start','Finish'])
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])
    return df


def read_procurement_tasks():
    if not os.path.exists(PROC_TASKS):
        raise FileNotFoundError(f"Procurement tasks not found at {PROC_TASKS}. Run generate_procurement_tasks_from_predictions.py first.")
    df = pd.read_csv(PROC_TASKS)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_vendor_db():
    if not os.path.exists(VENDOR_JSON):
        logger.info('vendor_data/vendors.json not found; vendors will be placeholders.')
        return {}
    try:
        with open(VENDOR_JSON, 'r', encoding='utf-8') as f:
            jd = json.load(f)
    except Exception:
        logger.warning('Failed to load vendor_data/vendors.json; vendors will be placeholders.')
        return {}
    mapping = {}
    for entry in jd.get('items', []):
        key = str(entry.get('MasterItemNo'))
        mapping[key] = entry.get('vendors', [])
    return mapping


PHASE_KEYWORDS = {
    'Electrical & HVAC': ['cable', 'tray', 'transformer', 'breaker', 'cable tray', 'hvac', 'duct', 'mcb', 'panel'],
    'IT Infrastructure': ['server', 'rack', 'network', 'switch', 'fiber', 'rack', 'server rack'],
    'Foundation & Civil Works': ['cement', 'steel', 'rebar', 'concrete', 'brick', 'block', 'mortar'],
    'Design & Planning': ['survey', 'drawing', 'design'],
    'Commissioning & Handover': ['commission', 'handover', 'testing']
}


def map_item_to_phase(item_desc, master_itemno, phases_df):
    s = (str(item_desc) if not pd.isna(item_desc) else "") + " " + str(master_itemno)
    s = s.lower()
    for phase, keywords in PHASE_KEYWORDS.items():
        for kw in keywords:
            if kw in s:
                return phase
    return 'Electrical & HVAC' if 'Electrical & HVAC' in phases_df['Task'].values else phases_df['Task'].iloc[0]


def choose_vendor_for_item(master_itemno, vendor_db):
    key = str(master_itemno)
    vendors = vendor_db.get(key) or vendor_db.get(int(master_itemno) if str(master_itemno).isdigit() else None) or []
    if vendors:
        return vendors[0]
    return {"company": "Placeholder Vendor", "url": None, "services": None, "location": None, "contact": None, "lead_time_days": 14, "source": "placeholder"}


def compute_procurement_schedule(project_df, proc_df, vendor_db):
    rows = []
    phase_map = {r['Task']: (r['Start'], r['Finish']) for _, r in project_df.iterrows()}

    for idx, r in proc_df.iterrows():
        item_desc = r.get('ItemDescription') or r.get('Item') or ""
        master = r.get('MasterItemNo') if 'MasterItemNo' in r else r.get('MasterItemNo')
        qty = r.get('QtyShipped') or r.get('Qty') or r.get('Quantity') or None

        needed_by = None
        for c in ('needed_by_date', 'needed_by', 'needed_by_dt', 'NeededBy'):
            if c in proc_df.columns:
                needed_by = parse_date_maybe(r.get(c))
                if needed_by is not None:
                    break

        phase = map_item_to_phase(item_desc, master, project_df)
        phase_start, phase_finish = phase_map.get(phase, (project_df['Start'].min(), project_df['Finish'].max()))
        needed_by = needed_by or phase_finish

        vendor = choose_vendor_for_item(master, vendor_db)
        lead = vendor.get('lead_time_days') or vendor.get('lead_time') or None
        try:
            lead = int(lead) if lead is not None else None
        except Exception:
            lead = None
        if lead is None:
            lead = 14 if vendor.get('source') == 'placeholder' else 21

        processing_buffer = 3
        proc_start = pd.to_datetime(needed_by) - pd.Timedelta(days=(lead + processing_buffer))
        proc_finish = pd.to_datetime(needed_by)
        project_earliest = project_df['Start'].min()
        if proc_start < project_earliest:
            proc_start = project_earliest

        rows.append({
            'Phase': phase,
            'PhaseStart': phase_start,
            'PhaseFinish': phase_finish,
            'ItemDescription': item_desc,
            'MasterItemNo': master,
            'Qty': qty,
            'VendorCompany': vendor.get('company'),
            'VendorSource': vendor.get('source'),
            'VendorContact': vendor.get('contact'),
            'VendorURL': vendor.get('url'),
            'VendorLeadDays': lead,
            'ProcurementStart': proc_start,
            'ProcurementFinish': proc_finish,
            'NeededBy': needed_by
        })

    enriched = pd.DataFrame(rows)
    return enriched


def build_gantt_dataframe(project_df, enriched_proc_df):
    rows = []
    for _, r in project_df.iterrows():
        rows.append({
            'Task': r['Task'],
            'Start': pd.to_datetime(r['Start']),
            'Finish': pd.to_datetime(r['Finish']),
            'Resource': 'Project Phase',
            'Type': 'Phase'
        })

    for idx, r in enriched_proc_df.iterrows():
        item_desc = safe_get(r, 'ItemDescription')
        master = safe_get(r, 'MasterItemNo')
        item_short = _short_str(item_desc)
        master_short = _short_str(master)
        task_name = f"Procure: {item_short or master_short or ('row_'+str(idx))}"
        vendor = safe_get(r, 'VendorCompany') or 'Unknown Vendor'

        start = parse_date_maybe(safe_get(r, 'ProcurementStart'))
        finish = parse_date_maybe(safe_get(r, 'ProcurementFinish'))
        if start is None and finish is not None:
            # assume 1-day duration if only finish known
            start = finish - pd.Timedelta(days=1)
        if finish is None and start is not None:
            finish = start + pd.Timedelta(days=1)
        if start is None and finish is None:
            # skip rows with no dates
            logger.warning('Skipping procurement row %s: no start/finish dates', idx)
            continue

        rows.append({
            'Task': task_name,
            'Start': pd.to_datetime(start),
            'Finish': pd.to_datetime(finish),
            'Resource': vendor,
            'Type': 'Procurement'
        })

    df = pd.DataFrame(rows)
    return df


def save_interactive_gantt(df_gantt, out_html=OUT_HTML):
    fig = px.timeline(df_gantt, x_start='Start', x_end='Finish', y='Task', color='Type', hover_data=['Resource'])
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(title_text='Integrated Project & Procurement Gantt', height=900, margin=dict(l=200, r=20, t=60, b=20))
    pio.write_html(fig, out_html, auto_open=False)
    logger.info('Wrote interactive Gantt: %s', out_html)
    return fig


def save_png_fallback(fig, df_gantt, out_png=OUT_PNG):
    try:
        img_bytes = pio.to_image(fig, format='png', width=1200, height=900, scale=1)
        with open(out_png, 'wb') as f:
            f.write(img_bytes)
        logger.info('Wrote PNG fallback (plotly): %s', out_png)
        return
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        gantt_df = df_gantt.sort_values('Start')
        fig_m, ax = plt.subplots(figsize=(12, max(6, len(gantt_df) * 0.25)))
        y_positions = range(len(gantt_df))
        for i, (_, row) in enumerate(gantt_df.iterrows()):
            start = row['Start']
            finish = row['Finish']
            ax.barh(i, (finish - start).days + 0.5, left=start, height=0.4, align='center')
            ax.text(start + (finish - start) / 2, i, str(row['Resource'])[:30], va='center', ha='center', fontsize=8, color='white')
        ax.set_yticks(list(y_positions))
        ax.set_yticklabels(gantt_df['Task'])
        ax.xaxis_date()
        fig_m.tight_layout()
        fig_m.savefig(out_png, dpi=150)
        plt.close(fig_m)
        logger.info('Wrote PNG fallback (matplotlib): %s', out_png)
    except Exception as e:
        logger.warning('Failed to write PNG fallback: %s', e)


if __name__ == '__main__':
    logger.info('Starting integrated Gantt generation...')
    project_df = read_or_create_project_schedule()
    proc_df = read_procurement_tasks()
    vendor_db = read_vendor_db()

    project_df['Start'] = pd.to_datetime(project_df['Start'])
    project_df['Finish'] = pd.to_datetime(project_df['Finish'])

    enriched = compute_procurement_schedule(project_df, proc_df, vendor_db)

    # ensure output folder exists
    out_parent = os.path.dirname(OUT_CSV) or ROOT
    os.makedirs(out_parent, exist_ok=True)

    enriched.to_csv(OUT_CSV, index=False)
    logger.info('Wrote enriched procurement tasks to %s', OUT_CSV)

    df_gantt = build_gantt_dataframe(project_df, enriched)
    if df_gantt.empty:
        logger.warning('No gantt rows to plot; exiting.')
        raise SystemExit(0)

    # ensure html/png parent exists
    os.makedirs(os.path.dirname(OUT_HTML) or ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG) or ROOT, exist_ok=True)

    fig = save_interactive_gantt(df_gantt, OUT_HTML)
    try:
        save_png_fallback(fig, df_gantt, OUT_PNG)
    except Exception as e:
        logger.warning('PNG fallback failed: %s', e)

    logger.info('Integrated Gantt generation complete.')
