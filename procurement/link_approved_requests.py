#!/usr/bin/env python3
"""
procurement/link_approved_requests.py

Link approved procurement requests into procurement/procurement_tasks.csv
and optionally regenerate the integrated Gantt.

Usage:
    python link_approved_requests.py           # default: regen gantt
    python link_approved_requests.py --no-gantt
    python link_approved_requests.py --debug
"""
import os
import json
import pandas as pd
from datetime import datetime
import subprocess
import sys
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REQUESTS_PATH = ROOT / "requests.json"
PROC_TASKS_PATH = ROOT / "procurement_tasks.csv"
REGENERATE_GANTT = True
GANTT_SCRIPT = ROOT / "generate_gantt_integrated.py"


def load_requests():
    """
    Return (reqs_list, root_obj)
    - If file is {"requests":[...]} -> returns (list, root_obj dict)
    - If file is [...] -> returns (list, None)
    - If missing or malformed -> returns ([], None)
    """
    if not REQUESTS_PATH.exists():
        return [], None
    try:
        with REQUESTS_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "requests" in obj and isinstance(obj["requests"], list):
            return obj["requests"], obj
        if isinstance(obj, list):
            return obj, None
        # try to coerce dict-of-dicts -> list
        if isinstance(obj, dict):
            candidates = []
            for v in obj.values():
                if isinstance(v, dict) and ("item" in v or "id" in v):
                    candidates.append(v)
            if candidates:
                return candidates, obj
        print("load_requests: unexpected JSON shape; returning empty list.")
        return [], None
    except Exception as e:
        print("load_requests: failed to read requests.json:", e)
        return [], None


def save_requests(reqs, root_obj=None):
    """
    Save list of requests. If root_obj provided, update its 'requests' key; otherwise
    write {"requests": reqs} to keep shape consistent.
    """
    try:
        if root_obj is None:
            out = {"requests": reqs}
            with REQUESTS_PATH.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        else:
            root_obj["requests"] = reqs
            with REQUESTS_PATH.open("w", encoding="utf-8") as f:
                json.dump(root_obj, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        print("save_requests: failed to write requests.json:", e)


def ensure_proc_tasks_exists():
    if not PROC_TASKS_PATH.exists():
        df = pd.DataFrame(columns=["MasterItemNo", "ItemDescription", "QtyShipped", "needed_by_date"])
        df.to_csv(PROC_TASKS_PATH, index=False)


def _coerce_master_item(itm_str):
    """
    Try to coerce item string to an integer MasterItemNo.
    Accepts strings like "  12345 ", "12,345", "12345.0".
    Returns int or None.
    """
    if itm_str is None:
        return None
    s = str(itm_str).strip()
    if not s:
        return None
    # Remove commas and other separators
    s_clean = re.sub(r"[,\s]", "", s)
    # If purely digits or digits with .0, convert
    try:
        if s_clean.isdigit():
            return int(s_clean)
        # handle float-looking ints
        f = float(s_clean)
        if f.is_integer():
            return int(f)
    except Exception:
        pass
    return None


def _coerce_qty(q):
    try:
        if q is None:
            return 1
        return int(float(str(q).replace(",", "").strip()))
    except Exception:
        return 1


def append_procurement_rows(rows, debug=False):
    ensure_proc_tasks_exists()
    try:
        df_existing = pd.read_csv(PROC_TASKS_PATH)
    except Exception:
        df_existing = pd.DataFrame(columns=["MasterItemNo", "ItemDescription", "QtyShipped", "needed_by_date"])

    df_new = pd.DataFrame(rows)

    # Ensure column compatibility: union columns, preserve order (existing cols first)
    existing_cols = list(df_existing.columns)
    new_cols = [c for c in df_new.columns if c not in existing_cols]
    final_cols = existing_cols + new_cols

    df_existing = df_existing.reindex(columns=final_cols)
    df_new = df_new.reindex(columns=final_cols)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
    df_combined.to_csv(PROC_TASKS_PATH, index=False)
    if debug:
        print(f"append_procurement_rows: wrote {len(df_new)} new rows, total {len(df_combined)} rows to {PROC_TASKS_PATH}")
    else:
        print(f"Appended {len(df_new)} rows to {PROC_TASKS_PATH}")


def main(run_gantt=True, debug=False):
    reqs, root_obj = load_requests()
    if not reqs:
        print("No requests found to process.")
        return

    to_link = []
    now_iso = datetime.now().isoformat()

    for r in reqs:
        if not isinstance(r, dict):
            if debug:
                print("Skipping non-dict request entry:", r)
            continue
        status = str(r.get("status", "")).strip().lower()
        linked = r.get("linked_to_procurement", False)
        if status == "approved" and not linked:
            itm = r.get("item") or r.get("Item") or r.get("item_description") or ""
            master = _coerce_master_item(itm)
            qty = _coerce_qty(r.get("qty") or r.get("quantity") or r.get("QtyShipped"))
            row = {
                "MasterItemNo": int(master) if master is not None else "",
                "ItemDescription": str(itm) if itm is not None else "",
                "QtyShipped": qty,
                "needed_by_date": r.get("needed_by") or r.get("NeededBy") or ""
            }
            to_link.append((r, row))
            if debug:
                print("Will link request id:", r.get("id"), "->", row)

    if not to_link:
        print("No approved/unlinked requests found.")
        return

    # Append the rows
    rows = [row for _, row in to_link]
    append_procurement_rows(rows, debug=debug)

    # Mark requests as linked and save
    for r, _ in to_link:
        r["linked_to_procurement"] = True
        r["linked_at"] = now_iso

    save_requests(reqs, root_obj)

    # Optionally regenerate integrated gantt
    if run_gantt and GANTT_SCRIPT.exists():
        try:
            # Call script; do not raise if it fails
            subprocess.run([sys.executable, str(GANTT_SCRIPT)], check=False)
            print("Invoked generate_gantt_integrated.py")
        except Exception as e:
            print("Failed to run generate_gantt_integrated.py:", e)
    elif run_gantt:
        print(f"Skipping gantt regeneration: {GANTT_SCRIPT} not found")

    print("Linking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gantt", action="store_true", help="Do not regenerate gantt")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = parser.parse_args()

    main(run_gantt=(not args.no_gantt), debug=args.debug)
