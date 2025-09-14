# procurement/generate_procurement_tasks_from_predictions.py
import pandas as pd
import os, json
from datetime import datetime, timedelta

def create_procurement_tasks(predictions_csv='submission/submission.csv',
                             vendor_json_dir='vendor_data/',
                             project_start='2025-10-01',
                             output_csv='procurement/procurement_tasks.csv'):
    # Diagnostics
    print(f'Looking for predictions file: {predictions_csv}')
    if not os.path.exists(predictions_csv):
        raise FileNotFoundError(f'Predictions file not found: {predictions_csv}. '
                                'Generate it using src.make_submission or create a valid CSV.')

    # Try reading and give helpful error if empty
    try:
        pred = pd.read_csv(predictions_csv)
    except pd.errors.EmptyDataError:
        raise ValueError(f'Predictions file {predictions_csv} is empty or malformed.')

    if pred.shape[0] == 0 or set(['id','MasterItemNo','QtyShipped']).issubset(pred.columns) == False:
        raise ValueError(f'Predictions file {predictions_csv} must have columns: id, MasterItemNo, QtyShipped and contain rows. '
                         f'Found columns: {list(pred.columns)} and {pred.shape[0]} rows.')

    # aggregate per item
    agg = pred.groupby('MasterItemNo', as_index=False)['QtyShipped'].sum()

    # load vendor mapping if present (expecting vendor_data/vendors.json or similar)
    vendor_map = {}
    vendor_file = os.path.join(vendor_json_dir, 'vendors.json')
    if os.path.exists(vendor_file):
        try:
            data = json.load(open(vendor_file, encoding='utf-8'))
            # flexible parsing: support list of entries or dict
            if isinstance(data, list):
                for e in data:
                    if 'MasterItemNo' in e:
                        vendor_map[int(e['MasterItemNo'])] = e.get('vendors', [])
            elif isinstance(data, dict):
                # if top-level dict maps item -> vendors
                if 'items' in data:
                    for e in data['items']:
                        vendor_map[int(e['MasterItemNo'])] = e.get('vendors', [])
                else:
                    # try parse keys
                    for k, v in data.items():
                        try:
                            mk = int(k)
                            vendor_map[mk] = v
                        except:
                            pass
        except Exception as e:
            print('Warning: failed to parse vendor file:', e)

    default_lead = 14
    tasks = []
    # assume installation starts after Design (30) + Procurement (60) days from project_start
    project_install_start = pd.to_datetime(project_start) + pd.Timedelta(days=30+60)
    for _, row in agg.iterrows():
        try:
            mid = int(row['MasterItemNo'])
        except:
            mid = row['MasterItemNo']
        qty = float(row['QtyShipped'])
        vendors = vendor_map.get(mid, [])
        lead = default_lead
        vendor_name = None
        if vendors and isinstance(vendors, list) and len(vendors) > 0:
            v0 = vendors[0]
            vendor_name = v0.get('company') if isinstance(v0, dict) else str(v0)
            lead = int(v0.get('lead_time_days', default_lead)) if isinstance(v0, dict) else default_lead
        procurement_end = project_install_start - pd.Timedelta(days=1)
        procurement_start = procurement_end - pd.Timedelta(days=max(lead, 7))
        tasks.append({
            'MasterItemNo': mid,
            'Qty': qty,
            'preferred_vendor': vendor_name or 'TBD',
            'lead_time_days': int(lead),
            'procurement_start': procurement_start.date().isoformat(),
            'procurement_end': procurement_end.date().isoformat(),
            'expected_delivery': project_install_start.date().isoformat()
        })

    df = pd.DataFrame(tasks)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print('Wrote', output_csv)
    return df

if __name__ == '__main__':
    create_procurement_tasks()
