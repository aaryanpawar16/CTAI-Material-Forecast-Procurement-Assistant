# create_demo_vendors.py
import pandas as pd
import json, os

sub_path = 'submission/submission.csv'
if not os.path.exists(sub_path):
    print('submission/submission.csv not found. Create submission first.')
    raise SystemExit(1)

sub = pd.read_csv(sub_path)
top = sub['MasterItemNo'].value_counts().head(20).index.tolist()

items = []
for it in top:
    items.append({
        "MasterItemNo": int(it) if str(it).isdigit() else str(it),
        "vendors": [
            {
                "company": f"Placeholder Vendor for {it} A",
                "url": "https://example-vendor-a.com",
                "location": "Nearest metro",
                "contact": f"procure+{it}@example.com",
                "lead_time_days": 14,
                "min_order_qty": 1
            },
            {
                "company": f"Placeholder Vendor for {it} B",
                "url": "https://example-vendor-b.com",
                "location": "Nearby city",
                "contact": f"sales+{it}@example.com",
                "lead_time_days": 21,
                "min_order_qty": 10
            }
        ]
    })

os.makedirs('vendor_data', exist_ok=True)
with open('vendor_data/vendors.json','w', encoding='utf-8') as f:
    json.dump({"items": items}, f, indent=2, ensure_ascii=False)

print('Wrote vendor_data/vendors.json with', len(items), 'items')
