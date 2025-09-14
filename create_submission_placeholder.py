# create_submission_placeholder.py
import os
import pandas as pd
import numpy as np

os.makedirs('submission', exist_ok=True)
test_path = os.path.join('data', 'test.csv')
train_path = os.path.join('data', 'train.csv')
out_path = os.path.join('submission', 'submission.csv')

if not os.path.exists(test_path):
    print(f"Error: test file not found at {test_path}. Create data/test.csv or use Option A (manual file).")
    # create tiny default
    df = pd.DataFrame({'id':[1,2,3,4], 'MasterItemNo':[101,102,101,104], 'QtyShipped':[5,3,7,2]})
    df.to_csv(out_path, index=False)
    print('Wrote default submission to', out_path)
else:
    df_test = pd.read_csv(test_path)
    if 'id' not in df_test.columns:
        # create id column if missing
        print('Warning: id column not found in test.csv — creating sequential ids')
        df_test = df_test.reset_index().rename(columns={'index':'id'})
        df_test['id'] = df_test['id'] + 1

    # decide fallback predictions
    default_item = 101
    default_qty = 1.0
    if os.path.exists(train_path):
        try:
            df_train = pd.read_csv(train_path)
            if 'MasterItemNo' in df_train.columns and df_train['MasterItemNo'].notna().any():
                default_item = int(pd.to_numeric(df_train['MasterItemNo'], errors='coerce').mode().iloc[0])
            if 'QtyShipped' in df_train.columns and df_train['QtyShipped'].notna().any():
                default_qty = float(pd.to_numeric(df_train['QtyShipped'], errors='coerce').median())
        except Exception as e:
            print('Could not read train.csv for statistics:', e)

    n = len(df_test)
    preds = pd.DataFrame({
        'id': df_test['id'].astype(int),
        'MasterItemNo': [default_item] * n,
        'QtyShipped': [default_qty] * n
    })
    preds.to_csv(out_path, index=False)
    print('Wrote placeholder submission to', out_path)
    print('Sample:')
    print(preds.head())
