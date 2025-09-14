# src/make_submission.py
import pandas as pd
import argparse
from src.predict import predict_on_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='data/test.csv')
    parser.add_argument('--out', type=str, default='submission/submission.csv')
    args = parser.parse_args()

    df_sub = predict_on_test(test_csv=args.test)
    # enforce column order and types
    df_sub['MasterItemNo'] = df_sub['MasterItemNo'].astype(int)
    df_sub['QtyShipped'] = pd.to_numeric(df_sub['QtyShipped'], errors='coerce').fillna(0)
    df_sub = df_sub[['id','MasterItemNo','QtyShipped']]
    df_sub.to_csv(args.out, index=False)
    print('Saved', args.out)