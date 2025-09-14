# src/data_preprocessing.py
"""
Improved preprocessing for CTAI CTD Hackathon.

Features:
- robust cleaning for currency/qty fields
- date parsing and calendar features
- derived features: project_duration_days, project_age_days, material_cost_per_unit,
  size_per_floor, is_large_project
- simple imputations and text cleaning for ItemDescription
"""

import pandas as pd
import numpy as np
import re

def clean_currency(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    # remove everything except digits, minus sign, and dot
    s = re.sub(r'[^\d\.\-]', '', s)
    if s in ['', '.', '-', '-.', None]:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def safe_to_numeric(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return np.nan

def parse_dates(df, date_cols):
    for col in date_cols:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], errors='coerce', dayfirst=False)
    return df

def add_calendar_features(df, date_col):
    """Add month/dayofweek/quarter/week features from date_col (expects date_col+'_dt')."""
    if date_col + '_dt' not in df.columns:
        return df
    s = df[date_col + '_dt']
    df[date_col + '_month'] = s.dt.month
    df[date_col + '_dow'] = s.dt.dayofweek
    df[date_col + '_week'] = s.dt.isocalendar().week.astype('Int64')
    df[date_col + '_quarter'] = s.dt.quarter
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- CLEAN numeric-like columns ----
    # Currency-like
    for col in ['invoiceTotal', 'UnitPrice', 'ExtendedPrice', 'REVISED_ESTIMATE']:
        if col in df.columns:
            df[col + '_clean'] = df[col].apply(clean_currency)

    # Quantity-like
    for col in ['QtyShipped', 'ExtendedQuantity', 'NUMFLOORS', 'NUMROOMS', 'NUMBEDS', 'SIZE_BUILDINGSIZE', 'MW']:
        if col in df.columns:
            df[col + '_clean'] = df[col].apply(safe_to_numeric)

    # Parse dates
    df = parse_dates(df, ['invoiceDate', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE'])

    # Calendar features for invoiceDate
    df = add_calendar_features(df, 'invoiceDate')

    # Derived features
    # project duration
    if 'CONSTRUCTION_START_DATE_dt' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE_dt' in df.columns:
        df['project_duration_days'] = (df['SUBSTANTIAL_COMPLETION_DATE_dt'] - df['CONSTRUCTION_START_DATE_dt']).dt.days

    # project_age_days: invoiceDate - construction_start
    if 'invoiceDate_dt' in df.columns and 'CONSTRUCTION_START_DATE_dt' in df.columns:
        df['project_age_days'] = (df['invoiceDate_dt'] - df['CONSTRUCTION_START_DATE_dt']).dt.days

    # size per floor
    if 'SIZE_BUILDINGSIZE_clean' in df.columns and 'NUMFLOORS_clean' in df.columns:
        df['size_per_floor'] = df['SIZE_BUILDINGSIZE_clean'] / df['NUMFLOORS_clean'].replace({0: np.nan})

    # material cost per unit (safely)
    if 'ExtendedPrice_clean' in df.columns and 'ExtendedQuantity_clean' in df.columns:
        df['material_cost_per_unit'] = df['ExtendedPrice_clean'] / df['ExtendedQuantity_clean'].replace({0: np.nan})
        # if material_cost_per_unit NaN, try ExtendedPrice / QtyShipped
        mask = df['material_cost_per_unit'].isna()
        if mask.any() and 'QtyShipped_clean' in df.columns:
            df.loc[mask, 'material_cost_per_unit'] = df.loc[mask, 'ExtendedPrice_clean'] / df.loc[mask, 'QtyShipped_clean'].replace({0: np.nan})

    # is_large_project (75th percentile)
    if 'SIZE_BUILDINGSIZE_clean' in df.columns:
        try:
            thresh = df['SIZE_BUILDINGSIZE_clean'].quantile(0.75)
            df['is_large_project'] = (df['SIZE_BUILDINGSIZE_clean'] > thresh).astype(int)
        except Exception:
            df['is_large_project'] = 0

    # ItemDescription cleaning
    if 'ItemDescription' in df.columns:
        df['ItemDescription_clean'] = df['ItemDescription'].astype(str).fillna('').str.lower()
        # remove non-alphanum except spaces
        df['ItemDescription_clean'] = df['ItemDescription_clean'].str.replace('[^a-z0-9 ]', ' ', regex=True)
        # collapse multiple spaces
        df['ItemDescription_clean'] = df['ItemDescription_clean'].str.replace('\\s+', ' ', regex=True).str.strip()
    else:
        df['ItemDescription_clean'] = ''

    # Simple imputation for numeric cleans (fill with median where appropriate)
    numeric_clean_cols = [c for c in df.columns if c.endswith('_clean') and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, 'float64', 'int64']]
    for c in numeric_clean_cols:
        try:
            med = df[c].median()
            df[c] = df[c].fillna(med)
        except Exception:
            pass

    # Keep original columns plus engineered ones
    return df

# small CLI to allow preprocessing from command line
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/train.csv')
    parser.add_argument('--output', type=str, default='data/train_preprocessed.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df2 = preprocess(df)
    df2.to_csv(args.output, index=False)
    print('Saved:', args.output)
