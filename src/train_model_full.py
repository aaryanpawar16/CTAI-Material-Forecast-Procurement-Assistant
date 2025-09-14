# src/train_model_full.py
import os, joblib, argparse, json
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from category_encoders import TargetEncoder
from src.data_preprocessing import preprocess
from src.features import fit_text_pipeline, transform_text_pipeline, add_basic_features

ARTIFACTS_DIR = 'artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

class Trainer:
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols or ['PROJECT_CITY','PROJECT_TYPE','CORE_MARKET','UOM']
        self.target_encoder = None
        self.models = []

    def fit_classifier(self, X, y, groups=None, n_splits=4, num_boost_round=300):
        te = TargetEncoder(cols=[c for c in self.cat_cols if c in X.columns])
        X_enc = te.fit_transform(X, y)
        self.target_encoder = te
        params = {
            'objective':'multiclass',
            'num_class': int(len(np.unique(y))),
            'metric':'multi_logloss',
            'verbosity': -1
        }
        gkf = GroupKFold(n_splits=n_splits)
        models=[]
        for tr, val in gkf.split(X_enc, y, groups=groups):
            dtr = lgb.Dataset(X_enc.iloc[tr], label=y.iloc[tr])
            dval = lgb.Dataset(X_enc.iloc[val], label=y.iloc[val])
            m = lgb.train(params, dtr, valid_sets=[dval], num_boost_round=num_boost_round)
            models.append(m)
        self.models = models
        return models

    def predict_classifier(self, X):
        X_enc = self.target_encoder.transform(X)
        preds=[]
        for m in self.models:
            preds.append(m.predict(X_enc))
        avg = np.mean(preds, axis=0)
        return np.argmax(avg, axis=1)

def train(args):
    df = pd.read_csv(args.train)
    df = preprocess(df)
    df = df.dropna(subset=['MasterItemNo']).copy()
    # preserve raw MasterItemNo strings
    df['MasterItemNo_raw'] = df['MasterItemNo'].astype(str).str.strip()
    le = LabelEncoder()
    y_raw = df['MasterItemNo_raw'].values
    y_encoded = le.fit_transform(y_raw)
    df['MasterItemNo_encoded'] = y_encoded
    joblib.dump(le, os.path.join(ARTIFACTS_DIR,'label_encoder.pkl'))
    map_str2int = {s: i for i, s in enumerate(le.classes_, start=1)}
    joblib.dump(map_str2int, os.path.join(ARTIFACTS_DIR,'map_str2int.pkl'))

    # TEXT pipeline fit
    print("Fitting text pipeline...")
    fit_text_pipeline(df, text_col='ItemDescription', max_features=15000, svd_n=50)
    X_text = transform_text_pipeline(df, text_col='ItemDescription')
    X_basic = add_basic_features(df)

    # combine features: we'll put SVD features as numeric columns
    X = X_basic.reset_index(drop=True).copy()
    for i in range(X_text.shape[1]):
        X[f'svd_{i}'] = X_text[:, i]

    groups = df['PROJECTNUMBER'] if 'PROJECTNUMBER' in df.columns else None
    y = pd.Series(df['MasterItemNo_encoded'], name='MasterItemNo_encoded')

    # Train classifier
    print("Training classifier...")
    trainer = Trainer()
    trainer.fit_classifier(X, y, groups=groups, n_splits=4, num_boost_round=args.num_boost_round)
    joblib.dump(trainer, os.path.join(ARTIFACTS_DIR,'classifier_trainer.pkl'))
    print("Saved classifier_trainer.pkl")

    # Train global regressor for QtyShipped
    print("Training global qty regressor...")
    # build regressor features (reuse X) and numeric target
    # Convert QtyShipped to numeric safely, compute median on numeric series (not on original object dtype)
    qty_numeric = pd.to_numeric(df['QtyShipped'], errors='coerce')
    median_val = qty_numeric.median()
    # If median is NaN (e.g., all values non-numeric), fall back to 0 to avoid errors
    if pd.isna(median_val):
        median_val = 0.0
    df['QtyShipped_num'] = qty_numeric.fillna(median_val).astype(float)

    reg_X = X.copy()
    # LightGBM requires numeric dtypes for training. Convert any object/category columns in reg_X
    # to integer codes using pandas.factorize() and save mappings for later use during inference.
    obj_cols = reg_X.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_mappings = {}
    for c in obj_cols:
        # fill missing with a sentinel so factorize keeps a consistent mapping
        reg_X[c] = reg_X[c].fillna('__MISSING__')
        codes, uniques = pd.factorize(reg_X[c])
        reg_X[c] = codes.astype(int)
        cat_mappings[c] = list(map(str, uniques))

    # persist mappings so deployment/inference can apply same encoding
    joblib.dump(cat_mappings, os.path.join(ARTIFACTS_DIR, 'regressor_cat_mappings.pkl'))

    reg_y = df['QtyShipped_num'].astype(float)
    dtrain = lgb.Dataset(reg_X, label=reg_y)
    reg_params = {'objective':'regression','metric':'l1','verbosity':-1}
    reg_global = lgb.train(reg_params, dtrain, num_boost_round=400)
    joblib.dump(reg_global, os.path.join(ARTIFACTS_DIR, 'regressor_global.pkl'))
    print("Saved regressor_global.pkl")

    # Per-item regressors for top-K frequent items
    K = args.topk if args.topk else 50
    top_items = df['MasterItemNo_raw'].value_counts().head(K).index.tolist()
    per_item_models = {}
    for itm in top_items:
        rows = df[df['MasterItemNo_raw'] == itm]
        if len(rows) < 20:
            continue
        X_it = reg_X.loc[rows.index]
        y_it = rows['QtyShipped_num'].astype(float)
        dtr = lgb.Dataset(X_it, label=y_it)
        m = lgb.train(reg_params, dtr, num_boost_round=200)
        per_item_models[str(itm)] = m
    joblib.dump(per_item_models, os.path.join(ARTIFACTS_DIR, 'regressors_per_item.pkl'))
    print("Saved per-item regressors for top items.")

    # Save some metadata
    meta = {'top_items': [str(x) for x in top_items]}
    with open(os.path.join(ARTIFACTS_DIR, 'meta.json'),'w') as f:
        json.dump(meta, f)
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='data/train.csv')
    parser.add_argument('--num_boost_round', type=int, default=200)
    parser.add_argument('--topk', type=int, default=50)
    args = parser.parse_args()
    train(args)
