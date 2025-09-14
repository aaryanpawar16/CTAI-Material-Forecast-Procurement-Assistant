# src/predict.py
import os
import pandas as pd
import numpy as np
import joblib
import importlib
import sys
import types

from src.data_preprocessing import preprocess

ARTIFACTS_DIR = 'artifacts'
CLASSIFIER_PATH = os.path.join(ARTIFACTS_DIR, 'classifier_trainer.pkl')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
MAP_STR2INT_PATH = os.path.join(ARTIFACTS_DIR, 'map_str2int.pkl')
REG_GLOBAL_PATH = os.path.join(ARTIFACTS_DIR, 'regressor_global.pkl')
REG_PER_ITEM_PATH = os.path.join(ARTIFACTS_DIR, 'regressors_per_item.pkl')
REG_CAT_MAPS_PATH = os.path.join(ARTIFACTS_DIR, 'regressor_cat_mappings.pkl')


def load_trainer(path=CLASSIFIER_PATH):
    """
    Load the saved trainer object. Handles pickles created under different module names by
    ensuring compatible class objects are present in sys.modules['__main__'] for unpickling.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trainer artifact not found at: {path}")

    # Try to import the legacy module first, then fallback to the full trainer module
    tm = None
    tried = []
    for modname in ('src.train_model', 'src.train_model_full'):
        try:
            tm = importlib.import_module(modname)
            break
        except Exception as e:
            tried.append((modname, str(e)))
    if tm is None:
        # As a last resort, try reloading package
        try:
            import src
            importlib.reload(src)
            tm = importlib.import_module('src.train_model')
        except Exception:
            # re-raise a helpful message
            raise ImportError(f"Could not import train model module. Tried: {tried}")

    # Ensure __main__ has compatible class names for unpickling
    main_mod = sys.modules.get('__main__')
    if main_mod is None:
        main_mod = types.ModuleType('__main__')
        sys.modules['__main__'] = main_mod

    # Common class name variations that older pickles might reference
    candidate_class_names = ['BaselineTrainer', 'Trainer']
    for cname in candidate_class_names:
        if not hasattr(main_mod, cname) and hasattr(tm, cname):
            setattr(main_mod, cname, getattr(tm, cname))

    # Also copy train function if present
    if not hasattr(main_mod, 'train') and hasattr(tm, 'train'):
        setattr(main_mod, 'train', getattr(tm, 'train'))

    trainer = joblib.load(path)
    return trainer


def load_label_encoder(path=LABEL_ENCODER_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_map_str2int(path=MAP_STR2INT_PATH):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _build_median_lookup(train_csv='data/train.csv'):
    """Median QtyShipped per MasterItemNo (stringified keys)."""
    if not os.path.exists(train_csv):
        return {}
    df = pd.read_csv(train_csv)
    if 'MasterItemNo' not in df.columns or 'QtyShipped' not in df.columns:
        return {}
    df['MasterItemNo_str'] = df['MasterItemNo'].astype(str).str.strip()
    df['QtyShipped_num'] = pd.to_numeric(df['QtyShipped'], errors='coerce')
    med = df.groupby('MasterItemNo_str')['QtyShipped_num'].median().to_dict()
    return med


def _load_regressors():
    reg_global = None
    per_item = {}
    cat_maps = {}
    if os.path.exists(REG_GLOBAL_PATH):
        try:
            reg_global = joblib.load(REG_GLOBAL_PATH)
        except Exception:
            reg_global = None
    if os.path.exists(REG_PER_ITEM_PATH):
        try:
            per_item = joblib.load(REG_PER_ITEM_PATH)
        except Exception:
            per_item = {}
    if os.path.exists(REG_CAT_MAPS_PATH):
        try:
            cat_maps = joblib.load(REG_CAT_MAPS_PATH)
        except Exception:
            cat_maps = {}
    return reg_global, per_item, cat_maps


def _build_text_features(df):
    """
    Attempt to transform text features using saved TF-IDF + SVD artifacts via src.features.
    Returns a numpy array (n_rows, n_components). If pipeline missing, returns zeros.
    """
    try:
        from src.features import transform_text_pipeline
        X_text = transform_text_pipeline(df, text_col='ItemDescription')
        return X_text
    except Exception:
        # fallback zero-array (50 dims assumed earlier); make it dynamic: try to load svd to infer dims
        svd_components = 50
        try:
            import joblib as _jl
            svd = _jl.load(os.path.join(ARTIFACTS_DIR, 'svd.pkl'))
            svd_components = getattr(svd, 'n_components', svd_components)
        except Exception:
            pass
        return np.zeros((len(df), svd_components))


def _build_basic_features(df):
    """
    Build basic feature frame that matches what trainer expects:
    columns: PROJECT_CITY, PROJECT_TYPE, CORE_MARKET, UOM, SIZE_BUILDINGSIZE, PROJECT_FREQ
    """
    from src.features import add_basic_features
    return add_basic_features(df)


def _encode_regressor_objects(X, cat_mappings):
    """Encode object/category columns using saved mappings (or factorize as fallback)."""
    X_enc = X.copy()
    obj_cols = X_enc.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in obj_cols:
        X_enc[c] = X_enc[c].fillna('__MISSING__').astype(str)
        if c in cat_mappings:
            uniques = cat_mappings[c]
            # map known -> index, unknown -> len(uniques)
            X_enc[c] = X_enc[c].apply(lambda v: int(uniques.index(str(v))) if str(v) in uniques else len(uniques))
        else:
            X_enc[c], _ = pd.factorize(X_enc[c])
            X_enc[c] = X_enc[c].astype(int)
    return X_enc


def predict_on_test(test_csv='data/test.csv', trainer_path=CLASSIFIER_PATH):
    """
    Predict items and quantities for test CSV.
    Returns a DataFrame with numeric MasterItemNo and QtyShipped.
    """
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    df_test = preprocess(df_test)

    # ensure id column exists
    if 'id' not in df_test.columns:
        df_test = df_test.reset_index().rename(columns={'index': 'id'})
        df_test['id'] = df_test['id'] + 1

    # Load trainer (classifier)
    trainer = load_trainer(trainer_path)

    # Build features as used in training
    # basic categorical + numeric frame
    X_basic = _build_basic_features(df_test)
    # text features (SVD)
    X_text = _build_text_features(df_test)

    # combine into a single dataframe X (columns svd_0..svd_N)
    X = X_basic.reset_index(drop=True).copy()
    # ensure shapes match
    if X_text is None:
        X_text = np.zeros((len(X), 0))
    for i in range(X_text.shape[1]):
        X[f'svd_{i}'] = X_text[:, i]

    # Prepare for categorical expected types (strings)
    for c in ['PROJECT_CITY', 'PROJECT_TYPE', 'CORE_MARKET', 'UOM']:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna('missing')

    # Classifier predictions (encoded labels)
    preds_encoded = trainer.predict_classifier(X)

    # Load label encoder to invert labels to original raw MasterItemNo strings
    le = load_label_encoder()
    if le is not None:
        try:
            preds_raw = le.inverse_transform(preds_encoded)
            preds_raw = [str(x).strip() for x in preds_raw]
        except Exception:
            preds_raw = [str(int(x)) for x in preds_encoded]
    else:
        preds_raw = [str(int(x)) for x in preds_encoded]

    # Map raw predicted IDs to numeric codes using map_str2int if available;
    # otherwise try to coerce string to int, else set 0.
    map_str2int = load_map_str2int()
    masteritem_out = []
    for s in preds_raw:
        if map_str2int is not None and s in map_str2int:
            try:
                masteritem_out.append(int(map_str2int[s]))
            except Exception:
                masteritem_out.append(0)
        else:
            # try to coerce to int (for cases where original ids were numeric strings)
            try:
                masteritem_out.append(int(float(s)))
            except Exception:
                masteritem_out.append(0)  # fallback code

    # Load regressors (global + per-item) and categorical mappings for regressor
    reg_global, per_item, cat_mappings = _load_regressors()

    # Build median lookup and global median fallback
    med_lookup = _build_median_lookup('data/train.csv')
    global_median = 1.0
    try:
        df_train = pd.read_csv('data/train.csv')
        global_median = float(pd.to_numeric(df_train['QtyShipped'], errors='coerce').median())
        if np.isnan(global_median):
            global_median = 1.0
    except Exception:
        pass

    # Predict QtyShipped: prefer per-item regressor -> global regressor -> median lookup -> global median
    preds_qty = []
    # create a numeric features frame for regressors; regressors expect same X as classifier used (i.e., X)
    X_reg = X.copy()

    # Encode object columns consistently using saved cat_mappings
    X_reg_enc = _encode_regressor_objects(X_reg, cat_mappings)

    for i, s in enumerate(preds_raw):
        q = None
        s_key = str(s)

        # per-item regressor expects the original label key (string). Try both raw and map key forms.
        if per_item and s_key in per_item:
            try:
                q = float(per_item[s_key].predict(X_reg_enc.iloc[[i]])[0])
            except Exception:
                q = None

        # try global regressor
        if q is None and reg_global is not None:
            try:
                q = float(reg_global.predict(X_reg_enc.iloc[[i]])[0])
            except Exception:
                q = None

        # fallback to train median per item
        if q is None:
            q = med_lookup.get(s_key)
        if q is None:
            # also try numeric masteritem mapping in med_lookup keys
            try:
                q = med_lookup.get(str(int(float(s_key))))
            except Exception:
                q = None

        # final fallback
        if q is None or (isinstance(q, float) and np.isnan(q)):
            q = global_median

        # postprocess: enforce min 1, integer rounding
        try:
            q = float(q)
            q = max(1.0, float(round(q)))
        except Exception:
            q = float(global_median)

        preds_qty.append(q)

    df_out = pd.DataFrame({
        'id': df_test['id'].astype(int),
        'MasterItemNo': masteritem_out,
        'QtyShipped': preds_qty
    })

    return df_out


if __name__ == '__main__':
    # quick CLI to generate submission
    out = predict_on_test()
    os.makedirs('submission', exist_ok=True)
    out.to_csv('submission/submission.csv', index=False)
    print('Wrote submission/submission.csv')
