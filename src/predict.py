# src/predict.py
import os
import pandas as pd
import numpy as np
import joblib
import importlib
import sys
import types
import traceback

from src.data_preprocessing import preprocess

ARTIFACTS_DIR = 'artifacts'
CLASSIFIER_PATH = os.path.join(ARTIFACTS_DIR, 'classifier_trainer.pkl')
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')
MAP_STR2INT_PATH = os.path.join(ARTIFACTS_DIR, 'map_str2int.pkl')
REG_GLOBAL_PATH = os.path.join(ARTIFACTS_DIR, 'regressor_global.pkl')
REG_PER_ITEM_PATH = os.path.join(ARTIFACTS_DIR, 'regressors_per_item.pkl')
REG_CAT_MAPS_PATH = os.path.join(ARTIFACTS_DIR, 'regressor_cat_mappings.pkl')

# optional download libs (import if available)
try:
    import gdown  # google drive friendly downloader
except Exception:
    gdown = None

try:
    import requests
except Exception:
    requests = None


def _download_model_from_url(url, dest_path):
    """
    Attempt to download url -> dest_path.
    Supports Google Drive share links via gdown (if available), else uses requests.
    Returns True on success.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Normalize Google Drive share links: accept 'https://drive.google.com/file/d/FILE_ID/view'
    try:
        if "drive.google.com" in url and gdown:
            # gdown accepts both full URL and id form
            gdown.download(url, dest_path, quiet=False)
            return os.path.exists(dest_path)
    except Exception:
        # continue to other methods
        pass

    # Try requests
    if requests:
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return os.path.exists(dest_path)
        except Exception:
            pass

    # Try gdown fallback even if not drive pattern
    if gdown:
        try:
            gdown.download(url, dest_path, quiet=False)
            return os.path.exists(dest_path)
        except Exception:
            pass

    return False


def ensure_model_available(path=CLASSIFIER_PATH):
    """
    Ensure the classifier pickle exists locally. If missing, attempt to download using
    the environment variable MODEL_URL (must be a public direct-download link).
    Returns (True, msg) on success, (False, error_msg) on failure.
    """
    if os.path.exists(path):
        return True, "Model artifact exists."

    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        return False, ("Model artifact not found at path and no MODEL_URL env var set. "
                       "Set MODEL_URL to a public downloadable URL (Google Drive / S3 / HF) "
                       "or place the classifier_trainer.pkl into the artifacts/ directory.")

    try:
        ok = _download_model_from_url(model_url, path)
        if not ok:
            return False, f"Attempted download from MODEL_URL but failed. URL: {model_url}"
        return True, f"Downloaded model from MODEL_URL to {path}."
    except Exception as e:
        return False, f"Exception while downloading model: {e}\n{traceback.format_exc()}"


def load_trainer(path=CLASSIFIER_PATH):
    """
    Load the saved trainer object. If the artifact is missing, attempt to download using MODEL_URL.
    Also handles unpickling issues by ensuring class definitions exist in __main__.
    """
    # ensure artifact present (or try download)
    if not os.path.exists(path):
        ok, msg = ensure_model_available(path)
        if not ok:
            raise FileNotFoundError(f"Trainer artifact not found at: {path}\n{msg}")
        # else continue to load

    # Try to import trainer module definitions so unpickling can find classes
    tried = []
    tm = None
    for modname in ('src.train_model', 'src.train_model_full'):
        try:
            tm = importlib.import_module(modname)
            break
        except Exception as e:
            tried.append((modname, str(e)))

    if tm is None:
        try:
            import src
            importlib.reload(src)
            tm = importlib.import_module('src.train_model')
        except Exception:
            raise ImportError(f"Could not import trainer module; attempted: {tried}")

    # Prepare __main__ compatibility for unpickling
    main_mod = sys.modules.get('__main__')
    if main_mod is None:
        main_mod = types.ModuleType('__main__')
        sys.modules['__main__'] = main_mod

    candidate_class_names = ['BaselineTrainer', 'Trainer']
    for cname in candidate_class_names:
        if not hasattr(main_mod, cname) and hasattr(tm, cname):
            setattr(main_mod, cname, getattr(tm, cname))

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


def _ensure_unpickle_compat():
    """
    Try importing modules that may define classes referenced by regressors.
    Also injects their classes into __main__ to help joblib/pickle unpickle objects.
    This is best-effort and will not raise on failure; it prints debug info.
    """
    tried = []
    for modname in ('src.train_model_full', 'src.train_model'):
        try:
            tm = importlib.import_module(modname)
            main_mod = sys.modules.get('__main__')
            if main_mod is None:
                main_mod = types.ModuleType('__main__')
                sys.modules['__main__'] = main_mod
            # copy likely class names into __main__
            for cname in ('BaselineTrainer', 'Trainer', 'QtyRegressor', 'PerItemRegressor'):
                if hasattr(tm, cname) and not hasattr(main_mod, cname):
                    setattr(main_mod, cname, getattr(tm, cname))
            return True, f"Imported {modname} for unpickle compatibility."
        except Exception as e:
            tried.append((modname, str(e)))
    # nothing imported
    print("Warning: could not import train modules for unpickle compatibility. Tried:", tried)
    return False, f"Failed imports: {tried}"


def _safe_load(path, friendly_name):
    """
    Try to joblib.load(path) but catch exceptions and return (obj, msg).
    On error returns (None, message-with-traceback)
    """
    if not os.path.exists(path):
        return None, f"{friendly_name} not found at {path}"
    try:
        # best-effort: ensure any custom classes are importable
        _ensure_unpickle_compat()
        obj = joblib.load(path)
        return obj, f"Loaded {friendly_name} from {path}"
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Failed to load {friendly_name} from {path}: {e}\n{tb}"


def _load_regressors():
    """
    Robustly load optional regressor artifacts (global + per-item + cat maps).
    If an artifact is missing or unpickling fails, return None / {} and print a helpful message.
    """
    reg_global = None
    per_item = {}
    cat_maps = {}

    rg, msg_rg = _safe_load(REG_GLOBAL_PATH, "global regressor")
    if rg is not None:
        reg_global = rg
    else:
        print(msg_rg)

    rp, msg_rp = _safe_load(REG_PER_ITEM_PATH, "per-item regressors")
    if rp is not None:
        per_item = rp
    else:
        print(msg_rp)

    rc, msg_rc = _safe_load(REG_CAT_MAPS_PATH, "regressor category mappings")
    if rc is not None:
        cat_maps = rc
    else:
        print(msg_rc)

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
    try:
        from src.features import add_basic_features
        return add_basic_features(df)
    except Exception:
        # Minimal fallback: select available columns and coerce types
        cols = ['PROJECT_CITY', 'PROJECT_TYPE', 'CORE_MARKET', 'UOM', 'SIZE_BUILDINGSIZE']
        X = pd.DataFrame()
        for c in cols:
            X[c] = df.get(c, pd.Series([np.nan]*len(df)))
        X['SIZE_BUILDINGSIZE'] = pd.to_numeric(X['SIZE_BUILDINGSIZE'], errors='coerce').fillna(0)
        return X


def _encode_regressor_objects(X, cat_mappings):
    """Encode object/category columns using saved mappings (or factorize as fallback)."""
    X_enc = X.copy()
    obj_cols = X_enc.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in obj_cols:
        X_enc[c] = X_enc[c].fillna('__MISSING__').astype(str)
        if c in cat_mappings:
            uniques = cat_mappings[c]
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
    X_basic = _build_basic_features(df_test)
    X_text = _build_text_features(df_test)

    X = X_basic.reset_index(drop=True).copy()
    if X_text is None:
        X_text = np.zeros((len(X), 0))
    for i in range(X_text.shape[1]):
        X[f'svd_{i}'] = X_text[:, i]

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
    map_str2int = load_map_str2int()
    masteritem_out = []
    for s in preds_raw:
        if map_str2int is not None and s in map_str2int:
            try:
                masteritem_out.append(int(map_str2int[s]))
            except Exception:
                masteritem_out.append(0)
        else:
            try:
                masteritem_out.append(int(float(s)))
            except Exception:
                masteritem_out.append(0)

    # Load regressors (global + per-item) and categorical mappings for regressor
    reg_global, per_item, cat_mappings = _load_regressors()

    med_lookup = _build_median_lookup('data/train.csv')
    global_median = 1.0
    try:
        df_train = pd.read_csv('data/train.csv')
        global_median = float(pd.to_numeric(df_train['QtyShipped'], errors='coerce').median())
        if np.isnan(global_median):
            global_median = 1.0
    except Exception:
        pass

    preds_qty = []
    X_reg = X.copy()
    X_reg_enc = _encode_regressor_objects(X_reg, cat_mappings)

    for i, s in enumerate(preds_raw):
        q = None
        s_key = str(s)
        if per_item and s_key in per_item:
            try:
                q = float(per_item[s_key].predict(X_reg_enc.iloc[[i]])[0])
            except Exception:
                # safe fallback
                print(f"Per-item regressor failed for item {s_key} at idx {i}:", traceback.format_exc())
                q = None

        if q is None and reg_global is not None:
            try:
                q = float(reg_global.predict(X_reg_enc.iloc[[i]])[0])
            except Exception:
                print(f"Global regressor failed at idx {i}:", traceback.format_exc())
                q = None

        if q is None:
            q = med_lookup.get(s_key)
        if q is None:
            try:
                q = med_lookup.get(str(int(float(s_key))))
            except Exception:
                q = None

        if q is None or (isinstance(q, float) and np.isnan(q)):
            q = global_median

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
    out = predict_on_test()
    os.makedirs('submission', exist_ok=True)
    out.to_csv('submission/submission.csv', index=False)
    print('Wrote submission/submission.csv')
