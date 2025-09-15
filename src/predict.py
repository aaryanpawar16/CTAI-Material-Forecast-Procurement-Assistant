# src/predict.py
import os
import pandas as pd
import numpy as np
import joblib
import importlib
import sys
import types
import traceback
import tempfile
from pathlib import Path
from urllib.parse import urlparse

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
    import gdown
except Exception:
    gdown = None

try:
    import requests
except Exception:
    requests = None


def _download_http_to_file(url: str, dest_path: str, timeout=120) -> bool:
    """Download HTTP/HTTPS url to dest_path, sanity-checking content-type."""
    if not requests:
        return False
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        # common failure: HTML login/error page — catch that early
        if "text/html" in content_type.lower() or resp.headers.get("Content-Length", "0") == "0":
            print(f"[download] Refusing to save remote content with Content-Type: {content_type}")
            return False
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return os.path.exists(dest_path)
    except Exception:
        print("[download] HTTP download failed:", traceback.format_exc())
        return False


def _download_model_from_url(url, dest_path):
    """
    Try multiple strategies to download a remote model file to dest_path.
    Supports:
      - direct HTTP/HTTPS (requests)
      - Google Drive via gdown (if url looks like drive)
      - fallback to gdown if present
    Returns True on success.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # direct HTTP/HTTPS
    parsed = urlparse(str(url))
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        ok = _download_http_to_file(url, dest_path)
        if ok:
            return True

    # google drive shortcuts / gdown
    try:
        if "drive.google.com" in str(url) and gdown:
            try:
                gdown.download(url, dest_path, quiet=False)
                return os.path.exists(dest_path)
            except Exception:
                print("[download] gdown (drive) attempt failed:", traceback.format_exc())
    except Exception:
        pass

    # fallback: try gdown for other url forms if available
    if gdown:
        try:
            gdown.download(url, dest_path, quiet=False)
            return os.path.exists(dest_path)
        except Exception:
            print("[download] gdown fallback failed:", traceback.format_exc())

    return False


def ensure_model_available(path=CLASSIFIER_PATH):
    """
    Ensure a model exists at `path`. If not present, try to download from MODEL_URL env var.
    Returns (bool_ok, message_string)
    """
    if os.path.exists(path):
        return True, "Model artifact exists locally."
    model_url = os.environ.get("MODEL_URL", "").strip()
    if not model_url:
        return False, ("Model artifact not found and MODEL_URL not set. "
                       f"Place {os.path.basename(path)} into {os.path.dirname(path) or '.'} or set MODEL_URL env var.")
    try:
        ok = _download_model_from_url(model_url, path)
        if not ok:
            return False, f"Attempted download from MODEL_URL but failed. URL: {model_url}"
        return True, f"Downloaded model from MODEL_URL to {path}."
    except Exception as e:
        return False, f"Exception while downloading model: {e}\n{traceback.format_exc()}"


def _ensure_unpickle_compat():
    """
    Best-effort: import training modules and inject likely classes into __main__
    so pickle can locate them. This helps when the pickled object references
    project-local classes.
    """
    tried = []
    for modname in ('src.train_model_full', 'src.train_model', 'train_model', 'train'):
        try:
            tm = importlib.import_module(modname)
            main_mod = sys.modules.get('__main__')
            if main_mod is None:
                main_mod = types.ModuleType('__main__')
                sys.modules['__main__'] = main_mod
            for cname in ('BaselineTrainer', 'Trainer', 'QtyRegressor', 'PerItemRegressor', 'TrainerFull'):
                if hasattr(tm, cname) and not hasattr(main_mod, cname):
                    setattr(main_mod, cname, getattr(tm, cname))
            return True, f"Imported {modname}"
        except Exception as e:
            tried.append((modname, str(e)))
    # nothing imported
    print("Warning: unpickle compatibility imports failed. Tried:", tried)
    return False, f"Failed imports: {tried}"


def _safe_load(path, friendly_name):
    """
    Try loading with joblib but catch and return (obj, message).
    Does unpickle compatibility preloading first.
    """
    if not os.path.exists(path):
        return None, f"{friendly_name} not found at {path}"
    try:
        _ensure_unpickle_compat()
        obj = joblib.load(path)
        return obj, f"Loaded {friendly_name} from {path}"
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"Failed to load {friendly_name} from {path}: {e}\n{tb}"


def load_label_encoder(path=LABEL_ENCODER_PATH):
    """Load label encoder safely using _safe_load."""
    le, msg = _safe_load(path, "label encoder")
    if le is None:
        print(f"[predict] {msg}")
    return le


def load_map_str2int(path=MAP_STR2INT_PATH):
    mapping, msg = _safe_load(path, "map_str2int")
    if mapping is None:
        print(f"[predict] {msg}")
    return mapping


def _build_median_lookup(train_csv='data/train.csv'):
    if not os.path.exists(train_csv):
        return {}
    try:
        df = pd.read_csv(train_csv)
        if 'MasterItemNo' not in df.columns or 'QtyShipped' not in df.columns:
            return {}
        df['MasterItemNo_str'] = df['MasterItemNo'].astype(str).str.strip()
        df['QtyShipped_num'] = pd.to_numeric(df['QtyShipped'], errors='coerce')
        med = df.groupby('MasterItemNo_str')['QtyShipped_num'].median().to_dict()
        return med
    except Exception:
        print("[predict] failed to build medians:", traceback.format_exc())
        return {}


def load_trainer(path=CLASSIFIER_PATH):
    """
    Robust trainer loader:
    - ensures artifact is present (download via MODEL_URL if not)
    - attempts to joblib.load the trainer (with unpickle compatibility)
    - on repeated failures, returns a DummyTrainer that yields safe trivial predictions
    """
    # ensure artifact present (or attempt download)
    if not os.path.exists(path):
        ok, msg = ensure_model_available(path)
        if not ok:
            print(f"[predict] trainer missing and could not download: {msg}")
        else:
            print(f"[predict] downloaded trainer: {msg}")

    # attempt to load actual trainer
    if os.path.exists(path):
        try:
            _ensure_unpickle_compat()
            trainer = joblib.load(path)
            print(f"[predict] Loaded trainer from {path}")
            return trainer
        except Exception:
            print("=== ERROR: failed to load trainer artifact ===")
            print(traceback.format_exc())

    # If we reach here, build and return a DummyTrainer fallback
    print("[predict] Returning DummyTrainer fallback — predictions will be trivial but app will remain functional.")

    # compute a reasonable fallback label
    fallback_label = None
    try:
        le = load_label_encoder()
        if le is not None and hasattr(le, 'classes_') and len(le.classes_) > 0:
            fallback_label = str(le.classes_[0])
            print("[predict] fallback_label from label_encoder:", fallback_label)
    except Exception:
        pass

    if fallback_label is None:
        try:
            df_train = pd.read_csv('data/train.csv')
            if 'MasterItemNo' in df_train.columns:
                mc = df_train['MasterItemNo'].astype(str).mode()
                if not mc.empty:
                    fallback_label = mc.iloc[0]
                    print("[predict] fallback_label from train.csv most-frequent:", fallback_label)
        except Exception:
            pass

    if fallback_label is None:
        fallback_label = "0"
        print("[predict] no fallback label found — using '0'")

    class DummyTrainer:
        def __init__(self, label):
            self._label = label
            try:
                self._label_int = int(float(label))
            except Exception:
                self._label_int = None

        def predict_classifier(self, X):
            """
            Return encoded predictions. Attempt to map via label encoder if available,
            otherwise return integer label or zeros.
            """
            n = len(X)
            try:
                le = load_label_encoder()
                if le is not None:
                    # If label encoder has transform:
                    try:
                        arr = le.transform([self._label] * n)
                        return np.array(arr, dtype=int)
                    except Exception:
                        pass
                    if hasattr(le, 'classes_'):
                        classes = [str(x) for x in le.classes_]
                        if str(self._label) in classes:
                            idx = classes.index(str(self._label))
                            return np.array([idx] * n, dtype=int)
            except Exception:
                print("[DummyTrainer] label encoder mapping failed:", traceback.format_exc())

            if self._label_int is not None:
                return np.array([self._label_int] * n, dtype=int)
            return np.zeros(n, dtype=int)

    return DummyTrainer(fallback_label)


def _load_regressors():
    """
    Robustly attempt to load optional regressors and category mappings.
    Returns (reg_global_or_None, per_item_dict_or_empty, cat_mappings_or_empty)
    """
    reg_global, msg1 = _safe_load(REG_GLOBAL_PATH, "global regressor")
    if reg_global is None:
        print(f"[predict] {msg1}")
    per_item, msg2 = _safe_load(REG_PER_ITEM_PATH, "per-item regressors")
    if per_item is None:
        per_item = {}
        print(f"[predict] {msg2}")
    cat_maps, msg3 = _safe_load(REG_CAT_MAPS_PATH, "regressor category mappings")
    if cat_maps is None:
        cat_maps = {}
        print(f"[predict] {msg3}")
    return reg_global, per_item, cat_maps


def _build_text_features(df):
    try:
        from src.features import transform_text_pipeline
        X_text = transform_text_pipeline(df, text_col='ItemDescription')
        # ensure numpy array
        return np.asarray(X_text)
    except Exception:
        svd_components = 50
        try:
            _svd = joblib.load(os.path.join(ARTIFACTS_DIR, 'svd.pkl'))
            svd_components = getattr(_svd, 'n_components', svd_components)
        except Exception:
            pass
        return np.zeros((len(df), svd_components))


def _build_basic_features(df):
    try:
        from src.features import add_basic_features
        return add_basic_features(df)
    except Exception:
        cols = ['PROJECT_CITY', 'PROJECT_TYPE', 'CORE_MARKET', 'UOM', 'SIZE_BUILDINGSIZE']
        X = pd.DataFrame()
        for c in cols:
            X[c] = df.get(c, pd.Series([np.nan] * len(df)))
        X['SIZE_BUILDINGSIZE'] = pd.to_numeric(X['SIZE_BUILDINGSIZE'], errors='coerce').fillna(0)
        return X


def _encode_regressor_objects(X, cat_mappings):
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
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    df_test = pd.read_csv(test_csv)
    df_test = preprocess(df_test)

    if 'id' not in df_test.columns:
        df_test = df_test.reset_index().rename(columns={'index': 'id'})
        df_test['id'] = df_test['id'] + 1

    trainer = load_trainer(trainer_path)

    X_basic = _build_basic_features(df_test)
    X_text = _build_text_features(df_test)

    X = X_basic.reset_index(drop=True).copy()
    if X_text is None:
        X_text = np.zeros((len(X), 0))
    # If X_text is numpy with shape (n, k), append as new columns
    if hasattr(X_text, "shape") and X_text.shape[1] > 0:
        for i in range(X_text.shape[1]):
            X[f'svd_{i}'] = X_text[:, i]
    else:
        # ensure at least 0 svd columns
        pass

    for c in ['PROJECT_CITY', 'PROJECT_TYPE', 'CORE_MARKET', 'UOM']:
        if c in X.columns:
            X[c] = X[c].astype(str).fillna('missing')

    try:
        preds_encoded = trainer.predict_classifier(X)
        preds_encoded = np.asarray(preds_encoded).astype(int)
    except Exception:
        print("[predict] trainer.predict_classifier raised exception; falling back to zeros. Traceback:")
        print(traceback.format_exc())
        preds_encoded = np.zeros(len(X), dtype=int)

    le = load_label_encoder()
    if le is not None:
        try:
            preds_raw = le.inverse_transform(preds_encoded)
            preds_raw = [str(x).strip() for x in preds_raw]
        except Exception:
            preds_raw = [str(int(x)) for x in preds_encoded]
    else:
        preds_raw = [str(int(x)) for x in preds_encoded]

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
                print(f"[predict] per-item regressor failed for {s_key} idx {i}:", traceback.format_exc())
                q = None

        if q is None and reg_global is not None:
            try:
                q = float(reg_global.predict(X_reg_enc.iloc[[i]])[0])
            except Exception:
                print(f"[predict] global regressor failed idx {i}:", traceback.format_exc())
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
    try:
        out = predict_on_test()
        os.makedirs('submission', exist_ok=True)
        out.to_csv('submission/submission.csv', index=False)
        print('Wrote submission/submission.csv')
    except Exception:
        print("[predict] Unhandled exception in main:", traceback.format_exc())
