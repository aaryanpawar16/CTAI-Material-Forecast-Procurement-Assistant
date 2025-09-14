# src/features.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ARTIFACTS_DIR = 'artifacts'
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, 'tfidf.pkl')
SVD_PATH = os.path.join(ARTIFACTS_DIR, 'svd.pkl')

def fit_text_pipeline(df, text_col='ItemDescription', max_features=20000, svd_n=50):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    texts = df[text_col].fillna('').astype(str).values
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=svd_n, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(svd, SVD_PATH)
    return X_svd

def transform_text_pipeline(df, text_col='ItemDescription'):
    if not os.path.exists(TFIDF_PATH) or not os.path.exists(SVD_PATH):
        raise FileNotFoundError("TF-IDF/SVD artifacts not found. Run fit_text_pipeline first.")
    tfidf = joblib.load(TFIDF_PATH)
    svd = joblib.load(SVD_PATH)
    texts = df[text_col].fillna('').astype(str).values
    X_tfidf = tfidf.transform(texts)
    X_svd = svd.transform(X_tfidf)
    return X_svd

def add_basic_features(df):
    # returns X dataframe with basic features: categorical raw + numeric
    X = pd.DataFrame()
    # standard feature list (keep consistent with train/predict)
    X['PROJECT_CITY'] = df.get('PROJECT_CITY','').fillna('missing').astype(str)
    X['PROJECT_TYPE'] = df.get('PROJECT_TYPE','').fillna('missing').astype(str)
    X['CORE_MARKET'] = df.get('CORE_MARKET','').fillna('missing').astype(str)
    X['UOM'] = df.get('UOM','').fillna('missing').astype(str)
    X['SIZE_BUILDINGSIZE'] = pd.to_numeric(df.get('SIZE_BUILDINGSIZE',0), errors='coerce').fillna(0)
    # frequency feature: how many times project appears (useful)
    if 'PROJECTNUMBER' in df.columns:
        proj_counts = df['PROJECTNUMBER'].fillna('NA').astype(str).value_counts()
        X['PROJECT_FREQ'] = df['PROJECTNUMBER'].fillna('NA').astype(str).map(proj_counts).fillna(0)
    else:
        X['PROJECT_FREQ'] = 0
    return X
