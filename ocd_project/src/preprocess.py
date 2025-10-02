import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """Basic cleaning: strip spaces, parse dates, handle missing values."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if 'OCD Diagnosis Date' in df.columns:
        df['OCD Diagnosis Date'] = pd.to_datetime(df['OCD Diagnosis Date'], errors='coerce')
    for c in df.select_dtypes('object').columns:
        df[c] = df[c].astype(str).str.strip().replace({'nan': np.nan})
    return df

def encode_features(df, target_col='Medications'):
    """Encode categorical features + label encode target."""
    df_enc = df.copy()
    # Example ordinal map
    edu_map = {'Some College':1,'College Degree':2,'High School':3,'Graduate Degree':4}
    if 'Education Level' in df_enc.columns:
        df_enc['Education Level'] = df_enc['Education Level'].map(edu_map).fillna(df_enc['Education Level'])
    bin_map = {'No':0,'Yes':1}
    for c in ['Family History of OCD','Depression Diagnosis','Anxiety Diagnosis']:
        if c in df_enc.columns:
            df_enc[c] = df_enc[c].map(bin_map).fillna(df_enc[c])
    # One-hot encoding
    cols_to_dummify = [c for c in df_enc.select_dtypes(include='object').columns if c != target_col]
    df_enc = pd.get_dummies(df_enc, columns=cols_to_dummify, drop_first=True)
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_enc[target_col])
    X = df_enc.drop(columns=[target_col])
    return X, y, le
