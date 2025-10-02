import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocess import clean_data, encode_features

# Load data
df = pd.read_csv("data/OCD Patient Dataset_ Demographics & Clinical Data.csv")
df = clean_data(df)

# Drop rows with missing Medications
df = df.dropna(subset=['Medications'])

# Drop ID/date columns
drop_cols = ['Patient ID','OCD Diagnosis Date']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode
X, y, le = encode_features(df, target_col='Medications')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train model (choose best, here XGBoost)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save pipeline
pipeline = {
    'model': model,
    'scaler': scaler,
    'columns': X.columns.tolist(),
    'label_encoder_classes': le.classes_.tolist()
}
joblib.dump(pipeline, "models/ocd_med_pipeline.pkl")
print("✅ Model saved to models/ocd_med_pipeline.pkl")
