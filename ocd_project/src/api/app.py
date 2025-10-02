from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
pipeline = joblib.load("models/ocd_med_pipeline.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    row = pd.DataFrame([data])
    row = pd.get_dummies(row)
    row = row.reindex(columns=pipeline['columns'], fill_value=0)
    x_s = pipeline['scaler'].transform(row)
    pred = pipeline['model'].predict(x_s)
    label = pipeline['label_encoder_classes'][pred[0]]
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
