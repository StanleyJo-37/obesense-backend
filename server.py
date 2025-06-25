from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pickle
import pandas as pd
from waitress import serve

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000", 
    "https://peter-louisx.github.io",
]}})

model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        input_df = pd.DataFrame([{
            'Age': float(request.json['age']),
            'Height': float(request.json['height']),
            'Weight': float(request.json['weight']),
            'FCVC': float(request.json['fcvc']),
            'NCP': float(request.json['ncp']),
            'CH2O': float(request.json['ch2o']),
            'FAF': float(request.json['faf']),
            'TUE': float(request.json['tue']),
            'Gender': request.json['gender'],
            'CAEC': request.json['caec'],
        }])
        
        pred = model.predict(input_df)

        target_labels = [
            'Insufficient Weight',
            'Normal Weight',
            'Overweight Level I',
            'Overweight Level II',
            'Obesity Type I',
            'Obesity Type II',
            'Obesity Type III',
        ]
        
        return jsonify({'prediction': target_labels[pred[0]]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)