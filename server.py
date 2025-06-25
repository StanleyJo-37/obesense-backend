from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": [
    "http://localhost:3000", 
    "https://peter-louisx.github.io"
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

        target_labels = {
            'Insufficient_Weight': 0,
            'Normal_Weight': 1,
            'Overweight_Level_I': 2,
            'Overweight_Level_II': 3,
            'Obesity_Type_I': 4,
            'Obesity_Type_II': 5,
            'Obesity_Type_III': 6,
        }

        for key, value in target_labels.items():
            if value == pred[0]:
                pred = key
                break
        
        return jsonify({'prediction': pred.replace("_", " ")})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')  # Runs on http://127.0.0.1:5000 by default
    #change host to 0.0.0.
    print('hello world')
