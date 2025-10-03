"""
nfl_model_api.py - Flask API for NFL Model Predictions
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load models
with open('nfl_betting_models.pkl', 'rb') as f:
    package = pickle.load(f)

models = package['models']
scaler = package['scalers']['main']
calibrator = package['calibrators']['win']

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 
        'models_loaded': True,
        'expected_features': len(scaler.feature_names_in_)
    })

@app.route('/api/nfl-model-predict', methods=['POST'])
def predict():
    try:
        data = request.json
        feature_cols = data.get('feature_columns', [])
        feature_values = data.get('feature_values', {})
        
        # Create DataFrame from provided features
        X = pd.DataFrame([feature_values])
        
        # CRITICAL FIX: Reorder columns to match training order
        expected_order = scaler.feature_names_in_
        
        # Check if all expected features are present
        missing_features = set(expected_order) - set(X.columns)
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Missing features: {missing_features}'
            }), 400
        
        # Reorder columns to match training
        X = X[expected_order]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        spread = float(models['spread'].predict(X_scaled)[0])
        total = float(models['total'].predict(X_scaled)[0])
        win_probs = calibrator.predict_proba(X_scaled)[0]
        
        return jsonify({
            'success': True,
            'predictions': {
                'spread': {
                    'value': spread,
                    'confidence': 'medium',
                    'model_r2': 0.156
                },
                'total': {
                    'value': total,
                    'confidence': 'low',
                    'model_r2': -0.099
                },
                'win_probability': {
                    'home': float(win_probs[1]),
                    'away': float(win_probs[0]),
                    'confidence': 'high',
                    'validation_accuracy': 0.625
                }
            }
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("PREDICTION ERROR:", error_trace)
        
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)

