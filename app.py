from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__, template_folder='.')  # Look in current directory for templates
CORS(app)

# Load artifacts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

FEATURES = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance\t', 'Previous qualification',
    'Previous qualification (grade)', 'Nacionality', "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation",
    'Admission grade', 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International', 'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
    'Inflation rate', 'GDP'
]

@app.route('/')
def home():
    return render_template('index.html')  # Now looks in current directory

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle both form data and JSON requests
        if request.is_json:
            input_data = request.get_json()
        else:
            input_data = request.form.to_dict()

        processed_data = {}
        for feature in FEATURES:
            value = input_data.get(feature)
            
            if not value:
                raise ValueError(f"Missing value for {feature}")
            
            # Handle categorical features
            if feature in label_encoders:
                try:
                    processed_data[feature] = label_encoders[feature].transform([value])[0]
                except ValueError:
                    valid = list(label_encoders[feature].classes_)
                    raise ValueError(f"Invalid {feature}. Valid options: {valid}")
            elif feature == 'Daytime/evening attendance\t':
                processed_data[feature] = 1 if str(value).lower() in ['daytime', '1'] else 0
            else:
                try:
                    processed_data[feature] = float(value)
                except ValueError:
                    raise ValueError(f"Invalid number for {feature}")

        # Create and process DataFrame
        input_df = pd.DataFrame([processed_data])[FEATURES]
        scaled = scaler.transform(input_df)
        pca_features = pca.transform(scaled)
        prediction = model.predict(pca_features)[0]
        prediction = label_encoders['Target'].inverse_transform([prediction])[0]

        # Return response
        if request.is_json:
            return jsonify({
                'prediction': prediction,
                'status': 'success',
                'confidence': float(np.max(model.predict_proba(pca_features)))
            })
        else:
            return render_template('index.html',
                                prediction_text=f'Prediction: {prediction}',
                                show_result=True)

    except Exception as e:
        error_msg = str(e)
        if request.is_json:
            return jsonify({
                'error': error_msg,
                'status': 'error',
                'required_fields': FEATURES
            }), 400
        else:
            return render_template('index.html',
                                error_text=f'Error: {error_msg}',
                                show_result=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
