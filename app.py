from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler (update the paths accordingly)
model = joblib.load(r'C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/rf_model.pkl')  # Update with the correct path
scaler = joblib.load(r'C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/scaler.pkl')  # Update with the correct path

SCALED_FEATURES = [
    'AI Adoption Rate (%)',
    'Job Loss Due to AI (%)',
    'Revenue Increase Due to AI (%)',
    'Human-AI Collaboration Rate (%)',
    'Consumer Trust in AI (%)',
    'Market Share of AI Companies (%)'
]

UNSCALED_FEATURES = [
    'AI Investment (in Billion $)',
    'Number of AI Startups',
    'AI Regulation Score'
]

ALL_FEATURES = SCALED_FEATURES + UNSCALED_FEATURES

@app.route('/')
def home():
    return render_template('index.html', features=ALL_FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all inputs
        input_data = {feature: float(request.form.get(feature, 0)) for feature in ALL_FEATURES}

        # Scale only the scaled features
        scaled_values = scaler.transform([[input_data[feature] for feature in SCALED_FEATURES]])[0]

        # Combine scaled + unscaled features in correct order
        final_input = list(scaled_values) + [input_data[feature] for feature in UNSCALED_FEATURES]

        # Prediction
        prediction = model.predict([final_input])[0]

        return render_template('index.html', prediction=prediction, input_data=input_data, features=ALL_FEATURES)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)