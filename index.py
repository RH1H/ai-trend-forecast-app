from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Dummy model & scaler loading (use your own .pkl paths)
model = pickle.load(open('C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/rf_model.pkl', 'rb'))
scaler = pickle.load(open('C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/scaler.pkl', 'rb'))
# model = joblib.load(r'C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/rf_model.pkl')  # Update with the correct path
# scaler = joblib.load(r'C:/Users/Rohith/OneDrive/Desktop/SRGECWorkshop/AI_Impact_App/models/scaler.pkl')  # Update with the correct path
# Features used by the model
features = [
    "AI Adoption Rate (%)",
    "Job Loss Due to AI (%)",
    "Revenue Increase Due to AI (%)",
    "Human-AI Collaboration Rate (%)",
    "Consumer Trust in AI (%)",
    "Market Share of AI Companies (%)",
    "AI Investment (in Billion $)",
    "Number of AI Startups",
    "AI Regulation Score"
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    input_data = {}

    if request.method == 'POST':
        for feature in features:
            input_data[feature] = float(request.form.get(feature, 0))
        
        input_values = [input_data[feature] for feature in features]
        scaled_values = scaler.transform([input_values])
        prediction = model.predict(scaled_values)[0]

    return render_template('form.html', features=features, prediction=prediction, input_data=input_data)

