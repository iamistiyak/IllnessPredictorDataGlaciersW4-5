from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import requests
import os
import gc

# Initialize Flask app
app = Flask(__name__)

# Load trained model and expected features
#model, expected_features = joblib.load("illness_model.pkl")
MODEL_PATH = "illness_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1dk31QrP6NxE2R5rO1brDW-mUKFpro32t"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)

model, expected_features = joblib.load(MODEL_PATH)

# Extract city column names from expected_features
city_columns = [col for col in expected_features if 'City_' in col]
gender_column = [col for col in expected_features if 'Gender_' in col][0]  # Only one dummy for Gender

@app.route("/", methods=["GET", "POST"])
def home():
    # Prepare dropdown options
    cities = [col.split("City_")[-1] for col in city_columns]
    genders = ['Male', 'Female']  # Assuming binary gender

    if request.method == "POST":
        try:
            # Get form data
            city = request.form["City"]
            gender = request.form["Gender"]
            age = int(request.form["Age"])
            income = int(request.form["Income"])

            # One-hot encode city
            city_encoded = np.zeros(len(city_columns))
            if f'City_{city}' in city_columns:
                index = city_columns.index(f'City_{city}')
                city_encoded[index] = 1

            # One-hot encode gender (assuming 'Male' is the dummy variable)
            gender_encoded = [1] if gender == "Male" else [0]

            # Combine all features
            features = np.hstack((city_encoded, gender_encoded, [age, income])).reshape(1, -1)

            print("Feature vector shape:", features.shape)
            print("Expected feature count:", len(expected_features))

            if features.shape[1] != len(expected_features):
                raise ValueError(f"Feature mismatch: Expected {len(expected_features)}, got {features.shape[1]}")

            # Make prediction
            probability = model.predict_proba(features)[0][1]
            probability = round(probability, 4)

            return render_template("index.html", prediction=probability, cities=cities, genders=genders)

        except Exception as e:
            return render_template("index.html", error=str(e), cities=cities, genders=genders)

    return render_template("index.html", cities=cities, genders=genders)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
