from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Use environment variable for the model path
MODEL_PATH = os.environ.get("MODEL_PATH", "illness_model.pkl")

# Global variables (initially None)
model = None
expected_features = None
city_columns = []
gender_columns = []

def load_model_once():
    global model, expected_features, city_columns, gender_columns

    if model is None or expected_features is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        print("Loading model...")
        model, expected_features = joblib.load(MODEL_PATH)
        city_columns = [col for col in expected_features if col.startswith('City_')]
        gender_columns = [col for col in expected_features if col.startswith('Gender_')]

@app.route("/", methods=["GET", "POST"])
def home():
    load_model_once()

    cities = [col.split("City_")[-1] for col in city_columns]
    genders = ['Male', 'Female']

    if request.method == "POST":
        try:
            city = request.form["City"]
            gender = request.form["Gender"]
            age = int(request.form["Age"])
            income = int(request.form["Income"])

            # One-hot encode city
            city_encoded = np.zeros(len(city_columns))
            city_feature = f'City_{city}'
            if city_feature in city_columns:
                index = city_columns.index(city_feature)
                city_encoded[index] = 1

            # One-hot encode gender (e.g., Gender_Female, Gender_Male)
            gender_encoded = np.zeros(len(gender_columns))
            gender_feature = f'Gender_{gender}'
            if gender_feature in gender_columns:
                index = gender_columns.index(gender_feature)
                gender_encoded[index] = 1

            # Combine all features
            features = np.hstack((city_encoded, gender_encoded, [age, income])).reshape(1, -1)

            if features.shape[1] != len(expected_features):
                raise ValueError(f"Feature mismatch: Expected {len(expected_features)}, got {features.shape[1]}")

            probability = model.predict_proba(features)[0][1]
            probability = round(probability, 4)

            return render_template("index.html", prediction=probability, cities=cities, genders=genders)

        except Exception as e:
            return render_template("index.html", error=str(e), cities=cities, genders=genders)

    return render_template("index.html", cities=cities, genders=genders)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
