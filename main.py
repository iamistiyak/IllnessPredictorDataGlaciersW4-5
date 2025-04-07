from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import requests
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "illness_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1QxpL2Vi_Lhoo2k3Vt0Q983nkHlEXOWj1"


# Global variables (initially None)
model = None
expected_features = None
city_columns = []
gender_column = None

def load_model_once():
    global model, expected_features, city_columns, gender_column

    if model is None or expected_features is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)

        print("Loading model...")
        model, expected_features = joblib.load(MODEL_PATH)
        city_columns = [col for col in expected_features if 'City_' in col]
        gender_column = [col for col in expected_features if 'Gender_' in col][0]

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
            if f'City_{city}' in city_columns:
                index = city_columns.index(f'City_{city}')
                city_encoded[index] = 1

            # One-hot encode gender
            gender_encoded = [1] if gender == "Male" else [0]

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
    app.run(host='0.0.0.0', debug=True)
