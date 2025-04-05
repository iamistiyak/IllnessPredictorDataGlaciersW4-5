from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
df = pd.read_csv(os.path.join(BASE_DIR, "toy_dataset.csv"))
# Load dataset

# Drop unnecessary columns
df = df.drop(columns=['Number'])

# Convert categorical columns to numeric
df = pd.get_dummies(df, columns=['City', 'Gender'], drop_first=True)
df['Illness'] = df['Illness'].map({'No': 0, 'Yes': 1})

# Split features and target
X = df.drop(columns=['Illness'])
y = df['Illness']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Features:", X_train.columns)
print("Number of features used in training:", X_train.shape[1])

print("Test Features:", X_test.columns)
print("Number of features used in test:", X_test.shape[1])
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'illness_model.pkl')
print("Model saved as 'illness_model.pkl'")

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("illness_model.pkl")

# Prepare column names for one-hot encoding the city input
city_columns = [col for col in df.columns if 'City_' in col]  # Based on the one-hot encoding from get_dummies


@app.route("/", methods=["GET", "POST"])
def home():
    # Extract unique cities and genders for dropdown
    city_columns = [col for col in df.columns if 'City_' in col]
    cities = [city.split('City_')[-1] for city in city_columns]  # Get city names
    genders = ['Male', 'Female']  # Assuming two genders: Male and Female

    if request.method == "POST":
        try:
            # Get form data
            city = request.form["City"]
            gender = request.form["Gender"]  # Ensure gender is captured
            age = int(request.form["Age"])
            income = int(request.form["Income"])

            # One-hot encoding for city
            city_encoded = np.zeros(len(city_columns))
            if f'City_{city}' in city_columns:
                city_index = city_columns.index(f'City_{city}')
                city_encoded[city_index] = 1

            # One-hot encoding for gender
            gender_encoded = [1] if gender == "Male" else [0]  # Assuming 'Male' is the dummy

            # Create feature vector (City + Gender + Age + Income)
            features = np.hstack((city_encoded, gender_encoded, np.array([age, income]))).reshape(1, -1)

            # Debugging prints
            print("Feature vector shape:", features.shape)
            print("Expected feature count:", len(X_train.columns))

            # Ensure correct shape
            if features.shape[1] != len(X_train.columns):
                raise ValueError(f"Feature mismatch: Expected {len(X_train.columns)}, got {features.shape[1]}")

            # Predict probability
            probability = model.predict_proba(features)[0][1]
            probability = round(probability, 4)

            return render_template("index.html", prediction=probability, cities=cities, genders=genders)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", cities=cities, genders=genders)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
