# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "toy_dataset.csv"))

# Preprocessing
df = df.drop(columns=['Number'])
df = pd.get_dummies(df, columns=['City', 'Gender'], drop_first=True)
df['Illness'] = df['Illness'].map({'No': 0, 'Yes': 1})

X = df.drop(columns=['Illness'])
y = df['Illness']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)




# Predict on the test set
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# Save model and feature names
joblib.dump((model, X_train.columns.tolist()), 'illness_model.pkl')

print("Model trained and saved.")
print("Features used:", X_train.columns.tolist())
