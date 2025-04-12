# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Read data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "toy_dataset.csv"))

# Simplify City values (keep top 10)
top_cities = df['City'].value_counts().nlargest(10).index
df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')

# One-hot encoding (no drop_first)
df = pd.get_dummies(df, columns=['City', 'Gender'], drop_first=False)

# Target encoding
df['Illness'] = df['Illness'].map({'No': 0, 'Yes': 1})

# Features & target
X = df.drop(columns=['Illness', 'Number'])
y = df['Illness']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Logistic Regression
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump((model, X_train.columns.tolist()), 'illness_model.pkl', compress=3)
print("Model saved. File size should now be much smaller.")