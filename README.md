# IllnessPredictor

**Description:** The Illness Prediction Web App is a machine learning-based Flask web application designed to predict the likelihood of a person being affected by a particular illness based on various input features such as city, gender, age, and income. This app utilizes a Random Forest Classifier trained on a dataset containing demographic and illness-related information.<br><br>

**DATASET:**https://www.kaggle.com/datasets/carlolepelaars/toy-dataset <br><br><br><br>

**Table of Contents:** <br>

•Project Overview <br>
•Features<br>
•Tech Stack<br>
•Setup Instructions<br>
•How to Use<br>
•Model Details<br>
•Use this model<br>


# Features
•User Input Form: Allows users to input their City, Gender, Age, and Income for illness prediction.

•Predictions: After input, the model predicts the probability of the user suffering from a particular illness.

•Machine Learning Model: A Random Forest Classifier is used for making the illness predictions based on historical data.

# Tech Stack
•Backend: Python, Flask

•Machine Learning: scikit-learn (Random Forest Classifier)

•Frontend: HTML, CSS (for creating the user interface)

•Database: None (Currently using static dataset)

•Version Control: Git, GitHub

# Setup Instructions
To run this project locally on your machine, follow the steps below:

Prerequisites
•Python (3.7 or higher)

•Git (for version control)

•pip (Python package installer)


# How to Use
Input Form: The app presents a form where users can select a City from the dropdown, choose their Gender, input their Age and Income.

Prediction: Once the user fills out the form and submits it, the Flask app collects the data and processes it through the Random Forest Classifier model to return a prediction on whether the user might be at risk for a particular illness.

Results: The app displays the predicted illness probability with a success or error message.

# Model Details
The prediction model is based on a Random Forest Classifier trained on a dataset with information about individuals' City, Gender, Age, and Income. The model outputs a probability value between 0 and 1, representing the likelihood that the user is at risk of the illness.

# Dataset
The dataset (toy_dataset.csv) contains historical data about people's characteristics and whether they had an illness or not. It was used to train the model. You can see a sample of this data below: <br>

| **City**   | **Gender** | **Age** | **Income ($)** | **Illness** |
|------------|------------|---------|----------------|-------------|
| City A     | Male       | 34      | 40000          | Yes         |
| City B     | Female     | 56      | 55000          | No          |
| City A     | Female     | 23      | 30000          | Yes         |
| City C     | Male       | 45      | 60000          | No          |
| City B     | Male       | 30      | 45000          | Yes         |

# Training the Model
The dataset was pre-processed by encoding categorical variables (like City and Gender) using one-hot encoding and mapping the target variable Illness (Yes/No) into numeric values (1 for Yes, 0 for No). The model was trained using 80% of the data for training and 20% for testing.

# Use this model
Anyone can install the dependencies using:
pip install -r requirements.txt





https://github.com/user-attachments/assets/c4c21a2d-39df-4a7f-8054-3ccbca3a12b6



