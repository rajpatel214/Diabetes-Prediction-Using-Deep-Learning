# Diabetes-Prediction-Using-Deep-Learning
## Overview
This project aims to predict diabetes using a deep learning model built with TensorFlow and Keras. The dataset contains patient health records, including features like age, BMI, HbA1c levels, and blood glucose levels. The model is trained to classify whether a patient has diabetes based on these medical indicators.

## Dataset
- **Source:** `diabetes_prediction_dataset.csv`
- **Features:**
  - `gender`: Male/Female
  - `age`: Patient's age
  - `hypertension`: 1 if the patient has hypertension, else 0
  - `heart_disease`: 1 if the patient has heart disease, else 0
  - `smoking_history`: Categorical (never, former, current, etc.)
  - `bmi`: Body Mass Index
  - `HbA1c_level`: Hemoglobin A1c level
  - `blood_glucose_level`: Blood glucose level
  - `diabetes`: 1 if the patient has diabetes, else 0 (Target variable)

## Exploratory Data Analysis (EDA)
The dataset is analyzed using Pandas, Matplotlib, and Seaborn:
- Missing values are checked and handled.
- Duplicate records are identified.
- Gender distribution is visualized using a pie chart.
- Histograms and count plots are created for key variables like age, hypertension, heart disease, smoking history, BMI, and blood glucose levels.
- Pivot tables analyze relationships between smoking history, gender, and health indicators.

## Model Architecture
A deep learning model is implemented using TensorFlow and Keras:
- **Input Layer:** 8 features
- **Hidden Layers:**
  - Dense(64, activation='relu')
  - Dense(32, activation='relu')
  - Dense(18, activation='relu')
- **Output Layer:** Dense(1, activation='sigmoid')
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (learning rate = 0.001)

## Training & Evaluation
- The dataset is split into **80% training** and **20% testing**.
- **Min-Max Scaling** is applied to normalize feature values.
- Early stopping is used to prevent overfitting.
- Performance metrics include:
  - Accuracy: **97%**
  - Classification Report:
    - Precision, Recall, and F1-score for diabetic and non-diabetic patients
    - Confusion matrix analysis

## Results
- The model achieved a **high accuracy of 97%**.
- The **precision for diabetic patients was 99%**, but recall was slightly lower (68%), indicating some false negatives.
- Model performance visualization includes:
  - Loss and accuracy plots for training and validation data.
