# Predicting Customer Churn with Random Forest

This project demonstrates the development and deployment of a customer churn prediction model using the Random Forest algorithm. The workflow covers everything from problem conception to model deployment.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Cross-Validation and Hyperparameter Optimization](#cross-validation-and-hyperparameter-optimization)
- [Final Model Evaluation](#final-model-evaluation)
- [Deployment](#deployment)
- [Author](#author)

## Overview
This project aims to predict customer churn using a machine learning model. The steps include data loading, preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment.

## Installation
To run this project, you'll need Python and the following libraries:
- joblib
- sklearn
- pandas
- numpy
- matplotlib
- seaborn

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/churn_prediction.git
cd churn_prediction
```

Run the Jupyter notebook:
```bash
jupyter notebook Churn_ML.ipynb
```

## Project Structure
1. **Defining the Business Problem**: Clearly outline the problem to solve.
2. **Loading the Dataset**: Load data used for training and testing.
3. **Exploratory Data Analysis (EDA)**: Analyze data to understand its structure and key characteristics.
4. **Automating Train-Test Split**: Automate the division of data into training and testing sets.
5. **Data Preprocessing**: Prepare the data by handling missing values, encoding categorical variables, and scaling features.
6. **Model Creation, Training, and Evaluation**: Create, train, and evaluate the model.
7. **Cross-Validation and Hyperparameter Optimization**: Apply cross-validation and optimize hyperparameters to improve model performance.
8. **Deployment**: Prepare the model for real-world use by deploying it.

## Data Preprocessing
Steps for data preprocessing include:
- Loading and renaming dataset columns
- Handling categorical values using One-Hot Encoding
- Normalizing numerical features using StandardScaler

## Modeling
We use the Random Forest algorithm to create and train the model. The steps include:
- Creating the model
- Training the model with training data
- Evaluating the model with test data

## Cross-Validation and Hyperparameter Optimization
To ensure robust model performance, we apply:
- Cross-validation to evaluate model stability
- Grid search for hyperparameter optimization

## Final Model Evaluation
Evaluate the final model using the best hyperparameters. Save the model and scaler for deployment.

## Deployment
Deploy the model using Streamlit. The deployment script (`deploy.py`) initializes a web application for making predictions based on user input.

Run the deployment script:
```bash
streamlit run deploy.py
```

## Author
Author: @panData
