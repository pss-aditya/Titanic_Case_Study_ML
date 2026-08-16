# Titanic Survival Prediction using Logistic Regression

A Machine Learning case study that predicts whether a Titanic passenger survived using **Logistic Regression**.

## Overview

This project demonstrates an end-to-end Machine Learning workflow:

* Data loading and preprocessing
* Missing value handling
* Categorical feature encoding
* Feature selection
* Train-test split
* Logistic Regression model training
* Model evaluation
* Model preservation using Joblib
* Prediction using a saved model

## Features

The model uses the following features:

`Age`, `Fare`, `Sex`, `sibsp`, `Parch`, `Pclass`, `Embarked_1.0`, `Embarked_2.0`

## Model Performance

**Accuracy:** 76.72%

**Confusion Matrix:**

```text
[[174  15]
 [46  27]]
```

## Project Structure

```text
Titanic_case_study/
│
├── TitanicDataset.csv
├── TitanicLogistic1.py
├── TitanicLogistic2_DataPreprocessing.py
├── TitanicLogistic3_SplitDataset.py
├── TitanicLogistic4_TrainModel.py
├── TitanicLogistic5_PreserveModel.py
├── ModelLoadTitanic.py
├── TitanicCaseStudy.pkl
└── README.md
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib

## Installation

```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

Train and preserve the model:

```bash
python TitanicLogistic5_PreserveModel.py
```

Load the saved model and make predictions:

```bash
python ModelLoadTitanic.py
```

## Author

**Aditya Govind Valekar**
