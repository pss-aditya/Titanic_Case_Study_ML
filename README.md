# Titanic Survival Prediction using Logistic Regression

A Machine Learning case study that predicts whether a Titanic passenger survived using **Logistic Regression**.

## Overview

This project demonstrates an end-to-end Machine Learning workflow, including data preprocessing, model training, evaluation, model preservation, and prediction using a saved model.

The project was developed **incrementally**, with each Python file representing a step in the learning and implementation process.

> **Important:** `TitanicLogistic5_PreserveModel.py` is the **final and complete implementation** of the project.
> The other Python files were created incrementally to understand and implement each stage of the Machine Learning workflow step by step.
>
> If you want to review or understand the **complete project implementation**, refer to **`TitanicLogistic5_PreserveModel.py`**.

## Machine Learning Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Preprocessing
   ↓
Missing Value Handling
   ↓
Categorical Feature Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Model Preservation
   ↓
Load Saved Model
   ↓
User Input
   ↓
Prediction
```

## Features

The model uses:

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
├── TitanicLogistic5_PreserveModel.py   ← Final Complete Implementation
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

### Train and Preserve the Model

```bash
python TitanicLogistic5_PreserveModel.py
```

### Load the Saved Model and Make Predictions

```bash
python ModelLoadTitanic.py
```

## Note

The project files `TitanicLogistic1.py` through `TitanicLogistic4_TrainModel.py` represent the **incremental development and learning process**.

For the complete implementation, refer to:

**`TitanicLogistic5_PreserveModel.py`**

## Author

**Aditya Valekar**
