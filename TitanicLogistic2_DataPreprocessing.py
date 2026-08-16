import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : Load the Data from CSV
#  Input         : Name of csv file
#  Output        : Data Frame
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#--------------------------------------------------------------------------
def LoadData(fileName):
    df =pd.read_csv(fileName)
    print("Dataset Loaded Successfully")
    print(df.head())
    return df

#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : It Perform Data Analysis
#  Input         : DataFrame
#  Output        : Updated DataFrame
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#--------------------------------------------------------------------------
def Preprocessing(df):
    df = df.drop([
        "Passengerid",
        "zero",
        "name"
    ],
    errors = "ignore"
    )
    
    # Handle missing value
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    # Convert categorical to numeric Data
    df = pd.get_dummies(
        df,
        columns=["Embarked"],
        drop_first= True,
        dtype = int        
    )
    
    print(df.head())
    print("Data Preprocessing Completed")
    return df
 
 
#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : Entry point Function
#  Input         : None
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#--------------------------------------------------------------------------
def main():
    df = LoadData("TitanicDataset.csv")
    
    df = Preprocessing(df)

if __name__ == "__main__":
    main()