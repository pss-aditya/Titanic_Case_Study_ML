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
    df = df.drop(columns =[
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
#  Description   : It Perform Spliting Activity
#  Input         : DataFrame
#  Output        : 4 subset for training and testing
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#-------------------------------------------------------------------------
def SplitData(df):
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.20,random_state = 42)
    
    print("Data Set Spliting Completed Successfully")
    return X_train,X_test,Y_train,Y_test
 

#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : It Perform Training Model
#  Input         : Training features and Labels 
#  Output        : Trained Model
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#-------------------------------------------------------------------------
def TrainModel(X_train,Y_train):
    model = LogisticRegression(max_iter = 1000)
    model = model.fit(X_train,Y_train)
    print("Model Trained Successfully")
    
    return model


#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : It Perform Model Testing
#  Input         : model, Testing Data (features,label)
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#-------------------------------------------------------------------------
def EvaluateModel(model,X_test,Y_test):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    
    print("Accuracy is      :",accuracy)
    print("Confusion Matrix :", confusion_matrix(Y_test,Y_pred))
 

#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : It Perform Model Preservation into .pkl file
#  Input         : model
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#-------------------------------------------------------------------------
def PreserveModel(model,filename):
    joblib.dump(model,filename)
    
    print("Model preserved with name :", filename)
 
   
#------------------------------------------------------------------------
#  Function Name : LoadData
#  Description   : Entry point Function
#  Input         : None
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#--------------------------------------------------------------------------
def main():
    # Step 1
    df = LoadData("TitanicDataset.csv")
    
    # Step 2
    df = Preprocessing(df)
    print("Columns after preprocessing :")
    print(df.columns)
    
    # Step 3
    X_train,X_test,Y_train,Y_test = SplitData(df)
    
    # Step 4
    model = TrainModel(X_train,Y_train)
    
    # Step 5
    EvaluateModel(model,X_test,Y_test)
    
    # Step 6
    PreserveModel(model,"TitanicCaseStudy.pkl")

if __name__ == "__main__":
    main()