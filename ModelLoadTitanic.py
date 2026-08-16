import pandas as pd
import numpy as np
import joblib

#------------------------------------------------------------------------
#  Function Name : LoadModel
#  Description   : Load the preserved model from .pkl file
#  Input         : Name of model file
#  Output        : Loaded Model
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#------------------------------------------------------------------------
def LoadModel(FileName):
    model = joblib.load(FileName)
    print("Model Loaded SuccessFully")
    print(model.feature_names_in_)
    return model


#------------------------------------------------------------------------
#  Function Name : PredictPassenger
#  Description   : Predict whether passenger survived or not
#  Input         : Trained Model
#  Output        : Prediction
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#------------------------------------------------------------------------
def PredictPassenger(model):
    print("\nEnter the information")
    Pclass = int(input("\nEnter PClass (1 = First, 2 = Second, 3 = Third) : "))
    Sex = int(input("\nEnter Sex (0 = Male, 1 = Female)          : "))
    Age = float(input("\nEnter Age                              : "))
    sibsp = int(input("\nEnter SibSp (siblings/spouse aboard)   : "))
    Parch = int(input("\nEnter Parch (parents/children aboard)  : "))
    Fare = float(input("\nEnter Fare                              : "))
    Embarked = int(input("\nEnter Embarked (0 = S, 1 = C, 2 = Q) : "))
    
    passenger = pd.DataFrame([{
        "Pclass": Pclass,
        "Sex": Sex,
        "Age": Age,
        "sibsp":sibsp,
        "Parch":Parch,
        "Fare":Fare,
        "Embarked_1.0":1 if Embarked == 1 else 0,
        "Embarked_2.0":1 if Embarked == 2 else 0       
    }])
    
    # Arrange features in the same order used during training
    passenger = passenger[model.feature_names_in_]
    
    # Make Prediction
    result = model.predict(passenger)
    
    print("\n--------------------------------------------------")
    print("                   Prediction Result                ")
    print("--------------------------------------------------")

    if result[0] == 1:
        print("Passenger Status : Survived")
    else:
        print("Passenger Status : Not Survived")
        
    print("----------------------------------------------------")
        
#------------------------------------------------------------------------
#  Function Name : main
#  Description   : Entry point Function
#  Input         : None
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#------------------------------------------------------------------------   
def main():
    model = LoadModel("TitanicCaseStudy.pkl")
    PredictPassenger(model)

if __name__ == "__main__":
    main()

