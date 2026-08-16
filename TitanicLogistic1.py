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
#  Description   : Entry point Function
#  Input         : None
#  Output        : None
#  Author        : Aditya Govind Valekar
#  Date          : 16/08/2026
#--------------------------------------------------------------------------
def main():
    LoadData("TitanicDataset.csv")

if __name__ == "__main__":
    main()