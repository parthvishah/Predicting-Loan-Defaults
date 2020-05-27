


#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import os
import numpy as np
from sklearn.utils import resample

class PreProcess: 
    #Get the dataset
    data = pd.DataFrame()
    def __init__(self):
        print("Reading the Dataset. Make sure it is in the same folder as the current directory.\n")
        
        #get current working directory
        self.cwd = pathlib.Path().absolute()
        
        #csv into DataFrame
        self.data = pd.read_csv(str(self.cwd)+ '\\' + 'MacroMicroData.csv')
        print("Data is converted into a dataframe\n")
        
        print('----------------------------------------------------------------------------------')
    
    #Create Training and Test Sets
    def splitStandardize(self):
        datacopy = self.data.copy()
        y = datacopy['loan_status']
        del datacopy['loan_status']
        X = datacopy
        X_train, X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2)
        
        #Standardise the numerical features
        print("Standardising the Numerical Features")
        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse = False, drop = 'first')
        x = ['loan_amnt','int_rate','installment','annual_inc','dti','delinq_2yrs','inq_last_6mths'
             ,'revol_util','total_acc',
             'oil', 'dollar','forex_reserve', 'fiscal_deficit','industrial_production',
             'inflation','GDP','monsoon','exports', 'gdp_growth','government_bond','trade_deficit',
             'compensation','unemployment','imports','gross_debt','government_exp','household_exp',
             'emp_length','term']
        x1 = ['sub_grade','home_ownership','verification_status']
        X_train_subset1 = pd.DataFrame(scaler.fit_transform(X_train[x]),columns = x)
        X_train_subset2 = pd.DataFrame(encoder.fit_transform(X_train[x1]), columns = encoder.get_feature_names(input_features = x1))
        X_train = pd.concat([X_train_subset1,X_train_subset2], axis = 1)
        X_test_subset1 = pd.DataFrame(scaler.transform(X_test[x]), columns = x)
        X_test_subset2 = pd.DataFrame(encoder.transform(X_test[x1]), columns = encoder.get_feature_names(input_features = x1))
        X_test = pd.concat([X_test_subset1,X_test_subset2], axis = 1)
        
        
        y_train = y_train.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        
        imputer_train = KNNImputer(n_neighbors=10, weights="uniform")
        X_train = pd.DataFrame(imputer_train.fit_transform(X_train),columns = X_train.columns)
        
        imputer_test = KNNImputer(n_neighbors=10, weights="uniform")
        X_test = pd.DataFrame(imputer_test.fit_transform(X_test),columns = X_test.columns)
        
        try:
            os.mkdir(str(self.cwd)+'\\TrainTestData')
        except:
            print("Making train test data files in the folder TrainTestData")
            
        X_train.to_csv(str(self.cwd)+'\\TrainTestData\\X_train.csv',index = False)
        X_test.to_csv(str(self.cwd)+'\\TrainTestData\\X_test.csv',index = False)
        y_train.to_csv(str(self.cwd)+'\\TrainTestData\\y_train.csv',index = False)
        y_test.to_csv(str(self.cwd)+'\\TrainTestData\\y_test.csv',index = False)
        print("Standardisation Done.\n")
        
        
        print('----------------------------------------------------------------------------------')



    def upSample(self):
        X_train = pd.read_csv(str(self.cwd)+'\\TrainTestData\\X_train.csv')
        y_train = pd.read_csv(str(self.cwd)+'\\TrainTestData\\y_train.csv')
        X = pd.concat([X_train, y_train], axis=1)

        # separate minority and majority classes
        not_default = X[X.loan_status==0]
        default = X[X.loan_status==1]

        # upsample minority
        default_upsampled = resample(default,
                          replace=True, # sample with replacement
                          n_samples=len(not_default)) # reproducible results

        # combine majority and upsampled minority
        upsampled = pd.concat([not_default, default_upsampled])

        # check new class counts
        #upsampled.Class.value_counts()
        X_train = upsampled.iloc[:,:-1]
        y_train = upsampled.iloc[:,-1]
        X_train.to_csv(str(self.cwd)+'\\TrainTestData\\X_train.csv',index = False)
        #X_test.to_csv(str(self.cwd)+'\\TrainTestData\\X_test.csv',index = False)
        y_train.to_csv(str(self.cwd)+'\\TrainTestData\\y_train.csv',index = False)






