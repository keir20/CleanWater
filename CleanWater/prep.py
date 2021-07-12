# Data Preperation
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# from data import 
from imblearn.under_sampling import InstanceHardnessThreshold

class DataPrep(data):
    
    def __init__(self, data):
        self.data = data
        
    def split(self, data):
        X = data.drop(columns=['Potability'])
        y = data['Potability']    
        return X, y
    
    def data_transform(self, data):    
        new_data_frame = pd.DataFrame()
        for column in data:
            new_data_frame[column] = data[column].fillna(data.groupby(['Potability'])[column].transform('mean'))
        return new_data_frame
    
    def sampling(self, X, y):
        X_instance, y_instance = InstanceHardnessThreshold().fit_resample(X,y)
        return X_instance, y_instance
    
    def tt_split(self, X, y, test_size=0.3, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
        return X_train, X_test, y_train, y_test
    
    def scale(self, data):
        scale = StandardScaler().fit_transform(data)
        return scale