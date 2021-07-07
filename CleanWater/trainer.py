# data analysis and wrangling
import numpy as np 
import pandas as pd

# data source
from CleanWater.data import load_data
from CleanWater.params import MLFLOW_URI, EXPERIMENT_NAME

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# mlflow
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from termcolor import colored


MLFLOW_URI='https://mlflow.lewagon.co/'


class Trainer(object):
    ESTIMATOR = "Classification"  # --> changed from 'Linear' to 'Classification'
    EXPERIMENT_NAME = "[UK] [London] [PKR] CleanWater"
    
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.3)
        self.kwargs = kwargs
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME
        
        
    def get_estimator(self):
        estimator = self.kwargs.get('estimator', self.ESTIMATOR)
        if estimator == 'Logistic_Regression':
            model = LogisticRegression()
            self.model_params = {'C': [0.01, 0.1, 1, 10, 100],
                                 'max_iter': [1, 10, 25, 50, 100, 200],
                                 'tol': [0.0001, 0.001, 0.01]}
        
        elif estimator == 'GBC':
            model = GradientBoostingClassifier()
            self.model_params = {'learning_rate': [0.01, 0.1, 1],
                                 'n_estimators': [800, 1000, 1200, 1500],
                                 'max_depth': [1, 5, 10],
                                 'validation_fraction': [0.01, 0.1, 1, 5, 7, 10],
                                 'tol': [1, 5, 10, 15, 20],
                                 'subsample': [0.01, 0.1, 0.17, 0.5, 1, 5, 10]}
            
        else:
            model = LogisticRegression()
        estimator_params = self.kwargs.get('estimator_params', {})
        self.mlflow_log_param('estimator', estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, 'red'))
        return model
    
    
    def set_experiment_name(self, experiment_name):
        """ defines the experiment name for MLFlow """
        self.experiment_name = experiment_name
    
    # feature engineering pipeline blocks
    def set_pipeline(self):
        """ defines the pipeline as a class attribute """
        feateng_steps = self.kwargs.get('feateng', ['ph', 'Hardness', 'Solids', 
                                                    'Chloramines','Sulfate', 
                                                    'Conductivity', 'Organic_carbon', 
                                                    'Trihalomethanes', 'Turbidity'])
        
        pipe_ph_features = Pipeline([
            ('ph', SimpleImputer(strategy='mean')),
            ('ph_scaler', StandardScaler())])
        
        pipe_hardness_features = Pipeline([
            ('hardness', SimpleImputer(strategy='mean')),
            ('hardness_scaler', StandardScaler())])
        
        pipe_solids_features = Pipeline([
            ('solids', SimpleImputer(strategy='mean')),
            ('solids_scaler', StandardScaler())])
        
        pipe_chloramines_features = Pipeline([
            ('chloramines', SimpleImputer(strategy='mean')),
            ('chloramines_scaler', StandardScaler())])
        
        pipe_sulfate_features = Pipeline([
            ('sulfate', SimpleImputer(strategy='mean')),
            ('sulfate_scaler', StandardScaler())])
        
        pipe_conductivity_features = Pipeline([
            ('conductivity', SimpleImputer(strategy='mean')),
            ('conductivity_scaler', StandardScaler())])
        
        pipe_carbon_features = Pipeline([
            ('carbon', SimpleImputer(strategy='mean')),
            ('carbon_scaler', StandardScaler())])
        
        pipe_trihalomethanes_features = Pipeline([
            ('trihalomethanes', SimpleImputer(strategy='mean')),
            ('trihalomethanes_scaler', StandardScaler())])
        
        pipe_turbidity_features = Pipeline([
            ('turbidity', SimpleImputer(strategy='mean')),
            ('turbidity_scaler', StandardScaler())])
    
    
    # define default feature engineering blocks
        feateng_blocks = [
            ('ph', pipe_ph_features, ['ph']),
            ('hardness', pipe_hardness_features, ['Hardness']),
            ('solids', pipe_solids_features, ['Solids']),
            ('chloramines', pipe_chloramines_features, ['Chloramines']),
            ('sulfate', pipe_sulfate_features, ['Sulfate']),
            ('conductivity', pipe_conductivity_features, ['Conductivity']),
            ('carbon', pipe_carbon_features, ['Organic_carbon']),
            ('trihalomethanes', pipe_trihalomethanes_features, ['Trihalomethanes']),
            ('turbidity', pipe_turbidity_features, ['Turbidity'])]
        
        # filter out some blocks according to input parameters
        for block in feateng_blocks:
            if block[0] not in feateng_steps:
                feateng_blocks.remove(block)

        features_encoder = ColumnTransformer(feateng_blocks,
                                             n_jobs=None,
                                             remainder='drop')

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

###--------------------------------------

# Run pipeline
    def run(self):
        self.set_pipeline()
        self.mlflow_log_param('model', 'Classification')
        self.pipeline.fit(self.X, self.y)
        print('pipeline fitted')
        
    # def evaluate(self, X_test, y_test):
    #     """ evaluates the pipeline on X and return the accuracy """
    #     y_pred_train = self.pipeline.predict(self.X)
    #     mse_train = mean_squared_error(self.y, y_pred_train)
    #     rmse_train = np.sqrt(mse_train)
    
    #     self.mlflow_log_metric('rmse_train', rmse_train)
        
    #     y_pred_test = self.pipeline.predict(X_test)
    #     mse_test = mean_squared_error(y_test, y_pred_test)
    #     rmse_test = np.sqrt(mse_test)
    #     self.mlflow_log_metric('rmse_test', rmse_test)
        
    #     return (round(rmse_train, 3) ,round(rmse_test, 3))
        
    def predict(self, X):
        y_pred = self.pipeline.predict(X)
        return y_pred
   
    def save_model(self):
        """ save the model into a .joblib format """
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored('model.joblib saved locally', 'green'))

###--------------------------------------

 # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

###--------------------------------------

if __name__ == "__main__":
    # store the data in a DataFrame
    N = 2000
    df = load_data(N)
        
    # set X and y
    y = df['Potability']
    X = df[['ph', 'Hardness', 'Solids', 'Chloramines','Sulfate', 
            'Conductivity', 'Organic_carbon','Trihalomethanes', 'Turbidity']]
    
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # train model
    estimators = ['Linear_Regression', 'GBC'] 

    for estimator in estimators:
        params = {'estimator': estimator,
                  'feateng': ['ph', 'hardness', 'solids', 
                              'chloramines','sulfate', 
                              'conductivity', 'carbon', 
                              'trihalomethanes', 'turbidity']}
        
        trainer = Trainer(X_train, y_train, **params)
        trainer.set_experiment_name(EXPERIMENT_NAME)
        trainer.run()
    
        # evaluate the pipeline
        accuracy = trainer.evaluate(X_test, y_test)
        print(f"accuracy: {accuracy}")
        
        # save model locally
        trainer.save_model()
