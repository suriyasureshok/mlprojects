import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initaite_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test  = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Catboost Regressor": CatBoostRegressor(verbose=False),
                "K Neighbours regressor": KNeighborsRegressor()
            }
            params={
                "Linear Regression":{},
                "Ridge":{
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],        
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'saga'], 
                    'fit_intercept': [True, False],                                
                    'max_iter': [1000, 5000, 10000],               
                    'tol': [1e-3, 1e-4, 1e-5]  
                },      
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Catboost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "K Neighbours regressor": {
                    'n_neighbors': [3, 5, 7]
                }
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model was found",sys)
            logging.info("Best found model on both training and test dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2
        
        except Exception as e:
            raise CustomException(e,sys)