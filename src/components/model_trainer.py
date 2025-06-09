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
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Ada Boost Regressor": AdaBoostRegressor(),
                "XGB Regressor": XGBRegressor(),
                "Catboost Regressor": CatBoostRegressor(verbose=False),
                "K Neighbours regressor": KNeighborsRegressor()
            }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

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