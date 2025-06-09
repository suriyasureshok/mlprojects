import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts',"preprocessin.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['reading_score','writing_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=True))
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns : {categorical_columns}")
            logging.info(f"Numerical Columns : {numerical_columns}")

            Preprocessor = ColumnTransformer(
                [
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )

            return Preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully")

            logging.info("Obtaining Preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = 'math_score'
            numerical_columns = ['reading_score','writing_score']

            input_feature_train_df = train_df.drop(target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing object")

            save_obj(
                file_path = self.data_transformation_config.preprocessing_obj_file_path,
                obj =  preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)