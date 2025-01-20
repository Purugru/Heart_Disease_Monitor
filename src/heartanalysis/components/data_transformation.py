import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.heartanalysis.utils import save_object
from src.heartanalysis.exception import CustomException
from src.heartanalysis.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        '''
        this function will perfrom data transformation
        '''
        try:
            numerical_columns = ["age","trestbps","chol","thalach","oldpeak"]
            categorical_columns = ["sex","cp","fbs","restecg","exang","slope","thal","ca"]
            num_pipeline=Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ('scalar',StandardScaler())
            ])

            cat_pipeline_onehot=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ])

            cat_pipeline_label=Pipeline(steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("label_encoder",OrdinalEncoder()),
            ])

            logging.info(f"Categorical Columns:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor_label = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline_label",cat_pipeline_label, categorical_columns)
                ]
            )
            preprocessor_onehot = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline_onehot",cat_pipeline_onehot,categorical_columns)
                ]
            )

            return preprocessor_onehot
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates data transformation for train and test datasets.
        """
        try:
            # Read train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test files")

            # Getting preprocessing object
            preprocessor = self.get_data_transformer_object()

            target_column_name = "target"
            numerical_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]

            # Separating input features and target variable
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing to train and test datasets")
            input_features_train = preprocessor.fit_transform(input_features_train_df)
            input_features_test = preprocessor.transform(input_features_test_df)

            # Combining features with target
            train_arr = np.c_[input_features_train, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test, np.array(target_feature_test_df)]

            logging.info("Data transformation completed")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)