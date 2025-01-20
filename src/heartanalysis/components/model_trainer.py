import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from src.heartanalysis.exception import CustomException
from src.heartanalysis.logger import logging
from src.heartanalysis.utils import save_object, evaluate_models
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score

import pickle

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='binary')
        return accuracy, precision
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "SVC": SVC()
            }

            params = {
                "Random Forest": {
                    "n_estimators": 100,  
                    "max_depth": 10,  
                },
                "Decision Tree": {
                    "max_depth": 10,  
                    "min_samples_split": 2,  
                },
                "Logistic Regression": {
                    "C": 1.0,  
                },
                "XGBoost": {
                    "learning_rate": 0.1,  
                    "n_estimators": 100,  
                },
                "SVC": {
                    "C": 1.0,  
                    "kernel": 'linear',  
                }
            }

            model_report, good_model_found = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Find best model
            if not good_model_found:
                print("No good model found. The models' performance is below the defined threshold.")
            else:
                best_model_name = max(model_report, key=lambda model: model_report[model]['accuracy'])  # Based on accuracy
                best_model = models[best_model_name]
                print(f"This is the best model: {best_model_name}")
                print(f"Best model performance: {model_report[best_model_name]}")
                logging.info(f"Best found model on both training and testing dataset")

                self.log_models_on_dagshub(model_report, models, params, X_test, y_test)  # Call the method with self

        except Exception as e:
            raise CustomException(e, sys)

    def log_models_on_dagshub(self, model_report, models, params, X_test, y_test):  # Add 'self' as the first argument
        # Set DagsHub URI
        mlflow.set_registry_uri("https://dagshub.com/Purugru/Heart_Disease_Monitor.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Iterate over each model and log its performance and parameters
        for model_name, metrics in model_report.items():
            best_model = models[model_name]
            best_params = params[model_name]

            with mlflow.start_run():
                # Evaluate and log the metrics for each model
                accuracy, precision = self.eval_metrics(y_test, best_model.predict(X_test))  # Calculate accuracy and precision

                # Log the metrics and parameters to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)

                # Log confusion matrix (optional, can be logged as a separate artifact if needed)
                confusion_mat = confusion_matrix(y_test, best_model.predict(X_test))
                mlflow.log_artifact(confusion_mat, "confusion_matrix.json")

                # Register the model with MLflow Model Registry if not using file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
