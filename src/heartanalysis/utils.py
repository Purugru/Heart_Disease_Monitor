import os
import sys
from src.heartanalysis.exception import CustomException
from src.heartanalysis.logger import logging
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pandas as pd
import pickle
import numpy as np



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, r2_score

def evaluate_models(X_train, y_train, X_test, y_test, models, param, threshold=0.5):
    """ 
    Evaluate models with simplified metrics for quick testing and debugging.
    
    Args:
    X_train: Training features
    y_train: Training target
    X_test: Testing features
    y_test: Testing target
    models: Dictionary of models to evaluate
    param: Hyperparameter dictionary (not used in this version for simplicity)
    threshold: Minimum performance threshold for a good model
    
    Returns:
    report: Dictionary of model performance metrics
    good_model_found: Boolean indicating if any model met the threshold
    """
    try:
        report = {}
        good_model_found = False  # Flag to check if any model meets the threshold

        for model_name, model in models.items():
            print(f"Training model: {model_name}")
            # Set parameters for each model from the params dictionary
            model.set_params(**param[model_name])
            # Fit the model without grid search for quick testing
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate basic metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
            confusion = confusion_matrix(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)

            # Check if the model meets the threshold
            if r2 >= threshold or accuracy >= threshold:
                good_model_found = True
            
            # Storing metrics in the report dictionary
            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'confusion_matrix': confusion,
                'r2_score': r2
            }

        return report, good_model_found

    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return {}, False
