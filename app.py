from src.heartanalysis.components.data_ingestion import DataIngestion, DataingestionConfig
from src.heartanalysis.components.data_transformation import DataTransformation, DataTransformationConfig
from src.heartanalysis.exception import CustomException
from src.heartanalysis.logger import logging
from src.heartanalysis.components.model_trainer import ModelTrainer, ModelTrainerConfig
import sys

if __name__=="__main__":
    logging.info("Transformation Test start")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys) 

