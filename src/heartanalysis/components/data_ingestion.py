import os 
import sys
from src.heartanalysis.exception import CustomException
from src.heartanalysis.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.path.join('notebooks/data','heart.csv'))

from dataclasses import dataclass

@dataclass
class DataingestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataingestionConfig()
    
    def initiate_data_ingestion(self):
        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        logging.info("Data Ingestion is completed")

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )