import pandas as pd
import json
import os,sys
import yaml
from sklearn.model_selection import train_test_split
from src.logging import logger
from src.exception_handling import custom_exception

def load_data(files)-> pd.DataFrame:
    df = []
    try:
        for i in files:
            with open(i, 'r',encoding="utf") as f:
                logger.info("Data ingestion started")
                data = json.load(f)
                df.extend(data)
                logger.info("Data ingestion started")
        return df
    except Exception as e:
        raise custom_exception(e,sys)

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    try:
        logger.info("data preprocessing started")
        review_details = [i['review_summary'] for i in df]
        review_rating = [i['rating'] for i in df]
        movie = [i['movie'] for i in df]
        review_date = [i['review_date'] for i in df]


        df1 = pd.DataFrame({
            'review_detail': review_details,
            'rating': review_rating,
            'movie': movie,
            'review_date': review_date
        })
        logger.info("data preprocessing completed")
        return df1
    except Exception as e:
        raise custom_exception(e,sys)

def load_params(params_path:str) -> float:
    try:
        logger.info("reading params file")
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['load_data']['test_size']
        logger.info("reading params complete")
        return test_size
    except Exception as e:
        raise custom_exception(e,sys)

def save_data(df:pd.DataFrame, test_size:float, processed_data_path:str) -> None:
    try:
        logger.info("train test started")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        train_data.to_csv(processed_data_path, index=False)
        test_data.to_csv(processed_data_path, index=False)
        logger.info("train test completed")
    except Exception as e:
        raise custom_exception(e,sys)

def main():
    try:
        ## get raw files path
        raw_files_path = os.getcwd() + os.path.join('\\data\\raw')
        json_files = [f for f in os.listdir(raw_files_path) if f.endswith('.json')]

        ## get directory of interim
        processed_data_path = os.getcwd() + os.path.join('\\data\\interim')

        ## data ingestion
        df = load_data(json_files)
        df1 = preprocess_data(df)
        #test_size = load_params(params_path='params1.yaml')
        save_data(df1, 0.2, processed_data_path)
    
    except Exception as e:
        raise custom_exception(e,sys)