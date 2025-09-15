import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import yaml


# create a log directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# create console handler 
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): URL or path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url}")
        return data
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
    except Exception as e:
        logger.error('An error occurred while loading the data: %s', e)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by 

    Args:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return data
    except KeyError as e:
        logger.error('Column not found during preprocessing: %s', e)
        raise
    except Exception as e:
        logger.error('An error occurred during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)-> None:
    """
    Save the training and testing data to CSV files.

    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing data.
        data_path (str): Directory path to save the CSV files.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.debug(f"Data saved successfully at {raw_data_path}")
    except Exception as e:
        logger.error('An error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
        data = load_data(data_url=data_path)
        final_data = preprocess_data(data)
        train_data, test_data = train_test_split(final_data, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
        logger.info("Data ingestion process completed successfully")
    except Exception as e:
        logger.error('Data ingestion process failed: %s', e)
        print(f"error: {e}")

if __name__ == "__main__":
    main()