import os
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import logging



# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(file_path:str)->pd.DataFrame:
    """
    Load data from a CSV file.
    Args:
        file_path (str): Path to the CSV file
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame
    
    """
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise



def train_model(X_train: np.ndarray, y_train: np.ndarray, params:dict)->RandomForestClassifier:
    """
    Train a RandomForestClassifier model.
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        params (dict): Hyperparameters for the RandomForestClassifier
    Returns:
        RandomForestClassifier: Trained model
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must be equal.")
        
        logger.debug("Initializing RandomForestClassifier with parameters: %s",params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug("Starting model training...")
        clf.fit(X_train, y_train)
        logger.debug("Model training completed.")

        return clf
    except ValueError as e:
        logger.error(f"ValueError during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        raise


def save_model(model, file_path:str) -> None:
    """
    Save the trained model to a file using pickle.
    Args:
        model: Trained model
        file_path (str): Path to save the model file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully at {file_path}")

    except FileNotFoundError as e:
        logger.error(f"File not found error while saving model: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the model: {e}")
        raise


def main():
    try:
        # Example parameters and data loading
        params = {
            'n_estimators': 22,
            'random_state': 2
        }
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values  
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()