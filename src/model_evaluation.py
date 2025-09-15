import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging


# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_model(file_path:str):
    """Load a pickled model from the specified file path."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise



def load_data(file_path:str)->pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    try:
        data = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: The file at {file_path} is empty")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the data: {e}")
        raise



def evaluate_model(clf, X_test:np.ndarray, y_test:np.ndarray)->dict:
    """
    Evaluate the model using various metrics.
    Args:
        clf: The trained classifier.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug(f"Model evaluation metrics calculated")
        return metrics
    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise


def save_metrics(metrics:dict, file_path:str)->None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving metrics: {e}")
        raise



def main():
    try:
        clf = load_model('models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
