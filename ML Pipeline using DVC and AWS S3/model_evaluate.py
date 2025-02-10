import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
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

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file or use default parameters if the file is missing or invalid."""
    default_params = {
        'random_state': 42
    }
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        
        # If the file is empty or invalid, params will be None
        if params is None:
            logger.warning('YAML file is empty or invalid. Using default parameters.')
            return default_params
        
        logger.debug('Parameters retrieved from %s', params_path)
        return params.get('model_evaluation', default_params)  # Use default if 'model_evaluation' key is missing
    except FileNotFoundError:
        logger.warning('File not found: %s. Using default parameters.', params_path)
        return default_params
    except yaml.YAMLError as e:
        logger.error('YAML error: %s. Using default parameters.', e)
        return default_params
    except Exception as e:
        logger.error('Unexpected error: %s. Using default parameters.', e)
        return default_params

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(data_path: str or pd.DataFrame) -> pd.DataFrame:
    """Load data from a CSV file or use an existing DataFrame."""
    if isinstance(data_path, str):  # If data_path is a file path
        try:
            df = pd.read_csv(data_path)
            logger.debug('Data loaded from %s', data_path)
            return df
        except pd.errors.ParserError as e:
            logger.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logger.error('Unexpected error occurred while loading the data: %s', e)
            raise
    elif isinstance(data_path, pd.DataFrame):  # If data_path is already a DataFrame
        logger.debug('Using provided DataFrame')
        return data_path
    else:
        raise ValueError("data_path must be a file path (str) or a DataFrame")

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)

        # Check if the model supports predict_proba
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test)
            if y_pred_proba.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:  # Multiclass classification
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        else:
            logger.warning("Model does not support probability prediction. Skipping AUC calculation.")
            auc = None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def main():
    try:
        # Load parameters (use default if params.yaml is missing or invalid)
        params = load_params(params_path='params.yaml')
        
        # Load the trained model
        clf = load_model('./models/model.pkl')
        
        # Load test data
        test_data = load_data(r'C:\Users\alisa\OneDrive\Desktop\MLOPs Note\data\raw\test.csv')
        
        # Prepare test data
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)

        # Experiment tracking using dvclive
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('recall', metrics['recall'])
            live.log_params(params)
        
        # Save metrics to a JSON file
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()