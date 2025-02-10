import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df['species'] = le.fit_transform(df['species'])
        logger.debug('Data preprocessing completed')
        return df
    except Exception as e:
        logger.error('Problem in LabelEncoder: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters (if needed)
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        
        test_size = 0.2
        data_path = sns.load_dataset('iris')
        
        # Load the data
        df = load_data(data_path)
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split into train and test sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test data
        save_data(train_data, test_data, data_path='./data')
    
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()