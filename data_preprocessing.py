import os
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import yaml

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Logging configuration
logging.basicConfig(
    filename=config['logging']['filename'].replace('data_acquisition', 'data_preprocessing'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directories
BASE_DATA_DIR = config['data']['base_output_dir']
STOCK_DATA_DIR = os.path.join(BASE_DATA_DIR, config['data']['stock_output_dir'])
ETF_DATA_DIR = os.path.join(BASE_DATA_DIR, config['data']['etf_output_dir'])
PREPROCESSED_DATA_DIR = 'preprocessed_data'

# Output filenames
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PREPROCESSED_ETFS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_etfs.csv')

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Datetime'])
        logging.info(f"Loaded data from {file_path} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def preprocess_asset_data(asset_type: str) -> pd.DataFrame:
    if asset_type == 'stocks':
        data_dir = STOCK_DATA_DIR
    elif asset_type == 'etfs':
        data_dir = ETF_DATA_DIR
    else:
        logging.error(f"Invalid asset type: {asset_type}. Choose 'stocks' or 'etfs'.")
        return pd.DataFrame()

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f".{config['data']['format']}")]
    logging.info(f"Found {len(all_files)} {asset_type} files to process.")

    asset_dict = {}

    for file in tqdm(all_files, desc=f"Loading {asset_type.capitalize()} Data"):
        df = load_data(file)
        if df.empty:
            continue
        ticker = df['Ticker'].iloc[0]
        df = df[['Datetime', 'Close']].rename(columns={'Close': ticker})
        df.set_index('Datetime', inplace=True)
        asset_dict[ticker] = df

    if not asset_dict:
        logging.error(f"No data loaded for {asset_type}.")
        return pd.DataFrame()

    # Combine all assets using outer join
    combined_df = pd.concat(asset_dict.values(), axis=1, join='outer')
    
    initial_shape = combined_df.shape
    logging.info(f"Initial combined {asset_type} data shape: {initial_shape}")

    # Calculate the percentage of missing data for each asset
    missing_percentage = combined_df.isnull().mean() * 100
    
    # Remove assets with more than 20% missing data
    assets_to_keep = missing_percentage[missing_percentage <= 20].index
    combined_df = combined_df[assets_to_keep]

    # Apply limited ffill and bfill
    combined_df = combined_df.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5)

    # Remove any remaining rows with NaN values
    combined_df.dropna(inplace=True)

    final_shape = combined_df.shape
    logging.info(f"Final combined {asset_type} data shape: {final_shape}")
    logging.info(f"Removed {initial_shape[1] - final_shape[1]} assets due to excessive missing data.")
    logging.info(f"Retained {final_shape[0]} valid rows out of {initial_shape[0]}.")

    return combined_df

def align_and_process_data(stocks_df: pd.DataFrame, etfs_df: pd.DataFrame) -> tuple:
    # Align data
    common_index = stocks_df.index.intersection(etfs_df.index)
    stocks_df = stocks_df.loc[common_index]
    etfs_df = etfs_df.loc[common_index]

    logging.info(f"Aligned data shape - Stocks: {stocks_df.shape}, ETFs: {etfs_df.shape}")

    # Calculate logarithmic returns
    stocks_returns = np.log(stocks_df / stocks_df.shift(1)).dropna()
    etfs_returns = np.log(etfs_df / etfs_df.shift(1)).dropna()

    # Ensure stocks and ETFs have the same index after calculating returns
    common_return_index = stocks_returns.index.intersection(etfs_returns.index)
    stocks_returns = stocks_returns.loc[common_return_index]
    etfs_returns = etfs_returns.loc[common_return_index]

    # Standardize returns (z-score normalization)
    stocks_returns = (stocks_returns - stocks_returns.mean()) / stocks_returns.std()
    etfs_returns = (etfs_returns - etfs_returns.mean()) / etfs_returns.std()

    logging.info(f"Calculated and standardized returns - Stocks: {stocks_returns.shape}, ETFs: {etfs_returns.shape}")

    return stocks_returns, etfs_returns

def main():
    logging.info("Starting data preprocessing.")

    stocks_df = preprocess_asset_data('stocks')
    etfs_df = preprocess_asset_data('etfs')

    processed_stocks, processed_etfs = align_and_process_data(stocks_df, etfs_df)

    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    logging.info(f"Ensuring directory {PREPROCESSED_DATA_DIR} exists for preprocessed data.")

    processed_stocks.to_csv(PREPROCESSED_STOCKS_FILE)
    logging.info(f"Saved preprocessed stocks data to {PREPROCESSED_STOCKS_FILE}.")

    processed_etfs.to_csv(PREPROCESSED_ETFS_FILE)
    logging.info(f"Saved preprocessed ETFs data to {PREPROCESSED_ETFS_FILE}.")

    logging.info("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()