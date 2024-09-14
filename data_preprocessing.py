import os
import pandas as pd
import logging
from tqdm import tqdm
import numpy as np

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Logging configuration
logging.basicConfig(
    filename='logs/data_preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Directories
STOCK_DATA_DIR = 'stock_data_minute'
ETF_DATA_DIR = 'etf_data_minute'
PREPROCESSED_DATA_DIR = 'preprocessed_data'

# Output filenames
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PREPROCESSED_ETFS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_etfs.csv')

# Data retrieval parameters
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

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
    """
    Preprocesses data for a given asset type (stocks or ETFs).

    Parameters:
        asset_type (str): 'stocks' or 'etfs'.

    Returns:
        pd.DataFrame: Preprocessed returns DataFrame.
    """
    if asset_type == 'stocks':
        data_dir = STOCK_DATA_DIR
        output_file = PREPROCESSED_STOCKS_FILE
    elif asset_type == 'etfs':
        data_dir = ETF_DATA_DIR
        output_file = PREPROCESSED_ETFS_FILE
    else:
        logging.error(f"Invalid asset type: {asset_type}. Choose 'stocks' or 'etfs'.")
        return pd.DataFrame()

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    logging.info(f"Found {len(all_files)} {asset_type} files to process.")

    # Dictionary to hold individual asset data
    asset_dict = {}

    # Sequentially load data
    for file in tqdm(all_files, desc=f"Loading {asset_type.capitalize()} Data"):
        df = load_data(file)
        if df.empty:
            continue
        ticker = df['Ticker'].iloc[0]
        # Use 'Adj Close' if available; otherwise, use 'Close'
        if 'Adj Close' in df.columns:
            df = df[['Datetime', 'Adj Close']].rename(columns={'Adj Close': ticker})
        else:
            df = df[['Datetime', 'Close']].rename(columns={'Close': ticker})
        asset_dict[ticker] = df.set_index('Datetime')

    if not asset_dict:
        logging.error(f"No data loaded for {asset_type}.")
        return pd.DataFrame()

    # Combine all assets into a single DataFrame
    combined_df = pd.concat(asset_dict.values(), axis=1, join='outer')
    logging.info(f"Combined {asset_type} data shape: {combined_df.shape}.")

    # Handle missing data: forward fill, then backward fill
    combined_df = combined_df.ffill().bfill()
    logging.info(f"Handled missing data for {asset_type}.")

    # Drop any remaining rows with NaN values
    combined_df.dropna(inplace=True)
    logging.info(f"After dropping NaNs, {asset_type} data shape: {combined_df.shape}.")

    # Calculate logarithmic returns
    # To avoid fragmentation, compute all returns first
    log_returns = np.log(combined_df / combined_df.shift(1))
    log_returns.dropna(inplace=True)
    logging.info(f"Calculated logarithmic returns for {asset_type}.")

    # Standardize returns (z-score normalization)
    returns_df = (log_returns - log_returns.mean()) / log_returns.std()
    logging.info(f"Standardized returns for {asset_type}.")

    # Save preprocessed returns
    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)
        logging.info(f"Created directory {PREPROCESSED_DATA_DIR} for preprocessed data.")

    try:
        returns_df.to_csv(output_file, index=True)
        logging.info(f"Saved preprocessed {asset_type} data to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving preprocessed {asset_type} data: {e}")

    return returns_df

def main():
    """
    Main function to orchestrate data preprocessing.
    """
    logging.info("Starting data preprocessing.")

    # Preprocess Stocks
    logging.info("Preprocessing stock data.")
    preprocessed_stocks = preprocess_asset_data('stocks')
    if preprocessed_stocks.empty:
        logging.error("Preprocessing stocks failed or resulted in empty DataFrame.")
    else:
        logging.info(f"Preprocessed stocks data shape: {preprocessed_stocks.shape}.")

    # Preprocess ETFs
    logging.info("Preprocessing ETF data.")
    preprocessed_etfs = preprocess_asset_data('etfs')
    if preprocessed_etfs.empty:
        logging.error("Preprocessing ETFs failed or resulted in empty DataFrame.")
    else:
        logging.info(f"Preprocessed ETFs data shape: {preprocessed_etfs.shape}.")

    logging.info("Data preprocessing completed successfully.")

# ------------------------------ #
#           Entry Point           #
# ------------------------------ #

if __name__ == "__main__":
    main()