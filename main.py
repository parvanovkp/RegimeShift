import pandas as pd
import yfinance as yf
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Logging configuration
logging.basicConfig(
    filename='data_acquisition.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Wikipedia URL for S&P 500 companies
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Date range for data retrieval
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=7)
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')

# Data retrieval parameters
INTERVAL = '1m'  # Minute-level data
MAX_THREADS = 8 # Number of cores (8 for my M2Pro)
STOCK_OUTPUT_DIR = 'stock_data_minute'
ETF_OUTPUT_DIR = 'etf_data_minute'
DATA_FORMAT = 'csv'  # Choose between 'csv' and 'parquet' ('csv' for now)

# ETF tickers (assuming the suggested tickers in the outline; could be expanded later)
ETF_TICKERS = [
    'XLV', 'XLK', 'XLP', 'XLF', 'XLB', 'XLY', 'XLE', 'XLI',
    'XLRE', 'XLU', 'SIZE', 'MTUM', 'USMV', 'XLC', 'VLUE',
    'QUAL', 'GLD', 'SLV', 'HYG', 'USO', 'TLT', 'EEM', 'EFA',
    'LQD', 'FXI'
]

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def fetch_sp500_tickers(wiki_url: str) -> list:
    """
    Fetches the list of S&P 500 tickers from the S&P 500 Wikipedia page.

    Parameters:
        wiki_url (str): https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

    Returns:
        list: List of ticker symbols formatted for yfinance.
    """
    try:
        tables = pd.read_html(wiki_url)
        # The first table contains the list of companies
        df = tables[0]
        tickers = df['Symbol'].tolist()
        logging.info(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        
        # Handle tickers with dots (e.g., BRK.B -> BRK-B)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        return []

def download_ticker_data(ticker: str, start_date: str, end_date: str, interval: str = '1m', max_retries: int = 3) -> pd.DataFrame:
    """
    Downloads historical minute-level data for a given ticker.

    Parameters:
        ticker (str): Stock or ETF ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (default '1m' for minute-level data).
        max_retries (int): Maximum number of retries for failed downloads.

    Returns:
        pd.DataFrame: DataFrame containing the historical data.
    """
    for attempt in range(1, max_retries + 1):
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, interval=interval, actions=False, prepost=False)
            if df.empty:
                logging.warning(f"No data fetched for {ticker}.")
                return pd.DataFrame()
            df.reset_index(inplace=True)
            df['Ticker'] = ticker
            return df
        except Exception as e:
            logging.error(f"Error downloading data for {ticker} (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                wait_time = 2 ** (attempt - 1)  # Exponential backoff
                logging.info(f"Retrying {ticker} after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download data for {ticker} after {max_retries} attempts.")
                return pd.DataFrame()

def save_data(df: pd.DataFrame, output_dir: str, format: str = 'csv'):
    """
    Saves the DataFrame to a file in the specified directory.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_dir (str): Directory where files will be saved.
        format (str): File format ('csv', 'parquet').
    """
    if df.empty:
        return
    ticker = df['Ticker'].iloc[0]
    if format == 'csv':
        output_path = os.path.join(output_dir, f"{ticker}.csv")
        try:
            df.to_csv(output_path, index=False)
            logging.info(f"Data for {ticker} saved successfully to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving data for {ticker}: {e}")
    elif format == 'parquet':
        output_path = os.path.join(output_dir, f"{ticker}.parquet")
        try:
            df.to_parquet(output_path, index=False)
            logging.info(f"Data for {ticker} saved successfully to {output_path}.")
        except Exception as e:
            logging.error(f"Error saving data for {ticker}: {e}")
    else:
        logging.error(f"Unsupported file format: {format}")

def download_all_data(tickers: list, start_date: str, end_date: str, interval: str = '1m',
                     max_threads: int = 20, output_dir: str = 'stock_data_minute', format: str = 'csv'):
    """
    Downloads historical minute-level data for all tickers using multithreading.

    Parameters:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (default '1m' for minute-level data).
        max_threads (int): Maximum number of concurrent threads.
        output_dir (str): Directory to save the downloaded data.
        format (str): File format for saving data ('csv', 'parquet').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created directory {output_dir} for saving data.")

    # Initialize ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # Submit download tasks
        future_to_ticker = {
            executor.submit(download_ticker_data, ticker, start_date, end_date, interval): ticker for ticker in tickers
        }

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker),
                           desc=f"Downloading {'Stocks' if 'stock' in output_dir else 'ETFs'} Data"):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                save_data(data, output_dir, format=format)
            except Exception as e:
                logging.error(f"Unhandled exception for {ticker}: {e}")

# ------------------------------ #
#            Main Function        #
# ------------------------------ #

def main():
    """
    Main function to orchestrate data acquisition.
    """
    # Fetch current S&P 500 tickers from Wikipedia
    sp500_tickers = fetch_sp500_tickers(WIKI_URL)
    if not sp500_tickers:
        logging.error("No S&P 500 tickers fetched. Exiting the script.")
        return

    # Initialize ETF tickers
    etf_tickers = ETF_TICKERS

    # Download S&P 500 stock data
    logging.info("Starting download of S&P 500 stock minute-level data.")
    download_all_data(
        tickers=sp500_tickers,
        start_date=START_DATE_STR,
        end_date=END_DATE_STR,
        interval=INTERVAL,
        max_threads=MAX_THREADS,
        output_dir=STOCK_OUTPUT_DIR,
        format=DATA_FORMAT
    )
    logging.info("Completed download of S&P 500 stock minute-level data.")

    # Download ETF data
    logging.info("Starting download of ETF minute-level data.")
    download_all_data(
        tickers=etf_tickers,
        start_date=START_DATE_STR,
        end_date=END_DATE_STR,
        interval=INTERVAL,
        max_threads=MAX_THREADS,
        output_dir=ETF_OUTPUT_DIR,
        format=DATA_FORMAT
    )
    logging.info("Completed download of ETF minute-level data.")

    logging.info("Data acquisition for Step 1 completed successfully.")

# ------------------------------ #
#           Entry Point           #
# ------------------------------ #

if __name__ == "__main__":
    main()