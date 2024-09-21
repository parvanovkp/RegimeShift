import pandas as pd
import yfinance as yf
import os
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Logging configuration
logging.basicConfig(
    filename=config['logging']['filename'],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Date range for data retrieval
END_DATE = datetime.now().date()
START_DATE = END_DATE - timedelta(days=config['data']['days'])
START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def fetch_sp500_tickers(wiki_url: str) -> list:
    """
    Fetches the list of S&P 500 tickers from the S&P 500 Wikipedia page.
    """
    try:
        tables = pd.read_html(wiki_url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        logging.info(f"Successfully fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        return [ticker.replace('.', '-') for ticker in tickers]
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
        return []

def download_ticker_data(ticker: str, start_date: str, end_date: str, interval: str = config['data']['interval']) -> pd.DataFrame:
    """
    Downloads historical data for a given ticker.
    """
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
        logging.error(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, output_dir: str, format: str = config['data']['format']):
    """
    Saves the DataFrame to a file in the specified directory.
    """
    if df.empty:
        return
    ticker = df['Ticker'].iloc[0]
    if format == 'csv':
        output_path = os.path.join(output_dir, f"{ticker}.csv")
        df.to_csv(output_path, index=False)
    elif format == 'parquet':
        output_path = os.path.join(output_dir, f"{ticker}.parquet")
        df.to_parquet(output_path, index=False)
    else:
        logging.error(f"Unsupported file format: {format}")
        return
    logging.info(f"Data for {ticker} saved successfully to {output_path}.")

def download_all_data(tickers: list, start_date: str, end_date: str, interval: str,
                      max_threads: int, output_dir: str, format: str):
    """
    Downloads historical data for all tickers using multithreading.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_ticker = {executor.submit(download_ticker_data, ticker, start_date, end_date, interval): ticker for ticker in tickers}
        
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker),
                           desc=f"Downloading Data"):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if not data.empty:
                    save_data(data, output_dir, format)
            except Exception as e:
                logging.error(f"Unhandled exception for {ticker}: {e}")

    successful_downloads = len([f for f in os.listdir(output_dir) if f.endswith(f'.{format}')])
    logging.info(f"Download summary: {successful_downloads}/{len(tickers)} tickers successfully downloaded.")

# ------------------------------ #
#            Main Function       #
# ------------------------------ #

def main():
    """
    Main function to orchestrate data acquisition.
    """
    sp500_tickers = fetch_sp500_tickers(config['data']['wiki_url'])
    if not sp500_tickers:
        logging.error("No S&P 500 tickers fetched. Exiting the script.")
        return

    etf_tickers = config['data']['etf_tickers']

    logging.info("Starting download of S&P 500 stock data.")
    download_all_data(
        tickers=sp500_tickers,
        start_date=START_DATE_STR,
        end_date=END_DATE_STR,
        interval=config['data']['interval'],
        max_threads=config['performance']['max_threads'],
        output_dir=os.path.join(config['data']['base_output_dir'], config['data']['stock_output_dir']),
        format=config['data']['format']
    )
    logging.info("Completed download of S&P 500 stock data.")

    logging.info("Starting download of ETF data.")
    download_all_data(
        tickers=etf_tickers,
        start_date=START_DATE_STR,
        end_date=END_DATE_STR,
        interval=config['data']['interval'],
        max_threads=config['performance']['max_threads'],
        output_dir=os.path.join(config['data']['base_output_dir'], config['data']['etf_output_dir']),
        format=config['data']['format']
    )
    logging.info("Completed download of ETF data.")

    logging.info("Data acquisition completed successfully.")

# ------------------------------ #
#           Entry Point          #
# ------------------------------ #

if __name__ == "__main__":
    main()