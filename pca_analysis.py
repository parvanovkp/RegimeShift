import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from datetime import timedelta

# Configuration
logging.basicConfig(
    filename='logs/pca_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PREPROCESSED_DATA_DIR = 'preprocessed_data'
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PREPROCESSED_ETFS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_etfs.csv')
PCA_RESULTS_DIR = 'pca_results'

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded preprocessed data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading preprocessed data: {e}")
        return pd.DataFrame()

def perform_pca(data: np.ndarray, n_components: float = 0.9) -> Tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    return pca, pca_result

def full_period_pca(data: pd.DataFrame) -> Tuple[PCA, np.ndarray]:
    logging.info("Performing full period PCA")
    pca, pca_result = perform_pca(data.values)
    logging.info(f"Full period PCA completed. Explained variance ratio: {pca.explained_variance_ratio_}")
    return pca, pca_result

def rolling_window_pca(data: pd.DataFrame, window_size: int, step_size: int) -> List[Tuple[pd.Timestamp, PCA, np.ndarray]]:
    logging.info(f"Performing rolling window PCA with window size {window_size} and step size {step_size}")
    results = []
    for start in tqdm(range(0, len(data) - window_size + 1, step_size)):
        end = start + window_size
        window_data = data.iloc[start:end]
        window_end_time = window_data.index[-1]
        pca, pca_result = perform_pca(window_data.values)
        results.append((window_end_time, pca, pca_result))
    logging.info("Rolling window PCA completed")
    return results

def intraday_pca(data: pd.DataFrame) -> List[Tuple[pd.Timestamp, PCA, np.ndarray]]:
    logging.info("Performing intraday PCA")
    results = []
    for date, day_data in tqdm(data.groupby(data.index.date)):
        pca, pca_result = perform_pca(day_data.values)
        results.append((pd.Timestamp(date), pca, pca_result))
    logging.info("Intraday PCA completed")
    return results

def plot_cumulative_explained_variance(pca: PCA, title: str, filename: str):
    plt.figure(figsize=(12, 6))
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'b-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(title)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Threshold')
    plt.legend()
    plt.savefig(os.path.join(PCA_RESULTS_DIR, filename))
    plt.close()

def plot_top_component_loadings(pca: PCA, data: pd.DataFrame, n_components: int, n_top_stocks: int, title: str, filename: str):
    plt.figure(figsize=(15, 10))
    loadings = pd.DataFrame(
        pca.components_.T[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    
    for i in range(n_components):
        plt.subplot(n_components, 1, i+1)
        top_stocks = loadings[f'PC{i+1}'].abs().nlargest(n_top_stocks)
        top_stocks.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Top {n_top_stocks} Stocks for PC{i+1}')
        plt.xlabel('')
        plt.ylabel('Loading')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PCA_RESULTS_DIR, filename))
    plt.close()

def plot_explained_variance_evolution(rolling_results: List[Tuple[pd.Timestamp, PCA, np.ndarray]], n_components: int, title: str, filename: str):
    dates = [result[0] for result in rolling_results]
    explained_variances = [result[1].explained_variance_ratio_[:n_components] for result in rolling_results]
    
    plt.figure(figsize=(15, 8))
    for i in range(n_components):
        plt.plot(dates, [ev[i] for ev in explained_variances], label=f'PC{i+1}')
    
    plt.xlabel('Date')
    plt.ylabel('Explained Variance Ratio')
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PCA_RESULTS_DIR, filename))
    plt.close()

def plot_component_heatmap(pca: PCA, data: pd.DataFrame, n_components: int, title: str, filename: str):
    loadings = pd.DataFrame(
        pca.components_.T[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    
    plt.figure(figsize=(20, 30))
    sns.heatmap(loadings, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PCA_RESULTS_DIR, filename))
    plt.close()

def main():
    os.makedirs(PCA_RESULTS_DIR, exist_ok=True)

    # Load preprocessed data
    stocks_data = load_preprocessed_data(PREPROCESSED_STOCKS_FILE)
    etfs_data = load_preprocessed_data(PREPROCESSED_ETFS_FILE)
    
    if stocks_data.empty or etfs_data.empty:
        logging.error("Failed to load preprocessed data. Exiting.")
        return

    # 2.1 Full Period PCA
    full_pca, full_pca_result = full_period_pca(stocks_data)
    plot_cumulative_explained_variance(full_pca, "Full Period PCA - Cumulative Explained Variance", "full_period_pca_variance.png")
    plot_top_component_loadings(full_pca, stocks_data, 5, 20, "Full Period PCA - Top 20 Stocks per Component", "full_period_pca_top_loadings.png")
    plot_component_heatmap(full_pca, stocks_data, 10, "Full Period PCA - Top 10 Components Heatmap", "full_period_pca_heatmap.png")

    # 2.2 Rolling Window PCA
    # 1-Day Window (assuming 2-minute data, 195 data points per trading day)
    one_day_results = rolling_window_pca(stocks_data, window_size=195, step_size=98)
    plot_explained_variance_evolution(one_day_results, 5, "1-Day Rolling Window PCA - Top 5 Components", "one_day_rolling_pca_evolution.png")

    # 5-Day Window
    five_day_results = rolling_window_pca(stocks_data, window_size=975, step_size=195)
    plot_explained_variance_evolution(five_day_results, 5, "5-Day Rolling Window PCA - Top 5 Components", "five_day_rolling_pca_evolution.png")

    # 2.3 Intraday PCA
    intraday_results = intraday_pca(stocks_data)
    
    # Plot explained variance for a sample of days
    sample_days = 5
    sample_intraday_results = intraday_results[1:sample_days + 1]
    for i, (date, pca, _) in enumerate(sample_intraday_results):
        plot_cumulative_explained_variance(pca, f"Intraday PCA - {date.date()}", f"intraday_pca_variance_day_{i+1}.png")

    logging.info("PCA analysis completed successfully")

if __name__ == "__main__":
    main()