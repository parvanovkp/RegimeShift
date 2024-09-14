import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Logging configuration
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'pca_analysis.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data directories and file paths
PREPROCESSED_DATA_DIR = 'preprocessed_data'
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PCA_RESULTS_DIR = 'pca_results'
if not os.path.exists(PCA_RESULTS_DIR):
    os.makedirs(PCA_RESULTS_DIR)

# PCA parameters
EXPLAINED_VARIANCE_THRESHOLD = 0.90  # Retain 90% of the variance

# Temporal PCA parameters
WINDOW_SIZE = 60  # in minutes
STEP_SIZE = 30    # in minutes

# ------------------------------ #
#        Helper Functions        #
# ------------------------------ #

def load_preprocessed_data(file_path: str) -> pd.DataFrame:
    """
    Loads the preprocessed returns data from a CSV file.
    
    Parameters:
        file_path (str): Path to the preprocessed CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing standardized returns.
    """
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded preprocessed data from {file_path} with shape {df.shape}.")
        return df
    except Exception as e:
        logging.error(f"Error loading preprocessed data from {file_path}: {e}")
        return pd.DataFrame()

def perform_basic_pca(returns_df: pd.DataFrame, variance_threshold: float = 0.90) -> PCA:
    """
    Performs PCA on the entire returns dataset and determines the number of components to retain.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing standardized returns.
        variance_threshold (float): Cumulative variance threshold to determine number of components.
    
    Returns:
        PCA: Fitted PCA object.
    """
    logging.info("Starting basic PCA on the entire dataset.")
    
    # Initialize PCA without specifying number of components
    pca = PCA()
    pca.fit(returns_df)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    logging.info(f"Number of components to retain to explain {variance_threshold*100}% variance: {n_components}")
    
    # Refit PCA with the determined number of components
    pca = PCA(n_components=n_components)
    pca.fit(returns_df)
    logging.info(f"PCA fitting completed with {n_components} components.")
    
    # Save explained variance
    explained_variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Explained Variance Ratio': pca.explained_variance_ratio_,
        'Cumulative Explained Variance': np.cumsum(pca.explained_variance_ratio_)
    })
    explained_variance_df.to_csv(os.path.join(PCA_RESULTS_DIR, 'explained_variance.csv'), index=False)
    logging.info("Saved explained variance to explained_variance.csv.")
    
    # Save principal components (loadings)
    loadings_df = pd.DataFrame(pca.components_.T, index=returns_df.columns, 
                               columns=[f'PC{i+1}' for i in range(n_components)])
    loadings_df.to_csv(os.path.join(PCA_RESULTS_DIR, 'pca_loadings.csv'))
    logging.info("Saved PCA loadings to pca_loadings.csv.")
    
    # Save PCA model using joblib
    joblib.dump(pca, os.path.join(PCA_RESULTS_DIR, 'pca_model.joblib'))
    logging.info("Saved PCA model to pca_model.joblib.")
    
    return pca

def perform_temporal_pca(returns_df: pd.DataFrame, window_size: int = 60, step_size: int = 30) -> pd.DataFrame:
    """
    Performs PCA over sliding windows to capture temporal shifts in principal components.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing standardized returns.
        window_size (int): Number of minutes in each sliding window.
        step_size (int): Number of minutes to step forward for each window.
    
    Returns:
        pd.DataFrame: DataFrame containing the explained variance of the first principal component over time.
    """
    logging.info("Starting temporal PCA (rolling window PCA).")
    
    # Initialize list to store results
    temporal_results = []
    
    # Calculate the number of windows
    total_minutes = returns_df.shape[0]
    num_windows = (total_minutes - window_size) // step_size + 1
    
    # Iterate over each window
    for i in tqdm(range(num_windows), desc="Performing Rolling PCA"):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        window_df = returns_df.iloc[start_idx:end_idx]
        
        # Perform PCA on the window
        pca = PCA(n_components=1)  # Focus on the first principal component
        pca.fit(window_df)
        
        # Store the timestamp and explained variance
        window_start = window_df.index[0]
        window_end = window_df.index[-1]
        temporal_results.append({
            'Window_Start': window_start,
            'Window_End': window_end,
            'Explained_Variance_PC1': pca.explained_variance_ratio_[0]
        })
    
    # Convert results to DataFrame
    temporal_pca_df = pd.DataFrame(temporal_results)
    
    # Save temporal PCA results
    temporal_pca_df.to_csv(os.path.join(PCA_RESULTS_DIR, 'temporal_pca.csv'), index=False)
    logging.info("Saved temporal PCA results to temporal_pca.csv.")
    
    return temporal_pca_df

def plot_explained_variance(explained_variance_file: str):
    """
    Plots the cumulative explained variance to determine the number of principal components.
    
    Parameters:
        explained_variance_file (str): Path to the explained_variance.csv file.
    """
    try:
        df = pd.read_csv(explained_variance_file)
        
        # Extract numeric component indices
        df['Component_Num'] = range(1, len(df) + 1)
        
        plt.figure(figsize=(14, 8))  # Increased figure size for better readability
        
        plt.plot(df['Component_Num'], df['Cumulative Explained Variance'], marker='o', linestyle='-')
        plt.axhline(y=EXPLAINED_VARIANCE_THRESHOLD, color='r', linestyle='--', 
                    label=f'{int(EXPLAINED_VARIANCE_THRESHOLD*100)}% Variance Threshold')
        
        plt.title('Cumulative Explained Variance by Principal Components', fontsize=16)
        plt.xlabel('Principal Components', fontsize=14)
        plt.ylabel('Cumulative Explained Variance', fontsize=14)
        
        # Determine tick interval based on number of components
        num_components = df['Component_Num'].max()
        if num_components <= 20:
            tick_interval = 1
        elif num_components <= 50:
            tick_interval = 5
        elif num_components <= 100:
            tick_interval = 10
        else:
            tick_interval = 20  # Adjust as needed for larger numbers
        
        ticks = range(1, num_components + 1, tick_interval)
        plt.xticks(ticks, [f'PC{tick}' for tick in ticks], rotation=45, fontsize=12)
        
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PCA_RESULTS_DIR, 'cumulative_explained_variance.png'))
        plt.close()
        logging.info("Saved cumulative explained variance plot to cumulative_explained_variance.png.")
    except Exception as e:
        logging.error(f"Error plotting explained variance: {e}")

def plot_temporal_pca(temporal_pca_file: str):
    """
    Plots the explained variance of the first principal component over time to detect regime shifts.
    
    Parameters:
        temporal_pca_file (str): Path to the temporal_pca.csv file.
    """
    try:
        df = pd.read_csv(temporal_pca_file, parse_dates=['Window_Start', 'Window_End'])
        plt.figure(figsize=(14, 7))
        plt.plot(df['Window_End'], df['Explained_Variance_PC1'], marker='o', linestyle='-')
        plt.title('Temporal PCA: Explained Variance of PC1 Over Time', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Explained Variance Ratio of PC1', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(PCA_RESULTS_DIR, 'temporal_pca_explained_variance.png'))
        plt.close()
        logging.info("Saved temporal PCA explained variance plot to temporal_pca_explained_variance.png.")
    except Exception as e:
        logging.error(f"Error plotting temporal PCA: {e}")

# ------------------------------ #
#            Main Function        #
# ------------------------------ #

def main():
    """
    Main function to orchestrate PCA analysis.
    """
    logging.info("Starting PCA analysis.")
    
    # Load preprocessed data
    returns_df = load_preprocessed_data(PREPROCESSED_STOCKS_FILE)
    if returns_df.empty:
        logging.error("Preprocessed returns data is empty. Exiting PCA analysis.")
        return
    
    # Perform basic PCA
    pca = perform_basic_pca(returns_df, variance_threshold=EXPLAINED_VARIANCE_THRESHOLD)
    
    # Plot explained variance
    explained_variance_file = os.path.join(PCA_RESULTS_DIR, 'explained_variance.csv')
    plot_explained_variance(explained_variance_file)
    
    # Perform temporal PCA
    temporal_pca_df = perform_temporal_pca(returns_df, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    
    # Plot temporal PCA results
    temporal_pca_file = os.path.join(PCA_RESULTS_DIR, 'temporal_pca.csv')
    plot_temporal_pca(temporal_pca_file)
    
    logging.info("PCA analysis completed successfully.")

# ------------------------------ #
#           Entry Point           #
# ------------------------------ #

if __name__ == "__main__":
    main()