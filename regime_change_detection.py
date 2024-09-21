import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
import joblib
from datetime import timedelta

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Logging configuration
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'regime_change_detection.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Data directories and file paths
PREPROCESSED_DATA_DIR = 'preprocessed_data'
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PREPROCESSED_ETFS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_etfs.csv')
PCA_RESULTS_DIR = 'pca_results'
if not os.path.exists(PCA_RESULTS_DIR):
    os.makedirs(PCA_RESULTS_DIR)

PCA_MODEL_FILE = os.path.join(PCA_RESULTS_DIR, 'pca_model.joblib')
EXPLAINED_VARIANCE_FILE = os.path.join(PCA_RESULTS_DIR, 'explained_variance.csv')

# Market events file (optional)
MARKET_EVENTS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'market_events.csv')  # Ensure this file exists or handle accordingly

# Output directories and file paths
REGIME_CHANGE_DIR = 'regime_change_results'
if not os.path.exists(REGIME_CHANGE_DIR):
    os.makedirs(REGIME_CHANGE_DIR)

REGIME_CHANGES_FILE = os.path.join(REGIME_CHANGE_DIR, 'regime_changes.csv')
REGIME_CHANGES_WITH_EVENTS_FILE = os.path.join(REGIME_CHANGE_DIR, 'regime_changes_with_events.csv')

# Interpretation parameters
ROLLING_WINDOW_SIZE = 30        # Number of days for rolling PCA
SHIFT_WINDOW_SIZE = 7           # Number of days to look back for detecting shifts
VARIANCE_CHANGE_THRESHOLD = 0.05  # 5% change in explained variance to detect a shift
TOP_N_PCS = 20                   # Number of top PCs to monitor
TOP_M_ETFS = 5                   # Number of top ETFs per PC to consider
EVENT_LOOKBACK_DAYS = 1          # Days before the shift to consider for events
EVENT_LOOKAHEAD_DAYS = 1         # Days after the shift to consider for events

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

def load_pca_model(pca_model_file: str) -> PCA:
    """
    Loads the saved PCA model from a joblib file.
    
    Parameters:
        pca_model_file (str): Path to the saved PCA model (.joblib file).
    
    Returns:
        PCA: Loaded PCA model.
    """
    try:
        pca = joblib.load(pca_model_file)
        logging.info(f"Loaded PCA model from {pca_model_file}.")
        return pca
    except Exception as e:
        logging.error(f"Error loading PCA model from {pca_model_file}: {e}")
        return None

def load_explained_variance(file_path: str, top_n: int) -> list:
    """
    Loads the explained variance CSV and returns the top N principal components.
    
    Parameters:
        file_path (str): Path to the explained_variance.csv file.
        top_n (int): Number of top PCs to select.
    
    Returns:
        list: List of top N principal component names.
    """
    try:
        df = pd.read_csv(file_path)
        top_pcs = df['Component'].head(top_n).tolist()
        logging.info(f"Selected top {top_n} principal components based on order.")
        return top_pcs
    except Exception as e:
        logging.error(f"Error loading explained variance from {file_path}: {e}")
        return []

def compute_principal_component_scores(pca: PCA, returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the returns data using the PCA model to obtain principal component scores.
    
    Parameters:
        pca (PCA): Fitted PCA model.
        returns_df (pd.DataFrame): DataFrame containing standardized returns.
    
    Returns:
        pd.DataFrame: DataFrame containing principal component scores.
    """
    try:
        pc_scores = pca.transform(returns_df)
        pc_columns = [f'PC{i+1}' for i in range(pc_scores.shape[1])]
        pc_scores_df = pd.DataFrame(pc_scores, index=returns_df.index, columns=pc_columns)
        logging.info(f"Computed principal component scores with shape {pc_scores_df.shape}.")
        return pc_scores_df
    except Exception as e:
        logging.error(f"Error computing principal component scores: {e}")
        return pd.DataFrame()

def align_data(pc_scores_df: pd.DataFrame, etf_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aligns the principal component scores and ETF returns based on timestamps.
    
    Parameters:
        pc_scores_df (pd.DataFrame): DataFrame containing principal component scores.
        etf_returns_df (pd.DataFrame): DataFrame containing ETF returns.
    
    Returns:
        pd.DataFrame: Aligned DataFrame containing PC scores and ETF returns.
    """
    try:
        aligned_df = pc_scores_df.join(etf_returns_df, how='inner')
        logging.info(f"Aligned data shape: {aligned_df.shape}.")
        return aligned_df
    except Exception as e:
        logging.error(f"Error aligning principal component scores with ETF returns: {e}")
        return pd.DataFrame()

def perform_rolling_pca(returns_df: pd.DataFrame, window_size: int, n_components: int) -> pd.DataFrame:
    """
    Performs rolling PCA on the returns data.
    
    Parameters:
        returns_df (pd.DataFrame): DataFrame containing standardized returns.
        window_size (int): Number of days for each rolling window.
        n_components (int): Number of principal components to retain.
    
    Returns:
        pd.DataFrame: DataFrame containing explained variance for each PC over time.
    """
    explained_variance_list = []
    dates = []
    
    for end_date in returns_df.index[window_size - 1:]:
        window_data = returns_df.loc[:end_date].tail(window_size)
        try:
            pca = PCA(n_components=n_components)
            pca.fit(window_data)
            explained_variance = pca.explained_variance_ratio_
            explained_variance_list.append(explained_variance)
            dates.append(end_date)
            logging.debug(f"Computed rolling PCA for window ending on {end_date.date()}.")
        except Exception as e:
            logging.error(f"Error performing rolling PCA for window ending on {end_date.date()}: {e}")
            explained_variance_list.append([np.nan]*n_components)
            dates.append(end_date)
    
    explained_variance_df = pd.DataFrame(explained_variance_list, index=dates, 
                                        columns=[f'PC{i+1}_EV' for i in range(n_components)])
    logging.info(f"Computed rolling explained variance with shape {explained_variance_df.shape}.")
    return explained_variance_df

def detect_shifts(rolling_ev_df: pd.DataFrame, shift_window: int, threshold: float) -> pd.DataFrame:
    """
    Detects significant shifts in explained variance.
    
    Parameters:
        rolling_ev_df (pd.DataFrame): DataFrame containing rolling explained variance.
        shift_window (int): Number of days to look back for detecting shifts.
        threshold (float): Threshold for significant change in explained variance.
    
    Returns:
        pd.DataFrame: DataFrame containing detected shifts.
    """
    shifts = []
    pcs = rolling_ev_df.columns.str.replace('_EV', '').tolist()
    
    for pc in pcs:
        current_ev = rolling_ev_df[f'{pc}_EV'].iloc[-shift_window:]
        baseline_ev = rolling_ev_df[f'{pc}_EV'].iloc[-(shift_window*2):-shift_window]
        
        # Compute average explained variance before and after the shift window
        baseline_mean = baseline_ev.mean()
        current_mean = current_ev.mean()
        change = current_mean - baseline_mean
        
        if abs(change) >= threshold:
            shift_date = current_ev.index[-1]
            shifts.append({
                'Principal_Component': pc,
                'Shift_Date': shift_date,
                'Change_in_Explained_Variance': change
            })
            logging.info(f"Detected shift in {pc} on {shift_date.date()}: Change = {change:.4f}")
    
    shifts_df = pd.DataFrame(shifts)
    logging.info(f"Detected {len(shifts_df)} regime shifts.")
    return shifts_df

def load_market_events(file_path: str) -> pd.DataFrame:
    """
    Loads market events from a CSV file.
    
    Parameters:
        file_path (str): Path to the market events CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing market events.
    """
    try:
        events_df = pd.read_csv(file_path, parse_dates=['Date'])
        logging.info(f"Loaded market events from {file_path} with shape {events_df.shape}.")
        return events_df
    except FileNotFoundError:
        logging.warning(f"Market events file {file_path} not found. Skipping market event validation.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading market events from {file_path}: {e}")
        return pd.DataFrame()

def correlate_shifts_with_events(shifts_df: pd.DataFrame, events_df: pd.DataFrame, 
                                 lookback: int, lookahead: int) -> pd.DataFrame:
    """
    Correlates detected regime shifts with market events.
    
    Parameters:
        shifts_df (pd.DataFrame): DataFrame containing detected shifts.
        events_df (pd.DataFrame): DataFrame containing market events.
        lookback (int): Days before the shift to consider for events.
        lookahead (int): Days after the shift to consider for events.
    
    Returns:
        pd.DataFrame: DataFrame containing shifts with correlated events.
    """
    if events_df.empty:
        logging.warning("No market events to correlate with shifts.")
        shifts_df['Related_Events'] = None
        return shifts_df
    
    shifts_with_events = shifts_df.copy()
    shifts_with_events['Related_Events'] = shifts_with_events['Shift_Date'].apply(
        lambda date: list(
            events_df[
                (events_df['Date'] >= (date - timedelta(days=lookback))) &
                (events_df['Date'] <= (date + timedelta(days=lookahead)))
            ]['Event'].values
        )
    )
    logging.info("Correlated regime shifts with market events.")
    return shifts_with_events

def save_regime_changes(shifts_df: pd.DataFrame, output_file: str):
    """
    Saves detected regime shifts to a CSV file.
    
    Parameters:
        shifts_df (pd.DataFrame): DataFrame containing detected shifts.
        output_file (str): Path to save the CSV file.
    """
    try:
        shifts_df.to_csv(output_file, index=False)
        logging.info(f"Saved regime changes to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving regime changes to {output_file}: {e}")

# ------------------------------ #
#            Main Function        #
# ------------------------------ #

def main():
    """
    Main function to perform regime change detection.
    """
    logging.info("Starting Regime Change Detection.")
    
    # Load preprocessed stock returns
    stock_returns_df = load_preprocessed_data(PREPROCESSED_STOCKS_FILE)
    if stock_returns_df.empty:
        logging.error("Preprocessed stock returns data is empty. Exiting regime change detection.")
        return
    
    # Load PCA model
    pca = load_pca_model(PCA_MODEL_FILE)
    if pca is None:
        logging.error("PCA model could not be loaded. Exiting regime change detection.")
        return
    
    # Load explained variance and select top N PCs
    top_pcs = load_explained_variance(EXPLAINED_VARIANCE_FILE, TOP_N_PCS)
    if not top_pcs:
        logging.error("Could not load top principal components based on explained variance. Exiting regime change detection.")
        return
    
    # Perform rolling PCA to track explained variance over time
    rolling_ev_df = perform_rolling_pca(stock_returns_df, ROLLING_WINDOW_SIZE, len(top_pcs))
    
    # Save rolling explained variance for reference
    rolling_ev_file = os.path.join(REGIME_CHANGE_DIR, 'rolling_explained_variance.csv')
    try:
        rolling_ev_df.to_csv(rolling_ev_file)
        logging.info(f"Saved rolling explained variance to {rolling_ev_file}.")
    except Exception as e:
        logging.error(f"Error saving rolling explained variance to {rolling_ev_file}: {e}")
    
    # Detect shifts in explained variance
    shifts_df = detect_shifts(rolling_ev_df, SHIFT_WINDOW_SIZE, VARIANCE_CHANGE_THRESHOLD)
    if shifts_df.empty:
        logging.info("No significant regime shifts detected.")
        return
    
    # Load market events
    events_df = load_market_events(MARKET_EVENTS_FILE)
    
    # Correlate detected shifts with market events
    shifts_with_events_df = correlate_shifts_with_events(shifts_df, events_df, EVENT_LOOKBACK_DAYS, EVENT_LOOKAHEAD_DAYS)
    
    # Save regime changes with events
    save_regime_changes(shifts_with_events_df, REGIME_CHANGES_WITH_EVENTS_FILE)
    
    logging.info("Regime Change Detection completed successfully.")

# ------------------------------ #
#           Entry Point           #
# ------------------------------ #

if __name__ == "__main__":
    main()