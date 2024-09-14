import os
import pandas as pd
import numpy as np
import logging
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------ #
#        Configuration           #
# ------------------------------ #

# Logging configuration
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'pca_interpretation.log'),
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

PCA_MODEL_FILE = os.path.join(PCA_RESULTS_DIR, 'pca_model.joblib')  # Updated file extension
PCA_LOADINGS_FILE = os.path.join(PCA_RESULTS_DIR, 'pca_loadings.csv')
EXPLAINED_VARIANCE_FILE = os.path.join(PCA_RESULTS_DIR, 'explained_variance.csv')

# Interpretation parameters
CORRELATION_THRESHOLD = 0.5  # Define a threshold for high correlation
TOP_N_PCS = 20               # Number of top PCs to consider based on explained variance
TOP_M_ETFS = 5               # Number of top ETFs per PC based on absolute correlation

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

def load_top_pcs(file_path: str, top_n: int) -> list:
    """
    Loads the explained variance CSV and returns the first N principal components.
    
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

def compute_correlations(aligned_df: pd.DataFrame, pc_columns: list, etf_columns: list) -> pd.DataFrame:
    """
    Computes Pearson correlation coefficients between principal components and ETFs.
    
    Parameters:
        aligned_df (pd.DataFrame): DataFrame containing both PC scores and ETF returns.
        pc_columns (list): List of principal component column names.
        etf_columns (list): List of ETF return column names.
    
    Returns:
        pd.DataFrame: Correlation matrix with PCs as rows and ETFs as columns.
    """
    try:
        # Initialize an empty list to collect correlation results
        results = []
        
        for pc in pc_columns:
            for etf in etf_columns:
                correlation = aligned_df[pc].corr(aligned_df[etf])
                results.append({'Principal_Component': pc, 'ETF': etf, 'Correlation': correlation})
        
        correlation_df = pd.DataFrame(results)
        logging.info(f"Computed correlation matrix with shape {correlation_df.shape}.")
        return correlation_df
    except Exception as e:
        logging.error(f"Error computing correlations: {e}")
        return pd.DataFrame()

def save_correlation_matrix(correlation_df: pd.DataFrame, output_file: str):
    """
    Saves the correlation matrix to a CSV file.
    
    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing correlation coefficients.
        output_file (str): Path to save the CSV file.
    """
    try:
        correlation_df.to_csv(output_file, index=False)
        logging.info(f"Saved correlation matrix to {output_file}.")
    except Exception as e:
        logging.error(f"Error saving correlation matrix to {output_file}: {e}")

def plot_correlation_heatmap(correlation_df: pd.DataFrame, output_file: str, top_n_pcs: int, top_m_etfs: int):
    """
    Plots a heatmap of the top correlated ETFs for the top principal components.
    
    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing correlation coefficients.
        output_file (str): Path to save the heatmap image.
        top_n_pcs (int): Number of top PCs to include.
        top_m_etfs (int): Number of top ETFs per PC to include.
    """
    try:
        # Select top N PCs
        top_pcs = [f'PC{i+1}' for i in range(top_n_pcs)]
        filtered_df = correlation_df[correlation_df['Principal_Component'].isin(top_pcs)]
        
        # For each PC, select top M ETFs based on absolute correlation
        filtered_df = filtered_df.copy()
        filtered_df['Abs_Correlation'] = filtered_df['Correlation'].abs()
        filtered_df = filtered_df.sort_values(['Principal_Component', 'Abs_Correlation'], ascending=[True, False])
        top_etfs_per_pc = filtered_df.groupby('Principal_Component').head(top_m_etfs)
        
        # Pivot the DataFrame to have PCs as rows and ETFs as columns
        heatmap_data = top_etfs_per_pc.pivot(index='Principal_Component', columns='ETF', values='Correlation')
        
        # Sort PCs by their order in top_pcs
        heatmap_data = heatmap_data.loc[top_pcs]
        
        # Plot heatmap
        plt.figure(figsize=(12, max(6, top_n_pcs * 0.3)))  # Adjust height based on number of PCs
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Top Correlated ETFs for Top Principal Components', fontsize=16)
        plt.xlabel('ETFs', fontsize=14)
        plt.ylabel('Principal Components', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        logging.info(f"Saved correlation heatmap to {output_file}.")
    except Exception as e:
        logging.error(f"Error plotting correlation heatmap: {e}")

def document_interpretations(correlation_df: pd.DataFrame, output_file: str, threshold: float = 0.5):
    """
    Documents the interpretations of principal components based on high correlations with ETFs.
    
    Parameters:
        correlation_df (pd.DataFrame): DataFrame containing correlation coefficients.
        output_file (str): Path to save the interpretations.
        threshold (float): Correlation threshold to consider as significant.
    """
    try:
        interpretations = []
        for pc in correlation_df['Principal_Component'].unique():
            pc_correlations = correlation_df[correlation_df['Principal_Component'] == pc]
            high_corr = pc_correlations[pc_correlations['Correlation'].abs() >= threshold]
            if not high_corr.empty:
                interpretations.append({
                    'Principal_Component': pc,
                    'High_Correlated_ETFs': high_corr[['ETF', 'Correlation']].to_dict('records')
                })
        
        # Save interpretations to a text file
        with open(output_file, 'w') as f:
            for interpretation in interpretations:
                pc = interpretation['Principal_Component']
                etfs = interpretation['High_Correlated_ETFs']
                f.write(f"{pc}:\n")
                for etf in etfs:
                    corr_value = etf['Correlation']
                    corr_sign = "positive" if corr_value >= 0 else "negative"
                    f.write(f"  - {etf['ETF']} ({corr_sign} correlation: {corr_value:.2f})\n")
                f.write("\n")
        
        logging.info(f"Saved principal component interpretations to {output_file}.")
    except Exception as e:
        logging.error(f"Error documenting interpretations: {e}")

# ------------------------------ #
#            Main Function        #
# ------------------------------ #

def main():
    """
    Main function to orchestrate interpretation of principal components.
    """
    logging.info("Starting interpretation of principal components.")
    
    # Load preprocessed stock returns
    stock_returns_df = load_preprocessed_data(PREPROCESSED_STOCKS_FILE)
    if stock_returns_df.empty:
        logging.error("Preprocessed stock returns data is empty. Exiting interpretation step.")
        return
    
    # Load PCA model
    pca = load_pca_model(PCA_MODEL_FILE)
    if pca is None:
        logging.error("PCA model could not be loaded. Exiting interpretation step.")
        return
    
    # Compute principal component scores
    pc_scores_df = compute_principal_component_scores(pca, stock_returns_df)
    if pc_scores_df.empty:
        logging.error("Principal component scores could not be computed. Exiting interpretation step.")
        return
    
    # Load preprocessed ETF returns
    etf_returns_df = load_preprocessed_data(PREPROCESSED_ETFS_FILE)
    if etf_returns_df.empty:
        logging.error("Preprocessed ETF returns data is empty. Exiting interpretation step.")
        return
    
    # Align principal component scores with ETF returns
    aligned_df = align_data(pc_scores_df, etf_returns_df)
    if aligned_df.empty:
        logging.error("Aligned DataFrame is empty. Exiting interpretation step.")
        return
    
    # Define principal components and ETFs
    pc_columns = [col for col in pc_scores_df.columns]
    etf_columns = [col for col in etf_returns_df.columns]
    
    # Load explained variance and select top N PCs
    top_pcs = load_top_pcs(EXPLAINED_VARIANCE_FILE, TOP_N_PCS)
    if not top_pcs:
        logging.error("Could not load top principal components based on explained variance. Exiting interpretation step.")
        return
    
    # Filter PC scores to include only top PCs
    pc_scores_df_top = pc_scores_df[top_pcs]
    
    # Re-align data with top PCs
    aligned_df_top = align_data(pc_scores_df_top, etf_returns_df)
    if aligned_df_top.empty:
        logging.error("Aligned DataFrame with top PCs is empty. Exiting interpretation step.")
        return
    
    # Define principal components and ETFs for top PCs
    pc_columns_top = [col for col in pc_scores_df_top.columns]
    etf_columns = [col for col in etf_returns_df.columns]
    
    # Debugging: Check for NaN values
    if aligned_df_top[pc_columns_top + etf_columns].isnull().values.any():
        logging.warning("Aligned DataFrame contains NaN values. Filling NaNs with zeros.")
        aligned_df_top.fillna(0, inplace=True)
    
    # Compute correlations
    correlation_df = compute_correlations(aligned_df_top, pc_columns_top, etf_columns)
    if correlation_df.empty:
        logging.error("Correlation matrix is empty. Exiting interpretation step.")
        return
    
    # Save correlation matrix
    correlation_matrix_file = os.path.join(PCA_RESULTS_DIR, 'pc_etf_correlation.csv')
    save_correlation_matrix(correlation_df, correlation_matrix_file)
    
    # Plot correlation heatmap with top PCs and top ETFs
    heatmap_file = os.path.join(PCA_RESULTS_DIR, 'pc_etf_correlation_heatmap.png')
    plot_correlation_heatmap(correlation_df, heatmap_file, TOP_N_PCS, TOP_M_ETFS)
    
    # Document interpretations
    interpretations_file = os.path.join(PCA_RESULTS_DIR, 'pc_interpretations.txt')
    document_interpretations(correlation_df, interpretations_file, threshold=CORRELATION_THRESHOLD)
    
    logging.info("Interpretation of principal components completed successfully.")

# ------------------------------ #
#           Entry Point           #
# ------------------------------ #

if __name__ == "__main__":
    main()