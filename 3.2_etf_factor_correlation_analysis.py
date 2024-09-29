import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import logging
from typing import List, Tuple, Dict
import os

# Configuration
logging.basicConfig(
    filename='logs/3.2_etf_factor_correlation_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
PREPROCESSED_DATA_DIR = 'preprocessed_data'
PCA_RESULTS_DIR = 'pca_results'
INTERPRETATION_RESULTS_DIR = 'interpretation_results'
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
PREPROCESSED_ETFS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_etfs.csv')
PCA_COMPONENTS_FILE = os.path.join(PCA_RESULTS_DIR, 'pca_components.csv')
SPARSE_PCA_COMPONENTS_FILE = os.path.join(PCA_RESULTS_DIR, 'sparse_pca_components.csv')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def load_pca_results(file_path: str) -> pd.DataFrame:
    """Load PCA results from CSV file."""
    try:
        df = pd.read_csv(file_path, index_col=0)
        logging.info(f"Loaded PCA results from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading PCA results from {file_path}: {e}")
        return pd.DataFrame()

def project_data_onto_components(data: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Project data onto PCA components to get component scores."""
    return pd.DataFrame(np.dot(data, components), index=data.index, columns=components.columns)

def correlate_pca_with_etfs(pca_scores: pd.DataFrame, etfs: pd.DataFrame) -> pd.DataFrame:
    """Correlate PCA component scores with ETFs."""
    correlation_matrix = pd.DataFrame(index=pca_scores.columns, columns=etfs.columns)
    for pc in pca_scores.columns:
        for etf in etfs.columns:
            correlation_matrix.loc[pc, etf] = stats.pearsonr(pca_scores[pc], etfs[etf])[0]
    return correlation_matrix

def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str) -> go.Figure:
    """Create a heatmap of correlations between PCA components and ETFs."""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig.update_layout(
        title=title,
        xaxis_title="ETFs",
        yaxis_title="PCA Components",
        height=600, width=800
    )
    return fig

def perform_lag_analysis(pca_scores: pd.DataFrame, etfs: pd.DataFrame, max_lag: int) -> Dict[str, pd.DataFrame]:
    """Perform lag analysis between PCA component scores and ETFs."""
    lag_correlations = {}
    for pc in pca_scores.columns:
        lag_corr = pd.DataFrame(index=range(-max_lag, max_lag + 1), columns=etfs.columns)
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = etfs.shift(-lag).corrwith(pca_scores[pc])
            else:
                corr = etfs.corrwith(pca_scores[pc].shift(lag))
            lag_corr.loc[lag] = corr
        lag_correlations[pc] = lag_corr
    return lag_correlations

def plot_lag_analysis(lag_correlations: Dict[str, pd.DataFrame], title: str) -> go.Figure:
    """Create a plot of lag analysis results."""
    fig = make_subplots(rows=len(lag_correlations), cols=1, 
                        subplot_titles=[f"PC{i+1}" for i in range(len(lag_correlations))],
                        shared_xaxes=True)
    
    for i, (pc, lag_corr) in enumerate(lag_correlations.items(), 1):
        for etf in lag_corr.columns:
            fig.add_trace(
                go.Scatter(x=lag_corr.index, y=lag_corr[etf], name=etf, mode='lines'),
                row=i, col=1
            )
    
    fig.update_layout(height=300*len(lag_correlations), width=800, title_text=title)
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="Correlation")
    
    return fig

def etf_correlation_analysis(pca_scores: pd.DataFrame, etfs: pd.DataFrame, max_lag: int = 5) -> Tuple[go.Figure, go.Figure]:
    """Perform ETF correlation analysis including lag analysis."""
    correlation_matrix = correlate_pca_with_etfs(pca_scores, etfs)
    heatmap = plot_correlation_heatmap(correlation_matrix, "PCA Components vs ETFs Correlation Heatmap")
    
    lag_correlations = perform_lag_analysis(pca_scores, etfs, max_lag)
    lag_plot = plot_lag_analysis(lag_correlations, "Lag Analysis: PCA Components vs ETFs")
    
    return heatmap, lag_plot

def main():
    # Ensure output directory exists
    os.makedirs(INTERPRETATION_RESULTS_DIR, exist_ok=True)

    # Load data
    stock_data = load_data(PREPROCESSED_STOCKS_FILE)
    etf_data = load_data(PREPROCESSED_ETFS_FILE)
    pca_components = load_pca_results(PCA_COMPONENTS_FILE)
    sparse_pca_components = load_pca_results(SPARSE_PCA_COMPONENTS_FILE)

    if stock_data.empty or etf_data.empty or pca_components.empty or sparse_pca_components.empty:
        logging.error("Failed to load necessary data. Exiting.")
        return

    # Project stock data onto PCA components
    pca_scores = project_data_onto_components(stock_data, pca_components)
    sparse_pca_scores = project_data_onto_components(stock_data, sparse_pca_components)

    # Perform ETF correlation analysis for regular PCA
    heatmap, lag_plot = etf_correlation_analysis(pca_scores, etf_data)

    # Save plots for regular PCA
    heatmap.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, 'pca_etf_correlation_heatmap.html'))
    lag_plot.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, 'pca_etf_lag_analysis.html'))

    # Perform ETF correlation analysis for Sparse PCA
    sparse_heatmap, sparse_lag_plot = etf_correlation_analysis(sparse_pca_scores, etf_data)

    # Save plots for Sparse PCA
    sparse_heatmap.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, 'sparse_pca_etf_correlation_heatmap.html'))
    sparse_lag_plot.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, 'sparse_pca_etf_lag_analysis.html'))

    logging.info("ETF correlation analysis completed successfully for both PCA and Sparse PCA.")

if __name__ == "__main__":
    main()