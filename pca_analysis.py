import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import logging
import os
import seaborn as sns
from typing import Tuple, List, Dict, Union
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

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
SECTOR_INFO_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'sector_info.csv')

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

def perform_sparse_pca(data: np.ndarray, n_components: int) -> Tuple[SparsePCA, np.ndarray]:
    sparse_pca = SparsePCA(n_components=n_components, random_state=42)
    sparse_pca_result = sparse_pca.fit_transform(data)
    return sparse_pca, sparse_pca_result

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

def perform_sparse_pca_window(args):
    start, end, data, n_components = args
    window_data = data.iloc[start:end]
    window_end_time = window_data.index[-1]
    sparse_pca, sparse_pca_result = perform_sparse_pca(window_data.values, n_components)
    return window_end_time, sparse_pca, sparse_pca_result

def rolling_window_sparse_pca(data: pd.DataFrame, window_size: int, step_size: int, n_components: int) -> List[Tuple[pd.Timestamp, SparsePCA, np.ndarray]]:
    logging.info(f"Performing parallel rolling window Sparse PCA with window size {window_size} and step size {step_size}")
    
    # Create a list of arguments for each window
    args_list = [(start, start + window_size, data, n_components) 
                 for start in range(0, len(data) - window_size + 1, step_size)]
    
    # Use all available cores except one
    num_processes = multiprocessing.cpu_count() - 1
    
    # Use multiprocessing to parallelize the computation
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(perform_sparse_pca_window, args_list), 
                            total=len(args_list), 
                            desc="Rolling Sparse PCA"))
    
    logging.info("Parallel rolling window Sparse PCA completed")
    return results

def sector_based_pca(data: pd.DataFrame, sector_mapping: Dict[str, str]) -> Dict[str, Tuple[PCA, np.ndarray]]:
    logging.info("Performing sector-based PCA")
    sector_results = {}
    for sector, stocks in sector_mapping.items():
        sector_data = data[stocks]
        pca, pca_result = perform_pca(sector_data.values)
        sector_results[sector] = (pca, pca_result)
    logging.info("Sector-based PCA completed")
    return sector_results

def plot_cumulative_explained_variance(pca: PCA, title: str, filename: str):
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio) + 1)), 
                             y=cumulative_variance_ratio, 
                             mode='lines', 
                             name='Cumulative Explained Variance'))
    
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% Threshold")
    fig.add_hline(y=0.9, line_dash="dash", line_color="green", annotation_text="90% Threshold")
    
    fig.update_layout(
        title=title,
        xaxis_title="Number of Components",
        yaxis_title="Cumulative Explained Variance Ratio",
        legend_title="Legend"
    )
    
    fig.write_html(os.path.join(PCA_RESULTS_DIR, filename))

def plot_top_component_loadings(pca: PCA, data: pd.DataFrame, n_components: int, n_top_stocks: int, title: str, filename: str):
    fig = make_subplots(
        rows=n_components, 
        cols=2, 
        subplot_titles=sum([[f'PC{i+1} Top {n_top_stocks}', f'PC{i+1} Bottom {n_top_stocks}'] for i in range(n_components)], [])
    )
    loadings = pd.DataFrame(
        pca.components_.T[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    for i in range(n_components):
        pc = f'PC{i+1}'
        top = loadings[pc].nlargest(n_top_stocks)
        bottom = loadings[pc].nsmallest(n_top_stocks)
        fig.add_trace(go.Bar(x=top.index, y=top.values, marker_color='blue'), row=i+1, col=1)
        fig.add_trace(go.Bar(x=bottom.index, y=bottom.values, marker_color='red'), row=i+1, col=2)
        fig.update_xaxes(tickangle=45, row=i+1, col=1)
        fig.update_xaxes(tickangle=45, row=i+1, col=2)
        fig.update_yaxes(title_text="Loading", row=i+1, col=1)
        fig.update_yaxes(title_text="Loading", row=i+1, col=2)
    fig.update_layout(height=300*n_components, width=1500, title_text=title, showlegend=False)
    fig.write_html(os.path.join(PCA_RESULTS_DIR, filename))

def plot_explained_variance_evolution(rolling_results: List[Tuple[pd.Timestamp, Union[PCA, SparsePCA], np.ndarray]], n_components: int, title: str, filename: str):
    dates = [result[0] for result in rolling_results]
    
    # Check if we're dealing with PCA or SparsePCA
    if isinstance(rolling_results[0][1], PCA):
        explained_variances = [result[1].explained_variance_ratio_[:n_components] for result in rolling_results]
    else:  # SparsePCA
        # For SparsePCA, we'll use the absolute sum of each component as a proxy for explained variance
        explained_variances = [np.abs(result[1].components_).sum(axis=1)[:n_components] for result in rolling_results]
        # Normalize to get a ratio
        explained_variances = [ev / ev.sum() for ev in explained_variances]
    
    fig = go.Figure()
    for i in range(n_components):
        fig.add_trace(go.Scatter(x=dates, y=[ev[i] for ev in explained_variances], mode='lines', name=f'PC{i+1}'))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Explained Variance Ratio",
        legend_title="Principal Components"
    )
    fig.write_html(os.path.join(PCA_RESULTS_DIR, filename))

def plot_component_heatmap(pca: PCA, data: pd.DataFrame, n_components: int, title: str, filename: str):
    loadings = pd.DataFrame(
        pca.components_.T[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    
    # Cluster stocks based on their loadings
    kmeans = KMeans(n_clusters=10, random_state=42)
    cluster_labels = kmeans.fit_predict(loadings)
    
    # Sort stocks based on cluster labels
    sorted_indices = np.argsort(cluster_labels)
    sorted_loadings = loadings.iloc[sorted_indices]
    
    fig = go.Figure(data=go.Heatmap(
        z=sorted_loadings.values,
        x=sorted_loadings.columns,
        y=sorted_loadings.index,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Principal Components",
        yaxis_title="Stocks",
        height=2000,
        width=1000
    )
    
    fig.write_html(os.path.join(PCA_RESULTS_DIR, filename))

def fetch_sector_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'Ticker': ticker,
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', 'Unknown')
        }
    except:
        return {'Ticker': ticker, 'Sector': 'Unknown', 'Industry': 'Unknown'}

def get_sector_mapping(stocks):
    if os.path.exists(SECTOR_INFO_FILE):
        sector_info = pd.read_csv(SECTOR_INFO_FILE)
    else:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_sector_info, ticker) for ticker in stocks]
            sector_info = [future.result() for future in tqdm(as_completed(futures), total=len(stocks), desc="Fetching sector info")]
        
        sector_info = pd.DataFrame(sector_info)
        sector_info.to_csv(SECTOR_INFO_FILE, index=False)
    
    # Merge similar sectors
    sector_mapping = {
    'Technology': ['Technology'],
    'Healthcare': ['Healthcare'],
    'Consumer': ['Consumer Cyclical', 'Consumer Defensive'],
    'Industrials': ['Industrials'],
    'Financial Services': ['Financial Services'],
    'Energy': ['Energy'],
    'Communication Services': ['Communication Services'],
    'Real Estate': ['Real Estate'],
    'Utilities': ['Utilities'],
    'Basic Materials': ['Basic Materials']
    }
    
    sector_info['MergedSector'] = sector_info['Sector'].map(
        {sector: merged_sector 
         for merged_sector, sectors in sector_mapping.items() 
         for sector in sectors}
    ).fillna(sector_info['Sector'])
    
    return {sector: group['Ticker'].tolist() 
            for sector, group in sector_info.groupby('MergedSector')}

def plot_sector_heatmaps(sparse_pca: SparsePCA, data: pd.DataFrame, sector_mapping: Dict[str, List[str]], n_components: int, title: str, filename: str):
    loadings = pd.DataFrame(
        sparse_pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    
    # Create a DataFrame with sector information
    sector_df = pd.DataFrame([(ticker, sector) for sector, tickers in sector_mapping.items() for ticker in tickers],
                             columns=['Ticker', 'Sector'])
    sector_df = sector_df.set_index('Ticker')
    
    # Merge loadings with sector information
    loadings_with_sector = loadings.join(sector_df)
    
    # Separate numeric loadings from sector information
    numeric_loadings = loadings_with_sector.drop('Sector', axis=1)
    sectors = loadings_with_sector['Sector']
    
    # Separate positive and negative loadings
    positive_mask = numeric_loadings >= 0
    negative_mask = numeric_loadings < 0
    
    positive_loadings = numeric_loadings[positive_mask].groupby(sectors).sum()
    negative_loadings = numeric_loadings[negative_mask].groupby(sectors).sum()
    
    # Normalize loadings
    positive_loadings = positive_loadings.div(positive_loadings.abs().sum(axis=0), axis=1)
    negative_loadings = negative_loadings.div(negative_loadings.abs().sum(axis=0), axis=1)
    
    # Create figure with two side-by-side subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Positive Loadings", "Negative Loadings"),
                        shared_yaxes=True, horizontal_spacing=0.1)
    
    # Add positive loadings heatmap
    fig.add_trace(
        go.Heatmap(z=positive_loadings.values, x=positive_loadings.columns, y=positive_loadings.index,
                   colorscale='Reds', zmin=0, zmax=positive_loadings.values.max(),
                   colorbar=dict(title="Positive<br>Loading<br>Strength", x=0.45, y=0.5)),
        row=1, col=1
    )
    
    # Add negative loadings heatmap
    fig.add_trace(
        go.Heatmap(z=np.abs(negative_loadings.values), x=negative_loadings.columns, y=negative_loadings.index,
                   colorscale='Blues', zmin=0, zmax=np.abs(negative_loadings.values).max(),
                   colorbar=dict(title="Negative<br>Loading<br>Strength", x=1.0, y=0.5)),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.535,  # Center the title
            y=0.95  # Adjust the vertical position if needed
        ),
        height=600,
        width=1200,
        yaxis=dict(title="Sectors"),
        xaxis=dict(title="Principal Components"),
        xaxis2=dict(title="Principal Components"),
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                showarrow=False,
                text="Interpretation: Darker colors indicate stronger influence of the sector on the principal component.",
                xref="paper",
                yref="paper",
                font=dict(size=12)
            )
        ]
    )
    
    fig.write_html(os.path.join(PCA_RESULTS_DIR, filename))

def save_pca_components(pca: PCA, data: pd.DataFrame, n_components: int, filename: str):
    loadings = pd.DataFrame(
        pca.components_.T[:, :n_components],
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    loadings.to_csv(os.path.join(PCA_RESULTS_DIR, filename))
    logging.info(f"Saved PCA components to {filename}")

def save_sparse_pca_components(sparse_pca: SparsePCA, data: pd.DataFrame, n_components: int, filename: str):
    loadings = pd.DataFrame(
        sparse_pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=data.columns
    )
    loadings.to_csv(os.path.join(PCA_RESULTS_DIR, filename))
    logging.info(f"Saved Sparse PCA components to {filename}")

def save_rolling_window_results(results: List[Tuple[pd.Timestamp, PCA, np.ndarray]], filename: str):
    """Save rolling window PCA results to a CSV file."""
    data = []
    max_components = 0
    max_features = 0

    # First pass: determine the maximum number of components and features
    for timestamp, pca, _ in results:
        max_components = max(max_components, pca.n_components_)
        max_features = max(max_features, pca.n_features_in_)
        
    # Second pass: create rows with consistent size
    for timestamp, pca, _ in results:
        row = [timestamp]
        flat_components = pca.components_.flatten()
        row.extend(flat_components)
        # Pad with zeros if necessary
        row.extend([0] * (max_components * max_features - len(flat_components)))
        data.append(row)
    
    columns = ['timestamp'] + [f'PC{i+1}_Stock{j+1}' for i in range(max_components) for j in range(max_features)]
    df = pd.DataFrame(data, columns=columns)
    df.set_index('timestamp', inplace=True)
    df.to_csv(os.path.join(PCA_RESULTS_DIR, filename))
    logging.info(f"Saved rolling window PCA results to {filename}")
    logging.info(f"Max components: {max_components}, Max features: {max_features}")

def save_rolling_window_sparse_pca_results(results: List[Tuple[pd.Timestamp, SparsePCA, np.ndarray]], filename: str):
    """Save rolling window Sparse PCA results to a CSV file."""
    data = []
    max_components = 0
    max_features = 0

    # First pass: determine the maximum number of components and features
    for timestamp, sparse_pca, _ in results:
        max_components = max(max_components, sparse_pca.n_components)
        max_features = max(max_features, sparse_pca.n_features_in_)
        
    # Second pass: create rows with consistent size
    for timestamp, sparse_pca, _ in results:
        row = [timestamp]
        flat_components = sparse_pca.components_.flatten()
        row.extend(flat_components)
        # Pad with zeros if necessary
        row.extend([0] * (max_components * max_features - len(flat_components)))
        data.append(row)
    
    columns = ['timestamp'] + [f'PC{i+1}_Stock{j+1}' for i in range(max_components) for j in range(max_features)]
    df = pd.DataFrame(data, columns=columns)
    df.set_index('timestamp', inplace=True)
    df.to_csv(os.path.join(PCA_RESULTS_DIR, filename))
    logging.info(f"Saved rolling window Sparse PCA results to {filename}")
    logging.info(f"Max components: {max_components}, Max features: {max_features}")

def save_sector_mapping(sector_mapping: Dict[str, List[str]], filename: str):
    """Save sector mapping to a CSV file."""
    df = pd.DataFrame([(stock, sector) for sector, stocks in sector_mapping.items() for stock in stocks],
                      columns=['Stock', 'Sector'])
    df.to_csv(os.path.join(PCA_RESULTS_DIR, filename), index=False)
    logging.info(f"Saved sector mapping to {filename}")

def main():
    os.makedirs(PCA_RESULTS_DIR, exist_ok=True)

    # Load preprocessed data
    stocks_data = load_preprocessed_data(PREPROCESSED_STOCKS_FILE)
    etfs_data = load_preprocessed_data(PREPROCESSED_ETFS_FILE)
    
    if stocks_data.empty or etfs_data.empty:
        logging.error("Failed to load preprocessed data. Exiting.")
        return

    # Get sector mapping
    sector_mapping = get_sector_mapping(stocks_data.columns)
    save_sector_mapping(sector_mapping, 'sector_mapping.csv')

    # 2.1 Full Period PCA
    full_pca, full_pca_result = full_period_pca(stocks_data)
    plot_cumulative_explained_variance(full_pca, "Full Period PCA - Cumulative Explained Variance", "full_period_pca_variance.html")
    plot_top_component_loadings(full_pca, stocks_data, 5, 20, "Full Period PCA - Top 20 Stocks per Component", "full_period_pca_top_loadings.html")
    plot_component_heatmap(full_pca, stocks_data, 10, "Full Period PCA - Top 10 Components Heatmap", "full_period_pca_heatmap.html")
    save_pca_components(full_pca, stocks_data, 10, 'pca_components.csv')

    # 2.2 Rolling Window PCA
    # 10-Day Window (assuming 2-minute data, 1950 data points for 10 trading days)
    ten_day_results = rolling_window_pca(stocks_data, window_size=1950, step_size=195)
    plot_explained_variance_evolution(ten_day_results, 5, "10-Day Rolling Window PCA - Top 5 Components", "ten_day_rolling_pca_evolution.html")
    save_rolling_window_results(ten_day_results, 'rolling_window_pca_results.csv')

    # New: Rolling Window Sparse PCA
    n_sparse_components = 10
    ten_day_sparse_results = rolling_window_sparse_pca(stocks_data, window_size=1950, step_size=195, n_components=n_sparse_components)
    plot_explained_variance_evolution(ten_day_sparse_results, 5, "10-Day Rolling Window Sparse PCA - Top 5 Components", "ten_day_rolling_sparse_pca_evolution.html")
    save_rolling_window_sparse_pca_results(ten_day_sparse_results, 'rolling_window_sparse_pca_results.csv')

    # 2.3 Sparse PCA
    sparse_pca, sparse_pca_result = perform_sparse_pca(stocks_data.values, n_sparse_components)
    plot_top_component_loadings(sparse_pca, stocks_data, n_sparse_components, 10, "Sparse PCA - Top 10 Stocks per Component", "sparse_pca_top_loadings.html")
    save_sparse_pca_components(sparse_pca, stocks_data, n_sparse_components, 'sparse_pca_components.csv')

    # 2.4 Sector-based PCA
    # Plot sector heatmaps for Sparse PCA
    plot_sector_heatmaps(sparse_pca, stocks_data, sector_mapping, n_sparse_components, 
                         "Sparse PCA - Sector Loadings Heatmap", "sparse_pca_sector_heatmaps.html")

    # Plot sector heatmaps for Full PCA as reference check
    n_components = n_sparse_components
    pca_full, pca_full_result = perform_pca(stocks_data.values, n_components)
    plot_sector_heatmaps(pca_full, stocks_data, sector_mapping, n_components, 
                         "Full PCA - Sector Loadings Heatmap", "full_pca_sector_heatmaps.html")

    # Plot cumulative explained variance for each sector
    sector_pca_results = sector_based_pca(stocks_data, sector_mapping)
    for sector, (pca, _) in sector_pca_results.items():
        plot_cumulative_explained_variance(pca, f"{sector} Sector PCA - Cumulative Explained Variance", f"{sector.lower()}_sector_pca_variance.html")

    logging.info("PCA analysis completed successfully")

if __name__ == "__main__":
    main()