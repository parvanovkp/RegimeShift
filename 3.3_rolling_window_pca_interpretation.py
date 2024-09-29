import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from typing import List, Tuple, Dict
from scipy.stats import zscore
from sklearn.cluster import KMeans

# Configuration
logging.basicConfig(
    filename='logs/3.3_rolling_window_pca_interpretation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCA_RESULTS_DIR = os.path.join(BASE_DIR, 'pca_results')
INTERPRETATION_RESULTS_DIR = os.path.join(BASE_DIR, 'interpretation_results')
ROLLING_WINDOW_PCA_FILE = os.path.join(PCA_RESULTS_DIR, 'rolling_window_pca_results.csv')
ROLLING_WINDOW_SPARSE_PCA_FILE = os.path.join(PCA_RESULTS_DIR, 'rolling_window_sparse_pca_results.csv')
FULL_PCA_FILE = os.path.join(PCA_RESULTS_DIR, 'pca_components.csv')
SPARSE_PCA_FILE = os.path.join(PCA_RESULTS_DIR, 'sparse_pca_components.csv')
SECTOR_MAPPING_FILE = os.path.join(PCA_RESULTS_DIR, 'sector_mapping.csv')

# Ensure directories exist
os.makedirs(INTERPRETATION_RESULTS_DIR, exist_ok=True)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path, index_col=0)
        if 'timestamp' in df.index.name.lower():
            df.index = pd.to_datetime(df.index)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def load_sector_mapping(file_path: str) -> Dict[str, str]:
    """Load sector mapping from CSV file."""
    try:
        df = pd.read_csv(file_path)
        mapping = dict(zip(df['Stock'], df['Sector']))
        logging.info(f"Loaded sector mapping for {len(mapping)} stocks")
        return mapping
    except Exception as e:
        logging.error(f"Error loading sector mapping: {e}")
        return {}

def analyze_sector_loading_evolution(rolling_results: pd.DataFrame, sector_mapping: Dict[str, str]) -> pd.DataFrame:
    """Analyze how sector loadings evolve over time."""
    n_components = len([col for col in rolling_results.columns if col.startswith('PC1_')])
    n_stocks = len(rolling_results.columns) // n_components

    sector_loadings = []
    for _, row in rolling_results.iterrows():
        loadings = row.values.reshape(n_components, n_stocks).T
        loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_components)])
        loadings_df['Sector'] = [sector_mapping.get(f'Stock{j+1}', 'Unknown') for j in range(n_stocks)]
        sector_avg_loadings = loadings_df.groupby('Sector').mean()
        sector_loadings.append(sector_avg_loadings)
    
    return pd.concat(sector_loadings, keys=rolling_results.index)

def plot_sector_loading_evolution(sector_loading_evolution: pd.DataFrame, title: str, filename: str):
    """Create a plot showing the evolution of sector loadings."""
    fig = go.Figure()
    for sector in sector_loading_evolution.columns:
        fig.add_trace(go.Scatter(x=sector_loading_evolution.index.get_level_values(0), 
                                 y=sector_loading_evolution[sector],
                                 mode='lines', name=sector))
    
    fig.update_layout(title=title,
                      xaxis_title="Date",
                      yaxis_title="Average Loading",
                      height=600, width=1000)
    fig.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, filename))

def identify_stability_change_periods(sector_loading_evolution: pd.DataFrame, window_size: int = 10) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Identify periods of stability and change in market structure."""
    rolling_std = sector_loading_evolution.groupby(level=0).std().rolling(window=window_size).mean().mean(axis=1)
    threshold = rolling_std.mean() + 2 * rolling_std.std()
    
    change_periods = []
    is_change_period = False
    start_date = None
    
    for date, value in rolling_std.items():
        if value > threshold and not is_change_period:
            is_change_period = True
            start_date = date
        elif value <= threshold and is_change_period:
            is_change_period = False
            change_periods.append((start_date, date, "Change"))
            start_date = date
    
    if is_change_period:
        change_periods.append((start_date, rolling_std.index[-1], "Change"))
    
    # Add stability periods
    all_periods = []
    prev_end = rolling_std.index[0]
    for start, end, period_type in change_periods:
        if start > prev_end:
            all_periods.append((prev_end, start, "Stability"))
        all_periods.append((start, end, period_type))
        prev_end = end
    
    if prev_end < rolling_std.index[-1]:
        all_periods.append((prev_end, rolling_std.index[-1], "Stability"))
    
    return sorted(all_periods)

def compare_with_full_period(rolling_results: pd.DataFrame, full_period_results: pd.DataFrame) -> Dict:
    """Compare rolling window results with full period analysis."""
    n_components = len(full_period_results.columns)
    n_stocks = len(full_period_results)

    full_period_loadings = full_period_results.values.T

    cosine_similarities = []
    for _, row in rolling_results.iterrows():
        rolling_loadings = row.values.reshape(n_components, n_stocks)
        similarity = np.sum(full_period_loadings * rolling_loadings) / (np.linalg.norm(full_period_loadings) * np.linalg.norm(rolling_loadings))
        cosine_similarities.append(similarity)
    
    return {
        'mean_similarity': np.mean(cosine_similarities),
        'std_similarity': np.std(cosine_similarities),
        'min_similarity': np.min(cosine_similarities),
        'max_similarity': np.max(cosine_similarities),
        'similarity_series': pd.Series(cosine_similarities, index=rolling_results.index)
    }

def plot_similarity_comparison(similarity_series: pd.Series, title: str, filename: str):
    """Plot the similarity between rolling window and full period results."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=similarity_series.index, y=similarity_series.values,
                             mode='lines', name='Cosine Similarity'))
    
    fig.update_layout(title=title,
                      xaxis_title="Date",
                      yaxis_title="Cosine Similarity",
                      height=400, width=800)
    fig.write_html(os.path.join(INTERPRETATION_RESULTS_DIR, filename))

def interpret_economic_significance(sector_loading_evolution: pd.DataFrame, stability_change_periods: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> str:
    """Interpret the economic significance of observed changes."""
    interpretation = "Economic Significance Interpretation:\n\n"
    
    for start, end, period_type in stability_change_periods:
        period_data = sector_loading_evolution.loc[start:end]
        dominant_sector = period_data.groupby(level=1).mean().mean().idxmax()
        
        interpretation += f"{period_type} Period ({start.date()} to {end.date()}):\n"
        interpretation += f"- Dominant sector: {dominant_sector}\n"
        
        if period_type == "Change":
            sector_changes = (period_data.groupby(level=1).mean().iloc[-1] - period_data.groupby(level=1).mean().iloc[0]).sort_values(ascending=False)
            interpretation += f"- Largest sector changes:\n"
            for sector, change in sector_changes.head(3).items():
                interpretation += f"  {sector}: {change:.4f}\n"
        
        interpretation += "\n"
    
    return interpretation

def main():
    # Load data
    rolling_pca_results = load_data(ROLLING_WINDOW_PCA_FILE)
    rolling_sparse_pca_results = load_data(ROLLING_WINDOW_SPARSE_PCA_FILE)
    full_pca_results = load_data(FULL_PCA_FILE)
    sparse_pca_results = load_data(SPARSE_PCA_FILE)
    sector_mapping = load_sector_mapping(SECTOR_MAPPING_FILE)

    # Analyze sector loading evolution
    pca_sector_evolution = analyze_sector_loading_evolution(rolling_pca_results, sector_mapping)
    sparse_pca_sector_evolution = analyze_sector_loading_evolution(rolling_sparse_pca_results, sector_mapping)

    # Plot sector loading evolution
    plot_sector_loading_evolution(pca_sector_evolution, "PCA Sector Loading Evolution", "pca_sector_loading_evolution.html")
    plot_sector_loading_evolution(sparse_pca_sector_evolution, "Sparse PCA Sector Loading Evolution", "sparse_pca_sector_loading_evolution.html")

    # Identify periods of stability and change
    pca_stability_change_periods = identify_stability_change_periods(pca_sector_evolution)
    sparse_pca_stability_change_periods = identify_stability_change_periods(sparse_pca_sector_evolution)

    # Compare with full period analysis
    pca_comparison = compare_with_full_period(rolling_pca_results, full_pca_results)
    sparse_pca_comparison = compare_with_full_period(rolling_sparse_pca_results, sparse_pca_results)

    # Plot similarity comparison
    plot_similarity_comparison(pca_comparison['similarity_series'], "PCA Rolling vs Full Period Similarity", "pca_rolling_vs_full_similarity.html")
    plot_similarity_comparison(sparse_pca_comparison['similarity_series'], "Sparse PCA Rolling vs Full Period Similarity", "sparse_pca_rolling_vs_full_similarity.html")

    # Interpret economic significance
    pca_interpretation = interpret_economic_significance(pca_sector_evolution, pca_stability_change_periods)
    sparse_pca_interpretation = interpret_economic_significance(sparse_pca_sector_evolution, sparse_pca_stability_change_periods)

    # Save results
    with open(os.path.join(INTERPRETATION_RESULTS_DIR, 'rolling_window_pca_interpretation.txt'), 'w') as f:
        f.write("Rolling Window PCA Interpretation Results\n\n")
        f.write("PCA Results:\n")
        f.write("Periods of Stability and Change:\n")
        for start, end, period_type in pca_stability_change_periods:
            f.write(f"{period_type}: {start.date()} to {end.date()}\n")
        f.write("\nComparison with Full Period Analysis:\n")
        for key, value in pca_comparison.items():
            if key != 'similarity_series':
                f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(pca_interpretation)
        
        f.write("\n\nSparse PCA Results:\n")
        f.write("Periods of Stability and Change:\n")
        for start, end, period_type in sparse_pca_stability_change_periods:
            f.write(f"{period_type}: {start.date()} to {end.date()}\n")
        f.write("\nComparison with Full Period Analysis:\n")
        for key, value in sparse_pca_comparison.items():
            if key != 'similarity_series':
                f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(sparse_pca_interpretation)

    logging.info("Rolling window PCA interpretation completed successfully.")

if __name__ == "__main__":
    main()