import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import SparsePCA
import logging
import os
from typing import List, Dict, Tuple
from datetime import time, datetime, timedelta

# Configuration
logging.basicConfig(
    filename='logs/3.4_intraday_pattern_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
PREPROCESSED_DATA_DIR = 'preprocessed_data'
PCA_RESULTS_DIR = 'pca_results'
INTERPRETATION_RESULTS_DIR = 'interpretation_results'
PREPROCESSED_STOCKS_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'preprocessed_stocks.csv')
SECTOR_MAPPING_FILE = os.path.join(PCA_RESULTS_DIR, 'sector_mapping.csv')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def load_sector_mapping(file_path: str) -> Dict[str, str]:
    """Load sector mapping from CSV file."""
    df = pd.read_csv(file_path)
    sector_mapping = dict(zip(df['Stock'], df['Sector']))
    logging.info(f"Loaded sector mapping for {len(sector_mapping)} stocks")
    return sector_mapping

def group_by_intraday_period(data: pd.DataFrame, n_periods: int = 13) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Group data into intraday periods with time-based labels."""
    market_open = time(9, 30)  # Assuming market opens at 9:30 AM
    market_close = time(16, 0)  # Assuming market closes at 4:00 PM
    trading_minutes = (market_close.hour * 60 + market_close.minute) - (market_open.hour * 60 + market_open.minute)
    period_duration = trading_minutes // n_periods

    def get_period_label(t):
        minutes_since_open = (t.hour * 60 + t.minute) - (market_open.hour * 60 + market_open.minute)
        period_number = minutes_since_open // period_duration
        period_start = (datetime.combine(data.index.date[0], market_open) + timedelta(minutes=period_number * period_duration)).time()
        period_end = (datetime.combine(data.index.date[0], market_open) + timedelta(minutes=(period_number + 1) * period_duration)).time()
        return f"{period_start.strftime('%H:%M')}-{period_end.strftime('%H:%M')}"

    data['period'] = data.index.map(lambda x: get_period_label(x.time()))
    grouped = {period: group.drop('period', axis=1) for period, group in data.groupby('period')}
    periods = sorted(grouped.keys())
    logging.info(f"Grouped data into {len(grouped)} intraday periods")
    return grouped, periods

def perform_sparse_pca(data: pd.DataFrame, n_components: int = 10) -> Tuple[SparsePCA, np.ndarray]:
    """Perform Sparse PCA on the data."""
    sparse_pca = SparsePCA(n_components=n_components, random_state=42)
    sparse_pca_result = sparse_pca.fit_transform(data)
    logging.info(f"Performed Sparse PCA with {n_components} components")
    return sparse_pca, sparse_pca_result

def calculate_sector_loadings(pca: SparsePCA, sector_mapping: Dict[str, str], stocks: List[str]) -> Dict[str, np.ndarray]:
    """Calculate sector loadings for PCA components using only positive loadings."""
    sector_loadings = {sector: np.zeros(pca.n_components_) for sector in set(sector_mapping.values())}
    for i, stock in enumerate(stocks):
        if stock in sector_mapping:
            sector = sector_mapping[stock]
            sector_loadings[sector] += np.maximum(pca.components_[:, i], 0)  # Only positive loadings
    
    # Normalize loadings across sectors for each component
    for i in range(pca.n_components_):
        total = sum(sector_loadings[sector][i] for sector in sector_loadings)
        for sector in sector_loadings:
            sector_loadings[sector][i] /= total if total != 0 else 1
    
    logging.info(f"Calculated sector loadings for {len(sector_loadings)} sectors")
    return sector_loadings

def analyze_intraday_patterns(stock_data: pd.DataFrame, sector_mapping: Dict[str, str]) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """Analyze intraday patterns in stock data."""
    intraday_groups, periods = group_by_intraday_period(stock_data)
    intraday_patterns = {}
    for period, group in intraday_groups.items():
        sparse_pca, _ = perform_sparse_pca(group)
        sector_loadings = calculate_sector_loadings(sparse_pca, sector_mapping, group.columns)
        intraday_patterns[period] = sector_loadings
    logging.info(f"Analyzed intraday patterns for {len(intraday_patterns)} periods")
    return intraday_patterns, periods

def plot_intraday_heatmap(intraday_patterns: Dict[str, Dict[str, np.ndarray]], title: str) -> go.Figure:
    """Create a heatmap of intraday patterns without color scales."""
    periods = list(intraday_patterns.keys())
    sectors = list(intraday_patterns[periods[0]].keys())
    components = list(range(1, len(intraday_patterns[periods[0]][sectors[0]]) + 1))
    
    fig = make_subplots(rows=len(components), cols=1, 
                        subplot_titles=[f"PC{i}" for i in components],
                        shared_xaxes=True,
                        vertical_spacing=0.03)
    
    for i, pc in enumerate(components, 1):
        z = [[intraday_patterns[period][sector][pc-1] for period in periods] for sector in sectors]
        heatmap = go.Heatmap(z=z, x=periods, y=sectors, colorscale='RdBu', zmid=np.mean(z),
                             showscale=False)  # Turn off color bar
        fig.add_trace(heatmap, row=i, col=1)
    
    fig.update_layout(height=300*len(components), width=1200, title_text=title)
    fig.update_xaxes(title_text="Intraday Period", row=len(components), col=1, tickangle=45)
    fig.update_yaxes(title_text="Sectors")
    
    logging.info(f"Created intraday heatmap with {len(components)} components")
    return fig

def analyze_sector_influence(intraday_patterns: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, List[float]]:
    """Analyze how sector influences change within the trading day using positive loadings."""
    sectors = list(intraday_patterns[list(intraday_patterns.keys())[0]].keys())
    sector_influence = {sector: [] for sector in sectors}
    
    for period in intraday_patterns:
        total_influence = sum(np.sum(intraday_patterns[period][sector]) for sector in sectors)
        for sector in sectors:
            sector_influence[sector].append(np.sum(intraday_patterns[period][sector]) / total_influence)
    
    logging.info(f"Analyzed sector influence for {len(sectors)} sectors")
    return sector_influence

def plot_sector_influence(sector_influence: Dict[str, List[float]], periods: List[str], title: str) -> go.Figure:
    """Create a line plot of sector influence changes throughout the day with percentage y-axis."""
    fig = go.Figure()
    
    for sector, influence in sector_influence.items():
        fig.add_trace(go.Scatter(x=periods, y=influence, mode='lines+markers', name=sector))
    
    fig.update_layout(
        title=title,
        xaxis_title="Intraday Period",
        yaxis_title="Average Sector Influence (%)",
        legend_title="Sectors",
        xaxis=dict(tickangle=45),
        yaxis=dict(
            tickformat='.2%',  # Display as percentage with 2 decimal places
            range=[0, max(max(influence) for influence in sector_influence.values()) * 1.1]
        )
    )
    
    logging.info(f"Created sector influence plot with {len(sector_influence)} sectors")
    return fig

def main():
    os.makedirs(INTERPRETATION_RESULTS_DIR, exist_ok=True)

    stock_data = load_data(PREPROCESSED_STOCKS_FILE)
    sector_mapping = load_sector_mapping(SECTOR_MAPPING_FILE)

    if stock_data.empty:
        logging.error("Failed to load necessary data. Exiting.")
        return

    intraday_patterns, periods = analyze_intraday_patterns(stock_data, sector_mapping)

    heatmap = plot_intraday_heatmap(intraday_patterns, "Intraday Patterns of Sector Influence on Sparse PCA Components")
    heatmap_path = os.path.join(INTERPRETATION_RESULTS_DIR, 'intraday_patterns_heatmap.html')
    heatmap.write_html(heatmap_path)
    logging.info(f"Saved intraday patterns heatmap to {heatmap_path}")

    sector_influence = analyze_sector_influence(intraday_patterns)

    influence_plot = plot_sector_influence(sector_influence, periods, "Changes in Sector Influence Throughout the Trading Day")
    influence_plot_path = os.path.join(INTERPRETATION_RESULTS_DIR, 'sector_influence_changes.html')
    influence_plot.write_html(influence_plot_path)
    logging.info(f"Saved sector influence changes plot to {influence_plot_path}")

    logging.info("Intraday pattern analysis completed successfully.")

if __name__ == "__main__":
    main()