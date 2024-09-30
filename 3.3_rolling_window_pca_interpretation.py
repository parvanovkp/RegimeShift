import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
import logging
import os

# Configuration
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/3.3_rolling_window_pca_interpretation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
PCA_RESULTS_DIR = 'pca_results'
PREPROCESSED_DATA_DIR = 'preprocessed_data'
OUTPUT_DIR = 'rolling_window_analysis'
N_COMPONENTS = 10

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    logging.info(f"Loaded data from {file_path} with shape {df.shape}")
    return df

def load_sector_mapping(file_path: str) -> Dict[str, str]:
    """Load sector mapping from CSV file."""
    df = pd.read_csv(file_path)
    sector_mapping = dict(zip(df['Stock'], df['Sector']))
    logging.info(f"Loaded sector mapping for {len(sector_mapping)} stocks")
    return sector_mapping

def extract_first_n_components(data: pd.DataFrame, n_components: int, n_stocks: int) -> List[np.ndarray]:
    """Extract the first n components from the rolling window data."""
    reshaped_data = []
    for _, row in data.iterrows():
        component_data = []
        for i in range(n_components):
            component_columns = [f'PC{i+1}_Stock{j+1}' for j in range(n_stocks)]
            component_data.append(row[component_columns].values)
        reshaped_data.append(np.array(component_data))
    return reshaped_data

def calculate_sector_loadings(loadings: np.ndarray, sector_mapping: Dict[str, str], stocks: List[str]) -> Dict[str, np.ndarray]:
    """Calculate weighted percentage of sector loadings for each component."""
    sector_loadings = {sector: np.zeros(loadings.shape[0]) for sector in set(sector_mapping.values())}
    for i in range(loadings.shape[0]):  # For each component
        positive_loadings = np.maximum(loadings[i], 0)
        total_positive = np.sum(positive_loadings)
        if total_positive > 0:
            for j, stock in enumerate(stocks):
                if stock not in sector_mapping:
                    raise ValueError(f"Stock {stock} is missing from the sector mapping.")
                sector = sector_mapping[stock]
                sector_loadings[sector][i] += positive_loadings[j] / total_positive
    return sector_loadings

def create_sector_heatmap(sector_loadings: Dict[str, np.ndarray], title: str, colorbar_x: float, hide_colorbar: bool = False) -> go.Figure:
    """Create a heatmap of sector loadings with an option to hide the color bar."""
    sectors = list(sector_loadings.keys())
    components = [f'PC{i+1}' for i in range(sector_loadings[sectors[0]].shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=np.array([sector_loadings[sector] for sector in sectors]),
        x=components,
        y=sectors,
        colorscale='burg',
        zmax=1.0,
        zmin=0.0,
        showscale=not hide_colorbar,  # Use showscale to toggle color bar visibility
        colorbar=dict(
            title="Sector Loadings",
            len=0.26,               # Match the height of the heatmap
            thickness=15,           # Control thickness of the color bar
            x=colorbar_x,           # Position the color bar
            y=0.55,                 # Vertically center the color bar
            yanchor="middle"
        ) if not hide_colorbar else None  # If hiding, set colorbar to None
    ))
    
    fig.update_layout(title=title, xaxis_title='Principal Components', yaxis_title='Sectors')
    return fig

def calculate_similarity(prev_loadings: Dict[str, np.ndarray], curr_loadings: Dict[str, np.ndarray]) -> float:
    """Calculate cosine similarity between two sets of loadings."""
    prev_flat = np.concatenate([prev_loadings[sector] for sector in sorted(prev_loadings.keys())])
    curr_flat = np.concatenate([curr_loadings[sector] for sector in sorted(curr_loadings.keys())])
    return 1 - cosine(prev_flat, curr_flat)

def create_stability_plot(similarities: List[float], dates: List[pd.Timestamp]) -> go.Figure:
    """Create a plot of market stability over time."""
    fig = go.Figure(data=go.Scatter(x=dates, y=similarities, mode='lines+markers', name="Market Stability"))
    fig.update_layout(title='Market Stability Over Time', xaxis_title='Date', yaxis_title='Similarity')
    return fig

def compare_to_full_period(rolling_loadings: Dict[str, np.ndarray], full_period_loadings: Dict[str, np.ndarray]) -> float:
    """Compare rolling window loadings to full period loadings."""
    rolling_flat = np.concatenate([rolling_loadings[sector] for sector in sorted(rolling_loadings.keys())])
    full_flat = np.concatenate([full_period_loadings[sector] for sector in sorted(full_period_loadings.keys())])
    return 1 - cosine(rolling_flat, full_flat)

def create_deviation_plot(deviations: List[float], dates: List[pd.Timestamp]) -> go.Figure:
    """Create a plot of deviations from full period analysis."""
    fig = go.Figure(data=go.Scatter(x=dates, y=deviations, mode='lines+markers', name="Deviation from Full Period"))
    fig.update_layout(title='Deviation from Full Period Analysis', xaxis_title='Date', yaxis_title='Similarity')
    return fig

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    rolling_results = load_data(os.path.join(PCA_RESULTS_DIR, 'rolling_window_pca_results.csv'))
    sparse_rolling_results = load_data(os.path.join(PCA_RESULTS_DIR, 'rolling_window_sparse_pca_results.csv'))
    full_period_results = load_data(os.path.join(PCA_RESULTS_DIR, 'pca_components.csv'))
    sparse_full_period_results = load_data(os.path.join(PCA_RESULTS_DIR, 'sparse_pca_components.csv'))
    sector_mapping = load_sector_mapping(os.path.join(PCA_RESULTS_DIR, 'sector_mapping.csv'))

    # Determine number of stocks
    n_stocks = len(full_period_results)
    logging.info(f"Number of stocks: {n_stocks}")

    # Extract first 10 components for full PCA and sparse PCA
    rolling_data = extract_first_n_components(rolling_results, N_COMPONENTS, n_stocks)
    sparse_rolling_data = extract_first_n_components(sparse_rolling_results, N_COMPONENTS, n_stocks)

    # Calculate sector loadings for each window
    stocks = full_period_results.index.tolist()
    rolling_sector_loadings = [calculate_sector_loadings(window, sector_mapping, stocks) for window in rolling_data]
    sparse_rolling_sector_loadings = [calculate_sector_loadings(window, sector_mapping, stocks) for window in sparse_rolling_data]

    # Create sector heatmaps for each window
    for i, (loadings, sparse_loadings) in enumerate(zip(rolling_sector_loadings, sparse_rolling_sector_loadings)):
        fig = create_sector_heatmap(loadings, f'Full PCA Sector Loadings - Window {i+1}', colorbar_x=1.0)
        fig.write_html(os.path.join(OUTPUT_DIR, f'full_pca_sector_heatmap_window_{i+1}.html'))
        
        sparse_fig = create_sector_heatmap(sparse_loadings, f'Sparse PCA Sector Loadings - Window {i+1}', colorbar_x=1.2)
        sparse_fig.write_html(os.path.join(OUTPUT_DIR, f'sparse_pca_sector_heatmap_window_{i+1}.html'))

    # Calculate stability
    similarities = [calculate_similarity(rolling_sector_loadings[i], rolling_sector_loadings[i+1]) 
                    for i in range(len(rolling_sector_loadings)-1)]
    dates = rolling_results.index[1:]  # Exclude the first date as we start from the second window for similarity
    stability_fig = create_stability_plot(similarities, dates)
    stability_fig.write_html(os.path.join(OUTPUT_DIR, 'market_stability.html'))

    # Compare with full period analysis
    full_period_loadings = calculate_sector_loadings(full_period_results.iloc[:, :N_COMPONENTS].values.T, sector_mapping, stocks)
    deviations = [compare_to_full_period(window_loadings, full_period_loadings) for window_loadings in rolling_sector_loadings]
    deviation_fig = create_deviation_plot(deviations, rolling_results.index)
    deviation_fig.write_html(os.path.join(OUTPUT_DIR, 'full_period_deviation.html'))

    # Create an interactive dashboard
    dashboard = make_subplots(rows=3, cols=2, subplot_titles=('Market Stability', 'Deviation from Full Period',
                                                              'Full PCA Sector Loadings', 'Sparse PCA Sector Loadings'),
                              row_heights=[0.3, 0.3, 0.4],  # Control row heights
                              horizontal_spacing=0.22)  # Increase the space between columns

    dashboard.add_trace(stability_fig.data[0], row=1, col=1)
    dashboard.add_trace(deviation_fig.data[0], row=1, col=2)
    dashboard.add_trace(create_sector_heatmap(rolling_sector_loadings[0], '', colorbar_x=1.0).data[0], row=2, col=1)
    dashboard.add_trace(create_sector_heatmap(sparse_rolling_sector_loadings[0], '', colorbar_x=1.2, hide_colorbar=True).data[0], row=2, col=2)

    dashboard.update_layout(
    height=950,  # Adjust height to fit content
    width=1100,  # You can tweak the width as needed
    autosize=True,  # Automatically size to fit the div/container
    title={
        'text': "Rolling Window PCA Analysis Dashboard",
        'x': 0.5,  # Center the title horizontally
        'xanchor': 'center',  # Set anchor to the center
        'yanchor': 'top'  # Optionally, set vertical anchor
    },
    margin=dict(l=50, r=50, t=100, b=50),  # Adjust margins to remove excess space
    xaxis=dict(domain=[0, 0.48]),  # Center first column
    xaxis2=dict(domain=[0.52, 1]),  # Center second column
    )

    # Add slider for time windows (remove play/pause buttons)
    steps = []
    for i in range(len(rolling_sector_loadings)):
        step = dict(
            method="update",
            args=[{"z": [None, None, 
                         [rolling_sector_loadings[i][sector] for sector in sorted(rolling_sector_loadings[i].keys())],
                         [sparse_rolling_sector_loadings[i][sector] for sector in sorted(sparse_rolling_sector_loadings[i].keys())]
                        ]},
                  {"title": f"Rolling Window PCA Analysis Dashboard - Window {i+1}"}],
            label=f"Window {i+1}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time Window: "},
        pad={"t": 50},
        y=.45,
        steps=steps
    )]

    dashboard.update_layout(sliders=sliders)

    dashboard.write_html(os.path.join(OUTPUT_DIR, 'interactive_dashboard.html'))

    logging.info("Analysis completed successfully. Results saved in the 'rolling_window_analysis' directory.")

if __name__ == "__main__":
    main()
