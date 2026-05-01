"""
Plotting configuration: colors, styles, and utility functions.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SAFE_SAMPLE_INTERVAL, DANGER_LABEL_SCALE, SCATTER_BIN_INTERVAL

# Color palettes
COLORS = {
    'safe': '#2ecc71',      # Green
    'danger': '#e74c3c',    # Red
    'danger_dark': '#c0392b',  # Dark red
}

# Bar chart colors (avoiding white)
BAR_COLORS_PAIRED = ['#2ecc71', '#e74c3c']  # Green, Red
BAR_COLORS_SINGLE = '#e74c3c'  # Dark red for danger rates

# Colormaps
COLORMAP_DANGER = 'RdYlGn_r'  # Red (danger) to Green (safe)

def get_percentile_range(data, lower_pct=5, upper_pct=95):
    """
    Get the range covering lower_pct to upper_pct percentiles.
    This focuses on main distribution while excluding extreme outliers.
    
    Args:
        data: Array-like data
        lower_pct: Lower percentile (default 5 = exclude bottom 5%)
        upper_pct: Upper percentile (default 95 = exclude top 5%)
    
    Returns:
        Tuple (min_val, max_val) for the percentile range
    """
    data_clean = np.asarray(data)[~np.isnan(data)]
    if len(data_clean) == 0:
        return (0, 1)
    min_val = np.percentile(data_clean, lower_pct)
    max_val = np.percentile(data_clean, upper_pct)
    # Add small margin
    margin = (max_val - min_val) * 0.05
    return (min_val - margin, max_val + margin)


def downsample_safe_samples(df, interval=None):
    """
    Downsample safe samples to reduce class imbalance in visualizations.
    Groups consecutive safe frames by (object_id, danger_label) and aggregates them.
    
    For every N safe frames of an object-child pair:
    - Average distance and relative_velocity
    - Count as 1 aggregated sample
    
    Danger frames are always kept (no downsampling).
    
    Args:
        df: DataFrame with columns [distance, relative_velocity, danger_label, object_id, frame_id]
        interval: Downsample interval (config.SAFE_SAMPLE_INTERVAL if None)
    
    Returns:
        DataFrame with downsampled safe samples and all danger samples
    """
    if interval is None:
        interval = SAFE_SAMPLE_INTERVAL
    
    if interval <= 1:
        return df.copy()
    
    # Separate danger and safe samples
    danger_df = df[df['danger_label'] == 1].copy()
    safe_df = df[df['danger_label'] == 0].copy()
    
    if len(safe_df) == 0:
        return df.copy()
    
    # Group safe samples by object_id and aggregate
    # Sort by frame_id to maintain sequence
    safe_df = safe_df.sort_values(['object_id', 'frame_id']).reset_index(drop=True)
    
    # Create groups of N consecutive frames
    safe_df['group_idx'] = safe_df.groupby('object_id').cumcount() // interval
    
    # Aggregate: average numeric features, keep first non-numeric values
    agg_dict = {
        'distance': 'mean',
        'relative_velocity': 'mean',
        'frame_id': 'first',  # Use first frame_id in group
        'video_id': 'first',
        'object_id': 'first',
        'object_type': 'first',
        'interaction': 'first',
        'danger_label': 'first',
    }
    
    # Only aggregate columns that exist
    agg_dict = {k: v for k, v in agg_dict.items() if k in safe_df.columns}
    
    downsampled_safe = safe_df.groupby(['object_id', 'group_idx'], as_index=False).agg(agg_dict)
    downsampled_safe = downsampled_safe.drop(columns=['group_idx'])
    
    # Combine danger and downsampled safe samples
    result_df = pd.concat([danger_df, downsampled_safe], ignore_index=True)
    result_df = result_df.sort_values('frame_id').reset_index(drop=True)
    
    return result_df


def apply_danger_scale(df, scale=None):
    """
    Apply visual weight scaling to danger samples.
    Repeats danger rows to emphasize them in visualizations.
    
    Args:
        df: DataFrame
        scale: Scale factor (config.DANGER_LABEL_SCALE if None)
    
    Returns:
        DataFrame with danger samples repeated scale times
    """
    if scale is None:
        scale = DANGER_LABEL_SCALE
    
    if scale <= 1:
        return df.copy()
    
    # Separate danger and safe
    danger_df = df[df['danger_label'] == 1]
    safe_df = df[df['danger_label'] == 0]
    
    # Repeat danger samples
    danger_df_repeated = pd.concat([danger_df] * int(scale), ignore_index=True)
    
    # Combine and return
    result_df = pd.concat([safe_df, danger_df_repeated], ignore_index=True)
    return result_df


def create_histogram2d_data(df, x_feature='distance', y_feature='relative_velocity', 
                           bin_interval=None):
    """
    Create 2D histogram data by grouping samples into bins based on x_feature intervals.
    Counts samples in each bin to determine color intensity.
    
    Args:
        df: DataFrame with numeric features
        x_feature: Feature for x-axis (default: distance)
        y_feature: Feature for y-axis (default: relative_velocity)
        bin_interval: Size of distance bins (config.SCATTER_BIN_INTERVAL if None)
    
    Returns:
        Dict with bin information:
        {
            'x_bins': list of bin edges,
            'y_data': dict mapping bin index to list of y values,
            'counts': dict mapping bin index to sample count in that bin,
            'bin_interval': interval used
        }
    """
    if bin_interval is None:
        bin_interval = SCATTER_BIN_INTERVAL
    
    data = df[[x_feature, y_feature, 'danger_label']].dropna()
    
    if len(data) == 0:
        return None
    
    # Create bins for x_feature (distance)
    x_min = data[x_feature].min()
    x_max = data[x_feature].max()
    
    # Create bin edges
    n_bins = int(np.ceil((x_max - x_min) / bin_interval))
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    
    # Assign each sample to a bin
    data['bin_idx'] = pd.cut(data[x_feature], bins=x_bins, labels=False, include_lowest=True)
    
    # Group by bin and count
    bin_counts = {}
    y_data = {}
    
    for bin_idx in range(n_bins):
        bin_data = data[data['bin_idx'] == bin_idx]
        bin_counts[bin_idx] = len(bin_data)
        y_data[bin_idx] = bin_data[y_feature].values
    
    return {
        'x_bins': x_bins,
        'y_data': y_data,
        'counts': bin_counts,
        'bin_interval': bin_interval,
        'x_min': x_min,
        'x_max': x_max
    }


def save_plot(output_dir, filename, dpi=150):
    """Save the current matplotlib figure to `output_dir/filename` and report path.

    Ensures the output directory exists, saves the current figure, closes it,
    and prints the absolute path of the saved file for reliable verification.
    """
    fig = plt.gcf()
    path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'  ✗ Failed to save {path}: {e}')
        return path

    if os.path.exists(path):
        print(f'  ✓ Saved: {os.path.abspath(path)}')
    else:
        print(f'  ✗ File missing after save: {os.path.abspath(path)}')
    return path


