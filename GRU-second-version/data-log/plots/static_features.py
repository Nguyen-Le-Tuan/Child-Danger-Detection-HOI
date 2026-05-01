"""
Static feature visualization module (histograms, scatter, boxplots, etc.)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from utils.plotting_config import COLORS, get_percentile_range, downsample_safe_samples, apply_danger_scale, create_histogram2d_data, save_plot


def plot_histograms(df, output_dir):
    """1. Histogram by label: overlaid density for safe/danger with percentile scaling and downsampling."""
    features = ['distance', 'relative_velocity']
    
    # Apply downsampling and scaling to reduce class imbalance
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)
    
    fig, axes = plt.subplots(1, len(features), figsize=(14, 5))
    
    for idx, feature in enumerate(features):
        if feature not in data_processed.columns:
            continue
        
        ax = axes[idx] if len(features) > 1 else axes
        
        # Remove NaN values
        data = data_processed[[feature, 'danger_label']].dropna()
        
        safe_data = data[data['danger_label'] == 0][feature]
        danger_data = data[data['danger_label'] == 1][feature]
        
        # Get percentile-based range (5th-95th) to focus on main distribution
        min_val, max_val = get_percentile_range(data[feature])
        
        ax.hist(safe_data, bins=30, alpha=0.6, label='Safe', density=True, 
                color=COLORS['safe'], range=(min_val, max_val))
        ax.hist(danger_data, bins=30, alpha=0.6, label='Danger', density=True, 
                color=COLORS['danger'], range=(min_val, max_val))
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'Histogram: {feature}\n(Downsampled & Scaled, n={len(data)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(output_dir, '01_histograms.png', dpi=150)


def plot_box_plots(df, output_dir):
    """2. Box plot: feature distribution by label with downsampling and scaling."""
    features = ['distance', 'relative_velocity']
    
    # Apply downsampling and scaling
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)
    
    fig, axes = plt.subplots(1, len(features), figsize=(12, 4))
    
    for idx, feature in enumerate(features):
        if feature not in data_processed.columns:
            continue
        
        ax = axes[idx] if len(features) > 1 else axes
        
        data = data_processed[[feature, 'danger_label']].dropna()
        data['Label'] = data['danger_label'].apply(lambda x: 'Safe' if x == 0 else 'Danger')
        
        sns.boxplot(data=data, x='Label', y=feature, ax=ax, 
                   palette=[COLORS['safe'], COLORS['danger']])
        ax.set_title(f'Box Plot: {feature} (Downsampled & Scaled)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot(output_dir, '02_boxplots.png', dpi=150)


def plot_violin_plots(df, output_dir):
    """3. Violin plot: density distribution by label with downsampling and scaling."""
    features = ['distance', 'relative_velocity']
    
    # Apply downsampling and scaling
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)
    
    fig, axes = plt.subplots(1, len(features), figsize=(12, 4))
    
    for idx, feature in enumerate(features):
        if feature not in data_processed.columns:
            continue
        
        ax = axes[idx] if len(features) > 1 else axes
        
        data = data_processed[[feature, 'danger_label']].dropna()
        data['Label'] = data['danger_label'].apply(lambda x: 'Safe' if x == 0 else 'Danger')
        
        sns.violinplot(data=data, x='Label', y=feature, ax=ax, 
                      palette=[COLORS['safe'], COLORS['danger']])
        ax.set_title(f'Violin Plot: {feature} (Downsampled & Scaled)')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot(output_dir, '03_violinplots.png', dpi=150)


def plot_2d_scatter(df, output_dir):
    """
    4. 2D Histogram Scatter: distance vs velocity with sample count coloring.
    Groups samples by distance intervals, colors represent sample count intensity.
    """
    # Apply downsampling and scaling
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)
    
    data = data_processed[['distance', 'relative_velocity', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No data for 2D scatter')
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get percentile ranges for axis limits
    x_range = get_percentile_range(data['distance'])
    y_range = get_percentile_range(data['relative_velocity'])
    
    # Create histogram2d data grouped by distance intervals
    hist2d_data = create_histogram2d_data(data, x_feature='distance', y_feature='relative_velocity')
    
    if hist2d_data is None:
        print(f'  ⚠ Failed to create histogram2d data')
        return
    
    x_bins = hist2d_data['x_bins']
    y_data = hist2d_data['y_data']
    counts = hist2d_data['counts']
    
    # Normalize counts for color mapping (0-1 range)
    max_count = max(counts.values()) if counts else 1
    norm = plt.Normalize(vmin=0, vmax=max_count)
    cmap = cm.get_cmap('YlOrRd')  # Yellow to Red (darker = more samples)
    
    # Plot each distance bin with color intensity based on sample count
    for bin_idx, y_vals in y_data.items():
        if len(y_vals) == 0:
            continue
        
        bin_count = counts[bin_idx]
        color_intensity = norm(bin_count)
        color = cmap(color_intensity)
        
        # Get x position (midpoint of bin)
        if bin_idx < len(x_bins) - 1:
            x_pos = (x_bins[bin_idx] + x_bins[bin_idx + 1]) / 2
        else:
            x_pos = x_bins[bin_idx]
        
        # Add jitter for better visibility
        x_jitter = np.random.normal(x_pos, hist2d_data['bin_interval'] * 0.05, len(y_vals))
        
        ax.scatter(x_jitter, y_vals, c=[color] * len(y_vals), 
                  s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add colorbar for sample count legend
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f'Sample Count (Bin Interval: {hist2d_data["bin_interval"]}m)', fontsize=11)
    
    # Set axis limits to percentile range
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Relative Velocity (m/s)', fontsize=12)
    ax.set_title(f'2D Histogram Scatter: Distance vs Velocity\nColor intensity = Sample count per bin (n={len(data)})', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    save_plot(output_dir, '04_scatter_2d.png', dpi=150)


def plot_correlation_heatmap(df, output_dir):
    """5. Correlation heatmap for numeric features."""
    numeric_features = ['distance', 'relative_velocity', 'danger_label']
    
    # Filter to available columns
    available = [f for f in numeric_features if f in df.columns]
    
    if len(available) < 2:
        print(f'  ⚠ Not enough numeric features for correlation, skipping')
        return
    
    data = df[available].dropna()
    corr = data.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
    
    ax.set_title('Correlation Heatmap: Numeric Features')
    
    plt.tight_layout()
    save_plot(output_dir, '05_correlation_heatmap.png', dpi=150)


def plot_feature_vs_time(df, output_dir):
    """6. Feature vs time: distance and velocity colored by danger label."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    features = ['distance', 'relative_velocity']
    
    for idx, feature in enumerate(features):
        if feature not in df.columns:
            continue
        
        ax = axes[idx]
        
        data = df[['frame_id', feature, 'danger_label']].dropna().sort_values('frame_id')
        
        # Plot safe and danger separately
        safe = data[data['danger_label'] == 0]
        danger = data[data['danger_label'] == 1]
        
        ax.scatter(safe['frame_id'], safe[feature], alpha=0.5, label='Safe', 
                  color=COLORS['safe'], s=10)
        ax.scatter(danger['frame_id'], danger[feature], alpha=0.5, label='Danger', 
                  color=COLORS['danger'], s=10)
        
        ax.set_xlabel('Frame ID')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} vs Frame ID')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot(output_dir, '06_feature_vs_time.png', dpi=150)


def plot_mean_std_over_time(df, output_dir, feature='distance'):
    """Mean ± Std over Time (per label).

    Align time to start of each video (relative frame), compute mean and std
    per relative frame grouped by `danger_label`, and plot mean curves with
    shaded ± std bands for safe vs danger.
    """
    if feature not in df.columns:
        print(f'  ⚠ Feature {feature} missing, skipping mean/std over time')
        return

    # Align frames per video
    data = df[['video_id', 'frame_id', 'danger_label', feature]].dropna().copy()
    data['rel_frame'] = data.groupby('video_id')['frame_id'].transform(lambda x: x - x.min())

    # Aggregate mean and std per relative frame and label
    agg = data.groupby(['rel_frame', 'danger_label'])[feature].agg(['mean', 'std', 'count']).reset_index()

    # Pivot for plotting
    labels = sorted(agg['danger_label'].unique())
    plt.figure(figsize=(12, 6))
    for lbl in labels:
        sub = agg[agg['danger_label'] == lbl]
        if len(sub) == 0:
            continue
        x = sub['rel_frame']
        y = sub['mean']
        yerr = sub['std'].fillna(0)
        label_name = 'Danger' if lbl == 1 else 'Safe'
        plt.plot(x, y, label=label_name, color=COLORS['danger'] if lbl == 1 else COLORS['safe'])
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=COLORS['danger'] if lbl == 1 else COLORS['safe'])

    plt.xlabel('Relative Frame')
    plt.ylabel(CONFIG_FEATURE_LABEL(feature))
    plt.title(f'Mean ± Std Over Time (aligned) - {feature}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(output_dir, f'06b_mean_std_{feature}_over_time.png', dpi=150)


def plot_temporal_heatmap(df, output_dir, feature='distance'):
    """Temporal Heatmap: rows = video, cols = time windows, value = mean feature.

    Uses `SAFE_SAMPLE_INTERVAL` from plotting_config as window size (frames).
    """
    from utils.plotting_config import SAFE_SAMPLE_INTERVAL

    if feature not in df.columns:
        print(f'  ⚠ Feature {feature} missing, skipping temporal heatmap')
        return

    data = df[['video_id', 'frame_id', feature]].dropna().copy()
    # Align frames per video
    data['rel_frame'] = data.groupby('video_id')['frame_id'].transform(lambda x: x - x.min())

    window = max(1, int(SAFE_SAMPLE_INTERVAL))
    data['window_idx'] = (data['rel_frame'] // window).astype(int)

    pivot = data.groupby(['video_id', 'window_idx'])[feature].mean().unstack(fill_value=np.nan)

    # Sort videos for consistent ordering
    pivot = pivot.reindex(sorted(pivot.index))

    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(pivot))))
    sns.heatmap(pivot, cmap='viridis', ax=ax, cbar_kws={'label': CONFIG_FEATURE_LABEL(feature)})
    ax.set_xlabel(f'Time Window (size={window} frames)')
    ax.set_ylabel('Video ID')
    ax.set_title(f'Temporal Heatmap: {feature} (per video windows)')
    plt.tight_layout()
    save_plot(output_dir, f'06c_temporal_heatmap_{feature}.png', dpi=150)


def CONFIG_FEATURE_LABEL(feature):
    """Helper to return xlabel from config if available."""
    try:
        from config import NUMERIC_FEATURES
        return NUMERIC_FEATURES.get(feature, {}).get('xlabel', feature)
    except Exception:
        return feature
