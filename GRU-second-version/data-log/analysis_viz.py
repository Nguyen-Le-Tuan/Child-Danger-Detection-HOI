"""
analysis_viz.py - Main orchestration file

Comprehensive data analysis and visualization for danger dataset patterns.
Combines static features (bbox, distance, velocity), semantic features (object, interaction),
and CLIP embeddings across all videos.

Refactored into modular components for easier maintenance and extension.
"""

import os
import json
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import DANGER_LABEL_SCALE, SAFE_SAMPLE_INTERVAL, SCATTER_BIN_INTERVAL
from utils.data_loader import load_all_danger_data, load_clip_embeddings
from utils.plotting_config import downsample_safe_samples, apply_danger_scale
from plots.static_features import (
    plot_histograms, plot_box_plots, plot_violin_plots, 
    plot_2d_scatter, plot_correlation_heatmap, plot_feature_vs_time
)
from plots.static_features import plot_mean_std_over_time, plot_temporal_heatmap
from plots.semantic_features import (
    plot_object_interaction_bars, plot_normalized_bars, plot_semantic_heatmap
)
from plots.embeddings import (
    plot_pca_visualization, plot_tsne_visualization,
    plot_pca_object_embeddings, plot_tsne_object_embeddings,
    plot_pca_object_type_embeddings, plot_tsne_object_type_embeddings,
    plot_pca_interaction_clip_embeddings, plot_tsne_interaction_clip_embeddings
)

OUTPUT_BASE = './data-log/output'


def run_analysis(fine_data_dir='./fine_data', output_subdir=None):
    """Run complete analysis and save results."""
    
    if output_subdir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_subdir = os.path.join(OUTPUT_BASE, timestamp)
    
    os.makedirs(output_subdir, exist_ok=True)
    print(f'[Analysis] Output directory: {output_subdir}')
    
    print(f'[Data] Loading CSV files...')
    df = load_all_danger_data(fine_data_dir)
    
    if df is None:
        print('[!] Failed to load data')
        return
    
    print(f'  -> Loaded {len(df)} rows from {df["video_id"].nunique()} videos')
    
    video_list = sorted(df['video_id'].unique())
    with open(os.path.join(output_subdir, 'videos.txt'), 'w') as f:
        f.write('\n'.join(video_list))
    print(f'  ✓ Saved videos.txt ({len(video_list)} videos)')
    
    print(f'[Plots] Generating static feature visualizations...')
    plot_histograms(df, output_subdir)
    plot_box_plots(df, output_subdir)
    plot_violin_plots(df, output_subdir)
    plot_2d_scatter(df, output_subdir)
    plot_correlation_heatmap(df, output_subdir)
    plot_feature_vs_time(df, output_subdir)
    # New temporal analyses
    plot_mean_std_over_time(df, output_subdir, feature='distance')
    plot_temporal_heatmap(df, output_subdir, feature='distance')
    
    print(f'[Plots] Generating semantic feature visualizations...')
    plot_object_interaction_bars(df, output_subdir)
    plot_normalized_bars(df, output_subdir)
    plot_semantic_heatmap(df, output_subdir)
    
    print(f'[Data] Loading embeddings...')
    embeddings_df = load_clip_embeddings(fine_data_dir)
    
    if embeddings_df is not None:
        print(f'  -> Loaded {len(embeddings_df)} embedding samples')
        print(f'[Plots] Generating INTERACTION embedding visualizations...')
        plot_pca_visualization(embeddings_df, output_subdir)
        plot_tsne_visualization(embeddings_df, output_subdir)
    
    print(f'[Plots] Generating OBJECT embedding visualizations...')
    plot_pca_object_embeddings(df, output_subdir)
    plot_tsne_object_embeddings(df, output_subdir)
    
    print(f'[Plots] Generating OBJECT_TYPE text embedding visualizations...')
    plot_pca_object_type_embeddings(df, output_subdir)
    plot_tsne_object_type_embeddings(df, output_subdir)
    
    print(f'[Plots] Generating INTERACTION CLIP text embedding visualizations...')
    plot_pca_interaction_clip_embeddings(df, output_subdir)
    plot_tsne_interaction_clip_embeddings(df, output_subdir)
    
    # Calculate processed data statistics for summary
    df_processed = downsample_safe_samples(df)
    df_processed = apply_danger_scale(df_processed)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'videos': video_list,
        'original_data': {
            'total_rows': len(df),
            'danger_count': int(df['danger_label'].sum()),
            'safe_count': int((1 - df['danger_label']).sum()),
            'danger_rate': float(df['danger_label'].mean())
        },
        'processed_data': {
            'total_rows': len(df_processed),
            'danger_count': int(df_processed['danger_label'].sum()),
            'safe_count': int((1 - df_processed['danger_label']).sum()),
            'danger_rate': float(df_processed['danger_label'].mean())
        },
        'config_parameters': {
            'DANGER_LABEL_SCALE': float(DANGER_LABEL_SCALE),
            'SAFE_SAMPLE_INTERVAL': int(SAFE_SAMPLE_INTERVAL),
            'SCATTER_BIN_INTERVAL': float(SCATTER_BIN_INTERVAL)
        },
        'total_embeddings': len(embeddings_df) if embeddings_df is not None else 0
    }
    
    with open(os.path.join(output_subdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\n[Summary - Original Data]')
    print(f'  Total rows: {summary["original_data"]["total_rows"]}')
    print(f'  Danger: {summary["original_data"]["danger_count"]} ({summary["original_data"]["danger_rate"]:.1%})')
    print(f'  Safe: {summary["original_data"]["safe_count"]} ({1-summary["original_data"]["danger_rate"]:.1%})')
    
    print(f'\n[Summary - Processed Data]')
    print(f'  Total rows: {summary["processed_data"]["total_rows"]}')
    print(f'  Danger: {summary["processed_data"]["danger_count"]} ({summary["processed_data"]["danger_rate"]:.1%})')
    print(f'  Safe: {summary["processed_data"]["safe_count"]} ({1-summary["processed_data"]["danger_rate"]:.1%})')
    
    print(f'\n[Summary - Configuration]')
    print(f'  DANGER_LABEL_SCALE: {DANGER_LABEL_SCALE}')
    print(f'  SAFE_SAMPLE_INTERVAL: {SAFE_SAMPLE_INTERVAL}')
    print(f'  SCATTER_BIN_INTERVAL: {SCATTER_BIN_INTERVAL}')
    print(f'  ✓ Analysis complete: {output_subdir}')
    
    return output_subdir


if __name__ == '__main__':
    fine_data_dir = sys.argv[1] if len(sys.argv) > 1 else './fine_data'
    run_analysis(fine_data_dir)
