"""
Semantic feature visualization module (object/interaction analysis)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.plotting_config import (
    COLORS, BAR_COLORS_PAIRED, BAR_COLORS_SINGLE, COLORMAP_DANGER,
    downsample_safe_samples, apply_danger_scale, save_plot
)


def plot_object_interaction_bars(df, output_dir):
    """1. Bar chart: frequency of object/interaction by label with downsampling."""
    # Apply downsampling to balance class distribution
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Object type distribution
    if 'object_type' in data_processed.columns:
        ax = axes[0]
        obj_counts = data_processed.groupby(['object_type', 'danger_label']).size().unstack(fill_value=0)
        obj_counts.plot(kind='bar', ax=ax, color=BAR_COLORS_PAIRED, alpha=0.8, width=0.8)
        ax.set_title(f'Object Type Distribution by Label (Downsampled, n={len(data_processed)})')
        ax.set_xlabel('Object Type')
        ax.set_ylabel('Count')
        ax.legend(['Safe', 'Danger'], title='Label')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')

    # Interaction distribution (exclude no-interaction)
    if 'interaction' in data_processed.columns:
        ax = axes[1]
        # Filter out "no-interaction"
        df_filtered = data_processed[data_processed['interaction'] != 'no-interaction'].copy()
        if len(df_filtered) > 0:
            inter_counts = df_filtered.groupby(['interaction', 'danger_label']).size().unstack(fill_value=0)
            inter_counts.plot(kind='bar', ax=ax, color=BAR_COLORS_PAIRED, alpha=0.8, width=0.8)
            ax.set_title(f'Interaction Distribution by Label\n(Downsampled, excluding no-interaction, n={len(df_filtered)})')
            ax.set_ylabel('Count')
            ax.legend(['Safe', 'Danger'], title='Label')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    save_plot(output_dir, '07_object_interaction_bars.png', dpi=150)


def plot_normalized_bars(df, output_dir):
    """2. Normalized bar chart: P(Danger | semantic) with downsampling."""
    # Apply downsampling to balance class distribution
    data_processed = downsample_safe_samples(df)
    data_processed = apply_danger_scale(data_processed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Object type danger probability
    if 'object_type' in data_processed.columns:
        ax = axes[0]
        obj_danger_rate = data_processed.groupby('object_type')['danger_label'].apply(lambda x: (x.sum() / len(x)) * 100)
        obj_danger_rate.plot(kind='bar', ax=ax, color=BAR_COLORS_SINGLE, alpha=0.8)
        ax.set_title(f'P(Danger | Object Type) - Downsampled (n={len(data_processed)})')
        ax.set_xlabel('Object Type')
        ax.set_ylabel('Danger Rate (%)')
        ax.set_ylim([0, 100])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')

    # Interaction danger probability (exclude no-interaction)
    if 'interaction' in data_processed.columns:
        ax = axes[1]
        df_filtered = data_processed[data_processed['interaction'] != 'no-interaction'].copy()
        if len(df_filtered) > 0:
            inter_danger_rate = df_filtered.groupby('interaction')['danger_label'].apply(lambda x: (x.sum() / len(x)) * 100)
            inter_danger_rate.plot(kind='bar', ax=ax, color=BAR_COLORS_SINGLE, alpha=0.8)
            ax.set_title(f'P(Danger | Interaction) - Downsampled\n(Excluding no-interaction, n={len(df_filtered)})')
            ax.set_ylabel('Danger Rate (%)')
            ax.set_ylim([0, 100])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    save_plot(output_dir, '08_normalized_bars.png', dpi=150)


def plot_semantic_heatmap(df, output_dir):
    """3. Heatmap: object_type × interaction → danger rate with downsampling."""
    # Apply downsampling to balance class distribution
    data_processed = downsample_safe_samples(df)

    if 'object_type' not in data_processed.columns or 'interaction' not in data_processed.columns:
        print(f'  ⚠ Missing object_type or interaction columns, skipping semantic heatmap')
        return

    # Filter out "no-interaction"
    df_filtered = data_processed[data_processed['interaction'] != 'no-interaction'].copy()

    if len(df_filtered) == 0:
        print(f'  ⚠ No interaction data after filtering, skipping semantic heatmap')
        return

    # Create crosstab of object × interaction with danger rate
    crosstab = pd.crosstab(
        df_filtered['object_type'],
        df_filtered['interaction'],
        values=df_filtered['danger_label'],
        aggfunc='mean'
    ) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(crosstab, annot=True, fmt='.1f', cmap=COLORMAP_DANGER, ax=ax,
                cbar_kws={'label': 'Danger Rate (%)'}, vmin=0, vmax=100)

    ax.set_title(f'Danger Rate: Object Type × Interaction (Downsampled)\n(Excluding no-interaction, n={len(df_filtered)})')
    ax.set_xlabel('Interaction')
    ax.set_ylabel('Object Type')

    plt.tight_layout()
    save_plot(output_dir, '09_semantic_heatmap.png', dpi=150)

