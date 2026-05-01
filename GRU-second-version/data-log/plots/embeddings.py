"""
Embedding visualization module (PCA and t-SNE for both embedding types)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.plotting_config import COLORS, COLORMAP_DANGER, get_percentile_range, save_plot


def plot_pca_visualization(embeddings_df, output_dir):
    """PCA visualization of CLIP embeddings (interaction) with percentile-based scaling."""
    if embeddings_df is None or len(embeddings_df) == 0:
        print(f'  ⚠ No embedding data, skipping PCA')
        return
    
    # Extract embeddings
    X = np.array(embeddings_df['embedding'].tolist())
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Get percentile-based range to focus on main cluster
    x_range = get_percentile_range(X_pca[:, 0])
    y_range = get_percentile_range(X_pca[:, 1])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=embeddings_df['danger_label'],
        cmap=COLORMAP_DANGER,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set limits to percentile range
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA: Interaction CLIP Embeddings\n(Focused on 5-95th Percentile, n={len(embeddings_df)})')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Danger Label')
    
    plt.tight_layout()
    save_plot(output_dir, '10_pca_interaction_embeddings.png', dpi=150)


def plot_tsne_visualization(embeddings_df, output_dir):
    """t-SNE visualization of CLIP embeddings (interaction) with percentile-based scaling."""
    if embeddings_df is None or len(embeddings_df) == 0:
        print(f'  ⚠ No embedding data, skipping t-SNE')
        return
    
    # Limit to 5000 samples for speed
    sample_df = embeddings_df.sample(min(5000, len(embeddings_df)))
    X = np.array(sample_df['embedding'].tolist())
    
    print(f'  [t-SNE] Computing on {len(X)} interaction embeddings...')
    tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    
    # Get percentile-based range
    x_range = get_percentile_range(X_tsne[:, 0])
    y_range = get_percentile_range(X_tsne[:, 1])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=sample_df['danger_label'],
        cmap=COLORMAP_DANGER,
        alpha=0.6,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set limits to percentile range
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE: Interaction CLIP Embeddings\n(Focused on 5-95th Percentile, n={len(X)})')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Danger Label')
    
    plt.tight_layout()
    save_plot(output_dir, '11_tsne_interaction_embeddings.png', dpi=150)


def plot_pca_object_embeddings(df, output_dir):
    """PCA visualization of object embeddings with percentile-based scaling."""
    if 'embedding' not in df.columns:
        print(f'  ⚠ No object embedding data, skipping PCA')
        return
    
    # Extract non-null embeddings
    data = df[['embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid object embeddings, skipping PCA')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['embedding']]
        X = np.array(embeddings_list)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_pca[:, 0])
        y_range = get_percentile_range(X_pca[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=data['danger_label'],
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'PCA: Object CLIP Embeddings\n(Focused on 5-95th Percentile, n={len(data)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '12_pca_object_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in PCA object embeddings: {e}')


def plot_tsne_object_embeddings(df, output_dir):
    """t-SNE visualization of object embeddings with percentile-based scaling."""
    if 'embedding' not in df.columns:
        print(f'  ⚠ No object embedding data, skipping t-SNE')
        return
    
    # Extract non-null embeddings
    data = df[['embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid object embeddings, skipping t-SNE')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['embedding']]
        X = np.array(embeddings_list)
        
        # Limit to 5000 for speed
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            X = X[indices]
            danger_labels = data['danger_label'].iloc[indices].values
        else:
            danger_labels = data['danger_label'].values
        
        print(f'  [t-SNE] Computing on {len(X)} object embeddings...')
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_tsne[:, 0])
        y_range = get_percentile_range(X_tsne[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=danger_labels,
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE: Object CLIP Embeddings\n(Focused on 5-95th Percentile, n={len(X)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '13_tsne_object_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in t-SNE object embeddings: {e}')


def plot_pca_object_type_embeddings(df, output_dir):
    """PCA visualization of object_type text embeddings with percentile-based scaling."""
    if 'object_type_embedding' not in df.columns:
        print(f'  ⚠ No object_type embedding data, skipping PCA')
        return
    
    # Extract non-null embeddings
    data = df[['object_type_embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid object_type embeddings, skipping PCA')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['object_type_embedding']]
        X = np.array(embeddings_list)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_pca[:, 0])
        y_range = get_percentile_range(X_pca[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=data['danger_label'],
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'PCA: Object Type Text Embeddings\n(Focused on 5-95th Percentile, n={len(data)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '14_pca_object_type_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in PCA object_type embeddings: {e}')


def plot_tsne_object_type_embeddings(df, output_dir):
    """t-SNE visualization of object_type text embeddings with percentile-based scaling."""
    if 'object_type_embedding' not in df.columns:
        print(f'  ⚠ No object_type embedding data, skipping t-SNE')
        return
    
    # Extract non-null embeddings
    data = df[['object_type_embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid object_type embeddings, skipping t-SNE')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['object_type_embedding']]
        X = np.array(embeddings_list)
        
        # Limit to 5000 for speed
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            X = X[indices]
            danger_labels = data['danger_label'].iloc[indices].values
        else:
            danger_labels = data['danger_label'].values
        
        print(f'  [t-SNE] Computing on {len(X)} object_type embeddings...')
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_tsne[:, 0])
        y_range = get_percentile_range(X_tsne[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=danger_labels,
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE: Object Type Text Embeddings\n(Focused on 5-95th Percentile, n={len(X)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '15_tsne_object_type_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in t-SNE object_type embeddings: {e}')


def plot_pca_interaction_clip_embeddings(df, output_dir):
    """PCA visualization of interaction CLIP text embeddings with percentile-based scaling."""
    if 'interaction_clip_embedding' not in df.columns:
        print(f'  ⚠ No interaction_clip embedding data, skipping PCA')
        return
    
    # Extract non-null embeddings
    data = df[['interaction_clip_embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid interaction_clip embeddings, skipping PCA')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['interaction_clip_embedding']]
        X = np.array(embeddings_list)
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_pca[:, 0])
        y_range = get_percentile_range(X_pca[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=data['danger_label'],
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title(f'PCA: Interaction CLIP Text Embeddings\n(Focused on 5-95th Percentile, n={len(data)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '16_pca_interaction_clip_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in PCA interaction_clip embeddings: {e}')


def plot_tsne_interaction_clip_embeddings(df, output_dir):
    """t-SNE visualization of interaction CLIP text embeddings with percentile-based scaling."""
    if 'interaction_clip_embedding' not in df.columns:
        print(f'  ⚠ No interaction_clip embedding data, skipping t-SNE')
        return
    
    # Extract non-null embeddings
    data = df[['interaction_clip_embedding', 'danger_label']].dropna()
    
    if len(data) == 0:
        print(f'  ⚠ No valid interaction_clip embeddings, skipping t-SNE')
        return
    
    try:
        # Parse embeddings
        embeddings_list = [json.loads(emb) if isinstance(emb, str) else emb for emb in data['interaction_clip_embedding']]
        X = np.array(embeddings_list)
        
        # Limit to 5000 for speed
        if len(X) > 5000:
            indices = np.random.choice(len(X), 5000, replace=False)
            X = X[indices]
            danger_labels = data['danger_label'].iloc[indices].values
        else:
            danger_labels = data['danger_label'].values
        
        print(f'  [t-SNE] Computing on {len(X)} interaction_clip embeddings...')
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X)
        
        # Get percentile-based range
        x_range = get_percentile_range(X_tsne[:, 0])
        y_range = get_percentile_range(X_tsne[:, 1])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=danger_labels,
            cmap=COLORMAP_DANGER,
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Set limits to percentile range
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title(f't-SNE: Interaction CLIP Text Embeddings\n(Focused on 5-95th Percentile, n={len(X)})')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Danger Label')
        
        plt.tight_layout()
        save_plot(output_dir, '17_tsne_interaction_clip_embeddings.png', dpi=150)
    except Exception as e:
        print(f'  ⚠ Error in t-SNE interaction_clip embeddings: {e}')
