# Complete Embedding Visualization Suite

## Overview
Enhanced the data-log analysis pipeline with comprehensive embedding visualizations for all 4 embedding types. The system now generates 17 distinct plots covering static features, semantic features, and 8 embedding projections.

## Embedding Types Visualized

### 1. **Object Embeddings** (from annotated objects)
   - **PCA**: 12_pca_object_embeddings.png (182 KB)
   - **t-SNE**: 13_tsne_object_embeddings.png (230 KB)
   - *Source*: `embedding` column in CSV (CLIP embeddings from annotated object data)

### 2. **Object Type Text Embeddings** (text encoding of object category)
   - **PCA**: 14_pca_object_type_embeddings.png (50 KB)
   - **t-SNE**: 15_tsne_object_type_embeddings.png (46 KB)
   - *Source*: `object_type_embedding` column in CSV (CLIP text encoding of object types)

### 3. **Interaction CLIP Text Embeddings** (text encoding of interaction label)
   - **PCA**: 16_pca_interaction_clip_embeddings.png (50 KB)
   - **t-SNE**: 17_tsne_interaction_clip_embeddings.png (89 KB)
   - *Source*: `interaction_clip_embedding` column in CSV (CLIP text encoding of interaction labels)

### 4. **Interaction Embeddings** (legacy, from JSON loading)
   - **PCA**: 10_pca_interaction_embeddings.png (49 KB)
   - **t-SNE**: 11_tsne_interaction_embeddings.png (71 KB)
   - *Source*: Loaded from JSON frames data

## Data Processing Pipeline

```
Load CSV (14,916 samples)
    ↓
Extract non-null embeddings (>13,000 per type)
    ↓
[PCA Branch]
  - Fit PCA to 2 components
  - Get percentile-based range (5-95th)
  - Plot with danger coloring (blue=safe, red=danger)
    ↓
  [t-SNE Branch]
  - Sample to 5,000 for speed
  - Compute t-SNE (max_iter=1000)
  - Get percentile-based range
  - Plot with danger coloring
```

## Plot Specifications

### All Embedding Visualizations Share:
- **Size**: 10×8 inches (figsize=(10, 8))
- **Color Mapping**: COLORMAP_DANGER (blue → red gradient)
- **Alpha**: 0.6 for transparency
- **Edge Colors**: Black with 0.5 linewidth
- **Marker Size**: 50 (PCA), variable (t-SNE)
- **Percentile Range**: 5-95th percentile focused view (excludes outliers)
- **Grid**: Enabled with 0.3 alpha
- **DPI**: 150

### Danger vs Safe Coloring:
- **Danger (1)**: Red shades (upper end of colormap)
- **Safe (0)**: Blue shades (lower end of colormap)

## File Organization

```
data-log/output/20260114_100323/
├── 01_histograms.png                          [Static Features]
├── 02_boxplots.png
├── 03_violinplots.png
├── 04_scatter_2d.png
├── 05_correlation_heatmap.png
├── 06_feature_vs_time.png
├── 07_object_interaction_bars.png             [Semantic Features]
├── 08_normalized_bars.png
├── 09_semantic_heatmap.png
├── 10_pca_interaction_embeddings.png          [Interaction Embeddings]
├── 11_tsne_interaction_embeddings.png
├── 12_pca_object_embeddings.png               [Object Embeddings]
├── 13_tsne_object_embeddings.png
├── 14_pca_object_type_embeddings.png          [Object Type Text]
├── 15_tsne_object_type_embeddings.png
├── 16_pca_interaction_clip_embeddings.png     [Interaction CLIP Text]
├── 17_tsne_interaction_clip_embeddings.png
├── summary.json
└── videos.txt
```

## Code Changes

### Modified Files:

#### 1. `data-log/plots/embeddings.py`
Added 4 new functions:
- `plot_pca_object_type_embeddings()` - PCA for object_type text embeddings
- `plot_tsne_object_type_embeddings()` - t-SNE for object_type text embeddings
- `plot_pca_interaction_clip_embeddings()` - PCA for interaction CLIP text embeddings
- `plot_tsne_interaction_clip_embeddings()` - t-SNE for interaction CLIP text embeddings

All functions follow the same pattern:
- Extract non-null embeddings from DataFrame
- Parse JSON-serialized vectors
- Apply dimensionality reduction
- Visualize with danger/safe coloring
- Save with percentile-based scaling

#### 2. `data-log/analysis_viz.py`
Updated imports:
```python
from plots.embeddings import (
    plot_pca_visualization, plot_tsne_visualization,
    plot_pca_object_embeddings, plot_tsne_object_embeddings,
    plot_pca_object_type_embeddings, plot_tsne_object_type_embeddings,
    plot_pca_interaction_clip_embeddings, plot_tsne_interaction_clip_embeddings
)
```

Added execution calls in `run_analysis()`:
```python
print(f'[Plots] Generating OBJECT_TYPE text embedding visualizations...')
plot_pca_object_type_embeddings(df, output_subdir)
plot_tsne_object_type_embeddings(df, output_subdir)

print(f'[Plots] Generating INTERACTION CLIP text embedding visualizations...')
plot_pca_interaction_clip_embeddings(df, output_subdir)
plot_tsne_interaction_clip_embeddings(df, output_subdir)
```

## Execution

Run the complete analysis:
```bash
cd /Volumes/Research/projects/SEF2526/GRU-second-version
python3 data-log/analysis_viz.py ./fine_data
```

Expected output:
```
[Analysis] Output directory: ./data-log/output/20260114_100323
[Data] Loading CSV files...
  -> Loaded 14916 rows from 3 videos
  ✓ Saved videos.txt (3 videos)
[Plots] Generating static feature visualizations... (6 plots)
[Plots] Generating semantic feature visualizations... (3 plots)
[Data] Loading embeddings...
  -> Loaded 14260 embedding samples
[Plots] Generating INTERACTION embedding visualizations... (2 plots)
[Plots] Generating OBJECT embedding visualizations... (2 plots)
[Plots] Generating OBJECT_TYPE text embedding visualizations... (2 plots)
[Plots] Generating INTERACTION CLIP text embedding visualizations... (2 plots)
✓ Analysis complete: ./data-log/output/20260114_100323
```

## Statistical Summary

From latest run (20260114_100323):

**Original Data**:
- Total: 14,916 rows
- Danger: 339 (2.3%)
- Safe: 14,577 (97.7%)

**Processed Data** (after downsampling):
- Total: 1,321 rows
- Danger: 339 (25.7%)
- Safe: 982 (74.3%)

**Embeddings Analyzed**:
- Interaction: 14,260 samples
- Object: 14,916 samples
- Object Type: 14,916 samples
- Interaction CLIP: 14,916 samples

## Key Features

✅ **All 4 embedding types visualized**
✅ **Both PCA (fast) and t-SNE (detailed) projections**
✅ **Danger/safe color coding across all plots**
✅ **Percentile-based scaling (5-95th) to focus on main clusters**
✅ **Automatic outlier exclusion for clearer patterns**
✅ **t-SNE sampling (5K max) for computational efficiency**
✅ **JSON serialization handling for embedded vectors**
✅ **Comprehensive error handling and warnings**
✅ **Consistent plot styling across all visualizations**
✅ **Sequential numbering for easy navigation**

## Interpretation Guide

### PCA Plots (02/04/06/16):
- Fast, deterministic 2D projection
- Shows global structure and variance explained
- Red clusters = danger regions, Blue clusters = safe regions
- PC1/PC2 explain ~40-60% of variance typically

### t-SNE Plots (11/13/15/17):
- Detailed, non-linear projection
- Shows local cluster structure
- Red islands = danger-associated embeddings
- Blue majority = safe-associated embeddings
- Computed on 5K random samples for speed

## Dependencies

- pandas, numpy: Data manipulation
- matplotlib, seaborn: Plotting
- sklearn.decomposition.PCA: Principal component analysis
- sklearn.manifold.TSNE: t-SNE projection
- json: Vector deserialization

## Next Steps

1. **Comparative Analysis**: Compare danger/safe separation across embedding types
2. **Cluster Analysis**: Identify specific object types or interactions most associated with danger
3. **Embedding Quality**: Measure cosine similarity within danger clusters
4. **Interactive Dashboard**: Create web-based exploration tool
5. **Real-time Integration**: Add live embedding projection during video playback

---
**Generated**: 2026-01-14 10:04 UTC
**Total Plots**: 17 (9 static/semantic + 8 embedding)
**Total Size**: ~1.5 MB
**Runtime**: ~2 minutes
