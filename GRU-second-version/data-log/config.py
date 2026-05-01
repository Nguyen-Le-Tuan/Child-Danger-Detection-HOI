# data-log/config.py

LABEL_COL = "status_label"   # 1 = danger, 0 = safe

# ==================== DANGER LABEL SCALING ====================
# Scale factor for danger samples vs safe samples in visualizations
# Higher values emphasize danger samples more prominently
# Example: DANGER_LABEL_SCALE = 5 means danger samples get 5x visual weight
DANGER_LABEL_SCALE = 0.8

# ==================== SAFE SAMPLE DOWNSAMPLING ====================
# Downsample safe (non-danger) samples to reduce class imbalance visualization
# For every N safe frames of an object-child pair, aggregate into 1 sample
# Safe frames: average distance and velocity, count as 1 aggregated sample
# Danger frames: always kept (no downsampling)
# 
# Examples:
#   SAFE_SAMPLE_INTERVAL = 1   → no downsampling (keep all samples)
#   SAFE_SAMPLE_INTERVAL = 5   → keep 1 out of every 5 safe frames
#   SAFE_SAMPLE_INTERVAL = 10  → keep 1 out of every 10 safe frames
# 
# NOTE: This affects histograms, scatter plots, boxplots, violinplots only.
#       NOT applied to: PCA, t-SNE, semantic features (object/interaction analysis)
SAFE_SAMPLE_INTERVAL = 5

# ==================== SCATTER PLOT BINNING ====================
# Group scatter plot samples into bins based on distance intervals
# Smaller values = finer resolution (more bins, more detail)
# Larger values = coarser resolution (fewer bins, less detail)
#
# Examples:
#   SCATTER_BIN_INTERVAL = 0.5  → bins: [0-0.5], [0.5-1.0], [1.0-1.5], etc.
#   SCATTER_BIN_INTERVAL = 1.0  → bins: [0-1], [1-2], [2-3], etc.
#   SCATTER_BIN_INTERVAL = 2.0  → bins: [0-2], [2-4], [4-6], etc.
#
# Color intensity represents sample count: darker = more samples in that bin
SCATTER_BIN_INTERVAL = 5.0

NUMERIC_FEATURES = {
    "distance": {
        "xlabel": "Distance (m)",
        "range": (0, 10)
    },
    "relative_velocity": {
        "xlabel": "Relative Velocity (m/s)",
        "range": (-10, 10)
    }
}
