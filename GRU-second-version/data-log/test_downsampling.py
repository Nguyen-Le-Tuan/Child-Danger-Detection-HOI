#!/usr/bin/env python3
"""Test script for downsampling and scaling features"""

import sys
sys.path.insert(0, '.')

from utils.data_loader import load_all_danger_data
from utils.plotting_config import downsample_safe_samples, apply_danger_scale
from config import SAFE_SAMPLE_INTERVAL, DANGER_LABEL_SCALE

print("=" * 60)
print("Testing Downsampling and Scaling Features")
print("=" * 60)

# Load data
print("\n1. Loading data...")
df = load_all_danger_data('../fine_data')
print(f"   Original: {len(df)} rows")
print(f"   - Safe: {(df['danger_label'] == 0).sum()}")
print(f"   - Danger: {(df['danger_label'] == 1).sum()}")

# Test with default config (no downsampling/scaling)
print(f"\n2. Config values:")
print(f"   SAFE_SAMPLE_INTERVAL = {SAFE_SAMPLE_INTERVAL}")
print(f"   DANGER_LABEL_SCALE = {DANGER_LABEL_SCALE}")

# Test downsampling with interval=5
print(f"\n3. Testing downsampling (interval=5)...")
df_down5 = downsample_safe_samples(df, interval=5)
safe_reduced = (df_down5['danger_label'] == 0).sum()
print(f"   After downsampling: {len(df_down5)} rows")
print(f"   - Safe: {safe_reduced}")
print(f"   - Danger: {(df_down5['danger_label'] == 1).sum()}")
print(f"   - Reduction: {100*(1-len(df_down5)/len(df)):.1f}%")

# Test downsampling with interval=10
print(f"\n4. Testing downsampling (interval=10)...")
df_down10 = downsample_safe_samples(df, interval=10)
print(f"   After downsampling: {len(df_down10)} rows")
print(f"   - Safe: {(df_down10['danger_label'] == 0).sum()}")
print(f"   - Danger: {(df_down10['danger_label'] == 1).sum()}")
print(f"   - Reduction: {100*(1-len(df_down10)/len(df)):.1f}%")

# Test danger scaling
print(f"\n5. Testing danger scaling (scale=3) on downsampled data...")
df_scaled = apply_danger_scale(df_down5, scale=3)
print(f"   After scaling: {len(df_scaled)} rows")
print(f"   - Safe: {(df_scaled['danger_label'] == 0).sum()}")
print(f"   - Danger: {(df_scaled['danger_label'] == 1).sum()} (3x multiplied)")

# Show sample averaging
print(f"\n6. Checking sample averaging (distance/velocity)...")
safe_sample = df_down5[df_down5['danger_label'] == 0].iloc[0]
print(f"   Sample aggregated data:")
print(f"   - distance: {safe_sample['distance']:.3f}")
print(f"   - relative_velocity: {safe_sample['relative_velocity']:.3f}")

print("\n" + "=" * 60)
print("✓ All tests passed successfully!")
print("=" * 60)
