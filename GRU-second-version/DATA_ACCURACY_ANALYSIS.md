# Data Accuracy Analysis & Issues Found

## Issues Identified

### 1. **Label Column Confusion**
**Problem**: The code uses `danger_label` (computed from interaction) but the CSV has a `label` column (direct safety annotation)

**Facts**:
- CSV has column: `label` (0=safe, 1=danger) - **Direct annotation from data**
- Code creates: `danger_label` (computed from interaction) - **Derived from 'interaction' values**
- These are DIFFERENT labels!

**Example from danger2**:
- Raw `label` danger rate: **3.6%** (83 dangerous out of 2,296)
- Derived `danger_label` rate: **~2.3%** (from interaction="drowning"/"near" only)

**Impact**: Graphs show under-represented danger because it only counts specific interactions

---

### 2. **Pool Object Correctly Exists** ✓
**Status**: NOT A BUG
- "pool" is a valid object_type in danger2 video
- Has 328 entries: 245 safe (0) + 83 danger (1) = **25.3% danger rate**
- Appears correctly in object_type distribution charts

---

### 3. **Downsampling Status**

**WHERE DOWNSAMPLING IS APPLIED** (with SAFE_SAMPLE_INTERVAL=20):
✅ plot_histograms() - Line 15
✅ plot_box_plots() - Line 54
✅ plot_violin_plots() - Line 105
✅ plot_2d_scatter() - Line 150
✅ plot_correlation_heatmap() - Line 195
✅ plot_feature_vs_time() - Line 220

**WHERE DOWNSAMPLING IS NOT APPLIED** ❌:
❌ plot_object_interaction_bars() - Uses raw df
❌ plot_normalized_bars() - Uses raw df
❌ plot_semantic_heatmap() - Uses raw df
❌ Embedding visualizations - Use raw df

**Impact**: Semantic plots show imbalanced class distribution (97.7% safe, 2.3% danger) while static plots show balanced distribution (after downsampling 20:1)

---

## Data Structure by Video

### danger1
- Total: 4,200 rows
- Danger: 651 (15.5%)
- Safe: 3,549 (84.5%)
- Objects: chair, bottle, dining table
- Interactions: no-interaction, reaching, touching, fall

### danger10
- Total: 8,420 rows
- Danger: 339 (4.0%)
- Safe: 8,081 (96.0%)
- Objects: bowl, remote, bed, bottle
- Interactions: no-interaction, lying, crawling, falling, none

### danger2
- Total: 2,296 rows
- Danger: 83 (3.6%)
- Safe: 2,213 (96.4%)
- Objects: chair, bowl, **pool** (valid!)
- Interactions: no-interaction, near, reaching, touching, clambering, drowning, NaN

### ALL COMBINED
- Total: 14,916 rows
- Using `label` column: 339 danger (2.3%), 14,577 safe (97.7%)
- Using derived `danger_label` from interaction: varies by interaction rules

---

## Recommendations

### Priority 1: Fix Label Mapping
Change data loading to use the CSV's `label` column instead of deriving from interaction:

```python
# In data_loader.py, change:
df['danger_label'] = df['interaction'].apply(...)  # WRONG

# To:
df['danger_label'] = df['label'].astype(int)  # CORRECT - uses CSV's direct annotation
```

**Why**: 
- CSV already has ground-truth danger labels
- Using `label` is more accurate than inferring from interaction text
- Ensures consistency across all plots

---

### Priority 2: Apply Downsampling to Semantic Plots
Add downsampling to object/interaction analysis:

```python
def plot_object_interaction_bars(df, output_dir):
    # Apply downsampling
    data_processed = downsample_safe_samples(df)
    
    # Then use data_processed instead of df
```

**Why**:
- Makes semantic plots comparable to static feature plots
- Balances class distribution in bar charts
- Prevents misinterpretation of rare but important danger patterns

---

### Priority 3: Add Downsampling Note to Embeddings
Consider adding downsampling to embedding visualizations or document why they're kept raw

---

## Verification Checklist

- [ ] Verify `label` column is used consistently across all plots
- [ ] Confirm pool object displays correctly after label fix
- [ ] Test that downsampling is applied to semantic plots
- [ ] Check that danger rates are consistent (should be ~2.3% across all plots after downsampling)
- [ ] Verify CSV columns match code expectations

---

## Current Configuration

From `data-log/config.py`:
- `SAFE_SAMPLE_INTERVAL` = 20 (downsample 1 out of every 20 safe frames)
- `DANGER_LABEL_SCALE` = 1 (no visual emphasis)
- `SCATTER_BIN_INTERVAL` = 5.0 (distance binning)

Expected result after downsampling:
- Raw: 14,916 samples (339 danger, 14,577 safe) → 2.3% danger
- Downsampled: ~850 samples (339 danger, ~500 safe) → 40% danger

---

## Files to Fix

1. **data-log/utils/data_loader.py** - Use CSV's `label` column
2. **data-log/plots/semantic_features.py** - Add downsampling to all 3 functions
3. **data-log/plots/embeddings.py** - Document choice to use raw data

---

Generated: 2026-01-14
