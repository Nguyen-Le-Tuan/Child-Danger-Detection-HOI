# Summary: Label Accuracy & Downsampling Fixes

## Problems Identified ✓

### 1. **Label Mapping Issue** 
- **Problem**: Danger labels derived from interaction text instead of CSV's ground-truth `label` column
- **Impact**: Danger samples undercounted by ~3x
- **Root cause**: Code used fuzzy text matching on interaction field

### 2. **Inconsistent Downsampling**
- **Problem**: Static feature plots downsampled, semantic plots did NOT
- **Impact**: Inconsistent class distribution across visualization suite
- **Root cause**: Semantic functions (`plot_object_interaction_bars`, etc.) operated on raw data

### 3. **"Pool" Object Confusion** 
- **Status**: ✅ NOT A BUG - legitimate object_type in danger2
- **Verification**: 328 samples, 83 dangerous (25.3% danger rate)

---

## Fixes Applied ✓

### Fix #1: Label Column Usage
**File**: `data-log/utils/data_loader.py`

```diff
- df['danger_label'] = df['interaction'].apply(
-     lambda x: 1 if isinstance(x, str) and x.lower() in {...} else 0
- )
+ df['danger_label'] = df['label'].astype(int)  # Use CSV's ground truth
```

**Result**: 
- Before: ~280 danger samples (inaccurate)
- After: **1,073 danger samples** (accurate)

---

### Fix #2: Semantic Plot Downsampling
**File**: `data-log/plots/semantic_features.py`

Updated 3 functions to apply downsampling:
1. `plot_object_interaction_bars()`
2. `plot_normalized_bars()`
3. `plot_semantic_heatmap()`

```diff
+ data_processed = downsample_safe_samples(df)
+ data_processed = apply_danger_scale(data_processed)
- obj_counts = df.groupby(...)
+ obj_counts = data_processed.groupby(...)
```

**Result**:
- Before: 14,916 raw samples in semantic charts
- After: **1,779 downsampled samples** (consistent with static plots)

---

## Key Statistics After Fixes

### Accurate Danger Counts
| Video | Total | Danger | Rate |
|-------|-------|--------|------|
| danger1 | 4,200 | 651 | 15.5% |
| danger2 | 2,296 | 83 | 3.6% |
| danger10 | 8,420 | 339 | 4.0% |
| **TOTAL** | **14,916** | **1,073** | **7.2%** |

### After Downsampling (×8.3 reduction)
- **1,779 total samples**
- **1,073 danger** (60.3%) - all kept
- **706 safe** (39.7%) - downsampled 15x

---

## What's Now Correct

✅ Danger labels from ground-truth CSV column
✅ All plots use consistent downsampling
✅ "Pool" object correctly displayed (25.3% danger rate)
✅ Class distribution balanced (7.2% → 60.3%)
✅ Semantic plots comparable to static plots

---

## Files Changed

1. **data-log/utils/data_loader.py** - Label source fix
2. **data-log/plots/semantic_features.py** - Downsampling fix

---

## Verification

Latest run output confirms:
```
Original Data: 1,073 danger (7.2%)  ← Correct count
Processed Data: 1,073 danger (60.3%) ← After downsampling

All 17 plots generated successfully
✓ Static features (6 plots) - downsampled
✓ Semantic features (3 plots) - NOW downsampled
✓ Embeddings (8 plots) - raw by design
```

---

## How to Regenerate

```bash
cd /Volumes/Research/projects/SEF2526/GRU-second-version
python3 data-log/analysis_viz.py ./fine_data
```

Output will be in: `./data-log/output/TIMESTAMP/`

---

**Status**: ✅ COMPLETE - All issues fixed and verified
