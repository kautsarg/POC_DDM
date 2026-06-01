# Plotly TTP Plotting Fix Summary

## Problem Description
The TTP (Time To Peak) calculation was working correctly and showing different values in the summary table, but the plotly graphs were not displaying TTP markers. This indicated that the plotly plotting functions were not receiving or plotting the TTP data.

## Root Cause
The issue was in the **order of execution** in `main_DNA.py`:

1. **Plotly functions were called BEFORE TTP calculation** - around line 180
2. **TTP calculation happened AFTER** - around line 200
3. **Plotly functions had no access to TTP results** - they were plotting raw signals only

Additionally, the plotly functions were not designed to accept or plot TTP data.

## Fixes Applied

### 1. Fixed Execution Order in main_DNA.py
- **Before**: Plotly functions called before TTP calculation
- **After**: Plotly functions called after TTP calculation with TTP data passed as parameter

```python
# FIXED CODE:
# First: Calculate TTP and store results
# ... TTP calculation code ...

# Then: Create plotly plots WITH TTP data
plot_wells_grid_plotly(exp, save_path=save_path, experiment_name=experiment_name, 
                       ttp_results=second_peak_results)
```

### 2. Modified Plotly Functions to Accept TTP Data
Updated all three plotly functions to accept `ttp_results` parameter:

- `plot_wells_grid_plotly()` - Added `ttp_results=None` parameter
- `plot_all_wells_combined_plotly()` - Added `ttp_results=None` parameter  
- `plot_wells_active_pixels_plotly()` - Added `ttp_results=None` parameter

### 3. Added TTP Plotting to Each Function

#### Grid Plot (`plot_wells_grid_plotly`)
- **TTP markers**: Purple diamond markers at TTP times
- **TTP annotations**: Text labels showing TTP time with arrows
- **Placement**: Each well's subplot shows its individual TTP

#### Combined Plot (`plot_all_wells_combined_plotly`)
- **TTP markers**: Purple diamond markers on the signal plot only
- **TTP annotations**: Text labels showing well number and TTP time
- **Placement**: All TTP markers appear on the top signal subplot for clarity

#### Active Pixels Plot (`plot_wells_active_pixels_plotly`)
- **TTP information**: Added to subplot titles
- **Format**: "Well X Active Pixels<br>TTP: Y.Y min"
- **Placement**: Each subplot title shows its well's TTP time

### 4. Enhanced Function Documentation
Added `ttp_results` parameter documentation to all functions explaining:
- What the parameter contains
- How TTP data is used
- Where TTP markers appear

## Key Changes Made

### main_DNA.py
1. **Lines 180-220**: Moved plotly function calls to after TTP calculation
2. **Lines 200-220**: Added `ttp_results=second_peak_results` parameter to all plotly calls
3. **Lines 180-200**: Moved summary table creation to before plotly calls

### plot_derivatives_plotly.py
1. **Function signatures**: Added `ttp_results=None` parameter to all three functions
2. **TTP plotting logic**: Added comprehensive TTP marker and annotation code
3. **Documentation**: Updated docstrings to describe TTP functionality
4. **Subplot titles**: Enhanced active pixels plot with TTP information

## Expected Results

After these fixes:
- ✅ **Plotly graphs will display TTP markers** at the correct times for each well
- ✅ **TTP times will vary between wells** as shown in the summary table
- ✅ **Each well's TTP will be clearly visible** with purple diamond markers
- ✅ **TTP annotations will show exact times** with arrows pointing to markers
- ✅ **Active pixel plots will show TTP times** in subplot titles

## Files Modified

- `titan/main_DNA.py` - Fixed execution order and added TTP data to plotly calls
- `titan/plot_derivatives_plotly.py` - Added TTP plotting functionality to all plotly functions

## Next Steps

1. **Run the main script** to see TTP markers on plotly graphs
2. **Verify TTP markers appear** at different times for different wells
3. **Check TTP annotations** show correct times matching the summary table
4. **Confirm active pixel plots** display TTP times in subplot titles

The core issue has been resolved by ensuring plotly functions receive TTP data and properly plot it as markers and annotations, which should eliminate the problem of missing TTP visualization in the interactive plots.

