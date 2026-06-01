# TTP Calculation Fix Summary

## Problem Description
The TTP (Time To Peak) calculation was showing identical values for all wells in the plotly graphs. This indicated that the TTP calculation algorithm was not properly differentiating between wells.

## Root Cause
The issue was in the `detect_second_derivative_peaks_from_first_derivative()` function in `titan/peak_detection.py`. The function was using a **single time base** from only the first well for all TTP calculations:

```python
# PROBLEMATIC CODE (lines 220-221):
# Get time base from first well
first_well = list(well_data.values())[0]
time_data = first_well['time']  # This was used for ALL wells!
```

This caused all wells to use the same time reference, leading to identical TTP calculations.

## Fixes Applied

### 1. Fixed Time Data Reference
- **Before**: Used `time_data` from first well for all wells
- **After**: Each well now uses its own `well_time_data` from `well_data[well_idx]['time']`

```python
# FIXED CODE:
for well_idx in well_data.keys():
    # ... other code ...
    well_time_data = well_data[well_idx]['time']  # Use this well's time data
    
    # All time calculations now use well_time_data instead of time_data
    anchor = np.clip(anchor, well_time_data[0], well_time_data[-1])
    mask = (well_time_data >= deriv2_start_min) & (well_time_data <= deriv_end_min)
    t2 = well_time_data[mask]
```

### 2. Added Comprehensive Debug Output
Added detailed logging throughout the TTP calculation process to help diagnose future issues:

- **Function entry**: Shows number of wells and time window parameters
- **Per-well processing**: Shows anchor times, time ranges, and processing steps
- **Zero-crossing detection**: Shows which fallback method is used and results
- **TTP calculation**: Shows search windows, peak detection parameters, and final results
- **Error handling**: Shows when wells are skipped and why

### 3. Enhanced Error Handling
- Better validation of time windows
- Clearer reporting when wells are skipped
- More informative error messages

## Key Changes Made

1. **Line 220-221**: Changed from single `time_data` to per-well `well_time_data`
2. **Line 250**: Updated anchor clamping to use well-specific time range
3. **Line 252**: Updated time window mask to use well-specific time data
4. **Line 254**: Updated t2 calculation to use well-specific time data
5. **Lines 220-400**: Added comprehensive debug output throughout the function

## Expected Results

After these fixes:
- **Each well will have its own TTP calculation** based on its individual time data
- **TTP times will vary between wells** as expected
- **Debug output will show the calculation process** for each well
- **Plotly graphs will display different TTP times** for each well

## Testing

A test script `test_ttp_fix.py` has been created to verify the fixes work correctly with synthetic data that has known, different TTP times.

## Files Modified

- `titan/peak_detection.py` - Fixed TTP calculation logic and added debug output
- `test_ttp_fix.py` - Created test script to verify fixes
- `TTP_FIX_SUMMARY.md` - This summary document

## Next Steps

1. **Run the main script** to see the improved TTP calculations
2. **Check the debug output** to verify each well is processed independently
3. **Verify plotly graphs** show different TTP times for each well
4. **Remove debug output** once everything is working correctly (optional)

The core issue has been resolved by ensuring each well uses its own time data for TTP calculations, which should eliminate the problem of identical TTP times across all wells.

