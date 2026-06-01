#!/usr/bin/env python3
"""
Test script to verify TTP calculation fixes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the titan directory to the path
sys.path.append('titan')

from peak_detection import detect_second_derivative_peaks_from_first_derivative, calculate_well_derivatives, calculate_well_second_derivatives

def create_test_data():
    """Create synthetic test data with known TTP times"""
    np.random.seed(42)  # For reproducible results
    
    # Create time array (40 minutes, 2400 samples)
    time = np.linspace(0, 40, 2400)
    dt = time[1] - time[0]
    
    # Create synthetic signal data for 3 wells with different TTP times
    well_data = {}
    
    # Well 1: TTP at ~15 minutes
    ttp1 = 15.0
    signal1 = 100 + 50 * np.exp(-0.5 * ((time - ttp1) / 2.0)**2) + 0.1 * np.random.randn(len(time))
    well_data[0] = {
        'signal': signal1,
        'derivative': np.gradient(signal1),
        'time': time
    }
    
    # Well 2: TTP at ~25 minutes  
    ttp2 = 25.0
    signal2 = 100 + 50 * np.exp(-0.5 * ((time - ttp2) / 2.0)**2) + 0.1 * np.random.randn(len(time))
    well_data[1] = {
        'signal': signal2,
        'derivative': np.gradient(signal2),
        'time': time
    }
    
    # Well 3: TTP at ~35 minutes
    ttp3 = 35.0
    signal3 = 100 + 50 * np.exp(-0.5 * ((time - ttp3) / 2.0)**2) + 0.1 * np.random.randn(len(time))
    well_data[2] = {
        'signal': signal3,
        'derivative': np.gradient(signal3),
        'time': time
    }
    
    return well_data, [ttp1, ttp2, ttp3]

def test_ttp_calculation():
    """Test TTP calculation with synthetic data"""
    print("Creating synthetic test data...")
    well_data, expected_ttp = create_test_data()
    
    print(f"Expected TTP times: {[f'{t:.1f}' for t in expected_ttp]} minutes")
    
    # Calculate second derivatives
    print("\nCalculating second derivatives...")
    second_deriv_data = calculate_well_second_derivatives(well_data, filter_order=1)
    
    # Create mock peak results (first derivative peaks)
    peak_results = {}
    for well_idx in well_data.keys():
        # Find the peak in the first derivative
        deriv = well_data[well_idx]['derivative']
        time = well_data[well_idx]['time']
        peak_idx = np.argmax(np.abs(deriv))
        peak_results[well_idx] = {
            'peak_detected': True,
            'peak_time': time[peak_idx],
            'peak_value': deriv[peak_idx]
        }
        print(f"Well {well_idx + 1}: First derivative peak at {time[peak_idx]:.2f} min")
    
    # Calculate TTP using the fixed function
    print("\nCalculating TTP times...")
    ttp_results = detect_second_derivative_peaks_from_first_derivative(
        well_data, second_deriv_data, peak_results,
        deriv2_start_min=1.0, deriv_end_min=35.0
    )
    
    # Display results
    print("\n" + "="*60)
    print("TTP CALCULATION RESULTS")
    print("="*60)
    
    for well_idx in range(len(well_data)):
        if well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
            actual_ttp = ttp_results[well_idx]['ttp_time']
            expected_ttp_well = expected_ttp[well_idx]
            error = abs(actual_ttp - expected_ttp_well)
            
            print(f"Well {well_idx + 1}:")
            print(f"  Expected TTP: {expected_ttp_well:.2f} min")
            print(f"  Actual TTP:   {actual_ttp:.2f} min")
            print(f"  Error:        {error:.2f} min")
            print(f"  Status:       {'✓ PASS' if error < 2.0 else '✗ FAIL'}")
            
            if ttp_results[well_idx]['found_zc']:
                print(f"  ZC time:      {ttp_results[well_idx]['zc_time']:.2f} min")
            else:
                print(f"  ZC:           Not found (used Fallback #3)")
        else:
            print(f"Well {well_idx + 1}: No TTP detected ✗")
    
    # Plot results
    plot_test_results(well_data, ttp_results, expected_ttp)

def plot_test_results(well_data, ttp_results, expected_ttp):
    """Plot the test results"""
    fig, axes = plt.subplots(len(well_data), 2, figsize=(15, 5*len(well_data)))
    if len(well_data) == 1:
        axes = axes.reshape(1, -1)
    
    for well_idx in range(len(well_data)):
        time = well_data[well_idx]['time']
        signal = well_data[well_idx]['signal']
        derivative = well_data[well_idx]['derivative']
        
        # Raw signal
        ax1 = axes[well_idx, 0]
        ax1.plot(time, signal, 'b-', linewidth=1.5, label='Signal')
        
        # Mark expected TTP
        ax1.axvline(x=expected_ttp[well_idx], color='green', linestyle='--', 
                   label=f'Expected TTP: {expected_ttp[well_idx]:.1f} min')
        
        # Mark actual TTP if found
        if well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
            actual_ttp = ttp_results[well_idx]['ttp_time']
            ax1.axvline(x=actual_ttp, color='red', linestyle='-', linewidth=2,
                       label=f'Actual TTP: {actual_ttp:.1f} min')
        
        ax1.set_title(f'Well {well_idx + 1} - Signal')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Signal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # First derivative
        ax2 = axes[well_idx, 1]
        ax2.plot(time, derivative, 'g-', linewidth=1.5, label='First Derivative')
        
        # Mark first derivative peak
        if well_idx in ttp_results and ttp_results[well_idx]['found_ttp']:
            peak_time = ttp_results[well_idx].get('peak_time', expected_ttp[well_idx])
            ax2.axvline(x=peak_time, color='orange', linestyle='--', 
                       label=f'First Deriv Peak: {peak_time:.1f} min')
        
        ax2.set_title(f'Well {well_idx + 1} - First Derivative')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('First Derivative')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ttp_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'ttp_test_results.png'")

if __name__ == "__main__":
    test_ttp_calculation()

