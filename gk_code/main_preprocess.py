import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pickle
import os 

import sys
sys.path.insert(0, '..')
from titan.Experiment import Experiment
from titan.plt_Experiment_summary import titan_plt_summary, titan_plt_summary_temp
from titan.load_and_preprocessing import titan_load_and_preprocessing
import sigmoid_fitting as sp
from titan.processing_DNA import titan_plt_infl
from scipy.ndimage import convolve1d
from sklearn.manifold import TSNE
from sklearn import preprocessing

import multiprocessing as mp
from functools import partial

def _fit_single_curve(y_row, x_time):
    """
    Worker function: fits one curve and returns the results.
    We pass x_time as a secondary argument.
    """
    # 1. Fit the parameters
    params, _ = sp.fit_5p(x_time, y_row, normalize=True)
    
    # 2. Generate the fitted curve
    fitted = sp.sigmoid_5p(x_time, *params)
    
    # 3. Calculate RMSE
    rmse = np.sqrt(np.nanmean((fitted - y_row)**2))
    
    return fitted, params, rmse

def plot_tsne(ori_curves_tsne, movavg_curves_tsne, sigmoid_5pl_tsne, well_ids, n_well=10, output_dir='.', plot_name=""):
    """
    Plot original and moving average t-SNE side by side
    
    Parameters
    ----------
    ori_curves_tsne : ndarray
        Original t-SNE (n_samples, 2)
    movavg_curves_tsne : ndarray
        Moving average t-SNE (n_samples, 2)
    well_ids : ndarray
        Well labels for each sample
    n_well : int
        Number of wells
    output_dir : str
        Directory to save the plot
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    
    colors = plt.cm.tab10(np.arange(n_well))
    
    # Plot 1: Original curves
    for label in range(n_well):
        mask = well_ids == label
        axes[0].scatter(ori_curves_tsne[mask, 0], ori_curves_tsne[mask, 1],
                       alpha=0.7, s=30, c=[colors[label]], 
                       label=f'Well-{label+1} (n={np.sum(mask)})',
                       edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    axes[0].set_title('Original Curves', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Plot 2: Moving average curves
    for label in range(n_well):
        mask = well_ids == label
        axes[1].scatter(movavg_curves_tsne[mask, 0], movavg_curves_tsne[mask, 1],
                       alpha=0.7, s=30, c=[colors[label]], 
                       label=f'Well-{label+1} (n={np.sum(mask)})',
                       edgecolors='black', linewidth=0.5)
    
    axes[1].set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    axes[1].set_title('Moving Average Curves', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

    # Plot 3: 5pl sigmoid fitting curves
    for label in range(n_well):
        mask = well_ids == label
        axes[2].scatter(sigmoid_5pl_tsne[mask, 0], sigmoid_5pl_tsne[mask, 1],
                       alpha=0.7, s=30, c=[colors[label]], 
                       label=f'Well-{label+1} (n={np.sum(mask)})',
                       edgecolors='black', linewidth=0.5)
    
    axes[2].set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    axes[2].set_title('Sigmoid 5P Fitting Curves', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Overall title
    fig.suptitle(f't-SNE Visualization: Original vs Moving Average vs Sigmoid 5P Fitting - {plot_name}', 
                 fontsize=15, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save to file
    output_path = f'{output_dir}/TSNE-{plot_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close(fig)


def moving_average(X, window_size=5):
    mask = np.ones(window_size) / window_size
    X_avg = convolve1d(X, mask, mode='nearest')
    return X_avg

def export_well_analysis(exp, output_dir='.'):
    well_stats = []
    total_pixels = 0
    total_active_pixels = 0
    
    for i, well in enumerate(exp.wells_list):
        n_well_pixels = well.well_nrows * well.well_ncols
        n_temp_pixels = well.well_temp_nrows * well.well_temp_ncols
        active_pixels = well.idx_active.sum()
        n_active_pixels = len(well.idx_active)
        active_pct = (active_pixels * 100 / n_active_pixels) if n_active_pixels > 0 else 0
        
        total_pixels += n_active_pixels
        total_active_pixels += active_pixels
        
        well_stats.append({
            'Well': f'Well-{i+1}',
            'well_nrows': well.well_nrows,
            'well_ncols': well.well_ncols,
            'well_total_pixels': n_well_pixels,
            'temp_nrows': well.well_temp_nrows,
            'temp_ncols': well.well_temp_ncols,
            'temp_total_pixels': n_temp_pixels,
            'active_pixels': active_pixels,
            'total_pixels': n_active_pixels,
            'active_percentage': f'{active_pct:.3f}%',
            'time_npr_shape': str(well.time_npr.shape),
            'well_3d_npr_shape': str(well.well_3d_npr.shape),
            'well_3d_temp_npr_shape': str(well.well_3d_temp_npr.shape),
            'well_3d_gain_shape': str(well.well_3d_gain.shape),
            'well_3d_lin_shape': str(well.well_3d_lin.shape),
            'idx_start': well.idx_start,
            'idx_settled': well.idx_settled,
            'idx_end': well.idx_end,
            'idx_active_shape': str(well.idx_active.shape),
        })
    
    df_wells = pd.DataFrame(well_stats)
    df_wells.to_csv(f'{output_dir}/well_statistics.csv', index=False)
    
    # SUMMARY
    total_active_pct = (total_active_pixels * 100 / total_pixels) if total_pixels > 0 else 0
    print(f"✓ Exported well statistics to: well_statistics.csv")
    print(f"\nTOTAL: {total_active_pixels}/{total_pixels} ({total_active_pct:.3f}%)")
    
    return df_wells

def exp_to_X(exp_data: Experiment, param_str: str):   
    vstacked = None
    well_ids = []

    for i, well in enumerate(exp_data.wells_list):
        temp_x = getattr(well, param_str).copy()
        temp_x = np.swapaxes(temp_x, 0, 1)

        if vstacked is None:
            vstacked = temp_x
        else:
            vstacked = np.vstack((vstacked, temp_x))

        well_ids.append(getattr(well, param_str).shape[1])
    if vstacked is None:
        return np.empty((0, 638))

    return vstacked, well_ids

def reconstruct_data(exp, chip_data_type="well_2d_bs_active"):
    X_2d_bs_active, X_2d_bs_active_ids = exp_to_X(exp, chip_data_type)
    if("npr" in chip_data_type):
        X_time = exp.wells_list[0].time_npr
    else:
        X_time = exp.wells_list[0].time
    
    Y_well = []
    for label, count in enumerate(X_2d_bs_active_ids):
        Y_well.extend([label] * count)

    Y_well = np.array(Y_well)

    return X_time, Y_well, X_2d_bs_active

def denoising_moving_average(X_2d_bs_active, window_size=50):
    print("-Moving Average Smoothing-")
    X_2d_bs_active_moving_avg = []
    rmse = []
    for i in range(X_2d_bs_active.shape[0]):
        avged = moving_average(X_2d_bs_active[i], window_size)
        
        X_2d_bs_active_moving_avg.append(avged)
        rmse.append(np.sqrt(np.nanmean((avged - X_2d_bs_active[i])**2)))

    return np.array(X_2d_bs_active_moving_avg), np.array(rmse)

# def sigmoid_fitting_5p(X_time, X_2d_bs_active):
#     print("-5P Sigmoid Fitting-")
#     X_2d_bs_active_sigmoid_5p = []
#     all_params = []
#     all_rmse = []
#     for i in range(X_2d_bs_active.shape[0]):
#         params, _ = sp.fit_5p(X_time, X_2d_bs_active[i, :], normalize=True)
#         fitted = sp.sigmoid_5p(X_time, *params)
#         X_2d_bs_active_sigmoid_5p.append(fitted)
#         rmse = np.sqrt(np.nanmean((fitted - X_2d_bs_active[i, :])**2))

#         all_params.append(params)
#         all_rmse.append(rmse)
#     return np.array(X_2d_bs_active_sigmoid_5p), np.array(all_params), np.array(all_rmse) 

def sigmoid_fitting_5p(X_time, X_2d_bs_active):
    print(f"-5P Sigmoid Fitting (Multiprocessing with {mp.cpu_count()} cores)-")
    
    # Create a partial function that already has X_time baked in
    worker_func = partial(_fit_single_curve, x_time=X_time)
    
    # Convert rows to a list for the pool (X_2d_bs_active is (n_samples, n_time))
    data_rows = [row for row in X_2d_bs_active]
    
    # Initialize the Pool
    # Using 'with' ensures the pool is closed and joined automatically
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(worker_func, data_rows)
    
    # results is a list of tuples: [(fitted1, params1, rmse1), (fitted2, ...)]
    # Unzip them into separate arrays
    curves_list, params_list, rmse_list = zip(*results)
    
    return np.array(curves_list), np.array(params_list), np.array(rmse_list)

def save_preprocessing_plots(exp, exp_path):
    titan_plt_summary(exp, exp_path, plt_save=True)
    titan_plt_summary_temp(exp, exp_path, plt_save=True)
    titan_plt_infl(exp, exp_path, plt_show=False, plt_save=True)


def curve_first_derivatives(X_time, curves, curve_params=None, fit="others"):
    first_derivatives = []
    if(fit == "sigmoid_5p"):
        for params in curve_params:
            first_derivatives.append(sp.sigmoid_5p_first_derivative(X_time, *params))
    else:
        for c in curves:
            first_derivatives.append(sp.calculate_first_derivative(X_time, c))

    return np.array(first_derivatives)


def curve_second_derivatives(X_time, curves, curve_params=None, fit="others"):
    second_derivatives = []
    if(fit == "sigmoid_5p"):
        for params in curve_params:
            second_derivatives.append(sp.sigmoid_5p_second_derivative(X_time, *params))
    else:
        for c in curves:
            second_derivatives.append(sp.calculate_second_derivative(X_time, c))

    return np.array(second_derivatives)


def extract_kinetic_params(X_time, curves, curve_params=None, fit="others"):
    kinetic_params = []
    if fit == "sigmoid_5p":
        for c, param in zip(curves, curve_params):
            kinetic_params.append(sp.extract_kinetic_parameters(X_time, c, param, threshold=0.05))
    else:
        for c in curves:
            kinetic_params.append(sp.extract_kinetic_parameters_original(X_time, c))
    
    return pd.DataFrame(kinetic_params)

def apply_tsne(data, random_state=42):
    """
    Apply t-SNE to data, removing NaN values first
    Returns aligned arrays with NaN samples as NaN in t-SNE space
    
    Parameters
    ----------
    data : array-like
        Input data (DataFrame or ndarray)
    random_state : int
        Random state for reproducibility
    
    Returns
    -------
    tsne_result : ndarray
        t-SNE transformed data (n_samples, 2)
        NaN rows are filled with NaN
    """
    
    # Convert DataFrame to ndarray if needed
    if isinstance(data, pd.DataFrame):
        data_array = data.values
    else:
        data_array = np.asarray(data)
    
    n_samples = data_array.shape[0]
    
    # Find rows with any NaN values
    nan_mask = np.any(np.isnan(data_array), axis=1)
    
    # Get clean data (remove NaN rows)
    clean_data = data_array[~nan_mask]
    
    print(f"  Input samples: {n_samples}")
    print(f"  NaN samples: {np.sum(nan_mask)}")
    print(f"  Clean samples: {clean_data.shape[0]}")
    
    # Initialize output array with NaN
    tsne_result = np.full((n_samples, 2), np.nan)
    
    # Apply t-SNE to clean data
    if clean_data.shape[0] > 0:
        clean_tsne = TSNE(
            n_components=2,
            init='pca',
            learning_rate='auto',
            random_state=random_state
        ).fit_transform(clean_data)
        
        # Fill in clean data t-SNE results at non-NaN positions
        tsne_result[~nan_mask] = clean_tsne
    else:
        print("  ⚠️  WARNING: No clean data after removing NaN!")
    
    return tsne_result

def curve_preprocessing(exp, n_wells, allpos_curves=False, output_dir=".", plt_save=True, 
                        curve_save=True, curve_name="kinetic_extract", chip_data_type="well_2d_bs_active"):
    """
    Optimized curve preprocessing pipeline
    
    Parameters
    ----------
    exp : Experiment
        Experiment object
    n_wells : int
        Number of wells
    output_dir : str
        Output directory for plots and pickle
    plt_save : bool
        Save plots
    curve_save : bool
        Save pickle file
    curve_name : str
        Name for pickle file
    
    Returns
    -------
    processed_curves : dict
        All processed data
    """
    
    # ========================================================================
    # INITIALIZE DATA STRUCTURE
    # ========================================================================
    
    curve_types = ["ori", "moving_avg", "sigmoid_5p"]
    data_keys = ["curves", "dy_dx", "d2y_dx2", "features"]
    tsne_keys = [f"{k}_tsne" for k in data_keys]
    
    # Create template for each curve type
    template = {k: [] for k in data_keys + tsne_keys}
    template.update({"params": 0, "rmse": []})
    
    processed_curves = {
        "X_time": [],
        "Y_well": [],
    }
    
    for curve_type in curve_types:
        processed_curves[curve_type] = template.copy()
    
    # ========================================================================
    # STEP 1: RECONSTRUCT & DENOISE DATA
    # ========================================================================
    
    print("=" * 60)
    print("STEP 1: Curves Denoising & Fitting")
    print("=" * 60)
    
    processed_curves["X_time"], processed_curves["Y_well"], processed_curves["ori"]["curves"] = reconstruct_data(exp, chip_data_type=chip_data_type)
    if allpos_curves:
        baseline_value = -np.min(processed_curves["ori"]["curves"]) + 1e-9
        processed_curves["ori"]["curves"] =  processed_curves["ori"]["curves"] + baseline_value
        processed_curves["ori"]["params"] = baseline_value

    # Moving Average
    processed_curves["moving_avg"]["params"] = 50
    processed_curves["moving_avg"]["curves"], processed_curves["moving_avg"]["rmse"] = denoising_moving_average(
        processed_curves["ori"]["curves"], 
        window_size=processed_curves["moving_avg"]["params"]
    )
    
    # Sigmoid 5P
    processed_curves["sigmoid_5p"]["curves"], processed_curves["sigmoid_5p"]["params"], processed_curves["sigmoid_5p"]["rmse"] = sigmoid_fitting_5p(
        processed_curves["X_time"], 
        processed_curves["ori"]["curves"]
    )
    
    # ========================================================================
    # STEP 2-4: DERIVATIVES & FEATURES (VECTORIZED)
    # ========================================================================
    
    derivative_types = [
        ("1st Derivatives", "dy_dx", curve_first_derivatives),
        ("2nd Derivatives", "d2y_dx2", curve_second_derivatives),
    ]
    
    for deriv_name, deriv_key, deriv_func in derivative_types:
        print("=" * 60)
        print(f"STEP 2+: {deriv_name}")
        print("=" * 60)
        
        for curve_type in curve_types:
            print(f"  {curve_type.upper()}...", end=" ", flush=True)
            
            if curve_type == "sigmoid_5p":
                result = deriv_func(
                    processed_curves["X_time"],
                    processed_curves["sigmoid_5p"]["curves"],
                    processed_curves["sigmoid_5p"]["params"],
                    fit="sigmoid_5p"
                )
            else:
                result = deriv_func(
                    processed_curves["X_time"],
                    processed_curves[curve_type]["curves"]
                )
            
            processed_curves[curve_type][deriv_key] = result
            print("✓")
    
    # Kinetic Features
    print("=" * 60)
    print("Kinetic Features")
    print("=" * 60)
    
    for curve_type in curve_types:
        print(f"  {curve_type.upper()}...", end=" ", flush=True)
        
        if curve_type == "sigmoid_5p":
            features = extract_kinetic_params(
                processed_curves["X_time"],
                processed_curves["sigmoid_5p"]["curves"],
                processed_curves["sigmoid_5p"]["params"],
                fit="sigmoid_5p"
            )
        else:
            features = extract_kinetic_params(
                processed_curves["X_time"],
                processed_curves[curve_type]["curves"]
            )
        
        processed_curves[curve_type]["features"] = features
        print("✓")
    
    # ========================================================================
    # STEP 5: t-SNE PROJECTION (VECTORIZED)
    # ========================================================================
    
    print("=" * 60)
    print("t-SNE Projection")
    print("=" * 60)
    
    data_to_project = ["curves", "dy_dx", "d2y_dx2", "features"]
    
    for curve_type in curve_types:
        print(f"\n{curve_type.upper()}:")
        
        for data_key in data_to_project:
            print(f"  {data_key}...", end=" ", flush=True)
            
            data = processed_curves[curve_type][data_key]
            
            # Convert DataFrame to ndarray
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            # Apply t-SNE
            tsne_key = f"{data_key}_tsne"
            processed_curves[curve_type][tsne_key] = apply_tsne(data)
            print("✓")
    
    # ========================================================================
    # STEP 6: VISUALIZATION (VECTORIZED)
    # ========================================================================
    
    if plt_save:
        print("=" * 60)
        print("Generating Plots")
        print("=" * 60)
        plt_dir = f'{output_dir}/{curve_name}'
        Path(plt_dir).mkdir(parents=True, exist_ok=True)
        plot_configs = [
            ("Curves", "curves_tsne"),
            ("1st Derivatives", "dy_dx_tsne"),
            ("2nd Derivatives", "d2y_dx2_tsne"),
            ("Kinetic Features", "features_tsne"),
        ]
        
        for plot_name, tsne_key in plot_configs:
            print(f"  {plot_name}...", end=" ", flush=True)
            
            plot_tsne(
                processed_curves["ori"][tsne_key],
                processed_curves["moving_avg"][tsne_key],
                processed_curves["sigmoid_5p"][tsne_key],
                processed_curves["Y_well"],
                n_wells,
                output_dir=plt_dir,
                plot_name=plot_name.replace(" ", "_")
            )
            print("✓")
    
    # ========================================================================
    # STEP 7: SAVE DATA
    # ========================================================================
    
    if curve_save:
        print("=" * 60)
        print("Saving Data")
        print("=" * 60)
        
        pickle_path = f'{output_dir}/{curve_name}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(processed_curves, f)
        
        print(f"✓ Pickle saved: {pickle_path}")
    
    return processed_curves

def normalise_curve(time, curves, wells):
    y_transposed = curves.T 

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    scaled_data_transposed = scaler.fit_transform(y_transposed)

    df_scaled = pd.DataFrame(
        scaled_data_transposed.T, 
        columns=np.char.add("Cycle", time.astype(str))
    )

    scaled_curves = pd.DataFrame({
        'row_min': scaler.data_min_,
        'row_max': scaler.data_max_,
        'row_range': scaler.data_range_,
        'well': wells
    })
    return pd.concat([scaled_curves, df_scaled], axis=1)

def denormalise_curve(df):
    df_unscaled = df.copy()

    time_cols = [col for col in df.columns if col.startswith("Cycle")]
    df_unscaled[time_cols] = df[time_cols].multiply(df['row_max'] - df['row_min'], axis=0).add(df['row_min'], axis=0)
    
    return df_unscaled

def plot_normalised_ac(df, output_dir, curve_name):
    unique_wells = sorted(df.well.unique())
    n_unique = len(unique_wells)
    well_to_idx = {well: i for i, well in enumerate(unique_wells)}
    
    y_wells = df.well.values
    cycle_cols = [col for col in df.columns if col.startswith('Cycle')]
    y = df[cycle_cols].values
    
    X = np.array([s.replace('Cycle', '') for s in cycle_cols]).astype(float)
    
    fig, ax = plt.subplots(2, n_unique, figsize=(n_unique * 5, 10), squeeze=False, sharey='row')
    
    df_denorm = denormalise_curve(df)
    y_denorm = df_denorm[cycle_cols].values

    for well in unique_wells:
        col_idx = well_to_idx[well]
        mask = (y_wells == well)
        
        # Row 0: Normalised
        ax[0][col_idx].plot(X, y[mask].T, color=plt.cm.tab10(col_idx % 10), alpha=0.1, linewidth=0.2, rasterized=True)
        ax[0][col_idx].plot(X, np.nanmean(y[mask], axis=0), color='black', linewidth=1.5)
        ax[0][col_idx].set_title(f"Well-{well} (Norm)")

        # Row 1: Denormalised
        ax[1][col_idx].plot(X, y_denorm[mask].T, color=plt.cm.tab10(col_idx % 10), alpha=0.1, linewidth=0.2)
        ax[1][col_idx].plot(X, np.nanmean(y_denorm[mask], axis=0), color='black', linewidth=1.5)
        ax[1][col_idx].set_title(f"Well-{well} (Original Scale)")
    
    fig.suptitle(f'Kinetic Profile Refinement: {curve_name.replace("_", " ").title()}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = f'{output_dir}/normalised-{curve_name}.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    n_wells = 10
    n_a_type = "v04"
    # exp_folder = "/Users/kautsarg/Library/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/Lim, Matthew - Gifari Run Data/trial test data"
    exp_folder = "/Users/kautsarg/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Final Project/Run Data/trial test data/"

    exp_paths = [Path(exp_folder, name) for name in os.listdir(exp_folder) if name != ".DS_Store"]
    # exp_paths = [Path(exp_folder, "D20250808_E00_C00_F4500KHz_U_Sample_7")]

    for i_path, exp_path in enumerate(exp_paths):
        path_readout_str = str(exp_path)
        print("=" * 10)
        print(exp_path)
        print(f"\n-------\nDEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")

        exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
                                            end_time_min=60, n_a_type=n_a_type,
                                            print_status=True, plt_gain_calib=False, save_gain_calib=True)
        
        # save_preprocessing_plots(exp, exp_path)
        # df_stats = export_well_analysis(exp, exp_path)

        chip_time, chip_well, chip_curves = reconstruct_data(exp, chip_data_type="well_2d_npr")
        chip_curve_data = normalise_curve(chip_time, chip_curves, chip_well)
        chip_curve_data.to_csv(f'{exp_path}/normalised_raw_data_l_df.csv',index=False)
        plot_normalised_ac(chip_curve_data, exp_path, "raw_data_l")

        chip_time, chip_well, chip_curves = reconstruct_data(exp, chip_data_type="well_2d_nl_bs_active")
        chip_curve_data = normalise_curve(chip_time, chip_curves, chip_well)
        chip_curve_data.to_csv(f'{exp_path}/normalised_raw_data_nl_df.csv',index=False)
        plot_normalised_ac(chip_curve_data, exp_path, "raw_data_nl")

        processed_curves = curve_preprocessing(exp, n_wells, allpos_curves=True, output_dir=exp_path, curve_name="min_val_inc_curves")

        ori_ac_data = normalise_curve(processed_curves["X_time"], processed_curves["ori"]["curves"], processed_curves["Y_well"])
        ori_ac_data.to_csv(f'{exp_path}/normalised_ori_ac_df.csv', index=False)
        plot_normalised_ac(ori_ac_data, exp_path, "ori_ac")

        moving_avg_data = normalise_curve(processed_curves["X_time"], processed_curves["moving_avg"]["curves"], processed_curves["Y_well"])
        moving_avg_data.to_csv(f'{exp_path}/normalised_moving_avg_df.csv',index=False)
        plot_normalised_ac(moving_avg_data, exp_path, "moving_avg")

        sigmoid_5p_data = normalise_curve(processed_curves["X_time"], processed_curves["sigmoid_5p"]["curves"], processed_curves["Y_well"])
        sigmoid_5p_data.to_csv(f'{exp_path}/normalised_sigmoid_5p_df.csv',index=False)
        plot_normalised_ac(sigmoid_5p_data, exp_path, "sigmoid_5p")

        