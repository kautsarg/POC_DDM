import pickle
import os
import numpy as np
import pandas as pd

class ExperimentDataLoader:
    def __init__(self, pkl_path):
        """Loads the restructured_results.pkl file and maps the data."""
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Could not find file at: {pkl_path}")
            
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
            
        # 1. Top-Level Data Blocks
        self.curves = self.data.get("curves", {})
        self.sigmoid_curves = self.data.get("sigmoid_curves", {})
        self.idxs = self.data.get("idxs", {})
        self.metadata = self.data.get("metadata", {})
        self.timestamps = self.data.get("timestamps", [])
        self.well_labels = self.data.get("well_labels", [])
        
        # 2. Top-Level Constants
        self.baseline = self.data.get("baseline_value", 0)
        self.window_size_ori = self.data.get("window_size_ori", 50)
        self.window_size_1stder = self.data.get("window_size_1stder", 200)
        self.margin = self.data.get("margin", 99)

        # Removed self._fit_map entirely!

    @property
    def baseline_value(self):
        """Returns the baseline shift applied before fitting."""
        return self.baseline

    # --- METADATA ---
    def get_metadata_df(self):
        """
        Returns the pixel metadata as a Pandas DataFrame.
        Includes well_id, row/col idx, and temp groupings.
        """
        df_meta = pd.DataFrame(self.metadata)
        df_meta.insert(0, 'well_id', self.well_labels)
        return df_meta

    # --- PROCESSED CURVES ---
    def get_curve(self, curve_name):
        """
        Returns the processed 2D numpy array for the requested curve type.
        Now includes 'well_2d_bs_active', 'well_temp_lin2d', etc.
        """
        if curve_name not in self.curves:
            raise ValueError(f"Invalid curve_name. Options: {self.list_curve_options()}")
        return self.curves.get(curve_name)

    def list_curve_options(self):
        """Returns a list of all available curve array keys."""
        return list(self.curves.keys())

    # --- FITTING RESULTS ---
    def get_fit(self, fit_name):
        """
        Returns a dictionary containing the fit results for the requested curve.
        Returns: {'fitted_full': array, 'fitted_stretched': array, 'params': array, 'rmse': array}
        """
        if fit_name not in self.sigmoid_curves:
            raise ValueError(f"Invalid fit_name. Options: {self.list_fit_options()}")
        
        # It is already perfectly formatted! Just return it directly.
        return self.sigmoid_curves[fit_name]
    
    def get_fitted_curve(self, curve_name, fit_component):
        """
        Directly extracts a specific array from the fitting results.

        Parameters:
        - curve_name: Which dataset's fit to extract (e.g., 'cleaned_lowest')
        - fit_component: Which component to extract ('fitted_full', 'fitted_stretched', 'params', 'rmse')
        """
        fit_results = self.get_fit(curve_name)
        if fit_component not in fit_results:
            raise ValueError(f"Invalid component. Options: {list(fit_results.keys())}")

        return fit_results[fit_component]

    def list_fit_options(self):
        return list(self.sigmoid_curves.keys())

    # --- CONSTANT ARRAYS ---
    def get_well_labels(self):
        return self.well_labels
    
    def get_timestamps(self):
        return self.timestamps
    
    # --- INDICES ---
    def get_cleaning_index(self, index_name):
        """Options: 'std', 'lowest', 'avg_std', 'avg_lowest', 'active', 'settled', 'start', 'end'"""
        idx_map = {
            "std": "cleaned_idx",
            "lowest": "cleaned_lowest_idx",
            "avg_std": "avg_cleaned_idx",
            "avg_lowest": "avg_cleaned_lowest_idx",
            "active": "idx_active",
            "settled": "idx_settled",
            "start": "idx_start",
            "end": "idx_end"
        }
        key = idx_map.get(index_name)
        if not key:
            raise ValueError(f"Invalid index_name. Options: {list(idx_map.keys())}")
        return self.idxs.get(key)