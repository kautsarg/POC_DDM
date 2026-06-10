import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from outlier_utils import init_html_report, fig_to_base64


# ====================================================================
# SHARED HELPERS
# ====================================================================

def _knn_neighbor_indices(coords_valid, k):
    """For each point, return the indices of its `k` nearest neighbors (excluding itself)."""
    n_neighbors = min(k + 1, len(coords_valid))  # +1 to account for self-match
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords_valid)
    _, neighbor_idx = nn.kneighbors(coords_valid)
    return neighbor_idx[:, 1:]


def _grid_neighbor_indices(coords_valid, window):
    """
    For each point, return the indices of points falling within a
    (2*window+1) x (2*window+1) grid window centered on it (excluding itself),
    based on exact (pixel_row_idx, pixel_col_idx) integer coordinates.
    """
    coord_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(coords_valid)}
    neighbor_lists = []
    for r, c in coords_valid:
        r, c = int(r), int(c)
        neighbors = []
        for dr in range(-window, window + 1):
            for dc in range(-window, window + 1):
                if dr == 0 and dc == 0:
                    continue
                key = (r + dr, c + dc)
                if key in coord_to_idx:
                    neighbors.append(coord_to_idx[key])
        neighbor_lists.append(neighbors)
    return neighbor_lists


def _neighbor_mean(X_valid, neighbor_indices):
    """Average curves over each point's neighbors. `neighbor_indices` may be a 2D
    array (fixed-size, KNN) or a list of variable-length lists (grid)."""
    if isinstance(neighbor_indices, np.ndarray):
        return X_valid[neighbor_indices].mean(axis=1), np.full(len(X_valid), True)

    has_neighbors = np.array([len(nbrs) > 0 for nbrs in neighbor_indices])
    means = np.zeros_like(X_valid)
    for i, nbrs in enumerate(neighbor_indices):
        if nbrs:
            means[i] = X_valid[nbrs].mean(axis=0)
    return means, has_neighbors


def _threshold_well(score, threshold_percentiles):
    """Compute per-threshold cutoffs (and elbow diagnostics) for one well's scores."""
    sorted_score = np.sort(score)
    indices_arr = np.arange(len(sorted_score))

    cutoffs = {}
    elbow_info = None
    for pct in threshold_percentiles:
        if str(pct).lower() == "elbow":
            kneedle = KneeLocator(indices_arr, sorted_score, curve="convex", direction="increasing")
            if kneedle.knee is not None:
                cutoffs[pct] = sorted_score[kneedle.knee]
                elbow_info = {"indices": indices_arr, "sorted_score": sorted_score,
                               "knee_idx": kneedle.knee, "threshold": cutoffs[pct]}
            else:
                cutoffs[pct] = np.percentile(score, 95)
                elbow_info = {"indices": indices_arr, "sorted_score": sorted_score,
                               "knee_idx": None, "threshold": cutoffs[pct]}
        else:
            cutoffs[pct] = np.percentile(score, float(pct))
    return cutoffs, elbow_info


def _save_html_report(spatial_plot_path, name, method_label, label_prefix, pct, sub_text,
                       unique_wells, Y_well, new_features, ref_curves, curves, coords_all,
                       elbow_data):
    label_col = f"{label_prefix}_label_{pct}"
    html = init_html_report(title=f"Spatial Consistency Outliers ({method_label}): {name}", subtitle=sub_text)

    for well in unique_wells:
        well_mask = (Y_well == well)
        is_outlier = (new_features.loc[well_mask, label_col] == -1).fillna(False).values
        well_coords = coords_all[well_mask]

        fig_curves, axes = plt.subplots(2, 2, figsize=(10, 8))

        if np.sum(~is_outlier) > 0:
            axes[0, 0].plot(ref_curves[well_mask][~is_outlier].T, c="blue", alpha=0.3, rasterized=True)
        axes[0, 0].set_title(f"Well {well} - Ref Normal")

        if np.sum(is_outlier) > 0:
            axes[0, 1].plot(ref_curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
        axes[0, 1].set_title(f"Well {well} - Ref Outliers")

        if np.sum(~is_outlier) > 0:
            axes[1, 0].plot(curves[well_mask][~is_outlier].T, c="green", alpha=0.3, rasterized=True)
        axes[1, 0].set_title("Curr Normal")

        if np.sum(is_outlier) > 0:
            axes[1, 1].plot(curves[well_mask][is_outlier].T, c="red", alpha=0.5, rasterized=True)
        axes[1, 1].set_title("Curr Outliers")

        plt.tight_layout()
        curves_b64 = fig_to_base64(fig_curves)

        # Spatial map: where on the chip are the flagged outliers?
        fig_map, ax_map = plt.subplots(figsize=(5, 5))
        if np.sum(~is_outlier) > 0:
            ax_map.scatter(well_coords[~is_outlier, 1], well_coords[~is_outlier, 0],
                           c="#2980b9", s=10, alpha=0.6, label="Normal")
        if np.sum(is_outlier) > 0:
            ax_map.scatter(well_coords[is_outlier, 1], well_coords[is_outlier, 0],
                           c="#e74c3c", s=14, alpha=0.9, label="Outlier")
        ax_map.set_title(f"Well {well} - Spatial Map")
        ax_map.set_xlabel("pixel_col_idx")
        ax_map.set_ylabel("pixel_row_idx")
        ax_map.invert_yaxis()
        ax_map.legend(fontsize=8)
        plt.tight_layout()
        map_b64 = fig_to_base64(fig_map)

        elbow_html = ""
        if str(pct).lower() == "elbow" and well in elbow_data:
            ed = elbow_data[well]
            fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
            ax_elbow.plot(ed["indices"], ed["sorted_score"], label="Sorted MSE", color="#2980b9", linewidth=2)
            if ed["knee_idx"] is not None:
                ax_elbow.axvline(ed["knee_idx"], color="#e74c3c", linestyle="--", linewidth=2,
                                 label=f"Elbow Cutoff (MSE={ed['threshold']:.5f})")
                ax_elbow.fill_between(ed["indices"][ed["knee_idx"]:], ed["sorted_score"][ed["knee_idx"]:],
                                      color="#e74c3c", alpha=0.2, label="Rejected Pixels")
            ax_elbow.set_title(f"Well {well} - Spatial Neighbor MSE Distribution", fontweight="bold")
            ax_elbow.set_xlabel("Pixels (Sorted by Lowest to Highest MSE)")
            ax_elbow.set_ylabel("Spatial Neighbor MSE")
            ax_elbow.grid(alpha=0.3)
            ax_elbow.legend()
            plt.tight_layout()
            elbow_b64 = fig_to_base64(fig_elbow)
            elbow_html = f"<div style='margin-top: 15px;'><img src='data:image/png;base64,{elbow_b64}' width='100%'></div>"

        html += f"""
        <div style='background: white; padding: 15px; border-radius: 8px; width: 45%; min-width: 500px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <img src='data:image/png;base64,{curves_b64}' width='100%'>
            <img src='data:image/png;base64,{map_b64}' width='60%'>
            {elbow_html}
        </div>
        """

    html += "</div></body></html>"
    save_path = os.path.join(spatial_plot_path, f"{name}_{label_prefix}_{pct}.html")
    with open(save_path, "w") as f:
        f.write(html)


def _run_spatial_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, metadata_df,
                           spatial_plot_path, threshold_percentiles, save_plot,
                           label_prefix, method_label, sub_text_fn, get_neighbor_indices,
                           min_valid_required):
    os.makedirs(spatial_plot_path, exist_ok=True)
    results_dfs = []

    unique_wells = np.unique(Y_well)
    coords_all = metadata_df[["pixel_row_idx", "pixel_col_idx"]].values.astype(float)

    for name, curves in zip(dataset_names, dataset_curves):
        clean_title = name.replace("_", " ").title()
        print(f"  -> Running Spatial Consistency [{method_label}] for {clean_title}...")

        results_filter = {}
        for pct in threshold_percentiles:
            results_filter[f"{label_prefix}_label_{pct}"] = np.full(len(Y_well), -1)

        elbow_data = {}

        for well in unique_wells:
            well_mask = (Y_well == well)
            well_indices = np.where(well_mask)[0]
            well_curves = curves[well_mask]
            well_coords = coords_all[well_mask]

            invalid_mask = (
                np.isnan(well_curves).any(axis=1) | np.isinf(well_curves).any(axis=1) |
                np.isnan(well_coords).any(axis=1)
            )
            valid_mask = ~invalid_mask
            X_valid = well_curves[valid_mask]
            coords_valid = well_coords[valid_mask]
            valid_indices = well_indices[valid_mask]

            n_valid = len(X_valid)
            if n_valid < min_valid_required:
                print(f"     [!] Skipping Well {well}: Not enough valid pixels ({n_valid}).")
                continue

            neighbor_indices = get_neighbor_indices(coords_valid)
            neighbor_mean, has_neighbors = _neighbor_mean(X_valid, neighbor_indices)

            score = np.mean((X_valid - neighbor_mean) ** 2, axis=1)
            score = score[has_neighbors]
            scored_indices = valid_indices[has_neighbors]

            if len(score) < 2:
                print(f"     [!] Skipping Well {well}: Not enough scored pixels ({len(score)}).")
                continue

            cutoffs, elbow_info = _threshold_well(score, threshold_percentiles)
            if elbow_info is not None:
                elbow_data[well] = elbow_info

            for pct in threshold_percentiles:
                label_col = f"{label_prefix}_label_{pct}"
                keep_mask = score <= cutoffs[pct]
                results_filter[label_col][scored_indices[keep_mask]] = 1

        new_features = pd.DataFrame(results_filter)
        results_dfs.append(new_features)

        if save_plot:
            for pct in threshold_percentiles:
                _save_html_report(
                    spatial_plot_path, name, method_label, label_prefix, pct,
                    sub_text_fn(pct), unique_wells, Y_well, new_features,
                    ref_curves, curves, coords_all, elbow_data
                )

        gc.collect()

    return results_dfs


# ====================================================================
# PUBLIC PIPELINES
# ====================================================================

def run_spatial_consistency_knn_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, metadata_df,
                                          spatial_plot_path, threshold_percentiles=["elbow", 90, 95],
                                          k_neighbors=24, save_plot=True):
    """
    For each pixel, finds its `k_neighbors` nearest neighbors by Euclidean distance
    in (pixel_row_idx, pixel_col_idx) space (within the same well), averages their
    curves, and scores the pixel by MSE against that local average. Pixels with the
    highest MSE per well are flagged as outliers via elbow/percentile thresholding.
    """
    def sub_text_fn(pct):
        if str(pct).lower() == "elbow":
            return f"Spatial Consistency (KNN, k={k_neighbors}) | Threshold dynamically set using the Knee/Elbow Method"
        return f"Spatial Consistency (KNN, k={k_neighbors}) | Kept bottom {pct}% of spatial-neighbor MSE per Well"

    return _run_spatial_pipeline(
        dataset_names, dataset_curves, Y_well, ref_curves, metadata_df,
        spatial_plot_path, threshold_percentiles, save_plot,
        label_prefix="spatial_knn", method_label=f"KNN k={k_neighbors}",
        sub_text_fn=sub_text_fn,
        get_neighbor_indices=lambda coords_valid: _knn_neighbor_indices(coords_valid, k_neighbors),
        min_valid_required=k_neighbors + 2,
    )


def run_spatial_consistency_grid_pipeline(dataset_names, dataset_curves, Y_well, ref_curves, metadata_df,
                                           spatial_plot_path, threshold_percentiles=["elbow", 90, 95],
                                           window=2, save_plot=True):
    """
    For each pixel, finds all pixels within a (2*window+1) x (2*window+1) grid window
    centered on it (within the same well), based on exact (pixel_row_idx, pixel_col_idx)
    coordinates, averages their curves, and scores the pixel by MSE against that local
    average. Pixels with the highest MSE per well are flagged as outliers via
    elbow/percentile thresholding.
    """
    grid_size = 2 * window + 1

    def sub_text_fn(pct):
        if str(pct).lower() == "elbow":
            return f"Spatial Consistency (Grid {grid_size}x{grid_size}) | Threshold dynamically set using the Knee/Elbow Method"
        return f"Spatial Consistency (Grid {grid_size}x{grid_size}) | Kept bottom {pct}% of spatial-neighbor MSE per Well"

    return _run_spatial_pipeline(
        dataset_names, dataset_curves, Y_well, ref_curves, metadata_df,
        spatial_plot_path, threshold_percentiles, save_plot,
        label_prefix="spatial_grid", method_label=f"Grid {grid_size}x{grid_size}",
        sub_text_fn=sub_text_fn,
        get_neighbor_indices=lambda coords_valid: _grid_neighbor_indices(coords_valid, window),
        min_valid_required=3,
    )
