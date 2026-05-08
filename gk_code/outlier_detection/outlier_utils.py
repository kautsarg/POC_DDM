import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg') # Prevents GUI crashes
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

def unsupervised_line_fitting(features):
    X = np.asarray(features)
    nan_mask = np.any(np.isnan(X), axis=1)
    X_clean = X[~nan_mask]
    if len(X_clean) < 2: return None
    
    mean_X = X_clean.mean(axis=0)
    X_centered = X_clean - mean_X
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    direction_ls = Vt[0, :]
    projections_ls = X_centered @ direction_ls
    X_line_ls_original = (projections_ls[:, np.newaxis] @ direction_ls[np.newaxis, :]) + mean_X
    
    return {
        'mean': mean_X, 'direction': direction_ls, 'projections': projections_ls,
        'line_points': X_line_ls_original
    }

def calculate_msc_mahalanobis(points, q1, q2, cov_matrix):
    points = np.atleast_2d(points)
    dq = q2 - q1
    P = np.dot(points - q1, dq) / np.dot(dq, dq)
    p_proj = q1 + np.outer(P, dq)
    residual = points - p_proj
    inv_cov = np.linalg.pinv(cov_matrix)
    d_squared = np.sum(np.dot(residual, inv_cov) * residual, axis=1)
    return np.sqrt(np.clip(d_squared, 0, None))

def calculate_chi2_threshold(p_value, df=2):
    return np.sqrt(chi2.ppf(1 - p_value, df))

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def init_html_report(title, subtitle):
    return f"""
    <html><head><title>{title}</title></head>
    <body style="font-family: Arial; background-color: #f4f4f9; padding: 20px; text-align: center;">
        <h1 style="color: #333;">{title}</h1>
        <p style="color: #666; font-size: 16px; margin-bottom: 30px;">{subtitle}</p>
        <div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 20px;'>
    """