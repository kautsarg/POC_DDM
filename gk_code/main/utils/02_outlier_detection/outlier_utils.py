import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg') # Prevents GUI crashes
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
import json
from bokeh.plotting import figure
from bokeh.layouts import column, row, Spacer
from bokeh.models import ColumnDataSource, CustomJS, Div, HoverTool, CrosshairTool, TapTool
from bokeh.embed import file_html
from bokeh.resources import CDN
import warnings

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

def safe_tolist(arr):
    if isinstance(arr, np.ndarray):
        arr = np.where(np.isnan(arr) | np.isinf(arr), None, arr)
        return arr.tolist()
    if isinstance(arr, list):
        return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in arr]
    return arr

def build_interactive_msc_html(save_path, title, msc_feats, Y_well, X_feats, Y_feats, Z_feats, is_outlier, curves, ref_curves, line_fits):
    plot_data = {}
    unique_wells = np.unique(Y_well)

    for well in unique_wells:
        well_mask = (Y_well == well)
        well_is_out = is_outlier[well_mask]

        X, Y, Z = X_feats[well_mask], Y_feats[well_mask], Z_feats[well_mask]
        curr_curves = curves[well_mask]
        ref_well = ref_curves[well_mask]

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_mean = np.nanmean(ref_well, axis=0)
            ref_std = np.nanstd(ref_well, axis=0)

        ref_upper = ref_mean + (3 * ref_std)
        ref_lower = ref_mean - (3 * ref_std)

        line_x, line_y, line_z = [], [], []
        well_fit = line_fits.get(well)
        if well_fit and len(well_fit.get('projections', [])) > 0:
            projections = well_fit['projections']
            p0 = well_fit['mean']
            v = well_fit['direction']
            t_min, t_max = np.min(projections), np.max(projections)
            margin = (t_max - t_min) * 0.1
            start, end = p0 + (t_min - margin) * v, p0 + (t_max + margin) * v
            line_x = [float(start[0]), float(end[0])]
            line_y = [float(start[1]), float(end[1])]
            line_z = [float(start[2]), float(end[2])]

        plot_data[int(well)] = {
            "norm": {
                "x": safe_tolist(X[~well_is_out]), "y": safe_tolist(Y[~well_is_out]), "z": safe_tolist(Z[~well_is_out]),
                "idx": safe_tolist(np.where(~well_is_out)[0])
            },
            "out": {
                "x": safe_tolist(X[well_is_out]), "y": safe_tolist(Y[well_is_out]), "z": safe_tolist(Z[well_is_out]),
                "idx": safe_tolist(np.where(well_is_out)[0])
            },
            "line": {"x": line_x, "y": line_y, "z": line_z},
            "curves": safe_tolist(curr_curves),
            "is_outlier": safe_tolist(well_is_out),
            "ref_mean": safe_tolist(ref_mean),
            "ref_upper": safe_tolist(ref_upper),
            "ref_lower": safe_tolist(ref_lower)
        }

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f0f2f5; padding: 20px; margin: 0; }}
            h2 {{ text-align: center; color: #1a1a1a; }}
            .instruction {{ text-align: center; color: #555; font-size: 15px; margin-bottom: 30px; }}
            .well-container {{ background: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-bottom: 40px; padding: 20px; max-width: 1600px; margin-left: auto; margin-right: auto; }}
            .well-header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eaeaea; padding-bottom: 12px; margin-bottom: 15px; height: 40px; }}
            h3 {{ color: #2c3e50; margin: 0; font-size: 20px; }}
            
            /* The Clear Selection Button CSS */
            .reset-btn {{ padding: 8px 16px; background-color: #e67e22; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; display: none; transition: background 0.2s; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
            .reset-btn:hover {{ background-color: #d35400; }}
            
            .info-banner {{ background: #f8f9fa; padding: 12px 15px; border-left: 4px solid #bac8d3; border-radius: 0 4px 4px 0; margin-bottom: 15px; font-family: "SFMono-Regular", Menlo, Monaco, Consolas, monospace; font-size: 14px; color: #333; }}
            .plots-wrapper {{ display: flex; width: 100%; height: 50vh; min-height: 450px; overflow: hidden; }}
            .plot3d {{ width: 45%; height: 100%; border-right: 1px solid #eaeaea; }}
            .plot2d {{ width: 55%; height: 100%; }}
        </style>
    </head>
    <body>
        <h2>Interactive MSC Outliers: {title}</h2>
        <p class="instruction">🖱️ <b>Hover</b> over points/curves for coordinates. <b>Click</b> to lock selection and cross-highlight. <b>Scroll</b> to view other wells.</p>
        
        <div id="all-wells-container"></div>

        <script>
            const plotData = {json.dumps(plot_data)};
            const mscFeats = {json.dumps(msc_feats)};
            const container = document.getElementById('all-wells-container');
            const state = {{}};

            // Global reset function attached to the button
            window.resetHighlight = function(well) {{
                const wd = plotData[well];
                const st = state[well];
                st.lockedIdx = null; 
                
                document.getElementById('btn_' + well).style.display = 'none'; // Hide button
                document.getElementById('info_' + well).innerHTML = "<i>Select a curve or 3D point to view details...</i>";
                
                // Reset 2D
                let traceIndices = Array.from({{length: wd.curves.length}}, (_, k) => k + 3);
                Plotly.restyle(st.plot2dId, {{'opacity': 0.25, 'line.width': 1.5}}, traceIndices);

                // Reset 3D (Colors, Opacity, Size)
                let normColors = wd.norm.idx.map(() => '#2980b9');
                let normOpacities = wd.norm.idx.map(() => 0.6);
                let normSizes = wd.norm.idx.map(() => 4);
                
                let outColors = wd.out.idx.map(() => '#e74c3c');
                let outOpacities = wd.out.idx.map(() => 0.8);
                let outSizes = wd.out.idx.map(() => 6);

                Plotly.restyle(st.plot3dId, {{
                    'marker.color': [normColors, outColors],
                    'marker.opacity': [normOpacities, outOpacities],
                    'marker.size': [normSizes, outSizes]
                }}, [0, 1]);
            }};

            for (const well in plotData) {{
                const wd = plotData[well];
                state[well] = {{ plot3dId: 'plot3d_' + well, plot2dId: 'plot2d_' + well, lockedIdx: null }};

                const wellDiv = document.createElement('div');
                wellDiv.className = 'well-container';
                wellDiv.innerHTML = `
                    <div class="well-header">
                        <h3>Well ${{well}}</h3>
                        <button id="btn_${{well}}" class="reset-btn" onclick="resetHighlight('${{well}}')">✖ Clear Selection</button>
                    </div>
                    <div class="info-banner" id="info_${{well}}"><i>Select a curve or 3D point to view details...</i></div>
                    <div class="plots-wrapper">
                        <div id="${{state[well].plot3dId}}" class="plot3d"></div>
                        <div id="${{state[well].plot2dId}}" class="plot2d"></div>
                    </div>
                `;
                container.appendChild(wellDiv);

                // --- 2D Plot Setup ---
                let traceMean = {{ y: wd.ref_mean, mode: 'lines', line: {{color: 'black', width: 2}}, name: 'Ref Mean', hoverinfo: 'none' }};
                let traceUp = {{ y: wd.ref_upper, mode: 'lines', line: {{color: 'gray', dash: 'dash', width: 1}}, name: '+3 Std', hoverinfo: 'none' }};
                let traceLow = {{ y: wd.ref_lower, mode: 'lines', line: {{color: 'gray', dash: 'dash', width: 1}}, name: '-3 Std', hoverinfo: 'none' }};

                let traces2d = [traceMean, traceUp, traceLow];
                
                for(let i=0; i<wd.curves.length; i++) {{
                    traces2d.push({{
                        y: wd.curves[i], mode: 'lines', line: {{ color: wd.is_outlier[i] ? '#e74c3c' : '#2980b9', width: 1.5 }},
                        opacity: 0.25, customdata: new Array(wd.curves[i].length).fill(i), name: 'Curve ' + i, showlegend: false,
                        hovertemplate: 'Idx: %{{customdata}}<br>Time: %{{x}}<br>Signal: %{{y:.2f}}<extra></extra>'
                    }});
                }}

                Plotly.newPlot(state[well].plot2dId, traces2d, {{
                    title: 'Kinetic Curves',
                    xaxis: {{ title: 'Time / Index', showspikes: true, spikemode: 'across', spikedash: 'solid', spikecolor: '#95a5a6', spikethickness: 1 }},
                    yaxis: {{ title: 'Signal', showspikes: true, spikemode: 'across', spikedash: 'solid', spikecolor: '#95a5a6', spikethickness: 1 }},
                    margin: {{t: 40, b: 40, l: 50, r: 120}}, 
                    hovermode: 'closest',
                    legend: {{ x: 1.05, y: 1 }}
                }});

                // --- 3D Plot Setup ---
                let traceNorm = {{ 
                    x: wd.norm.x, y: wd.norm.y, z: wd.norm.z, mode: 'markers', type: 'scatter3d', 
                    marker: {{color: '#2980b9', size: 4, opacity: 0.6}}, customdata: wd.norm.idx, name: 'Normal',
                    hovertemplate: 'Idx: %{{customdata}}<br>X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<extra></extra>'
                }};
                let traceOut = {{ 
                    x: wd.out.x, y: wd.out.y, z: wd.out.z, mode: 'markers', type: 'scatter3d', 
                    marker: {{color: '#e74c3c', size: 6, symbol: 'cross', opacity: 0.8}}, customdata: wd.out.idx, name: 'Outlier',
                    hovertemplate: 'Idx: %{{customdata}}<br>X: %{{x:.4f}}<br>Y: %{{y:.4f}}<br>Z: %{{z:.4f}}<extra></extra>'
                }};

                let traces3d = [traceNorm, traceOut];
                if (wd.line.x && wd.line.x.length > 0) {{
                    traces3d.push({{ x: wd.line.x, y: wd.line.y, z: wd.line.z, mode: 'lines', type: 'scatter3d', line: {{color: '#2c3e50', width: 4}}, name: 'Fitted Line', hoverinfo: 'none' }});
                }}

                Plotly.newPlot(state[well].plot3dId, traces3d, {{
                    title: 'Feature Space',
                    margin: {{l: 0, r: 0, b: 0, t: 40}},
                    scene: {{ xaxis: {{title: mscFeats[0]}}, yaxis: {{title: mscFeats[1]}}, zaxis: {{title: mscFeats[2]}} }},
                    hovermode: 'closest',
                    legend: {{ x: 1.05, y: 1 }}
                }});

                // --- Interaction Logic ---
                let plot3dDiv = document.getElementById(state[well].plot3dId);
                let plot2dDiv = document.getElementById(state[well].plot2dId);
                let infoDiv = document.getElementById('info_' + well);
                let btnDiv = document.getElementById('btn_' + well);

                function highlightIdx(localIdx) {{
                    let isOut = wd.is_outlier[localIdx];
                    let status = isOut ? "<span style='color:#e74c3c; font-weight:bold;'>OUTLIER</span>" : "<span style='color:#2980b9; font-weight:bold;'>NORMAL</span>";
                    
                    let srcArr = isOut ? wd.out : wd.norm;
                    let arrIdx = srcArr.idx.indexOf(localIdx);
                    let f1 = srcArr.x[arrIdx].toFixed(4);
                    let f2 = srcArr.y[arrIdx].toFixed(4);
                    let f3 = srcArr.z[arrIdx].toFixed(4);

                    infoDiv.innerHTML = `Curve Index: <b>${{localIdx}}</b> &nbsp;|&nbsp; Status: ${{status}} &nbsp;|&nbsp; Features (X,Y,Z): <b>${{f1}}, ${{f2}}, ${{f3}}</b>`;
                    btnDiv.style.display = 'inline-block'; // Show Clear Button

                    // Highlight 2D Curve
                    let opacities2d = new Array(wd.curves.length).fill(0.02);
                    let widths2d = new Array(wd.curves.length).fill(1);
                    opacities2d[localIdx] = 1.0;
                    widths2d[localIdx] = 4;
                    let traceIndices = Array.from({{length: wd.curves.length}}, (_, k) => k + 3);
                    Plotly.restyle(plot2dDiv, {{'opacity': opacities2d, 'line.width': widths2d}}, traceIndices);

                    // Highlight 3D Point (Change color to bright orange, massive size)
                    let highlightColor = '#f1c40f'; // Bright High-Contrast Yellow/Orange
                    
                    let normColors = wd.norm.idx.map(id => id === localIdx ? highlightColor : '#2980b9');
                    let normOpacities = wd.norm.idx.map(id => id === localIdx ? 1.0 : 0.05);
                    let normSizes = wd.norm.idx.map(id => id === localIdx ? 15 : 3);
                    
                    let outColors = wd.out.idx.map(id => id === localIdx ? highlightColor : '#e74c3c');
                    let outOpacities = wd.out.idx.map(id => id === localIdx ? 1.0 : 0.05);
                    let outSizes = wd.out.idx.map(id => id === localIdx ? 16 : 5);
                    
                    Plotly.restyle(plot3dDiv, {{ 
                        'marker.color': [normColors, outColors],
                        'marker.opacity': [normOpacities, outOpacities], 
                        'marker.size': [normSizes, outSizes] 
                    }}, [0, 1]);
                }}

                plot3dDiv.on('plotly_click', function(data) {{ if (data.points[0].curveNumber > 1) return; highlightIdx(data.points[0].customdata); }});
                plot2dDiv.on('plotly_click', function(data) {{ if (data.points[0].curveNumber < 3) return; highlightIdx(data.points[0].customdata); }});
                
            }}
        </script>
    </body>
    </html>
    """
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)