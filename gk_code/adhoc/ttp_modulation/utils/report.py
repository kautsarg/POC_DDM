import base64
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Persistence ────────────────────────────────────────────────────────────────

def save_result(result, pair_results, path):
    """Persist run_cv output to JSON so the report can be regenerated without retraining."""
    def _ser_pair(r):
        return {'label': r['label'], 'mean_acc': r['mean_acc'], 'std_acc': r['std_acc'],
                'cm': r['cm'].tolist(), 'classes': r['classes'].tolist(),
                'best_fold_acc': r.get('best_fold_acc'), 'best_fold_index': r.get('best_fold_index')}

    payload = {
        'fold_accs': result['fold_accs'],
        'mean_acc': result['mean_acc'],
        'std_acc': result['std_acc'],
        'precision': result['precision'],
        'recall': result['recall'],
        'f1': result['f1'],
        'pair_results': [_ser_pair(r) for r in pair_results] if pair_results else None,
        # multi-class only fields (absent in one-to-one agg result)
        'cm': result['cm'].tolist() if 'cm' in result else None,
        'classes': result['classes'].tolist() if 'classes' in result else None,
        'y_true': result['y_true'].tolist() if 'y_true' in result else None,
        'y_pred': result['y_pred'].tolist() if 'y_pred' in result else None,
        # best CV fold (its model is saved to disk when --save_models is passed); absent in one-to-one agg result
        'best_fold_acc': result.get('best_fold_acc'),
        'best_fold_index': result.get('best_fold_index'),
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_result(path):
    """Reload a saved result dict and pair_results from JSON."""
    with open(path) as f:
        p = json.load(f)

    result = {
        'fold_accs': p['fold_accs'],
        'mean_acc': p['mean_acc'],
        'std_acc': p['std_acc'],
        'precision': p['precision'],
        'recall': p['recall'],
        'f1': p['f1'],
        'best_fold_acc': p.get('best_fold_acc'),
        'best_fold_index': p.get('best_fold_index'),
    }
    for key in ('cm', 'classes', 'y_true', 'y_pred'):
        if p.get(key) is not None:
            result[key] = np.array(p[key])

    pair_results = None
    if p.get('pair_results'):
        pair_results = [
            {'label': r['label'], 'mean_acc': r['mean_acc'], 'std_acc': r['std_acc'],
             'cm': np.array(r['cm']), 'classes': np.array(r['classes']),
             'best_fold_acc': r.get('best_fold_acc'), 'best_fold_index': r.get('best_fold_index')}
            for r in p['pair_results']
        ]
    return result, pair_results


# ── Figures ────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _cm_figure(cm, classes, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(max(4, len(classes)), max(3, len(classes))))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * 0.6 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _fold_acc_figure(fold_accs, title='Per-Fold Accuracy'):
    fig, ax = plt.subplots(figsize=(max(4, len(fold_accs) * 0.8), 3))
    x = np.arange(len(fold_accs))
    bars = ax.bar(x, fold_accs, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.axhline(np.mean(fold_accs), color='tomato', linestyle='--', linewidth=1.4,
               label=f'Mean {np.mean(fold_accs):.1f}%')
    ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in x])
    ax.set_ylabel('Accuracy (%)'); ax.set_title(title)
    ax.set_ylim(0, 110); ax.legend(fontsize=9)
    for bar, v in zip(bars, fold_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    return fig


def _pair_acc_figure(pair_results, title='Per-Pair Accuracy'):
    labels = [r['label'] for r in pair_results]
    means  = [r['mean_acc'] for r in pair_results]
    stds   = [r['std_acc'] for r in pair_results]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue',
                  edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.2})
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='Chance (50%)')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)'); ax.set_title(title)
    ax.set_ylim(0, 115); ax.legend(fontsize=9)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f'{v:.1f}',
                ha='center', va='bottom', fontsize=7)
    fig.tight_layout()
    return fig


# ── HTML assembly ──────────────────────────────────────────────────────────────

_CSS = """
body { font-family: 'Helvetica Neue', Arial, sans-serif; margin: 32px; background: #f9f9f9; color: #222; }
h1   { font-size: 1.4em; margin-bottom: 4px; }
h2   { font-size: 1.1em; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 32px; }
h3   { font-size: 0.95em; color: #555; margin-top: 20px; }
.meta { font-size: 0.82em; color: #888; margin-bottom: 24px; }
.metrics-table { border-collapse: collapse; font-size: 0.88em; margin-top: 8px; }
.metrics-table th, .metrics-table td { border: 1px solid #ccc; padding: 6px 12px; }
.metrics-table th { background: #eef; }
.metrics-table td:not(:first-child) { text-align: right; }
.figure-row { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }
.figure-row img { border: 1px solid #ddd; border-radius: 4px; max-width: 100%; }
.chip { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;
        background: #dde; color: #335; margin-right: 4px; }
"""


def _metrics_table_html(res):
    best_fold_row = ''
    if res.get('best_fold_index') is not None:
        best_fold_row = (
            f'<tr><td>Best CV fold</td>'
            f'<td>Fold {res["best_fold_index"] + 1} — {res["best_fold_acc"]:.2f}%</td></tr>'
        )
    return (
        '<table class="metrics-table">'
        '<tr><th>Metric</th><th>Value</th></tr>'
        f'<tr><td>Accuracy (mean ± std)</td><td>{res["mean_acc"]:.2f} ± {res["std_acc"]:.2f} %</td></tr>'
        f'<tr><td>Precision (macro)</td><td>{res["precision"]:.4f}</td></tr>'
        f'<tr><td>Recall (macro)</td><td>{res["recall"]:.4f}</td></tr>'
        f'<tr><td>F1 (macro)</td><td>{res["f1"]:.4f}</td></tr>'
        f'{best_fold_row}'
        '</table>'
    )


def generate_html(dataset_name, combo_results, out_path):
    sections = []
    for c in combo_results:
        res = c['result']
        flags = c['flags']
        chip_html = (
            f'<span class="chip">{"one-to-one" if flags.get("one_to_one") else "multi-class"}</span>'
            f'<span class="chip">{"TTP-aligned" if flags.get("ttp_aligned") else "unaligned"}</span>'
            f'<span class="chip">{"normalised" if flags.get("normalised") else "raw"}</span>'
        )

        pair_results = c.get('pair_results')
        if pair_results:
            # one-to-one: fold acc bar + per-pair acc bar + one CM per pair (real labels)
            acc_b64 = _fig_to_b64(_fold_acc_figure(res['fold_accs'], title='Per-Fold Accuracy'))
            pair_b64 = _fig_to_b64(_pair_acc_figure(pair_results, title='Per-Pair Accuracy'))
            figures_html = (
                f'<div class="figure-row">'
                f'<img src="data:image/png;base64,{acc_b64}" alt="Fold accuracies" />'
                f'<img src="data:image/png;base64,{pair_b64}" alt="Per-pair accuracy" />'
                f'</div>'
                f'<h3>Per-Pair Confusion Matrices</h3><div class="figure-row">'
            )
            for pr in pair_results:
                title = pr['label']
                if pr.get('best_fold_index') is not None:
                    title += f" (best fold {pr['best_fold_index'] + 1}: {pr['best_fold_acc']:.1f}%)"
                b64 = _fig_to_b64(_cm_figure(pr['cm'], pr['classes'], title=title))
                figures_html += f'<img src="data:image/png;base64,{b64}" alt="{title}" />'
            figures_html += '</div>'
        else:
            # multi-class: CM + fold acc side by side
            acc_b64 = _fig_to_b64(_fold_acc_figure(res['fold_accs'], title='Per-Fold Accuracy'))
            cm_b64 = _fig_to_b64(_cm_figure(res['cm'], res['classes'], title='Confusion Matrix'))
            figures_html = (
                f'<div class="figure-row">'
                f'<img src="data:image/png;base64,{cm_b64}" alt="Confusion matrix" />'
                f'<img src="data:image/png;base64,{acc_b64}" alt="Fold accuracies" />'
                f'</div>'
            )

        sections.append(
            f'<h2>{c["label"]}</h2>'
            f'<div>{chip_html}</div>'
            f'<h3>Metrics</h3>'
            f'{_metrics_table_html(res)}'
            f'{figures_html}'
        )

    html = (
        '<!DOCTYPE html><html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>TTP Modulation Report — {dataset_name}</title>'
        f'<style>{_CSS}</style>'
        '</head><body>'
        f'<h1>TTP Modulation Report — {dataset_name}</h1>'
        f'<p class="meta">CNN-GRU dual model · stratified K-fold CV</p>'
        + ''.join(sections)
        + '</body></html>'
    )

    with open(str(out_path), 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Report saved → {out_path}")
