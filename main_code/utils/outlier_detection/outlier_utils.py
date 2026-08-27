import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Prevents GUI crashes
import matplotlib.pyplot as plt


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
