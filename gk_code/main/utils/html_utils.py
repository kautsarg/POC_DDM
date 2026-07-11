import base64
from io import BytesIO
import matplotlib.pyplot as plt


def _fig_to_buf(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _buf_to_img_html(buf, style="height:auto;"):
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f'<img src="data:image/png;base64,{img_b64}" style="{style}">'


def _panel(title, content_html, baseline=False):
    border = "border:2px solid #2c3e50;" if baseline else ""
    return (
        '<div class="panel" style="' + border + '">'
        f'<div class="panel-title">{title}</div>'
        f"{content_html}"
        "</div>"
    )


def build_tabbed_html(title, tabs, save_path):
    btn_html = pane_html = ""
    for i, (tab_id, tab_label, content) in enumerate(tabs):
        active = " active" if i == 0 else ""
        display = "" if i == 0 else 'style="display:none"'
        btn_html  += (f'<button class="tab-btn{active}" '
                      f'onclick="showTab(\'{tab_id}\', this)">{tab_label}</button>\n')
        pane_html += f'<div id="{tab_id}" class="tab-pane" {display}>{content}</div>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #eef0f4; margin: 0; padding: 20px; }}
  h1 {{ color: #2c3e50; font-size: 15px; margin-bottom: 14px; line-height: 1.5; }}
  .tab-nav {{ display: flex; gap: 4px; margin-bottom: 0; flex-wrap: wrap; }}
  .tab-btn {{ padding: 8px 18px; border: none; border-radius: 6px 6px 0 0; cursor: pointer;
    background: #bdc3c7; color: #2c3e50; font-size: 13px; font-weight: 600; transition: background 0.15s; }}
  .tab-btn:hover {{ background: #99a3a4; }}
  .tab-btn.active {{ background: #2c3e50; color: white; }}
  .tab-pane {{ background: white; border-radius: 0 8px 8px 8px; padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08); min-height: 200px; }}
  .panel-row {{ display: flex; flex-wrap: nowrap; overflow-x: auto;
    gap: 16px; padding: 8px 0 12px 0; align-items: flex-start; }}
  .panel {{ flex: 0 0 auto; background: #fafafa; padding: 12px;
    border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }}
  .panel-title {{ text-align: center; font-weight: 600; font-size: 13px;
    margin-bottom: 8px; color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: left; }}
  thead {{ background: #2c3e50; color: white; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
</style>
<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tab-pane').forEach(function(p) {{ p.style.display = 'none'; }});
  document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  document.getElementById(id).style.display = 'block';
  btn.classList.add('active');
}}
</script>
</head>
<body>
<h1>{title}</h1>
<div class="tab-nav">
{btn_html}</div>
{pane_html}
</body>
</html>"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
