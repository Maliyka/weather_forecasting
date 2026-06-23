"""
Corporate visual theme shared across every chart in the dashboard.

Provides:
  - A matplotlib/seaborn 'whitegrid' base (used by SHAP and decomposition fallbacks)
  - A matching Plotly template ('corporate_whitegrid') so interactive charts
    visually match static ones.
"""
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns

PRIMARY = "#1F4E79"      # corporate navy
SECONDARY = "#2E86AB"    # mid blue
ACCENT = "#E94F37"       # alert/anomaly red
SUCCESS = "#2A9D8F"      # teal
WARNING = "#F4A261"      # amber
NEUTRAL_GRID = "#D9D9D9"
PALETTE = [PRIMARY, SECONDARY, SUCCESS, WARNING, ACCENT, "#7B6D8D", "#A8DADC"]


def apply_corporate_theme() -> None:
    """Call once at app startup. Sets seaborn style + registers Plotly template."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": NEUTRAL_GRID,
        "grid.color": NEUTRAL_GRID,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
    })

    corporate_template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Segoe UI, Arial, sans-serif", size=12, color="#222222"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            colorway=PALETTE,
            xaxis=dict(gridcolor=NEUTRAL_GRID, zerolinecolor=NEUTRAL_GRID, showline=True, linecolor="#BFBFBF"),
            yaxis=dict(gridcolor=NEUTRAL_GRID, zerolinecolor=NEUTRAL_GRID, showline=True, linecolor="#BFBFBF"),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor=NEUTRAL_GRID, borderwidth=1),
            title=dict(font=dict(size=16, color=PRIMARY)),
            hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=PRIMARY),
        )
    )
    pio.templates["corporate_whitegrid"] = corporate_template
    pio.templates.default = "corporate_whitegrid"
