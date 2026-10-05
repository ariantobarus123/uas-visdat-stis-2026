"""Palet warna & template grafik terpusat agar seluruh visual konsisten."""
import plotly.graph_objects as go
import plotly.io as pio

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e8ecf3"
PRIMARY = "#4f46e5"
TEAL = "#0d9488"
AMBER = "#f59e0b"
ROSE = "#e11d48"
FONT = "Plus Jakarta Sans, Inter, Segoe UI, sans-serif"

ISLAND_COLORS = {
    "Sumatera": "#4f46e5",
    "Jawa": "#0ea5e9",
    "Bali & Nusa Tenggara": "#10b981",
    "Kalimantan": "#f59e0b",
    "Sulawesi": "#ec4899",
    "Maluku": "#8b5cf6",
    "Papua": "#ef4444",
}

# Klaster diurutkan dari IPM terendah -> tertinggi
CLUSTER_COLORS = ["#ef4444", "#f59e0b", "#10b981", "#4f46e5"]

CATEGORICAL = [
    "#4f46e5", "#0ea5e9", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#ef4444",
    "#14b8a6", "#f97316", "#84cc16", "#06b6d4", "#d946ef", "#64748b",
]

# Skala warna kontinu bertema
PALETTES = {
    "Indigo-Teal": [[0, "#eef2ff"], [0.35, "#818cf8"], [0.7, "#0d9488"], [1, "#064e3b"]],
    "Viridis": "Viridis",
    "Plasma": "Plasma",
    "Zamrud": ["#ecfdf5", "#a7f3d0", "#34d399", "#059669", "#064e3b"],
    "Safir": ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"],
    "Senja": "Sunsetdark",
}
PALETTE_NAMES = list(PALETTES.keys())

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family=FONT, size=12, color=INK),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=CATEGORICAL,
    title=dict(font=dict(size=15, color=INK), x=0.0, xanchor="left"),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=GRID),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f172a", font=dict(color="#f8fafc", family=FONT, size=12),
                    bordercolor="#0f172a"),
    margin=dict(l=40, r=20, t=50, b=40),
)
pio.templates["dash_bps"] = _template
pio.templates.default = "dash_bps"


SRC = "Sumber: BPS RI 2024"


def finish(fig, height=420, title=None, subtitle=None, **layout):
    """Terapkan layout standar; judul + subtitle (satuan/konteks/sumber) bila diberikan."""
    fig.update_layout(template="dash_bps", height=height, **layout)
    if title is not None:
        text = f"<b>{title}</b>"
        if subtitle:
            text += f"<br><span style='font-size:11px;color:{MUTED};font-weight:400'>{subtitle}</span>"
        fig.update_layout(title=dict(text=text))
        m = fig.layout.margin
        fig.update_layout(margin=dict(l=m.l if m.l is not None else 40, r=m.r if m.r is not None else 20,
                                      b=m.b if m.b is not None else 40, t=max(m.t or 0, 70 if subtitle else 50)))
    return fig
