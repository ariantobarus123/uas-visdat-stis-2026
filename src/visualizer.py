import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import INDICATORS_META, SHORT_LABELS, LOWER_IS_BETTER
from src.theme import (
    PALETTES, PALETTE_NAMES, ISLAND_COLORS, CLUSTER_COLORS, CATEGORICAL,
    INK, MUTED, GRID, PRIMARY, TEAL, AMBER, ROSE, SRC, finish,
)
from src.pca_analysis import CLUSTER_NAMES, LOG_FEATURES

__all__ = ["PALETTES", "PALETTE_NAMES"]

def _ver(mod):
    try:
        return tuple(int(x) for x in mod.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


NEW_PLOTLY = _ver(__import__("plotly")) >= (5, 24)   # go.Choroplethmap, px.scatter_map, cornerradius


def _round(r):
    return dict(cornerradius=r) if NEW_PLOTLY else {}


MAP_CONFIG_NOTE = "Peta bersih tanpa pin: hanya poligon wilayah administratif."

# ---------------------------------------------------------------- GAYA PETA DASAR
# Gaya bawaan tanpa token: carto-*, open-street-map, white-bg. Tambahan tanpa token: "satelit-esri".
# Gaya Mapbox (mapbox-*) memakai Mapbox Static Tiles API sebagai lapisan raster dan hanya aktif bila token
# tersedia di variabel lingkungan MAPBOX_TOKEN atau st.secrets["MAPBOX_TOKEN"]. Token TIDAK pernah ditulis di kode.
MAPBOX_STYLES = {"mapbox-dark": "dark-v11", "mapbox-light": "light-v11", "mapbox-streets": "streets-v12",
                 "mapbox-outdoors": "outdoors-v12", "mapbox-satellite": "satellite-streets-v12"}
MAP_STYLE_LABELS = {
    "carto-positron": "Terang (Positron)", "carto-darkmatter": "Gelap (Dark Matter)",
    "carto-voyager": "Voyager (berwarna lembut)", "satelit-esri": "Satelit (Esri)",
    "open-street-map": "Jalan (OpenStreetMap)", "white-bg": "Polos (siluet saja)",
    "mapbox-dark": "Mapbox · Gelap", "mapbox-light": "Mapbox · Terang", "mapbox-streets": "Mapbox · Jalan",
    "mapbox-outdoors": "Mapbox · Alam", "mapbox-satellite": "Mapbox · Satelit + jalan",
}
DARK_BASEMAPS = {"carto-darkmatter", "satelit-esri", "mapbox-dark", "mapbox-satellite"}
_ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def mapbox_token():
    """Token Mapbox dari lingkungan / st.secrets; None bila tidak ada."""
    import os
    tok = os.environ.get("MAPBOX_TOKEN")
    if not tok:
        try:
            import streamlit as st
            tok = st.secrets.get("MAPBOX_TOKEN")
        except Exception:
            tok = None
    return tok or None


def available_map_styles():
    """Daftar kunci gaya peta yang dapat dipakai (gaya Mapbox hanya bila token tersedia)."""
    keys = ["carto-positron", "carto-darkmatter", "carto-voyager", "satelit-esri", "open-street-map", "white-bg"]
    return keys + (list(MAPBOX_STYLES) if mapbox_token() else [])


def _basemap(map_style):
    """-> (style_dasar_plotly, daftar_layer_raster). Kunci tak dikenal diteruskan apa adanya sebagai style."""
    if map_style == "satelit-esri":
        return "white-bg", [dict(below="traces", sourcetype="raster", source=[_ESRI_IMAGERY],
                                 sourceattribution="Tiles © Esri — Maxar, Earthstar Geographics, dan kontributor GIS")]
    if map_style in MAPBOX_STYLES:
        tok = mapbox_token()
        if not tok:
            return "carto-darkmatter" if map_style in DARK_BASEMAPS else "carto-positron", []
        url = (f"https://api.mapbox.com/styles/v1/mapbox/{MAPBOX_STYLES[map_style]}/tiles/256/{{z}}/{{x}}/{{y}}@2x"
               f"?access_token={tok}")
        return "white-bg", [dict(below="traces", sourcetype="raster", source=[url],
                                 sourceattribution="© Mapbox © OpenStreetMap")]
    return map_style, []


def _set_layers(fig, layers):
    if layers:
        fig.update_layout(**({"map": {"layers": layers}} if NEW_PLOTLY else {"mapbox": {"layers": layers}}))


def get_meta_label(col):
    return INDICATORS_META.get(col, {}).get("label", str(col).replace("_", " ").title())


def get_short_label(col):
    return SHORT_LABELS.get(col, str(col).replace("_", " ").title())


def get_meta_unit(col):
    return INDICATORS_META.get(col, {}).get("unit", "")


def resolve_palette(name):
    return PALETTES.get(name, name)


def _discrete_map(color_by):
    if color_by == "Pulau":
        return ISLAND_COLORS
    if color_by == "Klaster":
        return dict(zip(CLUSTER_NAMES, CLUSTER_COLORS))
    return None


def _map_view(df, fallback_zoom=4.1):
    """Pusat & zoom peta menyesuaikan sebaran wilayah yang sedang ditampilkan."""
    if df is None or df.empty or len(df) > 400:
        return -2.4, 118.0, fallback_zoom
    lat_span = df["Latitude"].max() - df["Latitude"].min()
    lon_span = df["Longitude"].max() - df["Longitude"].min()
    span = max(lat_span * 1.15, lon_span * 0.62, 0.35)
    zoom = float(np.clip(np.log2(360 / (span * 1.9 + 0.9)) - 0.3, 3.9, 9.0))
    return float(df["Latitude"].mean()), float(df["Longitude"].mean()), zoom


# ---------------------------------------------------------------- PETA
def plot_choropleth_map(df, geojson, indicator="IPM", color_scale="Viridis",
                        highlight_kab=None, map_style="carto-positron", height=560, clip=(2, 98)):
    """Peta choropleth poligon kabupaten/kota (tanpa marker/pin)."""
    label = get_meta_label(indicator)
    unit = get_meta_unit(indicator)
    hover = (
        "<b>%{customdata[0]}</b><br>"
        "<span style='color:#94a3b8'>%{customdata[1]}</span><br><br>"
        f"{get_short_label(indicator)}: <b>%{{z:,.2f}}</b><br>"
        "IPM: <b>%{customdata[2]:.2f}</b><br>"
        "Kemiskinan: <b>%{customdata[3]:.2f}%</b><br>"
        "TPT: <b>%{customdata[4]:.2f}%</b><br>"
        "Penduduk: <b>%{customdata[5]:,.0f}</b> jiwa<extra></extra>"
    )
    customdata = df[["Kabupaten/Kota", "Provinsi", "IPM", "Persentase_Penduduk_Miskin_2024",
                     "TPT_Jumlah", "jumlah_penduduk"]].values
    Choro = go.Choroplethmap if NEW_PLOTLY else go.Choroplethmapbox
    vals = df[indicator].dropna()
    zmin, zmax = (np.percentile(vals, clip[0]), np.percentile(vals, clip[1])) if len(vals) > 20 else (vals.min(), vals.max())
    fig = go.Figure(Choro(
        geojson=geojson, locations=df["Kabupaten/Kota"], featureidkey="id", z=df[indicator], zmin=zmin, zmax=zmax,
        colorscale=color_scale, marker_opacity=0.74 if map_style in ("satelit-esri", "mapbox-satellite") else 0.86,
        marker_line_width=0.4,
        marker_line_color="rgba(255,255,255,0.7)" if map_style in DARK_BASEMAPS else "rgba(71,85,105,0.55)",
        customdata=customdata, hovertemplate=hover,
        colorbar=dict(title=dict(text=f"{unit}<br>(warna dipotong P{clip[0]}–P{clip[1]})" if len(vals) > 20 else unit, side="top", font=dict(size=10, color=MUTED)),
                      thickness=10, len=0.7, x=0.985, y=0.5, tickfont=dict(size=10, color=MUTED),
                      outlinewidth=0, bgcolor="rgba(255,255,255,0.0)"),
    ))
    if highlight_kab and highlight_kab in df["Kabupaten/Kota"].values:
        fig.add_trace(Choro(
            geojson=geojson, locations=[highlight_kab], featureidkey="id", z=[1],
            colorscale=[[0, ROSE], [1, ROSE]], showscale=False, marker_opacity=0.92,
            marker_line_width=2.4, marker_line_color="#ffffff", hoverinfo="skip",
        ))
    lat, lon, zoom = _map_view(df)
    base_style, layers = _basemap(map_style)
    view = dict(style=base_style, center=dict(lat=lat, lon=lon), zoom=zoom)
    fig.update_layout(**({"map": view} if NEW_PLOTLY else {"mapbox": view}),
                      margin=dict(r=0, t=0, l=0, b=0), height=height)
    _set_layers(fig, layers)
    return fig


def plot_proportional_symbol_map(df, size_col="jumlah_penduduk", color_col="IPM",
                                 color_scale="Plasma", map_style="carto-positron", height=560):
    """Peta simbol proporsional: ukuran = penduduk, warna = indikator."""
    df = df.dropna(subset=["Latitude", "Longitude", color_col])
    scatter_fn = px.scatter_map if NEW_PLOTLY else px.scatter_mapbox
    base_style, layers = _basemap(map_style)
    style_kw = {"map_style": base_style} if NEW_PLOTLY else {"mapbox_style": base_style}
    fig = scatter_fn(
        df, lat="Latitude", lon="Longitude", size=size_col, color=color_col,
        hover_name="Kabupaten/Kota",
        hover_data={"Provinsi": True, size_col: ":,.0f", color_col: ":.2f",
                    "Latitude": False, "Longitude": False},
        color_continuous_scale=color_scale, size_max=34,
        center={"lat": _map_view(df)[0], "lon": _map_view(df)[1]}, zoom=_map_view(df)[2],
        **style_kw,
    )
    fig.update_traces(marker=dict(opacity=0.72))
    fig.update_layout(
        margin=dict(r=0, t=0, l=0, b=0), height=height,
        coloraxis_colorbar=dict(title=dict(text=get_short_label(color_col), side="top",
                                           font=dict(size=11, color=MUTED)),
                                thickness=10, len=0.7, outlinewidth=0),
    )
    _set_layers(fig, layers)
    return fig


# ---------------------------------------------------------------- PCA
def plot_pca_scatter(df, x_col="PC1", y_col="PC2", color_by="Pulau", selected_kab=None,
                     show_biplot=False, loadings_df=None, height=600):
    custom = ["Kabupaten/Kota", "Provinsi", "IPM", "Persentase_Penduduk_Miskin_2024",
              "TPT_Jumlah", "pdrb_perkapita_adhk", "jumlah_penduduk"]
    hover = (
        "<b>%{customdata[0]}</b><br><span style='color:#94a3b8'>%{customdata[1]}</span><br><br>"
        f"{x_col}: <b>%{{x:.2f}}</b>  ·  {y_col}: <b>%{{y:.2f}}</b><br>"
        "IPM: <b>%{customdata[2]:.2f}</b><br>"
        "Kemiskinan: <b>%{customdata[3]:.2f}%</b><br>"
        "TPT: <b>%{customdata[4]:.2f}%</b><br>"
        "PDRB/kap ADHK: <b>%{customdata[5]:,.0f}</b><br>"
        "Penduduk: <b>%{customdata[6]:,.0f}</b><extra></extra>"
    )
    numeric = pd.api.types.is_numeric_dtype(df[color_by])
    kw = dict(color_continuous_scale="Viridis") if numeric else dict(
        color_discrete_map=_discrete_map(color_by), color_discrete_sequence=CATEGORICAL)
    if color_by == "Klaster":
        kw["category_orders"] = {"Klaster": CLUSTER_NAMES}
    fig = px.scatter(df, x=x_col, y=y_col, color=color_by, custom_data=custom, **kw)
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0.6, color="#ffffff")),
                      hovertemplate=hover, selector=dict(type="scatter"))
    fig.update_traces(unselected=dict(marker=dict(opacity=0.35)))
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#cbd5e1", line_width=1)

    if selected_kab and selected_kab in df["Kabupaten/Kota"].values:
        row = df[df["Kabupaten/Kota"] == selected_kab].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row[x_col]], y=[row[y_col]], mode="markers+text", name="Wilayah terpilih",
            marker=dict(size=19, color=INK, symbol="diamond-open", line=dict(width=3, color=INK)),
            text=[f"  <b>{selected_kab}</b>"], textposition="top right",
            textfont=dict(size=12, color=INK), hoverinfo="skip",
        ))

    if show_biplot and loadings_df is not None and x_col in loadings_df and y_col in loadings_df:
        sx = df[x_col].abs().max() * 0.8
        sy = df[y_col].abs().max() * 0.8
        for var in loadings_df.index:
            vx, vy = loadings_df.loc[var, x_col] * sx, loadings_df.loc[var, y_col] * sy
            fig.add_annotation(ax=0, ay=0, x=vx, y=vy, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.5,
                               arrowcolor="rgba(225,29,72,0.75)")
            fig.add_annotation(x=vx * 1.1, y=vy * 1.1, text=get_short_label(var), showarrow=False,
                               font=dict(size=10, color="#be123c"),
                               bgcolor="rgba(255,255,255,0.7)")
    finish(fig, height, xaxis_title=f"<b>{x_col}</b>", yaxis_title=f"<b>{y_col}</b>",
           legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
           margin=dict(l=40, r=20, t=40, b=40), clickmode="event+select")
    return fig


def plot_scree(explained_variance, cumulative_variance, pc_cols, height=380):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pc_cols, y=explained_variance * 100, name="Varians per komponen",
                         marker=dict(color=PRIMARY, **_round(6)),
                         text=[f"{v * 100:.1f}%" for v in explained_variance],
                         textposition="outside", textfont=dict(size=11, color=INK),
                         hovertemplate="%{x}: <b>%{y:.2f}%</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=pc_cols, y=cumulative_variance * 100, name="Kumulatif",
                             yaxis="y2", mode="lines+markers", line=dict(color=TEAL, width=3),
                             marker=dict(size=9, color=TEAL, line=dict(color="#fff", width=2)),
                             hovertemplate="%{x}: <b>%{y:.2f}%</b><extra></extra>"))
    fig.add_hline(y=70, line_dash="dot", line_color=AMBER, line_width=1.5, yref="y2",
                  annotation_text="Ambang 70%", annotation_position="bottom right",
                  annotation_font=dict(color="#b45309", size=10))
    finish(fig, height, title="Scree Plot: Varians yang Dijelaskan", subtitle="Persen varians total per komponen utama · garis putus = 70%",
           yaxis=dict(title="Varians (%)", range=[0, float(max(explained_variance) * 130)], gridcolor=GRID),
           yaxis2=dict(title="Kumulatif (%)", side="right", overlaying="y", range=[0, 105],
                       showgrid=False, tickmode="array", tickvals=[0, 25, 50, 75, 100]),
           legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
           margin=dict(t=85, b=40, l=40, r=50),
           bargap=0.45)
    return fig


def plot_loadings_heatmap(loadings_df, height=420):
    z = loadings_df.values
    fig = go.Figure(go.Heatmap(
        z=z, x=loadings_df.columns.tolist(), y=[get_short_label(i) + (" (log)" if i in LOG_FEATURES else "") for i in loadings_df.index],
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1, text=np.round(z, 2),
        texttemplate="%{text:.2f}", textfont=dict(size=11), xgap=3, ygap=3,
        colorbar=dict(title="Loading", thickness=10, outlinewidth=0),
        hovertemplate="%{y} → %{x}: <b>%{z:.3f}</b><extra></extra>"))
    finish(fig, height, title="Matriks Loading Komponen Utama", subtitle="Korelasi variabel (terstandardisasi; PDRB & penduduk di-log) dengan komponen",
           yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
           xaxis=dict(gridcolor="rgba(0,0,0,0)"), margin=dict(l=110, r=10, t=50, b=30))
    return fig


# ------------------------------------------------------------ MULTIVARIAT
def plot_parallel_coordinates(df, selected_dims=None, color_col="IPM", height=500):
    if not selected_dims or len(selected_dims) < 3:
        selected_dims = ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah",
                         "pdrb_perkapita_adhk", "sanitasi_layak", "air_minum_layak"]
    df = df.dropna(subset=list(selected_dims) + [color_col])
    dims = []
    for c in selected_dims:
        neg = c in LOWER_IS_BETTER
        dims.append(dict(range=[df[c].max(), df[c].min()] if neg else [df[c].min(), df[c].max()],
                         label=get_short_label(c) + (" ↓" if neg else ""), values=df[c]))
    fig = go.Figure(go.Parcoords(
        line=dict(color=df[color_col], colorscale="Viridis", showscale=True,
                  colorbar=dict(title=dict(text=get_short_label(color_col)), thickness=10,
                                outlinewidth=0)),
        dimensions=dims,
        labelfont=dict(size=12, color=INK), tickfont=dict(size=10, color=MUTED)))
    finish(fig, height, margin=dict(t=60, b=30, l=70, r=70))
    return fig


def plot_correlation_heatmap(df, cols=None, height=520, method="pearson"):
    if not cols or len(cols) < 2:
        cols = ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "pdrb_perkapita_adhk",
                "pertumbuhan_ekonomi", "sanitasi_layak", "air_minum_layak"]
    labels = [get_short_label(c) for c in cols]
    corr = df[cols].corr(method=method)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels, colorscale="RdBu_r", zmin=-1, zmax=1, zmid=0,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=11), xgap=3, ygap=3,
        colorbar=dict(title="r" if method == "pearson" else "ρ", thickness=10, outlinewidth=0),
        hovertemplate="%{y} × %{x}: <b>koef. = %{z:.3f}</b><extra></extra>"))
    finish(fig, height, title=f"Matriks Korelasi {method.capitalize()}", subtitle=f"Seluruh kabupaten/kota (tidak terpengaruh filter) · n = {len(df)} · {SRC}",
           yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
           xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-35),
           margin=dict(l=110, r=10, t=50, b=90))
    return fig


def plot_bivariate(df, x, y, height=450):
    sub = df[[x, y, "Pulau", "Kabupaten/Kota"]].dropna()
    r = sub[x].corr(sub[y])
    fig = px.scatter(sub, x=x, y=y, color="Pulau", color_discrete_map=ISLAND_COLORS,
                     hover_name="Kabupaten/Kota")
    fig.update_traces(marker=dict(size=8, opacity=0.78, line=dict(width=0.5, color="#fff")))
    if len(sub) > 2 and sub[x].nunique() > 1:
        m, b = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Tren linear (OLS)",
                                 line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
    finish(fig, height, title=f"{get_short_label(x)} vs {get_short_label(y)}", subtitle=f"r Pearson = {r:+.2f} · garis putus = tren linear (OLS) · n = {len(sub)} · {SRC}",
           xaxis_title=get_meta_label(x), yaxis_title=get_meta_label(y),
           legend=dict(orientation="h", y=-0.2, x=0, title=None))
    return fig, r


def plot_kabupaten_radar(df, kabupaten_name, indicators=None, height=380):
    indicators = indicators or ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah",
                                "sanitasi_layak", "air_minum_layak", "pertumbuhan_ekonomi"]
    row = df[df["Kabupaten/Kota"] == kabupaten_name].iloc[0]
    kab, nat, labels = [], [], []
    for ind in indicators:
        lo, hi = df[ind].min(), df[ind].max()
        scale = (hi - lo) or 1
        k = (row[ind] - lo) / scale * 100 if pd.notna(row[ind]) else np.nan
        n = (df[ind].mean() - lo) / scale * 100
        if ind in LOWER_IS_BETTER:           # dibalik: skor tinggi = kondisi lebih baik untuk SEMUA sumbu
            k, n = 100 - k, 100 - n
        kab.append(k)
        nat.append(n)
        labels.append(get_short_label(ind) + (" (dibalik)" if ind in LOWER_IS_BETTER else ""))
    close = lambda v: v + [v[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=close(nat), theta=close(labels), fill="toself",
                                  fillcolor="rgba(148,163,184,0.18)", name="Rata-rata nasional",
                                  line=dict(color="#94a3b8", width=1.6, dash="dot"),
                                  hovertemplate="%{theta}: %{r:.0f}/100<extra>Nasional</extra>"))
    fig.add_trace(go.Scatterpolar(r=close(kab), theta=close(labels), fill="toself",
                                  fillcolor="rgba(79,70,229,0.25)", name=kabupaten_name,
                                  line=dict(color=PRIMARY, width=3),
                                  hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>"))
    finish(fig, height, title="Profil vs Rata-rata Nasional",
           subtitle="Skor 0–100 (min–maks antar wilayah); semakin ke luar = kondisi semakin baik",
           polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False,
                                      gridcolor=GRID, linecolor=GRID),
                      angularaxis=dict(linecolor=GRID, gridcolor=GRID),
                      bgcolor="rgba(0,0,0,0)"),
           legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
           margin=dict(t=55, b=35, l=50, r=50))
    return fig


# ------------------------------------------------------------ RINGKASAN
def plot_rank_bar(df, indicator, n=5, top=True, height=260, ref=None):
    sub = (df.nlargest(n, indicator) if top else df.nsmallest(n, indicator)).iloc[::-1]
    color = TEAL if top else ROSE
    fig = go.Figure(go.Bar(
        x=sub[indicator], y=sub["Kabupaten/Kota"], orientation="h",
        marker=dict(color=color, **_round(5)),
        text=[f"{v:,.2f}" for v in sub[indicator]], textposition="auto",
        insidetextfont=dict(color="#fff", size=11), outsidetextfont=dict(color=INK, size=11),
        customdata=sub[["Provinsi"]].values,
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{x:,.2f}<extra></extra>"))
    if ref is not None:
        fig.add_vline(x=ref, line_dash="dot", line_color=INK, line_width=1.2,
                      annotation_text=f"Rerata {ref:,.1f}", annotation_position="top",
                      annotation_font=dict(size=10, color=MUTED))
    finish(fig, height, margin=dict(l=10, r=10, t=28 if ref is not None else 10, b=10),
           xaxis=dict(visible=False), yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True),
           bargap=0.35)
    return fig


def plot_island_bar(df, indicator, national_value=None, height=330):
    g = df.groupby("Pulau")[indicator].mean().sort_values()
    fig = go.Figure(go.Bar(
        x=g.values, y=g.index, orientation="h",
        marker=dict(color=[ISLAND_COLORS.get(i, PRIMARY) for i in g.index], **_round(5)),
        text=[f"{v:,.2f}" for v in g.values], textposition="auto",
        insidetextfont=dict(color="#fff", size=11), outsidetextfont=dict(color=INK, size=11),
        hovertemplate="<b>%{y}</b><br>Rata-rata: %{x:,.2f}<extra></extra>"))
    avg = df[indicator].mean()
    fig.add_vline(x=avg, line_dash="dot", line_color=INK, line_width=1.2,
                  annotation_text=f"Rerata {avg:,.1f}", annotation_position="top",
                  annotation_font=dict(size=10, color=MUTED))
    finish(fig, height, title="Rata-rata per Pulau / Wilayah", subtitle=f"{get_short_label(indicator)} · rerata tak berbobot penduduk · {SRC}",
           margin=dict(l=10, r=10, t=55, b=20), xaxis=dict(visible=False),
           yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True), bargap=0.3)
    return fig


def plot_distribution(df, indicator, height=330):
    fig = px.histogram(df, x=indicator, nbins=32, color_discrete_sequence=[PRIMARY],
                       marginal="box", hover_name="Kabupaten/Kota")
    fig.update_traces(marker_line_color="#fff", marker_line_width=1, selector=dict(type="histogram"))
    med = df[indicator].median()
    fig.add_vline(x=med, line_dash="dot", line_color=ROSE, line_width=1.5,
                  annotation_text=f"Median {med:,.1f}", annotation_font=dict(size=10, color=ROSE))
    finish(fig, height, title="Sebaran Nilai antar Kabupaten/Kota", subtitle=f"{get_short_label(indicator)} · n = {df[indicator].notna().sum()} · {SRC}",
           xaxis_title=get_short_label(indicator), yaxis_title="Jumlah wilayah",
           margin=dict(l=40, r=10, t=55, b=40), showlegend=False)
    return fig


def plot_bubble_overview(df, x="IPM", y="Persentase_Penduduk_Miskin_2024", height=430):
    fig = px.scatter(df, x=x, y=y, size="jumlah_penduduk", color="Pulau",
                     color_discrete_map=ISLAND_COLORS, hover_name="Kabupaten/Kota",
                     hover_data={"Provinsi": True, "jumlah_penduduk": ":,.0f", x: ":.2f", y: ":.2f",
                                 "Pulau": False},
                     size_max=38)
    fig.update_traces(marker=dict(opacity=0.62, line=dict(width=0.7, color="#fff")))
    sub = df[[x, y]].dropna()
    r = sub[x].corr(sub[y])
    if len(sub) > 2:
        m, b = np.polyfit(sub[x], sub[y], 1)
        xs = np.linspace(sub[x].min(), sub[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Tren linear",
                                 line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
    finish(fig, height, title=f"{get_short_label(x)} vs {get_short_label(y)}",
           subtitle=f"r Pearson = {r:+.2f} · ukuran gelembung = jumlah penduduk · n = {len(sub)} · {SRC}",
           xaxis_title=get_meta_label(x), yaxis_title=get_meta_label(y),
           legend=dict(orientation="h", y=-0.18, x=0, title=None))
    return fig


def plot_cluster_profile(df, height=380):
    """Profil rata-rata tiap klaster pada indikator utama (skala 0-100 min–maks)."""
    inds = ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "pdrb_perkapita_adhk",
            "sanitasi_layak", "air_minum_layak"]
    lo, hi = df[inds].min(), df[inds].max()
    fig = go.Figure()
    for name, color in zip(CLUSTER_NAMES, CLUSTER_COLORS):
        sub = df[df["Klaster"] == name]
        if sub.empty:
            continue
        # pdrb sangat miring -> gunakan median agar tidak didominasi outlier
        vals = [(sub[i].median() - lo[i]) / ((hi[i] - lo[i]) or 1) * 100 for i in inds]
        fig.add_trace(go.Bar(x=[get_short_label(i) for i in inds], y=vals, name=f"{name} (n={len(sub)})",
                             marker=dict(color=color, **_round(4)),
                             hovertemplate="%{x}: %{y:.0f}/100<extra>" + name + "</extra>"))
    finish(fig, height, title="Profil Median per Klaster", subtitle="Median tiap indikator, skala min–maks 0–100 (kemiskinan & TPT tidak dibalik: nilai tinggi = lebih buruk)", barmode="group",
           yaxis=dict(range=[0, 100], title=None), legend=dict(orientation="h", y=-0.2, x=0),
           bargap=0.2)
    return fig


# ------------------------------------------------------------ TEMUAN UTAMA
def plot_highlight_scatter(df, x, y, mask, highlight_label, title, subtitle, logx=False,
                           hline=None, vline=None, height=440):
    """Scatter posisi (x, y) dengan subset wilayah disorot; opsional garis kuadran."""
    sub = df.dropna(subset=[x, y]).copy()
    m = mask.reindex(sub.index).fillna(False)
    hover = ("<b>%{customdata[0]}</b><br><span style='color:#94a3b8'>%{customdata[1]}</span><br>"
             f"{get_short_label(x)}: <b>%{{x:,.2f}}</b><br>{get_short_label(y)}: <b>%{{y:,.2f}}</b><extra></extra>")
    fig = go.Figure()
    for flag, name, color, size, op in [(False, "Wilayah lain", "#94a3b8", 6, 0.45), (True, highlight_label, ROSE, 10, 0.95)]:
        s = sub[m == flag]
        fig.add_trace(go.Scatter(x=s[x], y=s[y], mode="markers", name=f"{name} ({len(s)})",
                                 marker=dict(size=size, color=color, opacity=op, line=dict(width=0.6, color="#fff")),
                                 customdata=s[["Kabupaten/Kota", "Provinsi"]].values, hovertemplate=hover))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color=MUTED, line_width=1.2)
    if vline is not None:
        fig.add_vline(x=vline, line_dash="dot", line_color=MUTED, line_width=1.2)
    finish(fig, height, title=title, subtitle=subtitle, xaxis_title=get_meta_label(x), yaxis_title=get_meta_label(y),
           legend=dict(orientation="h", y=-0.2, x=0, title=None),
           xaxis=dict(type="log" if logx else "linear", showgrid=False))
    return fig


def plot_gap_bar(df, n=10, height=380):
    """Selisih TPAK laki-laki - perempuan: n wilayah dengan kesenjangan terbesar."""
    g = df.assign(gap=df["TPAK_laki-laki"] - df["TPAK_perempuan"]).dropna(subset=["gap"]).nlargest(n, "gap").iloc[::-1]
    fig = go.Figure(go.Bar(x=g["gap"], y=g["Kabupaten/Kota"], orientation="h", marker=dict(color=PRIMARY, **_round(5)),
                           text=[f"{v:.1f}" for v in g["gap"]], textposition="auto",
                           insidetextfont=dict(color="#fff", size=11),
                           customdata=g[["Provinsi", "TPAK_laki-laki", "TPAK_perempuan"]].values,
                           hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>TPAK L: %{customdata[1]:.1f}% · P: %{customdata[2]:.1f}%<br>Selisih: <b>%{x:.1f}</b> poin<extra></extra>"))
    med = (df["TPAK_laki-laki"] - df["TPAK_perempuan"]).median()
    fig.add_vline(x=med, line_dash="dot", line_color=INK, line_width=1.2, annotation_text=f"Median nasional {med:.1f}",
                  annotation_position="top", annotation_font=dict(size=10, color=MUTED))
    finish(fig, height, title=f"{n} Wilayah dengan Selisih TPAK L–P Terbesar",
           subtitle=f"Selisih dalam poin persentase · {SRC}", xaxis=dict(visible=False),
           yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True), margin=dict(l=10, r=10, t=70, b=20), bargap=0.3)
    return fig


def plot_province_range(df, n=10, height=380):
    """Rentang IPM (maks - min) di dalam provinsi: seberapa timpang antar kab/kota dalam satu provinsi."""
    g = df.groupby("Provinsi")["IPM"].agg(["min", "max", "count"])
    g = g[g["count"] >= 3].assign(rentang=lambda t: t["max"] - t["min"]).nlargest(n, "rentang").iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["min"], y=g.index, mode="markers", name="IPM terendah", marker=dict(size=10, color=ROSE)))
    fig.add_trace(go.Scatter(x=g["max"], y=g.index, mode="markers", name="IPM tertinggi", marker=dict(size=10, color=TEAL)))
    for prov, row in g.iterrows():
        fig.add_shape(type="line", x0=row["min"], x1=row["max"], y0=prov, y1=prov, line=dict(color="#cbd5e1", width=3), layer="below")
    finish(fig, height, title=f"{n} Provinsi dengan Rentang IPM Terlebar",
           subtitle=f"Titik = IPM kab/kota terendah & tertinggi di provinsi (provinsi dengan ≥3 kab/kota) · {SRC}",
           xaxis_title="IPM (poin)", legend=dict(orientation="h", y=-0.2, x=0, title=None),
           yaxis=dict(gridcolor="rgba(0,0,0,0)", automargin=True), margin=dict(l=10, r=10, t=70, b=40))
    return fig


# ---------------------------------------------------------------- HIRARKI WILAYAH (ICICLE / TREEMAP / SUNBURST)
ICICLE_INDICATORS = [
    "IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "TPAK_laki-laki", "TPAK_perempuan",
    "pdrb_perkapita_adhk", "pertumbuhan_ekonomi", "sanitasi_layak", "air_minum_layak", "jumlah_penduduk",
]
HIERARCHY_INDICATORS = ICICLE_INDICATORS
HIERARCHY_KINDS = ["Icicle", "Treemap", "Sunburst"]


def _wmean(values, weights):
    """Rata-rata berbobot yang mengabaikan nilai kosong; NaN bila tidak ada data."""
    m = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[m], weights=weights[m])) if m.any() else np.nan


def _hierarchy_data(df, color_col, color_scale, root_label, clip):
    """Susun node hirarki Indonesia -> Provinsi -> Kabupaten/Kota dari salinan lokal dataframe (df tidak diubah)."""
    ind_cols = {"IPM": "IPM", "kemiskinan": "Persentase_Penduduk_Miskin_2024", "tpt": "TPT_Jumlah",
                "pdrb": "pdrb_perkapita_adhk"}
    need = ["Provinsi", "Kabupaten/Kota", "jumlah_penduduk", color_col, *ind_cols.values()]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Kolom tidak tersedia untuk visualisasi hirarki: {missing}")

    h_df = df.copy()
    h_df = h_df[h_df["jumlah_penduduk"].notna() & (h_df["jumlah_penduduk"] > 0)]
    h_df = h_df[h_df["Provinsi"].notna() & h_df["Kabupaten/Kota"].notna()]
    if h_df.empty:
        return None
    single_prov = h_df["Provinsi"].nunique() == 1
    if single_prov:                                   # satu provinsi: provinsi menjadi akar
        root_label = h_df["Provinsi"].iloc[0]
    pop_color = color_col == "jumlah_penduduk"        # warna = penduduk: rata-rata berbobot tidak bermakna -> rata-rata sederhana

    def stats(sub):
        w = sub["jumlah_penduduk"]
        out = {k: _wmean(sub[c], w) for k, c in ind_cols.items()}
        out["color"] = float(sub[color_col].mean()) if pop_color else _wmean(sub[color_col], w)
        out["pop"] = float(w.sum())
        return out

    def hover(name, prov, s, n_leaf, is_leaf):
        lines = [f"<b>{name}</b>"]
        if is_leaf:
            sub = f"Kabupaten/Kota · Provinsi: {prov}"
        elif n_leaf == "all":
            sub = "Tingkat wilayah terpilih (akar)"
        else:
            sub = "Provinsi"
        lines.append(f"<span style='color:#94a3b8'>{sub}</span>")
        lines.append("")
        lines.append(f"Jumlah Penduduk: <b>{s['pop']:,.0f}</b> jiwa")
        if not is_leaf:
            lines.append(f"Jumlah kab/kota: <b>{n_leaf if n_leaf != 'all' else len(h_df)}</b>")
            lines.append("<span style='color:#94a3b8'>Nilai indikator = "
                         + ("rata-rata per kab/kota" if pop_color else "rata-rata berbobot penduduk") + "</span>")
        if color_col not in ind_cols.values():
            val = s["color"]
            unit = get_meta_unit(color_col)
            lines.append(f"▸ {get_short_label(color_col)}: <b>{val:,.2f}</b> {unit}" if pd.notna(val) else f"▸ {get_short_label(color_col)}: n/a")
        for key, lab, f in (("IPM", "IPM", "{:.2f}"), ("kemiskinan", "Kemiskinan", "{:.2f}%"), ("tpt", "TPT", "{:.2f}%"),
                            ("pdrb", "PDRB/kap ADHK", "{:,.0f}")):
            if pd.notna(s[key]):
                mark = "▸ " if ind_cols[key] == color_col else ""
                lines.append(f"{mark}{lab}: <b>{f.format(s[key])}</b>")
        return "<br>".join(lines)

    ids, labels, parents, values, colors, texts = [], [], [], [], [], []
    s_root = stats(h_df)
    ids.append("root"); labels.append(root_label); parents.append("")
    values.append(s_root["pop"]); colors.append(s_root["color"])
    texts.append(hover(root_label, None, s_root, "all", False))

    for prov, g in h_df.groupby("Provinsi", sort=True):
        if not single_prov:
            s_p = stats(g)
            ids.append(f"P|{prov}"); labels.append(prov); parents.append("root")
            values.append(s_p["pop"]); colors.append(s_p["color"]); texts.append(hover(prov, None, s_p, len(g), False))
        for _, r in g.iterrows():
            s_k = stats(g.loc[[r.name]])
            ids.append(f"K|{prov}|{r['Kabupaten/Kota']}"); labels.append(r["Kabupaten/Kota"])
            parents.append("root" if single_prov else f"P|{prov}")
            values.append(s_k["pop"]); colors.append(s_k["color"]); texts.append(hover(r["Kabupaten/Kota"], prov, s_k, 1, True))

    leaf_vals = h_df[color_col].dropna()
    if len(leaf_vals) > 20:
        cmin, cmax = np.percentile(leaf_vals, clip[0]), np.percentile(leaf_vals, clip[1])
    elif len(leaf_vals):
        cmin, cmax = float(leaf_vals.min()), float(leaf_vals.max())
    else:
        cmin, cmax = 0.0, 1.0
    from plotly.colors import sample_colorscale
    colors_arr = np.array(colors, dtype=float)
    nan_mask = ~np.isfinite(colors_arr)
    norm = np.clip((np.where(nan_mask, cmin, colors_arr) - cmin) / ((cmax - cmin) or 1.0), 0, 1)
    scale = color_scale if isinstance(color_scale, (list, str)) else "Viridis"
    try:
        rgb = sample_colorscale(scale, list(norm))
    except Exception:
        rgb = sample_colorscale("Viridis", list(norm))
    rgb = ["#cbd5e1" if m else c for c, m in zip(rgb, nan_mask)]    # abu-abu = data tidak tersedia
    rgb[0] = "#e2e8f0"                                              # akar netral (nilai rata-rata ada di tooltip)
    return dict(ids=ids, labels=labels, parents=parents, values=values, text=texts, colors=rgb, cmin=cmin, cmax=cmax)


def _hierarchy_figure(kind, df, color_col, color_scale, root_label, maxdepth, height, clip):
    d = _hierarchy_data(df, color_col, color_scale, root_label, clip)
    if d is None:
        return go.Figure()
    common = dict(ids=d["ids"], labels=d["labels"], parents=d["parents"], values=d["values"], branchvalues="total",
                  text=d["text"], hoverinfo="text", textinfo="label", maxdepth=maxdepth if maxdepth and maxdepth > 0 else -1,
                  marker=dict(colors=d["colors"], line=dict(width=1, color="rgba(255,255,255,0.85)")),
                  textfont=dict(family="Plus Jakarta Sans, Inter, sans-serif", size=12))
    if kind == "Treemap":
        trace = go.Treemap(**common, tiling=dict(pad=2), root_color="rgba(0,0,0,0)", pathbar=dict(visible=True, thickness=22))
    elif kind == "Sunburst":
        trace = go.Sunburst(**common, insidetextorientation="radial", root_color="rgba(0,0,0,0)")
    else:
        trace = go.Icicle(**common, tiling=dict(orientation="h", pad=2), root_color="rgba(0,0,0,0)",
                          pathbar=dict(visible=True, thickness=22))
    fig = go.Figure(trace)
    unit = get_meta_unit(color_col)
    # Legenda warna: jejak kosong berskala warna (agar kab/kota tanpa data bisa berwarna abu-abu)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(colorscale=color_scale, cmin=d["cmin"], cmax=d["cmax"], color=[d["cmin"]], showscale=True,
                    colorbar=dict(title=dict(text=f"{get_short_label(color_col)}<br>{unit}", side="top", font=dict(size=10, color=MUTED)),
                                  thickness=10, len=0.7, x=1.0, outlinewidth=0, tickfont=dict(size=10, color=MUTED)))))
    fig.update_layout(margin=dict(l=4, r=70, t=4, b=4), height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def plot_hierarchical_icicle(df, color_col="IPM", color_scale="Viridis", root_label="Indonesia",
                             maxdepth=2, height=640, clip=(2, 98)):
    """Icicle: Indonesia -> Provinsi -> Kabupaten/Kota. Luas = jumlah_penduduk, warna = indikator terpilih
    (warna provinsi/akar = rata-rata berbobot penduduk, turunan dari data kab/kota)."""
    return _hierarchy_figure("Icicle", df, color_col, color_scale, root_label, maxdepth, height, clip)


def plot_hierarchy_treemap(df, color_col="IPM", color_scale="Viridis", root_label="Indonesia",
                           maxdepth=2, height=640, clip=(2, 98)):
    """Treemap dengan hirarki, ukuran, dan warna yang sama seperti Icicle."""
    return _hierarchy_figure("Treemap", df, color_col, color_scale, root_label, maxdepth, height, clip)


def plot_hierarchy_sunburst(df, color_col="IPM", color_scale="Viridis", root_label="Indonesia",
                            maxdepth=2, height=640, clip=(2, 98)):
    """Sunburst dengan hirarki, ukuran, dan warna yang sama seperti Icicle."""
    return _hierarchy_figure("Sunburst", df, color_col, color_scale, root_label, maxdepth, height, clip)


def plot_hierarchy(kind, df, **kw):
    """Pilih jenis grafik hirarki: 'Icicle' | 'Treemap' | 'Sunburst'."""
    fn = {"Icicle": plot_hierarchical_icicle, "Treemap": plot_hierarchy_treemap, "Sunburst": plot_hierarchy_sunburst}[kind]
    return fn(df, **kw)
