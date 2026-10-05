import os
import numpy as np
import pandas as pd
import streamlit as st
from src.data_loader import (
    load_dataset, load_geojson, INDICATORS_META, OFFICIAL_BPS_2024, PROV_TO_ISLAND, SHORT_LABELS,
    CHOROPLETH_OK, load_metadata, load_corrections, load_benchmark_table,
)
from src.pca_analysis import run_pca_pipeline, DEFAULT_PCA_FEATURES, CLUSTER_NAMES
from src.theme import PALETTE_NAMES, CLUSTER_COLORS
from src.visualizer import (
    plot_choropleth_map, plot_proportional_symbol_map, plot_pca_scatter, plot_scree,
    plot_loadings_heatmap, plot_parallel_coordinates, plot_correlation_heatmap, plot_bivariate,
    plot_kabupaten_radar, plot_rank_bar, plot_island_bar, plot_distribution, plot_bubble_overview,
    plot_cluster_profile, plot_highlight_scatter, plot_gap_bar, plot_province_range, get_meta_label, get_meta_unit,
    get_short_label, resolve_palette, plot_hierarchy, ICICLE_INDICATORS, HIERARCHY_KINDS,
    available_map_styles, MAP_STYLE_LABELS,
)

st.set_page_config(page_title="Eksplorasi Sosial-Ekonomi Indonesia 2024", page_icon="🇮🇩",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, select, textarea {font-family:'Plus Jakarta Sans',sans-serif !important;}
.stApp {background:
  radial-gradient(900px 520px at 92% -8%, rgba(157,61,143,.30), transparent 62%),
  radial-gradient(800px 520px at 0% 0%, rgba(124,58,237,.22), transparent 60%),
  linear-gradient(180deg,#eadbfb 0%,#d9c2f5 100%) !important;
  background-attachment: fixed !important;}
[data-testid="stMain"], section.main {background: transparent !important;}
header[data-testid="stHeader"] {background: transparent;}
header[data-testid="stHeader"] * {color:#5b2a86 !important;}
header[data-testid="stHeader"] svg {fill:#5b2a86 !important;}
.block-container {padding: 1.6rem 2rem 2.4rem; max-width: 1480px; width: calc(100% - 1.6rem); margin: .4rem auto 2rem;
  border-radius: 0; background: transparent; border: none; box-shadow: none;}
.block-container > div[data-testid="stVerticalBlockBorderWrapper"] {background:transparent !important; background-image:none !important; border:none !important; box-shadow:none !important;}
@media (max-width: 900px) {
  .block-container {padding:1rem .6rem 1.6rem; width:100%; margin:.2rem auto 1rem;}
  .hero {padding:24px 20px;} .hero h1 {font-size:1.45rem;} .hero-stats {gap:10px;}
  .hero-stats div {padding:9px 12px; min-width:0;}
}
h1,h2,h3,h4,h5 {letter-spacing:-0.01em; color:#2a1253;}
h3 {font-weight:800 !important;}
/* Hero */
.hero {position:relative; overflow:hidden; border-radius:22px; padding:34px 40px; color:#fff; margin-bottom:22px;
  background: radial-gradient(900px 300px at 85% -20%, rgba(162,77,114,.45), transparent 60%),
              linear-gradient(135deg,#17112F 0%,#32145F 45%,#63205F 80%,#A24D72 100%);
  box-shadow:0 18px 40px -12px rgba(15,23,42,.35); border:1px solid rgba(255,255,255,.07);}
.hero::after {content:""; position:absolute; inset:0; opacity:.07; pointer-events:none; z-index:1;
  background-image:radial-gradient(#fff 1px, transparent 1px); background-size:22px 22px;}
.hero h1 {font-size:2.05rem; font-weight:800; margin:6px 0 10px; color:#fff !important; line-height:1.18; max-width:900px;}
.hero p {font-size:.98rem; color:#cbd5e1; margin:0; line-height:1.6; max-width:820px;}
.chip {display:inline-block; padding:5px 13px; border-radius:999px; font-size:.74rem; font-weight:700;
  letter-spacing:.03em; margin:0 8px 8px 0; backdrop-filter:blur(6px);}
.chip.b {background:rgba(99,102,241,.25); color:#c7d2fe; border:1px solid rgba(129,140,248,.45);}
.chip.g {background:rgba(16,185,129,.2); color:#a7f3d0; border:1px solid rgba(52,211,153,.4);}
.chip.p {background:rgba(236,72,153,.18); color:#fbcfe8; border:1px solid rgba(244,114,182,.4);}
.hero > * {position:relative; z-index:2;}
.hero::before {content:""; position:absolute; inset:0; z-index:1; pointer-events:none; opacity:.95;
  background-repeat:no-repeat; background-position:right -10px center; background-size:auto 104%;
  -webkit-mask-image:linear-gradient(90deg, transparent 38%, #000 74%); mask-image:linear-gradient(90deg, transparent 38%, #000 74%);}
.hero p {max-width:640px !important;} .hero h1 {max-width:700px !important;}
.hero-stats {display:flex; gap:14px; margin-top:24px; flex-wrap:wrap;}
.hero-stats div {font-size:.7rem; color:#e9d5ff; text-transform:uppercase; letter-spacing:.09em; font-weight:700;
  background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); border-radius:14px; padding:12px 20px; min-width:132px;
  backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); box-shadow:0 8px 20px -10px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.1);}
.hero-stats b {display:block; font-size:1.45rem; color:#fff; letter-spacing:-.01em; text-transform:none; margin-top:3px;}
/* KPI */
.kpi {background:#fff; border-radius:18px; padding:18px 20px 16px; border:1px solid #ece4f6;
  box-shadow:0 1px 2px rgba(42,18,83,.05), 0 14px 28px -14px rgba(74,29,122,.26);
  transition:transform .2s, box-shadow .2s; height:100%; position:relative; overflow:hidden;}
.kpi:hover {transform:translateY(-4px); box-shadow:0 22px 38px -14px rgba(79,70,229,.40);}
.kpi::before {content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--c);}
.kpi-top {display:flex; justify-content:space-between; align-items:center;}
.kpi-label {font-size:.72rem; font-weight:800; color:#64748b; text-transform:uppercase; letter-spacing:.08em;}
.kpi-ico {width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:1.05rem;
  background:color-mix(in srgb, var(--c) 14%, white);}
.kpi-val {font-size:2.05rem; font-weight:800; color:#0f172a; margin:8px 0 4px; line-height:1.1; letter-spacing:-.02em;}
.kpi-val small {font-size:.95rem; font-weight:700; color:#64748b; margin-left:3px;}
.kpi-meta {font-size:.78rem; color:#64748b; display:flex; gap:8px; align-items:center; flex-wrap:wrap;}
.pill {display:inline-block; padding:2px 9px; border-radius:999px; font-size:.74rem; font-weight:700;}
.pill.up {background:#dcfce7; color:#166534;} .pill.down {background:#fee2e2; color:#991b1b;}
.pill.neutral {background:#e2e8f0; color:#334155;}
/* Insight */
.insight {background:linear-gradient(180deg,#fff,#fafaff); border:1px solid #e0e7ff; border-radius:16px; padding:14px 18px;
  font-size:.88rem; line-height:1.55; color:#334155; height:100%;}
.insight .t {font-size:.7rem; font-weight:800; color:#4f46e5; text-transform:uppercase; letter-spacing:.09em; margin-bottom:4px;}
.insight b {color:#0f172a;}
/* Temuan (storytelling) */
.story {background:linear-gradient(180deg,#fff,#fafaff); border:1px solid #e0e7ff; border-left:5px solid #4f46e5; border-radius:16px;
  padding:16px 20px; font-size:.88rem; line-height:1.6; color:#334155; height:100%;}
.story .q {font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:8px; line-height:1.35;}
.story p {margin:0 0 7px;} .story b.k {color:#4f46e5; text-transform:uppercase; font-size:.68rem; letter-spacing:.08em; display:block; margin-top:6px;}
.story .lim {color:#92400e; background:#fffbeb; border-radius:10px; padding:6px 10px; font-size:.82rem; margin-top:6px;}
/* Section title */
.sec {display:flex; align-items:center; gap:10px; margin:6px 0 4px;}
.sec .bar {width:5px; height:22px; border-radius:4px; background:linear-gradient(#4f46e5,#0d9488);}
.sec h4 {margin:0; font-size:1.08rem; font-weight:800;}
.sub {color:#64748b; font-size:.88rem; margin:0 0 12px 15px;}
/* Panel opsi peta: satu panel lavender yang rapi */
.opt-marker {display:none;}
:is(div[data-testid="stColumn"], div[data-testid="column"]):has(.opt-marker) {
  background:linear-gradient(180deg,#eadbfb 0%,#dcc8f6 100%);
  border:1px solid #cdb2f0; border-radius:16px; padding:18px 18px 14px;
  align-self:flex-start;}
/* hilangkan kartu putih bersarang di dalam panel */
:is(div[data-testid="stColumn"], div[data-testid="column"]):has(.opt-marker) div[data-testid="stVerticalBlockBorderWrapper"] {
  background:transparent !important; background-image:none !important; border:none !important;
  box-shadow:none !important; border-radius:0 !important;}
/* kotak statistik: seragam, angka lebih kecil */
:is(div[data-testid="stColumn"], div[data-testid="column"]):has(.opt-marker) [data-testid="stMetric"] {
  background:rgba(255,255,255,.55); border-radius:12px; padding:10px 12px;}
:is(div[data-testid="stColumn"], div[data-testid="column"]):has(.opt-marker) [data-testid="stMetricValue"] {
  font-size:1.35rem; font-weight:800; color:#2a1253;}
:is(div[data-testid="stColumn"], div[data-testid="column"]):has(.opt-marker) [data-testid="stMetricLabel"] p {
  font-size:.72rem; font-weight:700; color:#5b2a86; text-transform:uppercase; letter-spacing:.05em;}
/* Profile */
.profile {background:linear-gradient(160deg,#0f172a,#1e1b4b); color:#e2e8f0; border-radius:18px; padding:22px 22px 18px; height:100%;}
.profile .nm {font-size:1.35rem; font-weight:800; color:#fff; letter-spacing:-.01em;}
.profile .pv {color:#a5b4fc; font-size:.84rem; margin:2px 0 14px;}
.profile .row {display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid rgba(255,255,255,.08); font-size:.86rem;}
.profile .row:last-child {border:none;}
.profile .row span {color:#94a3b8;} .profile .row b {color:#fff;}
.profile .pc {display:flex; gap:8px; margin:0 0 12px;}
.profile .pc div {flex:1; background:rgba(255,255,255,.07); border-radius:12px; padding:9px 12px; font-size:.7rem; color:#94a3b8; font-weight:700; letter-spacing:.06em;}
.profile .pc b {display:block; font-size:1.15rem; color:#fff; letter-spacing:0;}
.tag {display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; font-weight:700; color:#fff; margin-top:8px;}
/* Tabs */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {gap:6px; background:linear-gradient(180deg,#fff,#faf6fd); padding:6px; border-radius:16px; border:1px solid #e6dcf2;
  box-shadow:0 1px 2px rgba(42,18,83,.05), 0 12px 26px -14px rgba(74,29,122,.28); margin-bottom:12px; overflow-x:auto;}
.stTabs [data-baseweb="tab"]:hover, .stTabs [role="tab"]:hover {background:#f3e8ff; color:#581c87;}
.stTabs [data-baseweb="tab"], .stTabs [role="tab"] {padding:9px 16px; font-weight:700; font-size:.88rem; border-radius:10px; color:#475569; height:auto;}
.stTabs [aria-selected="true"] {color:#fff !important; background:linear-gradient(135deg,#4c1d95,#9d3d8f) !important;
  box-shadow:0 10px 20px -8px rgba(110,40,140,.8), inset 0 1px 0 rgba(255,255,255,.25);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}
/* Sidebar */
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#ffffff,#f3f6fc); border-right:1px solid rgba(255,255,255,.2); box-shadow:12px 0 40px -18px rgba(0,0,0,.7);}
.brand {display:flex; align-items:center; gap:12px; margin-bottom:6px;}
.brand .logo {width:42px; height:42px; border-radius:13px; display:grid; place-items:center; font-size:1.3rem;
  background:linear-gradient(135deg,#4f46e5,#0d9488); box-shadow:0 8px 18px -8px rgba(79,70,229,.8);}
.brand .t {font-weight:800; font-size:1.02rem; line-height:1.15; color:#0f172a;}
.brand .s {font-size:.72rem; color:#64748b;}
.side-count {background:#eef2ff; border-radius:12px; padding:10px 14px; font-size:.82rem; color:#3730a3; font-weight:600;}
.stButton>button, .stDownloadButton>button {border-radius:12px; font-weight:700; border:1px solid #d8c4f0;}
.stDownloadButton>button {background:linear-gradient(135deg,#4c1d95,#9d3d8f); color:#fff; border:none;}
.stDownloadButton>button:hover {color:#fff; filter:brightness(1.08);}
.foot {text-align:center; color:#64748b; font-size:.8rem; padding:18px 0 4px;}
/* Wadah di dalam sidebar kadang ikut berkelas .block-container (tergantung versi Streamlit): netralkan gaya kartu konten utama */
section[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] .block-container, .stSidebarBlockContainer,
[data-testid="stSidebarUserContent"] {background:transparent !important; background-image:none !important; border:none !important;
  box-shadow:none !important; border-radius:0 !important; width:100% !important; max-width:none !important; margin:0 !important;}
section[data-testid="stSidebar"] .block-container {padding-top:1rem !important; padding-left:1rem !important; padding-right:1rem !important;}
/* Streamlit >= 1.4x membungkus isi sidebar dengan stVerticalBlockBorderWrapper: jangan beri gaya kartu putih (aturan "Panels" di atas) */
section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {background:transparent !important; background-image:none !important;
  border:none !important; box-shadow:none !important; border-radius:0 !important;}
/* Sidebar bertema ungu (selaras header) */
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#17112F 0%,#32145F 48%,#63205F 100%) !important;
  border-right:1px solid rgba(255,255,255,.08); box-shadow:12px 0 40px -18px rgba(23,17,47,.8);}
section[data-testid="stSidebar"] > div, section[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"], [data-testid="stSidebarHeader"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"], section[data-testid="stSidebar"] [data-testid="stElementContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {background:transparent !important; background-color:transparent !important;
  box-shadow:none !important; border-radius:0 !important; margin-left:0 !important; margin-right:0 !important;}
section[data-testid="stSidebar"] h5, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {color:#f3e8ff !important;}
section[data-testid="stSidebar"] h5 {font-size:.78rem !important; letter-spacing:.09em; text-transform:uppercase; font-weight:800;
  margin-top:6px; padding-bottom:6px; border-bottom:1px solid rgba(255,255,255,.12);}
section[data-testid="stSidebar"] .brand .t {color:#fff;} section[data-testid="stSidebar"] .brand .s {color:#d8b4fe;}
section[data-testid="stSidebar"] .brand .logo {background:linear-gradient(135deg,#A24D72,#7c3aed); box-shadow:0 8px 18px -8px rgba(217,70,168,.7);}
section[data-testid="stSidebar"] [data-baseweb="select"] > div, section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
  background:rgba(255,255,255,.09) !important; border:1px solid rgba(255,255,255,.18) !important; border-radius:12px !important; color:#fff !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] *, section[data-testid="stSidebar"] [data-testid="stSelectbox"] div {color:#fff !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {fill:#e9d5ff !important;}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] input, section[data-testid="stSidebar"] [data-baseweb="select"] input {
  color:#fff !important; -webkit-text-fill-color:#fff !important; opacity:1 !important;}
section[data-testid="stSidebar"] [data-baseweb="select"] [class*="singleValue"], section[data-testid="stSidebar"] [data-baseweb="select"] span {
  color:#fff !important; opacity:1 !important; font-weight:600;}
section[data-testid="stSidebar"] .side-count {background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.16); color:#fbcfe8;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"], section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] .stCaption {color:#d8b4fe !important;}
section[data-testid="stSidebar"] [data-testid="stExpander"] {background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.16) !important; border-radius:12px;}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary, section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] li, section[data-testid="stSidebar"] [data-testid="stExpander"] p {color:#f3e8ff !important;}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button, section[data-testid="stSidebar"] button svg {color:#e9d5ff !important;}
section[data-testid="stSidebar"] .stButton>button {background:rgba(255,255,255,.1); color:#fff; border:1px solid rgba(255,255,255,.25);}
section[data-testid="stSidebar"] .stButton>button:hover {background:rgba(255,255,255,.2); color:#fff;}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p, section[data-testid="stSidebar"] [data-testid="stToggle"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p {color:#f3e8ff !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def hero_art_css():
    """Ilustrasi peta Indonesia (SVG hasil tools/make_hero_svg.py) sebagai latar hero."""
    import base64
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hero_indonesia.svg"), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return f"<style>.hero::before{{background-image:url('data:image/svg+xml;base64,{b64}');}}</style>"
    except OSError:
        return ""


st.markdown(hero_art_css(), unsafe_allow_html=True)
ISLAND_ART = {"Jawa": "jawa", "Sumatera": "sumatra", "Kalimantan": "kalimantan", "Sulawesi": "sulawesi",
              "Bali & Nusa Tenggara": "bali_nusa_tenggara", "Maluku": "maluku", "Papua": "papua"}


@st.cache_resource
def island_art_css(island):
    """Ilustrasi dekoratif pulau (assets/islands/, hasil tools/make_island_art.py) untuk latar hero.
    Bila pulau tidak dipilih atau berkas tidak ada, latar peta Indonesia tetap dipakai."""
    import base64
    slug = ISLAND_ART.get(island)
    if not slug:
        return ""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "islands", f"hero_{slug}.svg"), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
    except OSError:
        return ""
    return ("<style>.hero::before{background-image:url('data:image/svg+xml;base64," + b64 + "');"
            "background-size:cover;background-position:right center;opacity:1;"
            "-webkit-mask-image:linear-gradient(90deg,transparent 32%,#000 68%);"
            "mask-image:linear-gradient(90deg,transparent 32%,#000 68%);}</style>")


def html(s):
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines()), unsafe_allow_html=True)


def section(title, sub=None):
    html(f'<div class="sec"><div class="bar"></div><h4>{title}</h4></div>' +
         (f'<p class="sub">{sub}</p>' if sub else ""))


def _st_ver():
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)


# Mode lebar "stretch" hanya ada di Streamlit >= 1.50; versi lama memakai use_container_width
STRETCH = {"width": "stretch"} if _st_ver() >= (1, 50) else {"use_container_width": True}
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def show(fig, key=None, cap=None, **kw):
    ev = st.plotly_chart(fig, config=PLOT_CFG, key=key, **STRETCH, **kw)
    if cap:
        st.caption(cap)
    return ev


def kpi(label, value, unit, color, meta, icon=""):
    ico = f'<div class="kpi-ico">{icon}</div>' if icon else ""
    html(f"""<div class="kpi" style="--c:{color}">
      <div class="kpi-top"><div class="kpi-label">{label}</div>{ico}</div>
      <div class="kpi-val">{value}<small>{unit}</small></div>
      <div class="kpi-meta">{meta}</div></div>""")

def delta_pill(diff, higher_is_better, fmt="{:+.2f}"):
    if abs(diff) < 0.005:
        return f'<span class="pill neutral">{fmt.format(0)}</span>'
    good = (diff > 0) == higher_is_better
    return f'<span class="pill {"up" if good else "down"}">{"▲" if diff > 0 else "▼"} {fmt.format(diff)}</span>'


# ------------------------------------------------------------------ DATA
df_master = load_dataset()
geojson = load_geojson()
pca_results = run_pca_pipeline(df_master, DEFAULT_PCA_FEATURES, n_components=4)
res = pca_results["result_df"]
for c in pca_results["pc_cols"] + ["Klaster"]:       # skor PCA & klaster selalu konsisten dgn pipeline
    df_master[c] = res[c].values
# Peta choropleth hanya untuk indikator rasio/persentase (angka absolut menyesatkan pada poligon beda luas)
INDICATOR_OPTS = [c for c in ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "sanitasi_layak",
                              "air_minum_layak", "pertumbuhan_ekonomi", "TPAK_perempuan", "TPAK_laki-laki"] if c in CHOROPLETH_OK]
BENCH = load_benchmark_table().set_index("indikator")
SRC_NOTE = "Sumber: BPS RI 2024"
NO_POLYGON = sorted(set(df_master["Kabupaten/Kota"]) - {f["id"] for f in geojson["features"]})
# ---- State seleksi bersama (cross-filter): klik pada peta atau scatter PCA memilih kabupaten/kota yang sama
st.session_state.setdefault("nonce", 0)
st.session_state.setdefault("last_click", {})
NONCE = st.session_state["nonce"]
KEY_MAP, KEY_PCA = f"map_ov_{NONCE}", f"pca_scatter_{NONCE}"


def _clicked(state_key, field):
    ev = st.session_state.get(state_key)
    try:
        pts = ev["selection"]["points"] if ev else []
    except (KeyError, TypeError):
        pts = []
    for p in pts:
        cd = p.get("customdata")
        cd0 = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
        v = p.get("location") or cd0 or p.get("hovertext")
        if isinstance(v, str) and v:
            return v
    return None


def _reset_selection():
    """Callback tombol reset: kosongkan seleksi klik (ganti key grafik) dan kembalikan wilayah default."""
    st.session_state["nonce"] += 1
    st.session_state["last_click"] = {}
    st.session_state["sel_kab"] = "Sleman"


def fmt(v, nd=2, suffix=""):
    return "n/a" if pd.isna(v) else f"{v:,.{nd}f}{suffix}"


# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    html("""<div class="brand"><div class="logo"><svg width="28" height="19" viewBox="0 0 3 2" xmlns="http://www.w3.org/2000/svg" style="display:block;border-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.45)"><rect width="3" height="1" fill="#E70011"/><rect y="1" width="3" height="1" fill="#FFFFFF"/></svg></div>
      <div><div class="t">Eksplorasi Indonesia</div><div class="s">Sosial-Ekonomi · BPS 2024</div></div></div>""")
    st.write("")
    st.markdown("#####  Filter Wilayah")
    islands = ["Semua Pulau"] + sorted(set(PROV_TO_ISLAND.values()))
    selected_island = st.selectbox("Pulau / Wilayah", islands)
    provs = sorted(df_master["Provinsi"].unique() if selected_island == "Semua Pulau"
                   else df_master.loc[df_master["Pulau"] == selected_island, "Provinsi"].unique())
    selected_prov = st.selectbox("Provinsi", ["Semua Provinsi"] + provs)
    st.markdown("#####  Tampilan Peta")
    map_style = st.selectbox("Gaya peta", available_map_styles(), format_func=MAP_STYLE_LABELS.get)
    st.caption("Peta dasar memuat tile daring. Gaya Mapbox muncul bila token Mapbox diatur (lihat README); "
               "tanpa internet, pilih *Polos*.")
df_f = df_master.copy()
if selected_island != "Semua Pulau":
    df_f = df_f[df_f["Pulau"] == selected_island]
if selected_prov != "Semua Provinsi":
    df_f = df_f[df_f["Provinsi"] == selected_prov]
with st.sidebar:
    html(f'<div class="side-count">📍 {len(df_f)} dari 514 kabupaten/kota terpilih</div>')
    st.write("")
    st.caption("Sumber: BPS RI, rilis 2024")
if df_f.empty:
    st.warning("Tidak ada wilayah yang cocok dengan filter.")
    st.stop()
KAB_LIST = sorted(df_f["Kabupaten/Kota"].unique())
if st.session_state.get("sel_kab") not in KAB_LIST:
    st.session_state["sel_kab"] = "Sleman" if "Sleman" in KAB_LIST else KAB_LIST[0]
for _src, (_key, _field) in {"map": (KEY_MAP, "location"), "pca": (KEY_PCA, "customdata")}.items():
    _c = _clicked(_key, _field)
    if _c and _c != st.session_state["last_click"].get(_src) and _c in KAB_LIST:
        st.session_state["sel_kab"] = _c
    st.session_state["last_click"][_src] = _c
# ------------------------------------------------------------------ HERO
scope = (selected_prov if selected_prov != "Semua Provinsi"
         else selected_island if selected_island != "Semua Pulau" else "Seluruh Indonesia")
st.markdown(island_art_css(selected_island), unsafe_allow_html=True)
html(f"""<div class="hero">
  <span class="chip b">BPS RI 2024</span><span class="chip g">514 Kabupaten / Kota · 38 Provinsi</span>
  <span class="chip p">PCA · Klaster · Geospasial</span>
  <h1>Eksplorasi Multidimensi Kondisi Sosial-Ekonomi Indonesia 2024</h1>
  <p>Telusuri kesenjangan pembangunan manusia, ketenagakerjaan, ekonomi daerah, dan infrastruktur dasar
  di seluruh kabupaten/kota, dari peta nasional hingga profil satu wilayah.</p>
  <div class="hero-stats">
    <div>Cakupan aktif<b>{scope}</b></div>
    <div>Wilayah<b>{len(df_f)}</b></div>
    <div>Penduduk<b>{df_f['jumlah_penduduk'].sum() / 1e6:,.1f} jt</b></div>
    <div>Indikator<b>13</b></div>
  </div></div>""")
tab1, tab_find, tab2, tab3, tab4, tab_hier, tab5 = st.tabs([" Ringkasan & Peta", " Temuan Utama", " Eksplorasi PCA",
                                                            " Profil Multivariat", "🗺️ Peta & Data", " Hirarki Wilayah",
                                                            " Metodologi & Sumber Data"])
# ================================================================== TAB 1
with tab1:
    ipm, pov, tpt = (df_f["IPM"].mean(), df_f["Persentase_Penduduk_Miskin_2024"].mean(), df_f["TPT_Jumlah"].mean())
    tpak_l, tpak_p = df_f["TPAK_laki-laki"].mean(), df_f["TPAK_perempuan"].mean()
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Wilayah", f"{len(df_f)}", "/ 514",  "#4f46e5",
            f"<span>{df_f['Provinsi'].nunique()} provinsi · {df_f['jumlah_penduduk'].sum() / 1e6:,.1f} jt jiwa</span>")
    with k2:
        kpi("IPM rata-rata", f"{ipm:.2f}", "poin",  "#0d9488",
            f"{delta_pill(ipm - OFFICIAL_BPS_2024['IPM'], True)}<span>vs nasional {OFFICIAL_BPS_2024['IPM']:.2f}</span>")
    with k3:
        kpi("Kemiskinan", f"{pov:.2f}", "%",  "#d97706",
            f"{delta_pill(pov - OFFICIAL_BPS_2024['Persentase_Penduduk_Miskin'], False)}<span>vs nasional {OFFICIAL_BPS_2024['Persentase_Penduduk_Miskin']:.2f}%</span>")
    with k4:
        kpi("Pengangguran (TPT)", f"{tpt:.2f}", "%",  "#4f46e5",
            f"{delta_pill(tpt - OFFICIAL_BPS_2024['TPT'], False)}<span>vs nasional {OFFICIAL_BPS_2024['TPT']:.2f}%</span>")
    with k5:
        kpi("TPAK", f"{(tpak_l + tpak_p) / 2:.2f}", "%",  "#0284c7",
            f'<span class="pill neutral">L {tpak_l:.1f}%</span><span class="pill neutral">P {tpak_p:.1f}%</span>')
    st.caption(f"Nilai utama = rata-rata tak berbobot kabupaten/kota terpilih (n = {len(df_f)}); pembanding \"nasional\" adalah angka agregat BPS "
               f"(IPM {BENCH.loc['IPM','periode']}, kemiskinan {BENCH.loc['Persentase_Penduduk_Miskin','periode']}, "
               f"TPT {BENCH.loc['TPT','periode']}) sehingga selisih bersifat indikatif. ▲▼ hijau = lebih baik, merah = lebih buruk. "
               "TPAK = rata-rata TPAK laki-laki dan perempuan. Data tidak tersedia dikecualikan dari rata-rata.")
    # Insight otomatis
    hi_r, lo_r = df_f.loc[df_f["IPM"].idxmax()], df_f.loc[df_f["IPM"].idxmin()]
    r_ipm_pov = df_f["IPM"].corr(df_f["Persentase_Penduduk_Miskin_2024"]) if len(df_f) > 2 else float("nan")
    isl = df_f.groupby("Pulau")["IPM"].mean()
    i1, i2, i3 = st.columns(3)
    with i1:
        html(f'<div class="insight"><div class="t">💡 Kesenjangan IPM</div>Tertinggi <b>{hi_r["Kabupaten/Kota"]}</b> '
             f'({hi_r["IPM"]:.2f}), terendah <b>{lo_r["Kabupaten/Kota"]}</b> ({lo_r["IPM"]:.2f}): selisih '
             f'<b>{hi_r["IPM"] - lo_r["IPM"]:.1f} poin</b>.</div>')
    with i2:
        txt = (f"Korelasi IPM dengan kemiskinan <b>r = {r_ipm_pov:+.2f}</b>: "
               f"{'semakin tinggi IPM, kemiskinan cenderung makin rendah' if r_ipm_pov < -0.3 else 'hubungan lemah di cakupan ini'}.")
        html(f'<div class="insight"><div class="t">🔗 Hubungan Indikator</div>{txt}</div>')
    with i3:
        html(f'<div class="insight"><div class="t">🏝️ Antar Wilayah</div>Rata-rata IPM tertinggi di <b>{isl.idxmax()}</b> '
             f'({isl.max():.2f}); terendah di <b>{isl.idxmin()}</b> ({isl.min():.2f}).</div>')
    st.write("")
    with st.container(border=True):
        section("Peta Choropleth Wilayah", "Klik sebuah wilayah untuk memilihnya; pilihan ikut berlaku di tab Eksplorasi PCA (brushing & linking).")
        c_opt, c_map = st.columns([1, 3.6], gap="medium")
        with c_opt:
            st.markdown('<div class="opt-marker"></div>', unsafe_allow_html=True)
            ov_ind = st.selectbox("Indikator", INDICATOR_OPTS, format_func=get_meta_label, key="ov_ind")
            ov_pal = st.selectbox("Palet warna", PALETTE_NAMES, key="ov_pal")
            st.caption(INDICATORS_META[ov_ind]["desc"])
            s = df_f[ov_ind]
            st.markdown("**Statistik ringkas**")
            m1, m2 = st.columns(2)
            m1.metric("Min", f"{s.min():,.2f}"); m2.metric("Maks", f"{s.max():,.2f}")
            m1.metric("Median", f"{s.median():,.2f}"); m2.metric("Std. dev", f"{s.std():,.2f}")
            st.caption(f"Data tidak tersedia: {int(s.isna().sum())} wilayah (ditampilkan kosong).")
        with c_map:
            show(plot_choropleth_map(df_f, geojson, ov_ind, resolve_palette(ov_pal), highlight_kab=st.session_state["sel_kab"],
                                     map_style=map_style, height=620),
                 key=KEY_MAP, on_select="rerun", selection_mode="points",
                 cap=f"{get_meta_label(ov_ind)} · {get_meta_unit(ov_ind)} · {SRC_NOTE} · warna dipotong pada persentil 2–98 agar outlier tidak menekan kontras."
                     + (f" Poligon tidak tersedia pada GeoJSON: {', '.join(NO_POLYGON)}." if NO_POLYGON else ""))
            sel_row = df_master[df_master["Kabupaten/Kota"] == st.session_state["sel_kab"]].iloc[0]
            html(f"""<div class="insight"><div class="t">📍 Wilayah terpilih</div><b>{sel_row['Kabupaten/Kota']}</b> ({sel_row['Provinsi']}):
              IPM <b>{sel_row['IPM']:.2f}</b> · kemiskinan <b>{sel_row['Persentase_Penduduk_Miskin_2024']:.2f}%</b> ·
              {get_short_label(ov_ind)} <b>{sel_row[ov_ind]:,.2f}</b> (rata-rata nasional {df_master[ov_ind].mean():,.2f}).
              Profil lengkap ada di tab <i>Eksplorasi PCA</i>.</div>""")
    st.write("")
    nat_mean = df_master[ov_ind].mean()
    cl, cr = st.columns(2, gap="medium")
    with cl:
        with st.container(border=True):
            section("Top 5 Tertinggi", get_meta_label(ov_ind))
            show(plot_rank_bar(df_f, ov_ind, 5, True, ref=nat_mean), key="top5",
                 cap=f"Garis putus = rata-rata nasional (514 kab/kota). {SRC_NOTE}")
    with cr:
        with st.container(border=True):
            section("Top 5 Terendah", get_meta_label(ov_ind))
            show(plot_rank_bar(df_f, ov_ind, 5, False, ref=nat_mean), key="bot5",
                 cap=f"Garis putus = rata-rata nasional (514 kab/kota). {SRC_NOTE}")
    cl, cr = st.columns(2, gap="medium")
    with cl:
        with st.container(border=True):
            show(plot_island_bar(df_f, ov_ind), key="island_bar")
    with cr:
        with st.container(border=True):
            show(plot_distribution(df_f, ov_ind), key="dist")
    with st.container(border=True):
        show(plot_bubble_overview(df_f), key="bubble")
# ================================================================== TAB TEMUAN UTAMA
with tab_find:
    section("Temuan Utama: dari Data ke Insight",
            "Lima pertanyaan analitis. Semua angka dihitung langsung dari data saat aplikasi berjalan (seluruh 514 kab/kota; tidak terpengaruh filter sidebar).")
    st.caption("Data bersifat *cross-sectional* tahun 2024: temuan menggambarkan keterkaitan antarwilayah pada satu waktu, bukan tren atau sebab-akibat. "
               "Data tidak tersedia dikecualikan pada tiap perhitungan.")
    D = df_master
    ipm_s, pov_s = D["IPM"], D["Persentase_Penduduk_Miskin_2024"]

    def story(q, finding, interp, impl, limit):
        html(f"""<div class="story"><div class="q">❓ {q}</div>
          <b class="k">Temuan</b><p>{finding}</p>
          <b class="k">Interpretasi</b><p>{interp}</p>
          <b class="k">Implikasi</b><p>{impl}</p>
          </div>""")

    def names(frame, n=5, by=None, asc=False):
        f = frame.sort_values(by, ascending=asc) if by else frame
        return ", ".join(f["Kabupaten/Kota"].head(n))

    # ---- Q1
    rho1 = ipm_s.corr(pov_s, method="spearman")
    qi, qp = ipm_s.quantile(.75), pov_s.quantile(.75)
    m1 = (ipm_s >= qi) & (pov_s >= qp)
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True):
            show(plot_highlight_scatter(D, "IPM", "Persentase_Penduduk_Miskin_2024", m1, "IPM & kemiskinan sama-sama kuartil teratas",
                                        "IPM vs Kemiskinan", f"Garis putus = kuartil ke-3 (IPM {qi:.1f}; kemiskinan {qp:.1f}%) · ρ Spearman = {rho1:+.2f} · n = {int(D[['IPM','Persentase_Penduduk_Miskin_2024']].dropna().shape[0])} · {SRC_NOTE}",
                                        hline=qp, vline=qi), key="story1")
    with c2:
        story("Apakah IPM tinggi berarti kemiskinan rendah, dan di mana pola itu patah?",
              f"Hubungan negatif ({rho1:+.2f}, Spearman). Namun <b>{int(m1.sum())}</b> wilayah berada di kuartil teratas IPM <i>sekaligus</i> kuartil teratas kemiskinan, "
              f"antara lain {names(D[m1], 6, 'Persentase_Penduduk_Miskin_2024')}.",
              "IPM mengukur kesehatan, pendidikan, dan standar hidup rata-rata; kemiskinan mengukur proporsi di bawah garis kemiskinan. Keduanya bisa berbeda ketika rata-rata baik tetapi distribusi pengeluaran timpang.",
              "Wilayah-wilayah ini layak dipelajari terpisah: kenaikan IPM saja belum menjamin kemiskinan turun.",
              "Korelasi bukan kausalitas; potongan kuartil adalah konvensi analitis, bukan batas resmi BPS.")
    st.write("")
    # ---- Q2
    pdrb = D["pdrb_perkapita_adhk"]
    rho2 = pdrb.corr(ipm_s, method="spearman")
    p90, med = pdrb.quantile(.9), ipm_s.median()
    m2 = (pdrb >= p90) & (ipm_s < med)
    n_top = int((pdrb >= p90).sum())
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True):
            show(plot_highlight_scatter(D, "pdrb_perkapita_adhk", "IPM", m2, "PDRB kuantil 10% teratas, IPM di bawah median",
                                        "PDRB per kapita vs IPM", f"Sumbu X logaritmik · garis putus = persentil ke-90 PDRB & median IPM ({med:.1f}) · ρ Spearman = {rho2:+.2f} · {SRC_NOTE}",
                                        logx=True, hline=med, vline=p90), key="story2")
    with c2:
        story("Apakah PDRB per kapita tinggi selalu diikuti IPM tinggi?",
              f"Keterkaitannya positif sedang ({rho2:+.2f}). Dari <b>{n_top}</b> wilayah dengan PDRB per kapita 10% teratas, <b>{int(m2.sum())}</b> memiliki IPM di bawah median nasional"
              + (f" (mis. {names(D[m2], 5, 'pdrb_perkapita_adhk')})." if m2.any() else "."),
              "Nilai ekonomi per kapita tidak otomatis menjadi kesejahteraan manusia; di wilayah berbasis sumber daya, PDRB tinggi dapat tidak merata atau mengalir keluar daerah (hipotesis; data ini tidak menguji penyebabnya).",
              "Pembangunan manusia perlu dipantau terpisah dari pertumbuhan ekonomi.",
              "Satuan PDRB per kapita belum terverifikasi terhadap tabel BPS (lihat tab Metodologi & Sumber Data); gunakan peringkat/rank, bukan nilai rupiah absolut.")
    st.write("")
    # ---- Q3
    gap = (D["TPAK_laki-laki"] - D["TPAK_perempuan"])
    tv = D.dropna(subset=["TPT_laki-laki", "TPT_perempuan"])
    n_fem = int((tv["TPT_perempuan"] > tv["TPT_laki-laki"]).sum())
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True):
            show(plot_gap_bar(D), key="story3")
    with c2:
        top2 = D.assign(g=gap).nlargest(2, "g")
        story("Seberapa besar kesenjangan gender di pasar kerja?",
              f"Median selisih TPAK laki-laki dan perempuan <b>{gap.median():.1f} poin</b>; terbesar di {top2.iloc[0]['Kabupaten/Kota']} ({top2.iloc[0]['g']:.1f}) dan "
              f"{top2.iloc[1]['Kabupaten/Kota']} ({top2.iloc[1]['g']:.1f}). TPT perempuan lebih tinggi daripada laki-laki di <b>{n_fem} dari {len(tv)}</b> wilayah.",
              "Partisipasi perempuan jauh lebih rendah daripada laki-laki hampir di semua wilayah, dan di sebagian besar wilayah mereka yang aktif mencari kerja pun lebih sulit terserap.",
              "Kebijakan ketenagakerjaan perlu memisahkan indikator menurut gender.",
              "TPAK/TPT tidak dipecah per sektor atau usia; selisih besar di wilayah tambang/perkebunan tidak dapat dijelaskan dari data ini.")
    st.write("")
    # ---- Q4
    gm = ipm_s.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for _, g in D.groupby("Provinsi")["IPM"])
    eta2 = ss_b / ((ipm_s - gm) ** 2).sum()
    rng = D.groupby("Provinsi")["IPM"].agg(["min", "max", "count"]).query("count >= 3").assign(r=lambda t: t["max"] - t["min"]).nlargest(2, "r")
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True):
            show(plot_province_range(D), key="story4")
    with c2:
        story("Apakah ketimpangan pembangunan manusia lebih ditentukan antar-provinsi atau di dalam provinsi?",
              f"Provinsi menjelaskan <b>{eta2 * 100:.0f}%</b> variasi IPM antar kab/kota (η² = {eta2:.2f}); sisanya <b>{(1 - eta2) * 100:.0f}%</b> adalah variasi di dalam provinsi. "
              f"Rentang IPM terlebar: {rng.index[0]} ({rng.iloc[0]['r']:.1f} poin) dan {rng.index[1]} ({rng.iloc[1]['r']:.1f} poin).",
              "Sebagian besar ketimpangan ada di dalam provinsi, sehingga rata-rata provinsi menyembunyikan wilayah tertinggal dan wilayah maju.",
              "Penargetan pada level kabupaten/kota lebih informatif daripada hanya level provinsi.",
              "η² tidak berbobot penduduk dan sensitif terhadap satu-dua wilayah ekstrem; provinsi baru hasil pemekaran memiliki sedikit kab/kota.")
    st.write("")
    # ---- Q5
    summ5 = D.groupby("Klaster").agg(n=("IPM", "size"), ipm=("IPM", "median"), miskin=("Persentase_Penduduk_Miskin_2024", "median")).reindex(CLUSTER_NAMES)
    cv = pca_results["cumulative_variance"]
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("**Ringkasan klaster (median)**")
            st.dataframe(summ5.reset_index().rename(columns={"Klaster": "Klaster", "n": "Wilayah", "ipm": "IPM", "miskin": "Miskin %"}).round(1),
                         hide_index=True, **STRETCH)
            st.caption(f"Visualisasi biplot, loading, dan profil klaster ada di tab Eksplorasi PCA. {SRC_NOTE}")
    with c2:
        story("Bagaimana pola sosial-ekonomi multidimensi antar kabupaten/kota?",
              f"PC1 menjelaskan <b>{pca_results['explained_variance'][0] * 100:.1f}%</b> variasi dan tiga komponen pertama <b>{cv[2] * 100:.1f}%</b>. "
              f"K-Means (k = 4) memisahkan {int(summ5.iloc[0]['n'])} wilayah ber-IPM terendah (median IPM {summ5.iloc[0]['ipm']:.1f}; kemiskinan {summ5.iloc[0]['miskin']:.1f}%) "
              f"dari {int(summ5.iloc[3]['n'])} wilayah ber-IPM tertinggi (median IPM {summ5.iloc[3]['ipm']:.1f}; kemiskinan {summ5.iloc[3]['miskin']:.1f}%).",
              "Dimensi utama pembeda wilayah adalah tingkat pembangunan; partisipasi kerja (PC2) dan pertumbuhan ekonomi (PC4) adalah dimensi terpisah yang tidak sejalan dengan pembangunan.",
              "Kelompok ber-IPM terendah dapat dijadikan prioritas pembangunan dasar, sementara dimensi lain butuh penanganan berbeda.",
              "Jumlah klaster ditetapkan k = 4 (bukan hasil optimasi formal); wilayah dengan data tidak tersedia memakai imputasi median.")
# ================================================================== TAB 2
with tab2:
    section("Reduksi Dimensi: Principal Component Analysis",
            f"{len(DEFAULT_PCA_FEATURES)} indikator saling berkorelasi diringkas menjadi dimensi ortogonal (PDRB & penduduk di-log; data tidak tersedia diimputasi median). Klik titik untuk memilih wilayah.")
    pca_df = df_f[["Kabupaten/Kota", "Kode_Wilayah", "Provinsi", "Pulau", "Latitude", "Longitude", "Klaster"]
                  + DEFAULT_PCA_FEATURES + pca_results["pc_cols"]]
    kab_list = KAB_LIST          # klik pada peta (Tab 1) atau scatter ini memilih wilayah yang sama
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1.3, 1.6, 0.9], gap="medium")
        pc_x = c1.selectbox("Sumbu X", pca_results["pc_cols"], index=0)
        pc_y = c2.selectbox("Sumbu Y", pca_results["pc_cols"], index=1)
        color_var = c3.selectbox("Warna titik", ["Pulau", "Klaster", "Provinsi", "IPM",
                                                 "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "pdrb_perkapita_adhk"],
                                 format_func=lambda c: get_short_label(c) if c in SHORT_LABELS else c)
        sel_kab = c4.selectbox("🎯 Kabupaten/Kota terpilih", kab_list, key="sel_kab")
        c5.write(""); c5.write("")
        biplot = c5.toggle("Biplot", value=True, help="Panah = loading variabel, diskalakan 0,8 × rentang skor agar terbaca (arah & panjang relatif bermakna, skala absolut tidak)")
        c5.button("↺ Reset", on_click=_reset_selection, help="Hapus pilihan klik dan kembali ke Sleman")
        fig_pca = plot_pca_scatter(pca_df, pc_x, pc_y, color_var, sel_kab, biplot, pca_results["loadings_df"])
        show(fig_pca, key=KEY_PCA, on_select="rerun", selection_mode="points",
             cap=f"Setiap titik = satu kabupaten/kota (n = {len(pca_df)}). Panah merah = arah loading variabel (biplot). {SRC_NOTE}")
    st.write("")
    row = pca_df[pca_df["Kabupaten/Kota"] == sel_kab].iloc[0]
    full = df_master[df_master["Kabupaten/Kota"] == sel_kab].iloc[0]
    kl_color = dict(zip(CLUSTER_NAMES, CLUSTER_COLORS))[row["Klaster"]]
    section(f"Profil Wilayah: {sel_kab}", "Terhubung dengan pilihan di grafik PCA (brushing & linking).")
    d1, d2, d3 = st.columns([1.15, 1.6, 1.8], gap="medium")
    with d1:
        html(f"""<div class="profile">
          <div class="nm">{sel_kab}</div>
          <div class="pv">{row['Provinsi']} · {row['Pulau']} · Kode {row['Kode_Wilayah']}</div>
          <div class="pc"><div>{pc_x}<b>{row[pc_x]:+.2f}</b></div><div>{pc_y}<b>{row[pc_y]:+.2f}</b></div></div>
          <div class="row"><span>Penduduk</span><b>{fmt(row['jumlah_penduduk'],0)}</b></div>
          <div class="row"><span>IPM</span><b>{fmt(row['IPM'])}</b></div>
          <div class="row"><span>Kemiskinan</span><b>{fmt(row['Persentase_Penduduk_Miskin_2024'],2,'%')}</b></div>
          <div class="row"><span>TPT</span><b>{fmt(row['TPT_Jumlah'],2,'%')}</b></div>
          <div class="row"><span>Sanitasi layak</span><b>{fmt(row['sanitasi_layak'],2,'%')}</b></div>
          <div class="row"><span>Air minum layak</span><b>{fmt(row['air_minum_layak'],2,'%')}</b></div>
          <span class="tag" style="background:{kl_color}">{row['Klaster']}</span>
          {'<div style="font-size:.74rem;color:#fcd34d;margin-top:8px">⚠ Ada indikator tidak tersedia; skor PCA memakai imputasi median.</div>' if bool(res.loc[res['Kabupaten/Kota'] == sel_kab, 'Imputasi'].iloc[0]) else ''}</div>""")
    with d2:
        with st.container(border=True):
            show(plot_kabupaten_radar(df_master, sel_kab, height=400), key="radar")
    with d3:
        with st.container(border=True):
            prov_df = df_master[df_master["Provinsi"] == row["Provinsi"]]
            st.markdown(f"**Peta Provinsi {row['Provinsi']}** · IPM, wilayah terpilih ditandai merah")
            show(plot_choropleth_map(prov_df, geojson, "IPM", resolve_palette("Indigo-Teal"),
                                     highlight_kab=sel_kab, map_style=map_style, height=358), key="minimap")
    st.write("")
    s1, s2 = st.columns(2, gap="medium")
    with s1:
        with st.container(border=True):
            show(plot_scree(pca_results["explained_variance"], pca_results["cumulative_variance"],
                            pca_results["pc_cols"]), key="scree",
                 cap="Jumlah komponen dipilih dari titik siku & ambang kumulatif ~70%; PC1–PC3 dipakai untuk interpretasi.")
    with s2:
        with st.container(border=True):
            show(plot_loadings_heatmap(pca_results["loadings_df"]), key="loadings")
    section("Interpretasi Komponen", "Dihasilkan otomatis dari nilai loading (|loading| > 0,2).")
    cols = st.columns(len(pca_results["interpretations"]), gap="small")
    for col, (pc, info) in zip(cols, pca_results["interpretations"].items()):
        with col:
            html(f"""<div class="insight"><div class="t">{pc} · {info['var_pct']:.1f}% varians</div>
              Kumulatif <b>{info['cum_pct']:.1f}%</b><br><br>
              <b>▲ Positif:</b> {info['pos_desc']}<br><br><b>▼ Negatif:</b> {info['neg_desc']}</div>""")
    ld = pca_results["loadings_df"]
    EXPECT = {"PC1": ({"IPM", "pdrb_perkapita_adhk", "Persentase_Penduduk_Miskin_2024"}, "tingkat pembangunan & kesejahteraan (IPM, PDRB, sanitasi, air minum tinggi; kemiskinan rendah)"),
              "PC2": ({"TPAK_laki-laki", "TPAK_perempuan"}, "partisipasi angkatan kerja (TPAK laki-laki & perempuan)"),
              "PC4": ({"pertumbuhan_ekonomi"}, "laju pertumbuhan ekonomi")}
    lines = []
    for pc, (names, label) in EXPECT.items():     # label hanya ditampilkan bila loading dominan benar-benar cocok
        if pc in ld.columns and ld[pc].abs().idxmax() in names:
            lines.append(f"**{pc}** ≈ {label}")
    cum3 = pca_results["cumulative_variance"][2] * 100
    st.info("**Ringkasan:** " + "; ".join(lines) + f". PC3 memuat campuran (sanitasi/pertumbuhan vs penduduk/PDRB/TPT) sehingga tidak diberi label tunggal. "
            f"Tiga komponen pertama menjelaskan **{cum3:.2f}%** variabilitas. "
            f"Catatan: TPT berloading positif pada PC1, artinya wilayah dengan pembangunan lebih tinggi cenderung memiliki TPT lebih tinggi (pola khas wilayah perkotaan).", icon="💡")
    st.write("")
    section("Tipologi Wilayah (K-Means, k = 4)",
            f"514 wilayah dikelompokkan menurut kemiripan {len(DEFAULT_PCA_FEATURES)} indikator; klaster diurutkan dari IPM rata-rata terendah ke tertinggi.")
    t1, t2 = st.columns([1.5, 1], gap="medium")
    with t1:
        with st.container(border=True):
            show(plot_cluster_profile(df_master), key="cluster_prof", cap="Klaster adalah ringkasan statistik, bukan label kebijakan. " + SRC_NOTE)
    with t2:
        summ = (df_master.groupby("Klaster").agg(
            Wilayah=("IPM", "size"), IPM=("IPM", "mean"), Miskin=("Persentase_Penduduk_Miskin_2024", "mean"),
            Penduduk=("jumlah_penduduk", "median")).reindex(CLUSTER_NAMES).round(1).reset_index())
        st.dataframe(summ, hide_index=True, **STRETCH, column_config={
            "Miskin": st.column_config.NumberColumn("Miskin %", format="%.1f"),
            "Penduduk": st.column_config.NumberColumn("Median penduduk", format="%d")})
        st.caption("Angka rata-rata kecuali penduduk (median).")
# ================================================================== TAB 3
with tab3:
    section("Parallel Coordinates Plot", "Seret pada sumbu vertikal untuk menyaring subpopulasi (brushing).")
    pcp_cols = ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "pdrb_perkapita_adhk",
                "sanitasi_layak", "air_minum_layak", "pertumbuhan_ekonomi"]
    with st.container(border=True):
        pc_col, _ = st.columns([1.2, 3])
        pcp_color = pc_col.selectbox("Warna garis", ["IPM", "Persentase_Penduduk_Miskin_2024", "pdrb_perkapita_adhk", "PC1"],
                                     format_func=lambda c: get_short_label(c) if c in SHORT_LABELS else c)
        show(plot_parallel_coordinates(df_f, pcp_cols, pcp_color), key="pcp",
             cap="Sumbu bertanda ↓ dibalik: pada SEMUA sumbu, posisi lebih atas = kondisi lebih baik. Wilayah dengan data tidak tersedia tidak digambar. Seret pada sumbu untuk menyaring. " + SRC_NOTE)
    st.write("")
    section("Analisis Korelasi Antar-Indikator")
    ch, cs = st.columns([1.1, 0.9], gap="medium")
    with ch:
        with st.container(border=True):
            corr_method = st.radio("Koefisien", ["spearman", "pearson"], horizontal=True,
                                   format_func=lambda m: "Spearman (rank, tahan outlier)" if m == "spearman" else "Pearson (linear)")
            show(plot_correlation_heatmap(df_master, pcp_cols, method=corr_method), key="corr",
                 cap="Korelasi menunjukkan keterkaitan, bukan sebab-akibat. Pearson peka terhadap outlier PDRB; Spearman disarankan.")
    with cs:
        with st.container(border=True):
            a, b = st.columns(2)
            sx = a.selectbox("Sumbu X", pcp_cols, 0, format_func=get_short_label)
            sy = b.selectbox("Sumbu Y", pcp_cols, 1, format_func=get_short_label)
            fig_b, r = plot_bivariate(df_f, sx, sy)
            show(fig_b, key="bivar", cap="Warna = pulau. Garis putus = tren linear (OLS) dari wilayah terfilter.")
            st.caption(f"Korelasi Pearson r = **{r:+.3f}**  ({'kuat' if abs(r) > .6 else 'sedang' if abs(r) > .3 else 'lemah'}).")
# ================================================================== TAB 4
with tab4:
    section("Peta Geospasial & Data Explorer", "Dua pendekatan spasial tanpa pin: choropleth poligon dan simbol proporsional.")
    with st.container(border=True):
        mt, mi, mp = st.columns([1.6, 1.4, 1], gap="medium")
        map_type = mt.radio("Representasi", ["Choropleth (poligon)", "Simbol proporsional (ukuran = penduduk)"], horizontal=True)
        is_choro = map_type.startswith("Choropleth")
        ind_opts = CHOROPLETH_OK if is_choro else list(INDICATORS_META)
        map_ind = mi.selectbox("Indikator", ind_opts, format_func=get_meta_label, key=f"t4_ind_{'c' if is_choro else 's'}",
                               help="Choropleth hanya untuk rasio/persentase; angka absolut (penduduk, PDRB) lebih tepat dengan simbol proporsional.")
        map_pal = mp.selectbox("Palet", PALETTE_NAMES, index=PALETTE_NAMES.index("Zamrud"), key="t4_pal")
        if map_type.startswith("Choropleth"):
            show(plot_choropleth_map(df_f, geojson, map_ind, resolve_palette(map_pal), map_style=map_style, height=600), key="map_t4_c",
                 cap=f"{get_meta_label(map_ind)} · {get_meta_unit(map_ind)} · {SRC_NOTE} · warna dipotong P2–P98.")
        else:
            show(plot_proportional_symbol_map(df_f, "jumlah_penduduk", map_ind, resolve_palette(map_pal), map_style, height=600), key="map_t4_s",
                 cap=f"Ukuran lingkaran = jumlah penduduk, warna = {get_meta_label(map_ind)} ({get_meta_unit(map_ind)}). Wilayah tanpa koordinat valid tidak digambar. {SRC_NOTE}")
    st.write("")
    section("Master Dataset", "Cari, urutkan, dan unduh data wilayah terpilih.")
    q = st.text_input("🔎 Cari kabupaten/kota atau provinsi", placeholder="mis. Sleman, Papua, Kalimantan …")
    tbl = df_f.copy()
    if q:
        m = tbl["Kabupaten/Kota"].str.contains(q, case=False) | tbl["Provinsi"].str.contains(q, case=False)
        tbl = tbl[m]
    with_extra = st.toggle("Tampilkan koordinat & skor PCA", value=False)
    drop = [] if with_extra else ["Latitude", "Longitude", "PC1", "PC2", "PC3", "PC4"]
    tbl = tbl.drop(columns=drop)
    cfg = {
        "Kabupaten/Kota": st.column_config.TextColumn("Kabupaten/Kota"),
        "Kode_Status": st.column_config.TextColumn("Status kode", help="valid / tidak_valid (kode asli tidak diketahui, lihat tab Metodologi & Sumber Data)"),
        "IPM": st.column_config.ProgressColumn("IPM", min_value=40, max_value=100, format="%.2f"),
        "sanitasi_layak": st.column_config.ProgressColumn("Sanitasi %", min_value=0, max_value=100, format="%.1f"),
        "air_minum_layak": st.column_config.ProgressColumn("Air minum %", min_value=0, max_value=100, format="%.1f"),
        "Persentase_Penduduk_Miskin_2024": st.column_config.NumberColumn("Miskin %", format="%.2f"),
        "jumlah_penduduk": st.column_config.NumberColumn("Penduduk", format="%d"),
        "Kode_Wilayah": st.column_config.TextColumn("Kode"),
    }
    for col_name in tbl.columns:                      # judul kolom jelas & format angka konsisten
        if col_name not in cfg and col_name in INDICATORS_META:
            cfg[col_name] = st.column_config.NumberColumn(get_meta_label(col_name), format="%d" if col_name == "jumlah_penduduk" else "%.2f")
    st.dataframe(tbl, **STRETCH, height=440, hide_index=True,
                 column_config={k: v for k, v in cfg.items() if k in tbl.columns})
    st.caption(f"Menampilkan {len(tbl)} baris. Sel kosong = data tidak tersedia / kode tidak diketahui (bukan nol).")
    st.download_button("📥 Unduh CSV", tbl.to_csv(index=False).encode("utf-8"),
                       "data_sosial_ekonomi_bps_2024.csv", "text/csv")
# ================================================================== TAB HIRARKI WILAYAH (ICICLE / TREEMAP / SUNBURST)
with tab_hier:
    section("Hirarki Wilayah Indonesia", "Indonesia → Provinsi → Kabupaten/Kota")
    ic_opts = [c for c in ICICLE_INDICATORS if c in df_f.columns]
    with st.container(border=True):
        hc1, hc2, hc3, hc4 = st.columns([1.5, 1.1, 1.1, 1.3], gap="medium")
        ic_ind = hc1.selectbox("Indikator Warna", ic_opts, index=ic_opts.index("IPM") if "IPM" in ic_opts else 0,
                               format_func=get_meta_label, key="ic_ind")
        ic_kind = hc2.selectbox("Tipe Visualisasi", HIERARCHY_KINDS, key="ic_kind")
        ic_pal = hc3.selectbox("Palet", PALETTE_NAMES, index=0, key="ic_pal")
        ic_depth = hc4.radio("Tampilan", ["Provinsi (klik untuk drill-down)", "Semua level"], key="ic_depth")
        if selected_prov != "Semua Provinsi":
            ic_root = selected_prov
        elif selected_island != "Semua Pulau":
            ic_root = f"Indonesia · {selected_island}"
        else:
            ic_root = "Indonesia"
        show(plot_hierarchy(ic_kind, df_f, color_col=ic_ind, color_scale=resolve_palette(ic_pal), root_label=ic_root,
                            maxdepth=2 if ic_depth.startswith("Provinsi") else -1, height=640),
             key=f"hier_{ic_kind}",
             cap="Klik wilayah untuk melakukan drill-down pada struktur hierarki.")
    with st.expander("ℹ️ Cara membaca grafik hirarki", expanded=False):
        st.markdown(
            "- **Hirarki:** Indonesia → Provinsi → Kabupaten/Kota; setiap kabupaten/kota adalah node paling bawah (daun).\n"
            "- **Ukuran area:** jumlah penduduk, sehingga wilayah berpenduduk besar tampak lebih luas. "
            "Area **tidak** menunjukkan nilai indikator.\n"
            "- **Warna:** indikator yang dipilih (dipotong pada persentil 2–98 agar outlier tidak menekan kontras). "
            f"Abu-abu = data tidak tersedia. Untuk kemiskinan dan TPT, nilai tinggi = lebih buruk.\n"
            "- **Warna provinsi/akar:** rata-rata berbobot penduduk dari kabupaten/kota di dalamnya (hitungan turunan, bukan angka resmi BPS). "
            "Bila warna = jumlah penduduk, yang ditampilkan adalah rata-rata sederhana per kabupaten/kota.\n"
            "- **Drill-down:** klik sebuah provinsi untuk membuka kabupaten/kota di dalamnya; klik bilah jalur (Icicle/Treemap) atau pusat lingkaran (Sunburst) untuk kembali. "
            "Arahkan kursor untuk melihat jumlah kab/kota, penduduk, IPM, kemiskinan, TPT, dan PDRB per kapita.\n"
            "- **Filter:** mengikuti filter Pulau/Provinsi di sidebar; bila satu provinsi dipilih, provinsi itu menjadi akar."
        )
    st.caption(f"{SRC_NOTE}. PDRB per kapita ditampilkan sebagai angka (satuan belum terverifikasi; lihat tab Metodologi & Sumber Data).")
# ================================================================== TAB 5
with tab5:
    meta, koreksi = load_metadata(), load_corrections()
    section("Sumber Data & Metadata", "Setiap indikator, satuan, sumber, dan status verifikasinya. Item yang belum jelas ditandai, tidak diisi dengan tebakan.")
    with st.container(border=True):
        md = meta.rename(columns={"kolom": "Kolom", "label": "Indikator", "satuan": "Satuan", "definisi": "Definisi", "sumber": "Sumber",
                                  "periode": "Periode", "arah_baik": "Arah baik", "status_verifikasi": "Status verifikasi"})
        st.dataframe(md, hide_index=True, **STRETCH, height=420)
        st.caption("Sumber tingkat-publikasi yang tercatat pada dokumentasi awal proyek. ID tabel dan bulan survei belum didokumentasikan; "
                   "berkas data mentah BPS tidak tersedia di repositori ini (hanya berkas staging).")
        st.markdown("**Angka nasional pembanding (kartu KPI)**")
        st.dataframe(load_benchmark_table().rename(columns={"indikator": "Indikator", "nilai": "Nilai", "periode": "Periode", "sumber": "Sumber", "catatan": "Catatan"}),
                     hide_index=True, **STRETCH)
    st.write("")
    section("Metodologi", "Ringkas dan sesuai dengan yang benar-benar dijalankan aplikasi.")
    with st.container(border=True):
        st.markdown(f"""
#### 1. Pra-pemrosesan untuk PCA
1. **Imputasi:** data tidak tersedia pada fitur PCA diisi dengan median nasional hanya untuk perhitungan PCA/klaster (ditandai pada profil wilayah).
2. **Transformasi log** pada `pdrb_perkapita_adhk` dan `jumlah_penduduk` (skewness awal 5,8 dan 2,8) agar komponen tidak didominasi segelintir kota besar.
3. **PDRB ADHB dikeluarkan** karena berkorelasi r = 0,997 dengan ADHK (redundan).
4. **Standardisasi** *Z-score*:""")
        st.latex(r"Z = \frac{X - \mu}{\sigma}")
        st.markdown(r"""
#### 2. Principal Component Analysis
Dekomposisi eigen matriks kovarians $\mathbf{\Sigma}$ (data terstandardisasi):""")
        st.latex(r"\mathbf{\Sigma}\,\mathbf{v}_i = \lambda_i\,\mathbf{v}_i")
        st.markdown(rf"""
$\lambda_i$ = varians komponen ke-$i$; $\mathbf{{v}}_i$ = bobot (*loading*). Fitur yang dipakai: {', '.join(get_short_label(c) for c in DEFAULT_PCA_FEATURES)}.

#### 3. Klasterisasi
**K-Means** (k = 4, `n_init=10`, `random_state=42`) pada fitur yang sama; klaster diurutkan menurut rata-rata IPM (K1 terendah, K4 tertinggi).

#### 4. Pilihan visual encoding
- **Choropleth** hanya untuk rasio/persentase; warna sekuensial dipotong pada persentil 2–98 agar outlier tidak menekan kontras.
- **Simbol proporsional** untuk angka absolut: ukuran = penduduk, warna = indikator.
- **Scatter PCA + biplot**: posisi = skor komponen; klik titik atau poligon peta memilih wilayah (brushing & linking).
- **Radar & Parallel coordinates**: indikator "makin tinggi makin buruk" dibalik sehingga arah *lebih baik* seragam.
- **Heatmap korelasi** diverging dengan titik tengah 0 (Spearman/Pearson).

#### 5. Keterbatasan
Data satu tahun (tanpa tren); rata-rata antarwilayah tidak berbobot penduduk; satuan PDRB belum terverifikasi; sumber penduduk belum terdokumentasi; 3 kabupaten tidak memiliki poligon peta.
""")
html('<div class="foot">🇮🇩 Dashboard Sosial-Ekonomi Indonesia 2024 · Sumber data: Badan Pusat Statistik (BPS) RI · Dokumentasi: lihat tab Metodologi &amp; Sumber Data</div>')