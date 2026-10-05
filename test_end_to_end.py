"""Uji asap (smoke test) end-to-end. Jalankan dari folder proyek mana pun:  python test_end_to_end.py"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))      # path relatif ke berkas ini (tanpa path absolut)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

print("1. Validasi data...")
assert subprocess.run([sys.executable, os.path.join("tools", "validate_data.py")], capture_output=True).returncode == 0, \
    "tools/validate_data.py gagal; jalankan langsung untuk melihat detail"

from src.data_loader import load_dataset, load_geojson
from src.pca_analysis import run_pca_pipeline, DEFAULT_PCA_FEATURES
from src import visualizer as v

print("2. Memuat data & GeoJSON...")
df, geo = load_dataset(), load_geojson()
assert len(df) == 514 and len(geo["features"]) >= 500

print("3. PCA & klaster...")
pca = run_pca_pipeline(df, DEFAULT_PCA_FEATURES, n_components=4)
assert pca["result_df"].shape[0] == 514 and pca["result_df"]["Klaster"].nunique() == 4
d = df.copy()
for c in pca["pc_cols"] + ["Klaster"]:
    d[c] = pca["result_df"][c].values

print("4. Membangun semua figure...")
figs = [
    v.plot_choropleth_map(d, geo, "IPM"), v.plot_proportional_symbol_map(d), v.plot_pca_scatter(pca["result_df"], color_by="Klaster",
        show_biplot=True, loadings_df=pca["loadings_df"], selected_kab="Sleman"),
    v.plot_scree(pca["explained_variance"], pca["cumulative_variance"], pca["pc_cols"]), v.plot_loadings_heatmap(pca["loadings_df"]),
    v.plot_parallel_coordinates(d), v.plot_correlation_heatmap(d), v.plot_bivariate(d, "IPM", "TPT_Jumlah")[0],
    v.plot_kabupaten_radar(d, "Sleman"), v.plot_rank_bar(d, "IPM"), v.plot_island_bar(d, "IPM"), v.plot_distribution(d, "IPM"),
    v.plot_bubble_overview(d), v.plot_cluster_profile(d), v.plot_gap_bar(d), v.plot_province_range(d),
    *[v.plot_hierarchy(k, d, color_col=c) for k in v.HIERARCHY_KINDS for c in v.ICICLE_INDICATORS],
        *[v.plot_choropleth_map(d, geo, 'IPM', map_style=s) for s in v.available_map_styles()],
        *[v.plot_proportional_symbol_map(d, map_style=s) for s in v.available_map_styles()],
]
print(f"   {len(figs)} figure berhasil dibuat")

print("5. Smoke test aplikasi Streamlit...")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=240).run()
assert not at.exception, [e.value for e in at.exception]
print("SEMUA UJI LOLOS")
