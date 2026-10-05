"""Membuat ilustrasi hero (siluet peta Indonesia bercahaya + jaringan titik) dari GeoJSON & CSV proyek.
Jalankan:  python tools/make_hero_svg.py   -> assets/hero_indonesia.svg
"""
import json, math, random, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LON0, LON1, LAT0, LAT1, K = 94.6, 141.4, -11.4, 6.4, 10.0
W, H = (LON1 - LON0) * K, (LAT1 - LAT0) * K

def proj(lon, lat):
    return (lon - LON0) * K, (LAT1 - lat) * K

def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy) or 1e-9
    dmax, idx = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * pts[i][0] - dx * pts[i][1] + x2 * y1 - y2 * x1) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        return rdp(pts[:idx + 1], eps)[:-1] + rdp(pts[idx:], eps)
    return [pts[0], pts[-1]]

gj = json.load(open(os.path.join(ROOT, "data", "indonesia_kabkota.geojson"), encoding="utf-8"))
paths = []
for f in gj["features"]:
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        ring = [proj(x, y) for x, y in poly[0]]
        xs, ys = [p[0] for p in ring], [p[1] for p in ring]
        if math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < 1.6:   # buang pulau sangat kecil
            continue
        m = len(ring) // 2
        s = rdp(ring[:m + 1], 0.5)[:-1] + rdp(ring[m:], 0.5)
        if len(s) < 4:
            continue
        paths.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in s) + "Z")
d = "".join(paths)

df = pd.read_csv(os.path.join(ROOT, "data", "master_data_2024.csv"))
random.seed(7)
pts = df.sort_values("jumlah_penduduk", ascending=False)
big = pts.head(28)
rest = pts.iloc[28:].sample(52, random_state=7)
nodes = [proj(r.Longitude, r.Latitude) for r in pd.concat([big, rest]).itertuples()]
nodes = [n for n in nodes if 0 <= n[0] <= W and 0 <= n[1] <= H]
edges = set()
for i, a in enumerate(nodes):
    near = sorted(range(len(nodes)), key=lambda j: (nodes[j][0] - a[0]) ** 2 + (nodes[j][1] - a[1]) ** 2)[1:4]
    for j in near[:random.choice([2, 3])]:
        edges.add(tuple(sorted((i, j))))
lines = "".join(f'<line x1="{nodes[i][0]:.1f}" y1="{nodes[i][1]:.1f}" x2="{nodes[j][0]:.1f}" y2="{nodes[j][1]:.1f}"/>' for i, j in edges)
dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.5 if k < 28 else 0.9}"/>' for k, (x, y) in enumerate(nodes))
halo = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2"/>' for k, (x, y) in enumerate(nodes) if k < 28)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="xMaxYMid meet">
<defs>
<linearGradient id="fill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6366f1" stop-opacity=".42"/><stop offset="1" stop-color="#14b8a6" stop-opacity=".34"/></linearGradient>
<filter id="glow" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="4"/></filter>
<filter id="glow2" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.2"/></filter>
<path id="land" d="{d}"/>
</defs>
<use href="#land" fill="#818cf8" opacity=".55" filter="url(#glow)"/>
<use href="#land" fill="url(#fill)" stroke="#a5b4fc" stroke-opacity=".75" stroke-width=".35" stroke-linejoin="round"/>
<g stroke="#c7d2fe" stroke-opacity=".5" stroke-width=".4">{lines}</g>
<g fill="#67e8f9" opacity=".45" filter="url(#glow2)">{halo}</g>
<g fill="#e0f2fe">{dots}</g>
</svg>'''
out = os.path.join(ROOT, "assets", "hero_indonesia.svg")
open(out, "w", encoding="utf-8").write(svg)
print("OK", out, f"{len(svg)/1024:.0f} KB", len(paths), "poligon", len(nodes), "node", len(edges), "garis")
