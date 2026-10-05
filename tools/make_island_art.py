"""Membuat 7 ilustrasi header pulau (siluet dari GeoJSON proyek + lanskap pegunungan + partikel).
Gaya seragam: editorial data visualization, ungu-magenta gelap, tanpa teks/logo/label/marker, latar transparan.
Jalankan:  python tools/make_island_art.py   -> assets/islands/hero_<pulau>.svg
Hanya dekoratif; peta choropleth tetap memakai GeoJSON asli. Hasil deterministik (seed tetap).
"""
import json, math, os, random
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "islands")
W, H = 1200, 520                      # kanvas seragam
MAP_BOX = (430, 40, 1140, 400)        # area siluet (x0, y0, x1, y1) di sisi kanan; kiri dibiarkan lapang untuk teks

ISLANDS = {                           # berkas -> nama 'Pulau' pada data proyek
    "jawa": "Jawa", "sumatra": "Sumatera", "kalimantan": "Kalimantan", "sulawesi": "Sulawesi",
    "bali_nusa_tenggara": "Bali & Nusa Tenggara", "maluku": "Maluku", "papua": "Papua",
}


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


def simplify(ring, eps):
    m = len(ring) // 2
    return rdp(ring[:m + 1], eps)[:-1] + rdp(ring[m:], eps)


def load():
    gj = json.load(open(os.path.join(ROOT, "data", "indonesia_kabkota.geojson"), encoding="utf-8"))
    df = pd.read_csv(os.path.join(ROOT, "data", "master_data_2024.csv"))
    by_name = df.drop_duplicates("Kabupaten/Kota").set_index("Kabupaten/Kota")["Provinsi"].to_dict()
    sys_path = os.path.join(ROOT, "src")
    import sys
    sys.path.insert(0, ROOT)
    from src.data_loader import PROV_TO_ISLAND
    feats = {}
    for f in gj["features"]:
        p = f["properties"]
        prov = by_name.get(p.get("kabupaten_kota")) or p.get("provinsi")
        isl = PROV_TO_ISLAND.get(prov)
        if isl:
            feats.setdefault(isl, []).append(f["geometry"])
    return feats


def rings_of(geoms):
    out = []
    for g in geoms:
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            out.append(poly[0])
    return out


def make(slug, island, geoms, seed):
    rnd = random.Random(seed)
    rings = rings_of(geoms)
    xs = [x for r in rings for x, _ in r]
    ys = [y for r in rings for _, y in r]
    lon0, lon1, lat0, lat1 = min(xs), max(xs), min(ys), max(ys)
    k = math.cos(math.radians((lat0 + lat1) / 2))
    gw, gh = (lon1 - lon0) * k, (lat1 - lat0)
    bx0, by0, bx1, by1 = MAP_BOX
    sc = min((bx1 - bx0) / gw, (by1 - by0) / gh)
    ox = bx0 + ((bx1 - bx0) - gw * sc) / 2
    oy = by0 + ((by1 - by0) - gh * sc) / 2

    def proj(x, y):
        return ox + (x - lon0) * k * sc, oy + (lat1 - y) * sc

    tiny = 3.0 if island in ("Maluku", "Bali & Nusa Tenggara") else 6.0
    paths = []
    for r in rings:
        pr = [proj(x, y) for x, y in r]
        px, py = [p[0] for p in pr], [p[1] for p in pr]
        if math.hypot(max(px) - min(px), max(py) - min(py)) < tiny:
            continue
        s = simplify(pr, 0.9)
        if len(s) >= 4:
            paths.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in s) + "Z")
    land = "".join(paths)

    def ridge(base, amp, seg, rough, phase):
        pts, n = [], 60
        for i in range(n + 1):
            x = W * i / n
            y = base - amp * (0.55 * math.sin(i / n * seg * math.pi + phase) + 0.3 * math.sin(i / n * seg * 2.3 * math.pi + phase * 1.7)
                              + rough * (rnd.random() - 0.5))
            pts.append((x, y))
        return "M0 %d " % H + "L".join(f"{x:.1f} {y:.1f}" for x, y in pts) + f"L{W} {H}Z"

    m1 = ridge(H - 128, 46, 3.1, 0.25, rnd.random() * 6)
    m2 = ridge(H - 88, 38, 4.4, 0.3, rnd.random() * 6)
    m3 = ridge(H - 48, 26, 6.2, 0.35, rnd.random() * 6)

    stars = []
    for _ in range(70):
        x, y = rnd.uniform(0, W), rnd.uniform(0, H * 0.78)
        r = rnd.choice([0.6, 0.8, 1.0, 1.3, 1.7])
        o = rnd.uniform(0.25, 0.85)
        stars.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" opacity="{o:.2f}"/>')
    dust = []
    for _ in range(26):
        x, y = rnd.uniform(bx0 - 40, bx1 + 40), rnd.uniform(by0 - 20, by1 + 60)
        dust.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{rnd.uniform(1.8, 3.4):.1f}" opacity="{rnd.uniform(.15, .4):.2f}"/>')

    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMaxYMid slice" role="img" aria-label="Ilustrasi dekoratif {island.replace("&","dan")}">
<defs>
<radialGradient id="aura" cx="{cx / W:.3f}" cy="{cy / H:.3f}" r="0.55"><stop offset="0" stop-color="#d946a8" stop-opacity=".42"/><stop offset=".45" stop-color="#7c2d9c" stop-opacity=".22"/><stop offset="1" stop-color="#2a1160" stop-opacity="0"/></radialGradient>
<linearGradient id="isl2" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e9a6da"/><stop offset=".55" stop-color="#9d4b9a"/><stop offset="1" stop-color="#4c1d95"/></linearGradient>
<linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7e3a96" stop-opacity=".55"/><stop offset="1" stop-color="#2b1263" stop-opacity=".0"/></linearGradient>
<linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5b2a86" stop-opacity=".75"/><stop offset="1" stop-color="#1e0f4a" stop-opacity=".0"/></linearGradient>
<linearGradient id="g3" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#341563" stop-opacity=".95"/><stop offset="1" stop-color="#17112f" stop-opacity=".0"/></linearGradient>
<filter id="glow" x="-15%" y="-15%" width="130%" height="130%"><feGaussianBlur stdDeviation="9"/></filter>
<path id="land" d="{land}"/>
</defs>
<rect width="{W}" height="{H}" fill="url(#aura)"/>
<g fill="#f5d0fe">{"".join(stars)}</g>
<use href="#land" fill="#e879c6" opacity=".5" filter="url(#glow)"/>
<g opacity=".78"><use href="#land" fill="none" stroke="#fbcfe8" stroke-width="2.4" stroke-linejoin="round"/>
<use href="#land" fill="url(#isl2)" stroke="url(#isl2)" stroke-width=".9" stroke-linejoin="round"/></g>
<g fill="#fdf4ff">{"".join(dust)}</g>
<path d="{m1}" fill="url(#g1)"/><path d="{m2}" fill="url(#g2)"/><path d="{m3}" fill="url(#g3)"/>
</svg>'''
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"hero_{slug}.svg")
    open(path, "w", encoding="utf-8").write(svg)
    return path, len(svg) / 1024, len(paths)


if __name__ == "__main__":
    feats = load()
    for i, (slug, isl) in enumerate(ISLANDS.items()):
        p, kb, n = make(slug, isl, feats[isl], seed=100 + i)
        print("OK", os.path.relpath(p, ROOT), f"{kb:.0f} KB", n, "poligon")
