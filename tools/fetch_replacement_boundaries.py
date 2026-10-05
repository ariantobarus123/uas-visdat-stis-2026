"""Mengambil batas poligon pengganti untuk kabupaten yang di GeoJSON asli identik dengan poligon kotanya.

Sumber batas: GADM 2.x (batas administrasi Indonesia, tingkat-2) yang dimuat ulang pada repo publik
https://github.com/rifani/geojson-political-indonesia  (berkas IDN_adm_2_kabkota.json).
Catatan: batas GADM bersifat lama (sekitar 2010) -> hanya dipakai untuk mengisi poligon yang HILANG,
bukan untuk menimpa poligon yang sudah benar.

Keluaran: data/raw/batas_pengganti_gadm.geojson (hanya fitur pengganti, sudah disederhanakan).
Jalankan:  python tools/fetch_replacement_boundaries.py
"""
import json, os, sys, urllib.request, hashlib, collections, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://raw.githubusercontent.com/rifani/geojson-political-indonesia/master/IDN_adm_2_kabkota.json"
ORIG = os.path.join(ROOT, "data", "raw", "indonesia_kabkota_original.geojson")
OUT = os.path.join(ROOT, "data", "raw", "batas_pengganti_gadm.geojson")

# Kasus khusus yang hasil pemeriksaan (jarak centroid) menunjukkan poligon bersama milik salah satu nama.
#  nama CSV -> (nama GADM, provinsi GADM, tipe GADM)
MANUAL = {
    "Pegunungan Bintang": ("Pegunungan Bintang", "Papua", "Kabupaten"),
    "Kota Banjar": ("Banjar", "Jawa Barat", "Kotamadya"),
}
KEEP = {"Bintan", "Banjar"}   # centroid poligon bersama berada di wilayah ini, bukan di padanannya
norm = lambda s: s.lower().replace(" ", "").replace("kota", "")


def centroid(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    cx = cy = tot = 0.0
    for poly in polys:
        r = np.array(poly[0]); x, y = r[:, 0], r[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]; a = 0.5 * cr.sum()
        if abs(a) < 1e-12:
            continue
        cx += abs(a) * (np.sum((x[:-1] + x[1:]) * cr) / (6 * a)); cy += abs(a) * (np.sum((y[:-1] + y[1:]) * cr) / (6 * a)); tot += abs(a)
    return cy / tot, cx / tot


def rdp(pts, eps):
    if len(pts) < 3:
        return pts
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    dx, dy = x2 - x1, y2 - y1; n = math.hypot(dx, dy) or 1e-12
    dmax, idx = 0, 0
    for i in range(1, len(pts) - 1):
        d = abs(dy * pts[i][0] - dx * pts[i][1] + x2 * y1 - y2 * x1) / n
        if d > dmax:
            dmax, idx = d, i
    return rdp(pts[:idx + 1], eps)[:-1] + rdp(pts[idx:], eps) if dmax > eps else [pts[0], pts[-1]]


def simplify(geom, eps=0.004):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    out = []
    for poly in polys:
        ring = [tuple(p[:2]) for p in poly[0]]
        m = len(ring) // 2
        s = rdp(ring[:m + 1], eps)[:-1] + rdp(ring[m:], eps)
        if len(s) >= 4:
            out.append([[[round(x, 4), round(y, 4)] for x, y in s]])
    return {"type": "MultiPolygon", "coordinates": out}


def main(src=None):
    gadm = json.load(open(src, encoding="utf-8")) if src else json.load(urllib.request.urlopen(URL, timeout=90))
    cand = [(p["properties"], p["geometry"]) for p in gadm["features"] if p.get("geometry")]
    cand = [(pr, g, centroid(g)) for pr, g in cand]
    orig = json.load(open(ORIG, encoding="utf-8"))
    groups = collections.defaultdict(list)
    for f in orig["features"]:
        groups[hashlib.md5(json.dumps(f["geometry"], sort_keys=True).encode()).hexdigest()].append(f)
    feats, log, status = [], [], {}
    for grp in [g for g in groups.values() if len(g) > 1]:
        c0 = centroid(grp[0]["geometry"])
        for f in grp:
            name = f["id"]
            is_kota = name.startswith("Kota ")
            if name in MANUAL:
                nm, prov, typ = MANUAL[name]
                pick = [c for c in cand if c[0]["NAME_2"] == nm and c[0]["NAME_1"] == prov and c[0]["TYPE_2"] == typ]
            elif not is_kota:      # kabupaten: poligon bersama adalah milik kota padanannya
                pick = [c for c in cand if norm(c[0]["NAME_2"]) == norm(name) and c[0]["TYPE_2"] == "Kabupaten"]
            else:
                status[name] = "dipertahankan"
                continue            # poligon bersama dipertahankan untuk kota
            if name in KEEP:        # pemilik sah poligon bersama (hasil pemeriksaan jarak centroid)
                log.append((name, "poligon asli dipertahankan"))
                status[name] = "dipertahankan"
                continue
            if not pick:
                log.append((name, "TIDAK TERSEDIA di sumber pengganti (poligon dikosongkan)"))
                status[name] = "tidak_tersedia"
                continue
            pr, g, c = pick[0]
            feats.append({"type": "Feature", "id": name,
                          "properties": {"kabupaten_kota": name, "sumber": f"GADM 2.x {pr['TYPE_2']} {pr['NAME_2']} ({pr['NAME_1']})"},
                          "geometry": simplify(g)})
            log.append((name, f"diganti dari GADM: {pr['NAME_2']} / {pr['NAME_1']}"))
            status[name] = "diganti"
    json.dump({"type": "FeatureCollection", "features": feats}, open(OUT, "w", encoding="utf-8"), separators=(",", ":"))
    json.dump(status, open(os.path.join(ROOT, "data", "raw", "batas_status.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for l in log:
        print(*l, sep=" -> ")
    print(f"OK: {len(feats)} poligon pengganti -> {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
