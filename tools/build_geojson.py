"""GeoJSON asli + poligon pengganti -> data/indonesia_kabkota.geojson (tanpa poligon kembar).
Urutan fitur: kabupaten dahulu, kota terakhir, sehingga poligon kota (lebih kecil) berada di atas saat tumpang tindih.
Jalankan:  python tools/build_geojson.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
orig = json.load(open(os.path.join(RAW, "indonesia_kabkota_original.geojson"), encoding="utf-8"))
repl = {f["id"]: f for f in json.load(open(os.path.join(RAW, "batas_pengganti_gadm.geojson"), encoding="utf-8"))["features"]}
status = json.load(open(os.path.join(RAW, "batas_status.json"), encoding="utf-8"))
out = []
for f in orig["features"]:
    st = status.get(f["id"])
    if st == "diganti":
        f = {**f, "geometry": repl[f["id"]]["geometry"]}
    elif st == "tidak_tersedia":
        continue                      # tidak ada poligon yang benar -> jangan menggambar poligon milik wilayah lain
    out.append(f)
out.sort(key=lambda f: f["id"].startswith("Kota "))
json.dump({"type": "FeatureCollection", "features": out}, open(os.path.join(ROOT, "data", "indonesia_kabkota.geojson"), "w", encoding="utf-8"), separators=(",", ":"))
print(f"OK: {len(out)} fitur ({len(orig['features']) - len(out)} dikosongkan: {[k for k, v in status.items() if v == 'tidak_tersedia']})")
