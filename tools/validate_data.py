"""Validasi data/master_data_2024.csv terhadap GeoJSON. Keluar dengan kode != 0 bila ada pelanggaran.
Jalankan:  python tools/validate_data.py
"""
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "master_data_2024.csv"), dtype={"Kode_Wilayah": str})
geo = json.load(open(os.path.join(ROOT, "data", "indonesia_kabkota.geojson"), encoding="utf-8"))
ids = {f["id"] for f in geo["features"]}
fails = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


check(len(df) == 514, f"514 kabupaten/kota (ditemukan {len(df)})")
check(df["Kabupaten/Kota"].is_unique, "nama kabupaten/kota unik")
check(df["Provinsi"].nunique() == 38, f"38 provinsi (ditemukan {df['Provinsi'].nunique()})")
kode = df["Kode_Wilayah"].dropna()
check(kode.is_unique, "kode wilayah (non-kosong) unik")
check(((df["Kode_Status"] == "valid") == df["Kode_Wilayah"].notna()).all(), "Kode_Status konsisten dengan kode terisi")
check(kode.str.fullmatch(r"\d{2}\.\d{2}").all(), "format kode = PP.KK (teks, nol di depan/belakang terjaga)")
pref = df.dropna(subset=["Kode_Wilayah"]).assign(p=lambda t: t["Kode_Wilayah"].str[:2]).groupby("Provinsi")["p"].nunique()
check((pref == 1).all(), "satu awalan kode untuk tiap provinsi")
for c in ["TPT_Jumlah", "sanitasi_layak", "air_minum_layak"]:
    check((df[c] == 0).sum() == 0, f"tidak ada nilai 0,00 pada {c} (0 dianggap tidak tersedia)")
for c in ["sanitasi_layak", "air_minum_layak", "TPAK_laki-laki", "TPAK_perempuan", "Persentase_Penduduk_Miskin_2024"]:
    check(df[c].dropna().between(0, 100).all(), f"{c} berada pada rentang 0-100")
t = df.dropna(subset=["TPT_Jumlah", "TPT_laki-laki", "TPT_perempuan"])
lo = t[["TPT_laki-laki", "TPT_perempuan"]].min(axis=1) - 0.01
hi = t[["TPT_laki-laki", "TPT_perempuan"]].max(axis=1) + 0.01
check(((t["TPT_Jumlah"] >= lo) & (t["TPT_Jumlah"] <= hi)).all(), "TPT total berada di antara TPT laki-laki dan perempuan (baris lengkap)")
check((df["pdrb_perkapita_adhb"] >= df["pdrb_perkapita_adhk"]).all(), "PDRB ADHB >= ADHK")
check(ids <= set(df["Kabupaten/Kota"]), "semua poligon GeoJSON punya baris data")
missing = sorted(set(df["Kabupaten/Kota"]) - ids)
print(f"INFO  wilayah tanpa poligon ({len(missing)}): {missing}")
print(f"INFO  koordinat kosong: {int(df['Latitude'].isna().sum())} | kode tidak valid: {int((df['Kode_Status'] == 'tidak_valid').sum())}")
print("\nSELESAI:", "SEMUA LOLOS" if not fails else f"{len(fails)} PELANGGARAN")
sys.exit(1 if fails else 0)
