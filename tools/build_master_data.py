"""Membangun data/master_data_2024.csv dari berkas staging (data/raw/master_data_2024_original.csv).

Seluruh koreksi dicatat di data/koreksi_data.csv sehingga dapat diaudit. Skrip bersifat deterministik.
Urutan:  python tools/fetch_replacement_boundaries.py ; python tools/build_geojson.py ; python tools/build_master_data.py

Aturan koreksi (setiap aturan berbasis bukti dari data itu sendiri, bukan tebakan):
 R1  Kode_Wilayah dibaca sebagai TEKS (kode adalah pengenal, bukan angka).
 R2  Kode ganda: pada pasangan kabupaten/kota berkode sama, kode dipertahankan untuk baris yang sesuai konvensi
     Kemendagri (kota: nomor kab/kota >= 71; kabupaten: < 71). Baris lain dikosongkan (kode asli tidak diketahui,
     TIDAK dikarang) dan diberi Kode_Status = 'tidak_valid'.
 R3  Provinsi: 'Pegunungan Bintang' (IPM 49,36; penduduk 83.330) berlabel Kepulauan Riau -> Papua Pegunungan;
     'Kota Banjar' berlabel Kalimantan Selatan -> Jawa Barat (poligon sumber & kota Jawa Barat tidak lengkap tanpa baris ini).
 R4  Latitude/Longitude kabupaten yang berbagi koordinat dengan kotanya dihitung ulang dari centroid poligon yang
     benar; bila poligon tidak tersedia, koordinat dikosongkan.
 R5  Nilai 0,00 pada TPT_Jumlah (beserta TPT L/P baris itu), sanitasi_layak, dan air_minum_layak diperlakukan sebagai
     'tidak tersedia' (NaN). Nol pada TPT_laki-laki / TPT_perempuan saja dipertahankan (dapat valid pada sampel kecil).
"""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
df = pd.read_csv(os.path.join(RAW, "master_data_2024_original.csv"), dtype={"Kode_Wilayah": str})
status = json.load(open(os.path.join(RAW, "batas_status.json"), encoding="utf-8"))
geo = json.load(open(os.path.join(ROOT, "data", "indonesia_kabkota.geojson"), encoding="utf-8"))
log = []


def note(name, col, old, new, why):
    log.append({"Kabupaten/Kota": name, "kolom": col, "nilai_lama": old, "nilai_baru": new, "alasan": why})


def centroid(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    cx = cy = tot = 0.0
    for poly in polys:
        r = np.array(poly[0]); x, y = r[:, 0], r[:, 1]
        cr = x[:-1] * y[1:] - x[1:] * y[:-1]; a = 0.5 * cr.sum()
        if abs(a) < 1e-12:
            continue
        cx += abs(a) * (np.sum((x[:-1] + x[1:]) * cr) / (6 * a)); cy += abs(a) * (np.sum((y[:-1] + y[1:]) * cr) / (6 * a)); tot += abs(a)
    return round(cy / tot, 5), round(cx / tot, 5)


cent = {f["id"]: centroid(f["geometry"]) for f in geo["features"]}
df["Kode_Status"] = "valid"

# R3 provinsi
for name, prov in {"Pegunungan Bintang": "Papua Pegunungan", "Kota Banjar": "Jawa Barat"}.items():
    i = df.index[df["Kabupaten/Kota"] == name][0]
    note(name, "Provinsi", df.at[i, "Provinsi"], prov, "R3: profil & jumlah penduduk tidak sesuai provinsi tercatat")
    df.at[i, "Provinsi"] = prov

# R2 kode ganda
dup_codes = df.index[df.Kode_Wilayah.duplicated(keep=False)]
for kode, grp in df.loc[dup_codes].groupby("Kode_Wilayah"):
    nomor = int(kode.split(".")[1])
    for i in grp.index:
        is_kota = df.at[i, "Kabupaten/Kota"].startswith("Kota ")
        nama = df.at[i, "Kabupaten/Kota"]
        valid = ((is_kota and nomor >= 71) or ((not is_kota) and nomor < 71)) and status.get(nama) not in ("diganti", "tidak_tersedia")
        if not valid:
            note(df.at[i, "Kabupaten/Kota"], "Kode_Wilayah", kode, "", "R2: kode ganda, tidak sesuai konvensi kab/kota; kode asli tidak diketahui")
            df.at[i, "Kode_Wilayah"] = np.nan
            df.at[i, "Kode_Status"] = "tidak_valid"

# R4 koordinat
for name, st in status.items():
    i = df.index[df["Kabupaten/Kota"] == name]
    if len(i) == 0 or st == "dipertahankan":
        continue
    i = i[0]
    new = cent.get(name, (np.nan, np.nan))
    note(name, "Latitude/Longitude", f"{df.at[i,'Latitude']},{df.at[i,'Longitude']}",
         "" if np.isnan(new[0]) else f"{new[0]},{new[1]}", "R4: koordinat sama dengan kota padanan; dihitung ulang dari poligon")
    df.at[i, "Latitude"], df.at[i, "Longitude"] = new

# R5 nol -> NaN
z = df["TPT_Jumlah"] == 0
for i in df.index[z]:
    for c in ["TPT_Jumlah", "TPT_laki-laki", "TPT_perempuan"]:
        note(df.at[i, "Kabupaten/Kota"], c, df.at[i, c], "", "R5: TPT total 0,00 dianggap tidak tersedia")
        df.at[i, c] = np.nan
for c in ["sanitasi_layak", "air_minum_layak"]:
    for i in df.index[df[c] == 0]:
        note(df.at[i, "Kabupaten/Kota"], c, 0.0, "", "R5: akses 0,00% dianggap tidak tersedia (perlu verifikasi ke BPS)")
        df.loc[i, c] = np.nan

df.to_csv(os.path.join(ROOT, "data", "master_data_2024.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(ROOT, "data", "koreksi_data.csv"), index=False)
print(f"OK: {len(df)} baris, {len(log)} koreksi tercatat -> data/koreksi_data.csv")
print("kode unik (non-kosong):", df.Kode_Wilayah.dropna().is_unique, "| NaN kode:", int(df.Kode_Wilayah.isna().sum()))
