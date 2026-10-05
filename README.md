# Dashboard Sosial-Ekonomi Indonesia 2024 (BPS)

Dashboard Streamlit interaktif untuk 514 kabupaten/kota (38 provinsi), 13 indikator numerik.
Proyek UAS Visualisasi Data.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tanpa API key. Peta dasar memakai tile daring gratis (Positron, Dark Matter, Voyager, Satelit Esri, OpenStreetMap); bila offline pilih gaya peta **Polos** di sidebar.

**Opsional — gaya Mapbox:** buat token publik di akun Mapbox, lalu salin `.streamlit/secrets.toml.example` menjadi `.streamlit/secrets.toml` dan isi `MAPBOX_TOKEN` (atau set variabel lingkungan `MAPBOX_TOKEN`). Lima gaya Mapbox (Gelap, Terang, Jalan, Alam, Satelit+jalan) lalu muncul pada pilihan *Gaya peta*. Tanpa token, aplikasi berjalan normal dan gaya itu tidak ditampilkan. Token tidak ada di kode; batasi token pada URL aplikasi Anda di dasbor Mapbox.
Kompatibel dengan Streamlit ≥ 1.40 (parameter lebar otomatis `width="stretch"` atau `use_container_width`)
dan Plotly ≥ 5.20. Diuji pada Streamlit 1.65 dan Plotly 7.1.

## Isi dashboard (7 tab)

| Tab | Isi |
|---|---|
| Ringkasan & Peta | KPI (IPM, kemiskinan, TPT, TPAK) vs angka nasional, choropleth, peringkat, wilayah terpilih |
| Temuan Utama | 5 pertanyaan analitis; semua angka dihitung langsung dari data, lengkap dengan keterbatasan |
| Eksplorasi PCA | Scree, biplot, loading, klaster K-Means (k=4) |
| Profil Multivariat | Radar, parallel coordinates, heatmap korelasi (Spearman/Pearson) |
| Peta & Data | Simbol proporsional, tabel data, unduh CSV |
| Hirarki Wilayah | Icicle / Treemap / Sunburst Indonesia → Provinsi → Kabupaten/Kota; luas = jumlah penduduk, warna = indikator pilihan; klik provinsi untuk drill-down |
| Metodologi & Sumber Data | Metadata indikator, angka pembanding, kualitas data, log koreksi, metodologi |

Brushing & linking: klik poligon peta atau titik scatter PCA untuk memilih wilayah; seleksi dipakai bersama.

## Struktur

```
app.py                  aplikasi utama
src/                    data_loader, pca_analysis, visualizer, theme
data/                   master_data_2024.csv, indonesia_kabkota.geojson, koreksi_data.csv,
                        metadata_indikator.csv, benchmark_nasional.csv, raw/ (berkas asli/staging)
assets/                 hero_indonesia.svg; islands/hero_<pulau>.svg (7 ilustrasi header dekoratif per pulau)
tools/                  validate_data.py, build_master_data.py, build_geojson.py,
                        fetch_replacement_boundaries.py, make_hero_svg.py, make_island_art.py
test_end_to_end.py      uji menyeluruh (data, figure, smoke test aplikasi)
```

Urutan pipeline data: `fetch_replacement_boundaries.py → build_geojson.py → build_master_data.py`.

## Validasi

```bash
python tools/validate_data.py
python test_end_to_end.py
```

## Deploy (Streamlit Community Cloud)

1. Unggah folder ini ke repositori GitHub (jangan unggah `.env`/secrets).
2. Di share.streamlit.io pilih repositori, branch, dan file utama `app.py`.
3. `requirements.txt` dibaca otomatis.

## Pembersihan data (deterministik, tercatat di `data/koreksi_data.csv`)

- Kode wilayah dibaca sebagai teks; kode ganda yang tidak sesuai konvensi Kemendagri dikosongkan.
- Koreksi provinsi: Pegunungan Bintang → Papua Pegunungan; Kota Banjar → Jawa Barat.
- Nilai 0,00 pada TPT total, sanitasi, dan air minum dianggap tidak tersedia (NaN).
- PDRB ADHB dikeluarkan dari PCA (r = 0,997 dengan ADHK); imputasi median hanya untuk PCA/klaster.
- Polygon 24 kabupaten diganti dengan batas GADM (± 2010).

## Keterbatasan & hal yang belum terverifikasi

- Satu tahun (cross-sectional): temuan adalah keterkaitan, bukan sebab-akibat.
- Satuan PDRB per kapita belum diverifikasi terhadap tabel BPS asli; gunakan untuk peringkat, bukan nilai rupiah absolut.
- Angka nasional pembanding (IPM 75,02; kemiskinan 9,03; TPT 4,91) dan periodenya perlu dicek ke publikasi BPS sebelum dikutip.
- Sumber `jumlah_penduduk` belum terdokumentasi; ID tabel dan bulan survei BPS belum tercatat.
- 3 kabupaten tanpa poligon (Blitar, Jayapura, Sorong); 27 kode wilayah tidak valid (kode asli tidak diketahui, tidak dikarang).
