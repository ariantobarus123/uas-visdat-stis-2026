import os
import json
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV_PATH = os.path.join(DATA_DIR, "master_data_2024.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, "indonesia_kabkota.geojson")

INDICATORS_META = {
    "IPM": {
        "label": "Indeks Pembangunan Manusia (IPM)",
        "unit": "Poin (0-100)",
        "format": "{:.2f}",
        "desc": "Mengukur capaian pembangunan manusia berbasis kesehatan, pendidikan, dan standar hidup layak.",
        "higher_is_better": True
    },
    "Persentase_Penduduk_Miskin_2024": {
        "label": "Persentase Penduduk Miskin",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Proporsi penduduk dengan pengeluaran per kapita di bawah Garis Kemiskinan pada tahun 2024.",
        "higher_is_better": False
    },
    "TPT_Jumlah": {
        "label": "Tingkat Pengangguran Terbuka (TPT Total)",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Persentase jumlah pengangguran terhadap total angkatan kerja.",
        "higher_is_better": False
    },
    "TPT_laki-laki": {
        "label": "TPT Laki-laki",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Tingkat Pengangguran Terbuka untuk penduduk laki-laki.",
        "higher_is_better": False
    },
    "TPT_perempuan": {
        "label": "TPT Perempuan",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Tingkat Pengangguran Terbuka untuk penduduk perempuan.",
        "higher_is_better": False
    },
    "TPAK_laki-laki": {
        "label": "TPAK Laki-laki",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Tingkat Partisipasi Angkatan Kerja laki-laki usia 15 tahun ke atas.",
        "higher_is_better": True
    },
    "TPAK_perempuan": {
        "label": "TPAK Perempuan",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Tingkat Partisipasi Angkatan Kerja perempuan usia 15 tahun ke atas.",
        "higher_is_better": True
    },
    "pdrb_perkapita_adhb": {
        "label": "PDRB Per Kapita ADHB",
        "unit": "Rp*",
        "format": "Rp {:,.2f}",
        "desc": "PDRB per kapita atas dasar harga berlaku (ADHB). *Satuan belum terverifikasi terhadap tabel BPS.",
        "higher_is_better": True
    },
    "pdrb_perkapita_adhk": {
        "label": "PDRB Per Kapita ADHK",
        "unit": "Rp*",
        "format": "Rp {:,.2f}",
        "desc": "PDRB per kapita atas dasar harga konstan (ADHK). *Satuan belum terverifikasi terhadap tabel BPS.",
        "higher_is_better": True
    },
    "pertumbuhan_ekonomi": {
        "label": "Laju Pertumbuhan Ekonomi",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Pertumbuhan riil PDRB kabupaten/kota dari tahun sebelumnya.",
        "higher_is_better": True
    },
    "jumlah_penduduk": {
        "label": "Jumlah Penduduk",
        "unit": "Jiwa",
        "format": "{:,.0f}",
        "desc": "Estimasi jumlah penduduk kabupaten/kota pada pertengahan tahun 2024.",
        "higher_is_better": None
    },
    "sanitasi_layak": {
        "label": "Akses Sanitasi Layak",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Persentase rumah tangga yang memiliki akses terhadap fasilitas sanitasi layak.",
        "higher_is_better": True
    },
    "air_minum_layak": {
        "label": "Akses Air Minum Layak",
        "unit": "%",
        "format": "{:.2f}%",
        "desc": "Persentase rumah tangga yang memiliki akses terhadap sumber air minum layak.",
        "higher_is_better": True
    }
}

# Indikator berbentuk rasio/persentase -> layak untuk choropleth. Angka absolut (penduduk, PDRB) memakai simbol proporsional / scatter.
CHOROPLETH_OK = ["IPM", "Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "TPT_laki-laki", "TPT_perempuan",
                 "TPAK_laki-laki", "TPAK_perempuan", "pertumbuhan_ekonomi", "sanitasi_layak", "air_minum_layak"]
# Indikator "makin tinggi makin buruk" -> dibalik pada radar/PCP agar arah "lebih baik" seragam
LOWER_IS_BETTER = ["Persentase_Penduduk_Miskin_2024", "TPT_Jumlah", "TPT_laki-laki", "TPT_perempuan"]

PROV_TO_ISLAND = {
    "Aceh": "Sumatera", "Sumatera Utara": "Sumatera", "Sumatera Barat": "Sumatera",
    "Riau": "Sumatera", "Jambi": "Sumatera", "Sumatera Selatan": "Sumatera",
    "Bengkulu": "Sumatera", "Lampung": "Sumatera", "Kepulauan Bangka Belitung": "Sumatera",
    "Kepulauan Riau": "Sumatera",
    "DKI Jakarta": "Jawa", "Jawa Barat": "Jawa", "Jawa Tengah": "Jawa",
    "DI Yogyakarta": "Jawa", "Jawa Timur": "Jawa", "Banten": "Jawa",
    "Bali": "Bali & Nusa Tenggara", "Nusa Tenggara Barat": "Bali & Nusa Tenggara",
    "Nusa Tenggara Timur": "Bali & Nusa Tenggara",
    "Kalimantan Barat": "Kalimantan", "Kalimantan Tengah": "Kalimantan",
    "Kalimantan Selatan": "Kalimantan", "Kalimantan Timur": "Kalimantan",
    "Kalimantan Utara": "Kalimantan",
    "Sulawesi Utara": "Sulawesi", "Sulawesi Tengah": "Sulawesi",
    "Sulawesi Selatan": "Sulawesi", "Sulawesi Tenggara": "Sulawesi",
    "Gorontalo": "Sulawesi", "Sulawesi Barat": "Sulawesi",
    "Maluku": "Maluku", "Maluku Utara": "Maluku",
    "Papua": "Papua", "Papua Barat": "Papua", "Papua Selatan": "Papua",
    "Papua Tengah": "Papua", "Papua Pegunungan": "Papua", "Papua Barat Daya": "Papua"
}

SHORT_LABELS = {
    "IPM": "IPM",
    "Persentase_Penduduk_Miskin_2024": "Kemiskinan",
    "TPT_Jumlah": "TPT",
    "TPT_laki-laki": "TPT Laki-laki",
    "TPT_perempuan": "TPT Perempuan",
    "TPAK_laki-laki": "TPAK Laki-laki",
    "TPAK_perempuan": "TPAK Perempuan",
    "pdrb_perkapita_adhb": "PDRB/kap ADHB",
    "pdrb_perkapita_adhk": "PDRB/kap ADHK",
    "pertumbuhan_ekonomi": "Pertumbuhan Ekon.",
    "jumlah_penduduk": "Penduduk",
    "sanitasi_layak": "Sanitasi",
    "air_minum_layak": "Air Minum",
}

BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark_nasional.csv")
METADATA_PATH = os.path.join(DATA_DIR, "metadata_indikator.csv")
CORRECTIONS_PATH = os.path.join(DATA_DIR, "koreksi_data.csv")


def _load_benchmark():
    """Angka nasional pembanding dibaca dari data/benchmark_nasional.csv (nilai + periode + sumber)."""
    bm = pd.read_csv(BENCHMARK_PATH)
    return {r.indikator: r.nilai for r in bm.itertuples()}


OFFICIAL_BPS_2024 = _load_benchmark()

@st.cache_data
def load_dataset():
    """Load the cleaned 2024 master dataset."""
    df = pd.read_csv(CSV_PATH, dtype={"Kode_Wilayah": str})   # kode = pengenal teks, bukan angka
    df["Pulau"] = df["Provinsi"].map(PROV_TO_ISLAND)
    return df


@st.cache_data
def load_benchmark_table():
    return pd.read_csv(BENCHMARK_PATH)


@st.cache_data
def load_metadata():
    return pd.read_csv(METADATA_PATH)


@st.cache_data
def load_corrections():
    return pd.read_csv(CORRECTIONS_PATH)

@st.cache_data
def load_geojson():
    """Load the pre-optimized GeoJSON for Indonesian regencies/cities."""
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
