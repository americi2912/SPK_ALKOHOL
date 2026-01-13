import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from topsis import topsis_rank, normalize_weights

st.set_page_config(page_title="SPK Alkohol - TOPSIS", layout="wide")
st.title("Sistem Pendukung Keputusan Alkohol (TOPSIS)")
st.caption("Rekomendasi otomatis dari dataset bawaan menggunakan 5 kriteria: Harga, Brand, Komposisi, Estetika Botol, Ketersediaan.")

DATASET_FILENAME = "Dataset Alkohol.xlsx"


# ------------------ BACA EXCEL (HEADER DI BARIS product_id) ------------------
def read_dataset(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)

    header_row = None
    for i in range(min(80, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains("product_id", na=False).any():
            header_row = i
            break

    if header_row is None:
        df = pd.read_excel(file_path)
    else:
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = raw.iloc[header_row].tolist()

    # rapikan
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan" and not c.startswith("Unnamed")]]
    df = df.dropna(axis=1, how="all")

    # pastikan kolom penting ada
    required = ["product_name", "category", "main_ingredients", "price_idr", "origin_country", "packaging"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dataset tidak sesuai format. Kolom hilang: {missing}")
        st.stop()

    return df


# ------------------ KONVERSI TEKS -> SKOR 1–5 (SESUAI 5 KRITERIA AWAL) ------------------
def score_brand(row) -> int:
    """
    Proxy BRAND (1–5) dari kombinasi:
    - origin_country: impor cenderung lebih tinggi dari lokal
    - category: Spirit/Wine cenderung 'brand image' lebih tinggi daripada Beer/RTD
    """
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    # skor kategori (brand image)
    cat_score_map = {
        "spirit": 5,
        "wine": 5,
        "sake": 4,
        "cider": 3,
        "rtd": 3,
        "beer": 2,
    }
    cat_score = cat_score_map.get(category, 3)

    # skor asal negara (impor vs lokal)
    # Indonesia = 3 (netral), selain Indonesia = 4 (impor)
    origin_score = 3 if origin == "indonesia" else 4

    score = round((cat_score + origin_score) / 2)
    return int(max(1, min(5, score)))


def score_komposisi(row) -> int:
    """
    Proxy KOMPOSISI (1–5) dari kompleksitas main_ingredients:
    - hitung jumlah bahan (dipisah koma)
    """
    ing = str(row.get("main_ingredients", "")).strip()
    if ing.lower() in ["", "nan"]:
        return 1
    parts = [p.strip() for p in ing.split(",") if p.strip()]
    n = len(parts)

    if n >= 4:
        return 5
    if n == 3:
        return 4
    if n == 2:
        return 3
    if n == 1:
        return 2
    return 1


def score_estetika(row) -> int:
    """
    Proxy ESTETIKA BOTOL (1–5) dari packaging.
    """
    pack = str(row.get("packaging", "")).strip().lower()
    if pack == "bottle":
        return 5
    if pack == "can":
        return 3
    if pack in ["glass", "jar"]:
        return 4
    return 4  # default netral


def score_ketersediaan(row) -> int:
    """
    Proxy KETERSEDIAAN (1–5) berdasarkan:
    - kategori yang umumnya lebih mudah ditemukan (Beer/RTD lebih mudah)
    - bonus jika origin Indonesia (lebih mudah tersedia)
    """
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    cat_avail_map = {
        "beer": 5,
        "rtd": 5,
        "cider": 4,
        "spirit": 3,
        "wine": 2,
        "sake": 2,
    }
    base = cat_avail_map.get(category, 3)

    if origin == "indonesia":
        base = min(5, base + 1)

    return int(max(1, min(5, base)))


def build_spk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bentuk matriks keputusan 5 kriteria sesuai aturan awal:
    - Harga: pakai price_idr (akan diperlakukan COST)
    - Brand: skor 1–5 (proxy)
    - Komposisi: skor 1–5 (proxy)
    - Estetika Botol: skor 1–5 (proxy)
    - Ketersediaan: skor 1–5 (proxy)
    """
    out = pd.DataFrame()
    out["Alternatif"] = df["product_name"].astype(str)

    # Harga: gunakan price_idr asli (COST)
    out["Harga"] = pd.to_numeric(df["price_idr"], errors="coerce")

    # Skor 1–5 dari kolom teks
    out["Brand"] = df.apply(score_brand, axis=1)
    out["Komposisi"] = df.apply(score_komposisi, axis=1)
    out["Estetika Botol"] = df.apply(score_estetika, axis=1)
    out["Ketersediaan"] = df.apply(score_ketersediaan, axis=1)

    out = out.dropna(subset=["Harga"]).reset_index(drop=True)
    return out


# ================== LOAD DATASET ==================
dataset_path = Path(__file__).parent / DATASET_FILENAME
if not dataset_path.exists():
    st.error(f"File dataset tidak ditemukan: {DATASET_FILENAME}. Pastikan file ada 1 folder dengan app.py.")
    st.stop()

df = read_dataset(dataset_path)

st.subheader("Preview Dataset (Asli)")
st.dataframe(df.head(10), use_container_width=True)

# ================== BENTUK MATRIKS 5 KRITERIA ==================
spk_df = build_spk_matrix(df)

st.subheader("Matriks Keputusan (5 Kriteria Sesuai Aturan Awal)")
st.dataframe(spk_df.head(20), use_container_width=True)

with st.expander("Lihat aturan pembentukan skor (Brand/Komposisi/Estetika/Ketersediaan)", expanded=False):
    st.markdown(
        """
**Harga (Cost)**: menggunakan `price_idr` langsung.  
**Brand (Benefit)**: rata-rata skor dari *category* (Spirit/Wine lebih tinggi) dan *origin_country* (impor sedikit lebih tinggi dari lokal).  
**Komposisi (Benefit)**: jumlah bahan pada `main_ingredients` (semakin banyak bahan, skor semakin tinggi).  
**Estetika Botol (Benefit)**: `packaging` Bottle=5, Can=3, lainnya=4.  
**Ketersediaan (Benefit)**: Beer/RTD lebih mudah (skor tinggi), Wine/Sake lebih rendah; bonus jika origin Indonesia.
"""
    )

# ================== PILIH ALTERNATIF YANG DIBANDINGKAN ==================
st.markdown("## Pilih Alternatif yang Dibandingkan")

all_alts = spk_df["Alternatif"].tolist()
chosen = st.multiselect(
    "Pilih alternatif:",
    options=all_alts,
    default=all_alts[:10]
)

if len(chosen) < 2:
    st.warning("Pilih minimal 2 alternatif.")
    st.stop()

data_df = spk_df[spk_df["Alternatif"].isin(chosen)].reset_index(drop=True)

st.subheader("Data yang Digunakan untuk TOPSIS")
st.dataframe(data_df, use_container_width=True)

# ================== BOBOT & TIPE KRITERIA (SESUIAI ATURAN AWAL) ==================
st.sidebar.header("Bobot & Tipe Kriteria")

# default bobot sesuai pola awal (5,4,3,2,1)
weights_raw = {
    "Harga": st.sidebar.slider("Bobot Harga", 1, 5, 5),
    "Brand": st.sidebar.slider("Bobot Brand", 1, 5, 4),
    "Komposisi": st.sidebar.slider("Bobot Komposisi", 1, 5, 3),
    "Estetika Botol": st.sidebar.slider("Bobot Estetika Botol", 1, 5, 2),
    "Ketersediaan": st.sidebar.slider("Bobot Ketersediaan", 1, 5, 1),
}

# tipe sesuai aturan: Harga COST, lainnya BENEFIT
types = {
    "Harga": "Cost",
    "Brand": "Benefit",
    "Komposisi": "Benefit",
    "Estetika Botol": "Benefit",
    "Ketersediaan": "Benefit",
}

weights = normalize_weights(weights_raw)

st.subheader("Bobot (Normalisasi)")
w_df = pd.DataFrame({
    "Kriteria": list(weights.keys()),
    "Bobot (raw)": [weights_raw[k] for k in weights.keys()],
    "Bobot (normalisasi)": [weights[k] for k in weights.keys()],
})
st.dataframe(w_df, use_container_width=True)

# ================== HITUNG TOPSIS ==================
criteria_cols = ["Harga", "Brand", "Komposisi", "Estetika Botol", "Ketersediaan"]

X = data_df[criteria_cols].to_numpy(float)
w = np.array([weights[c] for c in criteria_cols], dtype=float)
is_benefit = np.array([types[c] == "Benefit" for c in criteria_cols], dtype=bool)

result = topsis_rank(X, w, is_benefit)

out = pd.DataFrame({
    "Alternatif": data_df["Alternatif"].astype(str),
    "D_plus": result["D_plus"],
    "D_minus": result["D_minus"],
    "V": result["V"],
})
out["Ranking"] = out["V"].rank(ascending=False, method="min").astype(int)
out = out.sort_values(["Ranking", "Alternatif"]).reset_index(drop=True)

st.subheader("Hasil TOPSIS")
st.dataframe(out, use_container_width=True)

best = out.iloc[0]
st.success(f"Rekomendasi terbaik: **{best['Alternatif']}** (V = {best['V']:.4f})")
