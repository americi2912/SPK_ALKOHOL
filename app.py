import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from topsis import topsis_rank, normalize_weights

st.set_page_config(page_title="SPK Alkohol - TOPSIS", layout="wide")
st.title("Sistem Pendukung Keputusan Alkohol (TOPSIS)")
st.caption(
    "Rekomendasi otomatis dari dataset bawaan menggunakan 5 kriteria: "
    "Harga, Brand, Komposisi, Estetika Botol, Ketersediaan."
)

DATASET_FILENAME = "Dataset Alkohol.xlsx"

# ================== ALTERNATIF PATEN ==================
FIXED_ALTERNATIVES = ["Wine Merah", "Vodka", "Baileys", "Tequila", "Aperol"]


# ------------------ BACA EXCEL (HEADER DI BARIS product_id) ------------------
def read_dataset(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None, engine="openpyxl")

    header_row = None
    for i in range(min(80, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains("product_id", na=False).any():
            header_row = i
            break

    if header_row is None:
        df = pd.read_excel(file_path, engine="openpyxl")
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
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    cat_score_map = {"spirit": 5, "wine": 5, "sake": 4, "cider": 3, "rtd": 3, "beer": 2}
    cat_score = cat_score_map.get(category, 3)

    origin_score = 3 if origin == "indonesia" else 4
    score = round((cat_score + origin_score) / 2)
    return int(max(1, min(5, score)))


def score_komposisi(row) -> int:
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
    pack = str(row.get("packaging", "")).strip().lower()
    if pack == "bottle":
        return 5
    if pack == "can":
        return 3
    if pack in ["glass", "jar"]:
        return 4
    return 4


def score_ketersediaan(row) -> int:
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    cat_avail_map = {"beer": 5, "rtd": 5, "cider": 4, "spirit": 3, "wine": 2, "sake": 2}
    base = cat_avail_map.get(category, 3)

    if origin == "indonesia":
        base = min(5, base + 1)

    return int(max(1, min(5, base)))


def build_spk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Alternatif"] = (
    df["product_name"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)  # rapikan spasi ganda
    .str.strip()                           # buang spasi depan/belakang
    )

    out["Harga"] = pd.to_numeric(df["price_idr"], errors="coerce")
    out["Brand"] = df.apply(score_brand, axis=1)
    out["Komposisi"] = df.apply(score_komposisi, axis=1)
    out["Estetika Botol"] = df.apply(score_estetika, axis=1)
    out["Ketersediaan"] = df.apply(score_ketersediaan, axis=1)

    out = out.dropna(subset=["Harga"]).reset_index(drop=True)
    return out


# ================== LOAD DATASET (DEPLOY SAFE) ==================
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()

dataset_path = base_dir / DATASET_FILENAME

if not dataset_path.exists():
    xlsx_files = list(base_dir.glob("*.xlsx"))
    if xlsx_files:
        dataset_path = xlsx_files[0]

if not dataset_path.exists():
    st.error(
        "Dataset Excel tidak ditemukan di repo.\n\n"
        "Pastikan file Excel ada di root repo (sefolder app.py), contoh: Dataset Alkohol.xlsx"
    )
    st.stop()

st.sidebar.caption(f"Dataset dipakai: {dataset_path.name}")

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
**Harga (Cost)**: menggunakan `price_idr`.  
**Brand (Benefit)**: skor dari *category* dan *origin_country*.  
**Komposisi (Benefit)**: jumlah bahan pada `main_ingredients`.  
**Estetika Botol (Benefit)**: `packaging` Bottle=5, Can=3, lainnya=4.  
**Ketersediaan (Benefit)**: Beer/RTD lebih tinggi; bonus jika origin Indonesia.
"""
    )

# ================== ALTERNATIF PATEN (TAMPIL LIST) ==================
st.markdown("## Alternatif yang Dibandingkan (Tetap)")

st.markdown(
    """
- Wine Merah  
- Vodka  
- Baileys  
- Tequila  
- Aperol  
"""
)

# normalisasi untuk pencocokan (lower + strip + rapikan spasi)
spk_df["_alt_norm"] = (
    spk_df["Alternatif"]
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
)

fixed_norm = [a.strip().lower() for a in FIXED_ALTERNATIVES]

data_df = spk_df[spk_df["_alt_norm"].isin(fixed_norm)].copy()

# pastikan urutannya sesuai FIXED_ALTERNATIVES
order_map = {name.strip().lower(): i for i, name in enumerate(FIXED_ALTERNATIVES)}
data_df["_order"] = data_df["_alt_norm"].map(order_map)
data_df = data_df.sort_values("_order").drop(columns=["_alt_norm", "_order"]).reset_index(drop=True)

if len(data_df) != len(FIXED_ALTERNATIVES):
    found_norm = set(
        spk_df["Alternatif"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
        .tolist()
    )
    missing = [a for a in FIXED_ALTERNATIVES if a.strip().lower() not in found_norm]
    st.error(f"Alternatif berikut tidak ditemukan di dataset: {missing}")
    st.stop()


st.subheader("Data yang Digunakan untuk TOPSIS")
st.dataframe(data_df, use_container_width=True)

# ================== BOBOT & TIPE ==================
st.sidebar.header("Bobot & Tipe Kriteria")

weights_raw = {
    "Harga": st.sidebar.slider("Bobot Harga", 1, 5, 5),
    "Brand": st.sidebar.slider("Bobot Brand", 1, 5, 4),
    "Komposisi": st.sidebar.slider("Bobot Komposisi", 1, 5, 3),
    "Estetika Botol": st.sidebar.slider("Bobot Estetika Botol", 1, 5, 2),
    "Ketersediaan": st.sidebar.slider("Bobot Ketersediaan", 1, 5, 1),
}

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
