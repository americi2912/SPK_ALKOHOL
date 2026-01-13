import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from topsis import topsis_rank, normalize_weights

st.set_page_config(page_title="SPK Alkohol - TOPSIS", layout="wide")
st.title("Sistem Pendukung Keputusan Alkohol (TOPSIS)")
st.caption("Rekomendasi dihitung otomatis dari dataset bawaan (Dataset Alkohol.xlsx).")

# ================== CONFIG DATASET ==================
DATASET_FILENAME = "Dataset Alkohol.xlsx"  # nama file dataset di folder project


# ================== HELPER: BACA EXCEL AUTO HEADER ==================
def read_excel_auto_header(file_path: Path, sheet_name=0) -> pd.DataFrame:
    """
    Membaca Excel yang header-nya tidak di baris pertama.
    Cari baris yang mengandung 'product_id' lalu jadikan header.
    Jika tidak ketemu, fallback ke header normal.
    """
    raw = pd.read_excel(file_path, header=None, sheet_name=sheet_name)

    header_row = None
    for i in range(min(50, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains("product_id", na=False).any():
            header_row = i
            break

    if header_row is None:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = raw.iloc[header_row].tolist()

    df = df.dropna(how="all")
    df = df.loc[:, ~pd.Series(df.columns).astype(str).str.contains("^Unnamed", na=False)]
    df = df.reset_index(drop=True)
    return df


# ================== LOAD DATASET ==================
dataset_path = Path(__file__).parent / DATASET_FILENAME

if not dataset_path.exists():
    st.error(
        f"File dataset tidak ditemukan: {DATASET_FILENAME}\n\n"
        "Pastikan file tersebut berada 1 folder dengan app.py (satu repo saat deploy)."
    )
    st.stop()

df = read_excel_auto_header(dataset_path)

if df.empty:
    st.error("Dataset terbaca kosong.")
    st.stop()

st.subheader("Preview Dataset (Bawaan)")
st.dataframe(df.head(10), use_container_width=True)

# ================== PILIH ALTERNATIF (NAMA PRODUK) ==================
st.markdown("## Pengaturan Alternatif & Kriteria")

name_col = st.selectbox(
    "Pilih kolom nama produk (Alternatif):",
    options=df.columns.tolist(),
    index=df.columns.tolist().index("product_name") if "product_name" in df.columns else 0,
)

# ================== PILIH KRITERIA (KOLUM NUMERIK) ==================
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if str(c).lower() not in ["product_id"]]

if len(numeric_cols) < 2:
    st.error("Dataset harus punya minimal 2 kolom numerik untuk TOPSIS.")
    st.stop()

criteria_cols = st.multiselect(
    "Pilih kolom kriteria (Numerik) untuk TOPSIS:",
    options=numeric_cols,
    default=[c for c in ["price_idr", "alcohol_content_percent", "volume_ml", "sugar_content_g"] if c in numeric_cols]
    or numeric_cols[:3],
)

if len(criteria_cols) < 2:
    st.warning("Pilih minimal 2 kriteria.")
    st.stop()

# ================== PILIH ALTERNATIF YANG DIBANDINGKAN ==================
opts = df[name_col].astype(str).tolist()

chosen = st.multiselect(
    "Pilih alternatif yang ingin dibandingkan:",
    options=opts,
    default=opts[:10]  # default ambil 10 pertama biar user langsung lihat hasil
)

if len(chosen) < 2:
    st.warning("Pilih minimal 2 alternatif.")
    st.stop()

data_df = df[df[name_col].astype(str).isin(chosen)][[name_col] + criteria_cols].copy()
data_df = data_df.rename(columns={name_col: "Alternatif"})

st.subheader("Matriks Keputusan (Data Asli dari Dataset)")
st.dataframe(data_df, use_container_width=True)

# ================== BOBOT & TIPE KRITERIA ==================
st.sidebar.header("Bobot & Tipe Kriteria")

weights_raw = {}
types = {}

for c in criteria_cols:
    weights_raw[c] = st.sidebar.slider(f"Bobot {c}", 1, 5, 3)

    # default otomatis (bisa kamu sesuaikan)
    default_type = "Benefit"
    if str(c).lower() in ["price_idr", "sugar_content_g", "harga", "price"]:
        default_type = "Cost"

    types[c] = st.sidebar.selectbox(
        f"Tipe {c}",
        ["Benefit", "Cost"],
        index=0 if default_type == "Benefit" else 1,
    )

weights = normalize_weights(weights_raw)

st.sidebar.markdown("---")
st.sidebar.caption("Catatan: Cost = semakin kecil semakin baik, Benefit = semakin besar semakin baik.")

st.subheader("Bobot (Normalisasi)")
w_df = pd.DataFrame({
    "Kriteria": list(weights.keys()),
    "Bobot (raw)": [weights_raw[k] for k in weights.keys()],
    "Bobot (normalisasi)": [weights[k] for k in weights.keys()],
})
st.dataframe(w_df, use_container_width=True)

# ================== HITUNG TOPSIS ==================
matrix_numeric = data_df.copy()
for c in criteria_cols:
    matrix_numeric[c] = pd.to_numeric(matrix_numeric[c], errors="raise")

X = matrix_numeric[criteria_cols].to_numpy(float)
w = np.array([weights[c] for c in criteria_cols], dtype=float)
is_benefit = np.array([types[c] == "Benefit" for c in criteria_cols], dtype=bool)

result = topsis_rank(X, w, is_benefit)

out = pd.DataFrame({
    "Alternatif": matrix_numeric["Alternatif"].astype(str),
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
