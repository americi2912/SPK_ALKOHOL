import re
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from topsis import topsis_rank, normalize_weights

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SPK Alkohol - TOPSIS", layout="wide")

# CSS kecil biar aksen biru + kartu rapi
st.markdown(
    """
<style>
/* Aksen biru */
div[data-testid="stSidebar"] { background: #f6f9ff; }
h1, h2, h3 { color: #0b3d91; }
.badge {
  display:inline-block; padding: 0.15rem 0.5rem; border-radius: 0.75rem;
  background:#e8f0ff; color:#0b3d91; font-size:0.85rem; border:1px solid #cfe0ff;
}
.card {
  border: 1px solid #d7e6ff;
  border-left: 6px solid #2b6fff;
  background: #fbfdff;
  padding: 1rem;
  border-radius: 0.75rem;
  margin-bottom: 0.9rem;
}
.smallmuted { color:#5a6b85; font-size:0.9rem; }
hr { border: none; border-top: 1px solid #e5eefc; margin: 0.8rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Sistem Pendukung Keputusan Alkohol (TOPSIS)")
st.caption(
    "Rekomendasi otomatis dari dataset bawaan (Dataset Alkohol.xlsx) menggunakan 5 kriteria: "
    "Harga, Brand, Komposisi, Estetika Botol, Ketersediaan."
)

DATASET_FILENAME = "Dataset Alkohol.xlsx"
TOP_N = 10

CRITERIA_COLS = ["Harga", "Brand", "Komposisi", "Estetika Botol", "Ketersediaan"]
CRITERIA_TYPES = {
    "Harga": "Cost",
    "Brand": "Benefit",
    "Komposisi": "Benefit",
    "Estetika Botol": "Benefit",
    "Ketersediaan": "Benefit",
}


# =========================
# HELPERS
# =========================
def read_dataset(file_path: Path) -> pd.DataFrame:
    """
    Membaca Excel yang header-nya bisa jadi tidak di baris pertama.
    Deteksi baris yang mengandung 'product_id' sebagai header.
    """
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

    # rapikan kolom
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan" and not str(c).startswith("Unnamed")]]
    df = df.dropna(axis=1, how="all").reset_index(drop=True)

    required = ["product_name", "category", "main_ingredients", "price_idr", "origin_country", "packaging"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dataset tidak sesuai format. Kolom hilang: {missing}")
        st.stop()

    return df


def parse_price_to_number(x) -> float:
    """
    price_idr di dataset bisa berbentuk:
    - 450000
    - "Rp 450,000"
    - "Rp" terpisah di kolom lain (kadang)
    Kita ambil angka saja.
    """
    if pd.isna(x):
        return np.nan
    s = str(x)
    # ambil semua digit
    digits = re.sub(r"[^\d]", "", s)
    if digits == "":
        return np.nan
    return float(digits)


def score_brand(row) -> int:
    """
    Proxy BRAND (1–5) dari kombinasi:
    - category: Spirit/Wine cenderung brand image lebih tinggi daripada Beer/RTD
    - origin_country: non-Indonesia sedikit lebih tinggi
    """
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    cat_score_map = {
        "spirit": 5,
        "wine": 5,
        "liquor": 4,
        "liqueur": 4,
        "liquer": 4,   # jaga typo
        "sake": 4,
        "cider": 3,
        "rtd": 3,
        "beer": 2,
        "traditional": 3,
        "mead": 3,
    }
    cat_score = cat_score_map.get(category, 3)
    origin_score = 3 if origin == "indonesia" else 4
    score = round((cat_score + origin_score) / 2)
    return int(max(1, min(5, score)))


def score_komposisi(row) -> int:
    """
    Proxy KOMPOSISI (1–5) dari kompleksitas main_ingredients:
    semakin banyak bahan (dipisah koma) semakin tinggi.
    """
    ing = str(row.get("main_ingredients", "")).strip()
    if ing.lower() in ["", "nan", "none"]:
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
    return 4


def score_ketersediaan(row) -> int:
    """
    Proxy KETERSEDIAAN (1–5) berdasarkan:
    - kategori yang umum lebih mudah ditemukan (Beer/RTD lebih mudah)
    - bonus jika origin Indonesia
    """
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    cat_avail_map = {
        "beer": 5,
        "rtd": 5,
        "cider": 4,
        "liquor": 3,
        "liqueur": 3,
        "liquer": 3,
        "spirit": 3,
        "wine": 2,
        "sake": 2,
        "traditional": 4,
        "mead": 3,
    }
    base = cat_avail_map.get(category, 3)
    if origin == "indonesia":
        base = min(5, base + 1)
    return int(max(1, min(5, base)))


def build_spk_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Alternatif"] = (
        df["product_name"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out["Harga"] = df["price_idr"].apply(parse_price_to_number)
    out["Brand"] = df.apply(score_brand, axis=1)
    out["Komposisi"] = df.apply(score_komposisi, axis=1)
    out["Estetika Botol"] = df.apply(score_estetika, axis=1)
    out["Ketersediaan"] = df.apply(score_ketersediaan, axis=1)

    out = out.dropna(subset=["Harga"]).reset_index(drop=True)
    return out


def best_per_criterion(data_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CRITERIA_COLS:
        if CRITERIA_TYPES[c] == "Cost":
            idx = data_df[c].astype(float).idxmin()
        else:
            idx = data_df[c].astype(float).idxmax()

        r = data_df.loc[idx]
        rows.append(
            {
                "Kriteria": c,
                "Tipe": CRITERIA_TYPES[c],
                "Alternatif Terbaik": r["Alternatif"],
                "Nilai": float(r[c]),
            }
        )
    return pd.DataFrame(rows)


# =========================
# LOAD DATASET (AUTO)
# =========================
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
        "Dataset Excel tidak ditemukan.\n\n"
        "Pastikan file Excel ada di root repo (sefolder app.py), contoh: Dataset Alkohol.xlsx"
    )
    st.stop()

st.sidebar.caption(f"Dataset dipakai: {dataset_path.name}")

df_raw = read_dataset(dataset_path)
spk_df = build_spk_matrix(df_raw)

# Dataset tampil hanya jika expander dibuka
with st.expander("📂 View Full Dataset (All Products)", expanded=False):
    st.dataframe(df_raw, use_container_width=True)

with st.expander("📊 View Decision Matrix (5 Criteria)", expanded=False):
    st.dataframe(spk_df, use_container_width=True)

with st.expander("ℹ️ Aturan pembentukan skor (Brand/Komposisi/Estetika/Ketersediaan)", expanded=False):
    st.markdown(
        """
- **Harga (Cost)**: menggunakan `price_idr` (dibersihkan jadi angka).
- **Brand (Benefit)**: skor dari *category* + *origin_country* (impor sedikit lebih tinggi).
- **Komposisi (Benefit)**: jumlah bahan pada `main_ingredients` (semakin banyak, semakin tinggi).
- **Estetika Botol (Benefit)**: Bottle=5, Can=3, lainnya=4.
- **Ketersediaan (Benefit)**: Beer/RTD lebih tinggi; bonus jika origin Indonesia.
"""
    )

st.markdown("---")

# =========================
# SIDEBAR: WEIGHTS + BUTTON
# =========================
st.sidebar.header("Bobot & Tipe Kriteria")

weights_raw = {
    "Harga": st.sidebar.slider("Bobot Harga", 1, 5, 5),
    "Brand": st.sidebar.slider("Bobot Brand", 1, 5, 4),
    "Komposisi": st.sidebar.slider("Bobot Komposisi", 1, 5, 3),
    "Estetika Botol": st.sidebar.slider("Bobot Estetika Botol", 1, 5, 2),
    "Ketersediaan": st.sidebar.slider("Bobot Ketersediaan", 1, 5, 1),
}

weights = normalize_weights(weights_raw)

st.sidebar.markdown("---")
calc = st.sidebar.button("Calculate Recommendation", type="primary")

# tampil bobot normalisasi (boleh sebelum hitung)
st.subheader("Bobot (Normalisasi)")
w_df = pd.DataFrame(
    {
        "Kriteria": list(weights.keys()),
        "Bobot (raw)": [weights_raw[k] for k in weights.keys()],
        "Bobot (normalisasi)": [weights[k] for k in weights.keys()],
        "Tipe": [CRITERIA_TYPES[k] for k in weights.keys()],
    }
)
st.dataframe(w_df, use_container_width=True)

# =========================
# RUN TOPSIS ONLY WHEN BUTTON CLICKED
# =========================
if not calc:
    st.info("Atur bobot di sidebar, lalu klik **Calculate Recommendation** untuk menampilkan hasil.")
    st.stop()

# TOPSIS untuk SEMUA produk → ambil Top 10
data_df = spk_df.copy()

X = data_df[CRITERIA_COLS].to_numpy(float)
w = np.array([weights[c] for c in CRITERIA_COLS], dtype=float)
is_benefit = np.array([CRITERIA_TYPES[c] == "Benefit" for c in CRITERIA_COLS], dtype=bool)

result = topsis_rank(X, w, is_benefit)

rank_df = pd.DataFrame(
    {
        "Alternatif": data_df["Alternatif"].astype(str),
        "Harga": data_df["Harga"].astype(float),
        "Brand": data_df["Brand"].astype(float),
        "Komposisi": data_df["Komposisi"].astype(float),
        "Estetika Botol": data_df["Estetika Botol"].astype(float),
        "Ketersediaan": data_df["Ketersediaan"].astype(float),
        "D_plus": result["D_plus"],
        "D_minus": result["D_minus"],
        "V": result["V"],
    }
)
rank_df["Ranking"] = rank_df["V"].rank(ascending=False, method="min").astype(int)
rank_df = rank_df.sort_values(["Ranking", "Alternatif"]).reset_index(drop=True)

top10 = rank_df.head(TOP_N).copy()
best = top10.iloc[0]

st.success(f"Rekomendasi terbaik (TOPSIS): **{best['Alternatif']}** (V = {best['V']:.4f})")

st.markdown(f"## 🏆 Top {TOP_N} Recommendations")

# tampil Top 10 ala “card”
for i, row in top10.iterrows():
    rank_no = int(row["Ranking"])
    alt = row["Alternatif"]
    v = float(row["V"])

    st.markdown(
        f"""
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap: 1rem;">
    <div>
      <div class="badge">Rank #{rank_no}</div>
      <h3 style="margin:0.4rem 0 0.2rem 0;">{alt}</h3>
      <div class="smallmuted">TOPSIS Score (V): <b>{v:.4f}</b></div>
    </div>
  </div>
  <hr/>
</div>
""",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    cols[0].metric("Harga (IDR)", f"{int(row['Harga']):,}".replace(",", "."))
    cols[1].metric("Brand", f"{row['Brand']:.0f}/5")
    cols[2].metric("Komposisi", f"{row['Komposisi']:.0f}/5")
    cols[3].metric("Estetika", f"{row['Estetika Botol']:.0f}/5")
    cols[4].metric("Ketersediaan", f"{row['Ketersediaan']:.0f}/5")

    with st.expander("Show Calculation Details", expanded=False):
        st.write("Nilai TOPSIS untuk alternatif ini:")
        det = row[["D_plus", "D_minus", "V"]].to_frame("Value")
        st.dataframe(det, use_container_width=True)

st.markdown("---")

st.markdown("## Rekomendasi Terbaik per Kriteria (1 pemenang tiap kriteria)")
bestcrit = best_per_criterion(rank_df[["Alternatif"] + CRITERIA_COLS].copy())
st.dataframe(bestcrit, use_container_width=True)

with st.expander("📌 View Full Ranking (All Products)", expanded=False):
    st.dataframe(rank_df, use_container_width=True)
