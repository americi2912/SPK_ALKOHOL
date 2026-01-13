import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from topsis import topsis_rank, normalize_weights

# ================== CONFIG ==================
st.set_page_config(page_title="SPK Alkohol - TOPSIS", layout="wide")

# ================== BLUE UI (NOT PINK) ==================
st.markdown(
    """
<style>
.card {
  border: 2px solid #4da3ff33;
  background: #4da3ff12;
  border-radius: 14px;
  padding: 16px 16px 10px 16px;
  margin-bottom: 14px;
}
.card-title {
  font-size: 18px;
  font-weight: 800;
  margin-bottom: 6px;
  color: #0b3c78;
}
.card-sub {
  font-size: 13px;
  opacity: 0.85;
  margin-bottom: 10px;
  color: #1f4f8a;
}
.badge-row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 6px;
  margin-bottom: 4px;
}
.badge {
  background: #ffffff;
  border: 1px solid #00000012;
  border-radius: 10px;
  padding: 8px 10px;
  min-width: 180px;
}
.badge b { font-weight: 800; color: #0b3c78; }
.small { font-size: 12px; opacity: 0.85; color: #1f4f8a; }
hr.soft { border: none; border-top: 1px solid #00000012; margin: 12px 0; }

button[kind="primary"] {
  background-color: #1e88e5 !important;
  color: white !important;
  border-radius: 10px !important;
}
details summary {
  background-color: #e3f2fd;
  border-radius: 10px;
  padding: 8px 12px;
  font-weight: 600;
  color: #0b3c78;
}
</style>
""",
    unsafe_allow_html=True,
)

# ================== HEADER ==================
st.title("Sistem Pendukung Keputusan Alkohol (TOPSIS)")
st.caption(
    "User mengisi bobot kriteria di sidebar, lalu klik tombol **Hitung Rekomendasi** untuk menampilkan hasil."
)

DATASET_FILENAME = "Dataset Alkohol.xlsx"

# Alternatif paten (HARUS ini)
FIXED_ALTERNATIVES = ["Wine Merah", "Vodka", "Baileys", "Tequila", "Aperol"]

CRITERIA = ["Harga", "Brand", "Komposisi", "Estetika Botol", "Ketersediaan"]
TYPES = {
    "Harga": "Cost",
    "Brand": "Benefit",
    "Komposisi": "Benefit",
    "Estetika Botol": "Benefit",
    "Ketersediaan": "Benefit",
}


# ================== DATASET LOADER ==================
def find_dataset_path(filename: str) -> Path:
    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()

    p = base_dir / filename
    if p.exists():
        return p

    # fallback: ambil xlsx pertama di repo
    xlsx_files = list(base_dir.glob("*.xlsx"))
    if xlsx_files:
        return xlsx_files[0]
    return p


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

    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan" and not str(c).startswith("Unnamed")]]
    df = df.dropna(axis=1, how="all")

    required = ["product_name", "category", "main_ingredients", "price_idr", "origin_country", "packaging"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Dataset tidak sesuai format. Kolom hilang: {missing}")
        st.stop()

    return df


# ================== SCORING (TEXT -> 1..5) ==================
def score_brand(row) -> int:
    origin = str(row.get("origin_country", "")).strip().lower()
    category = str(row.get("category", "")).strip().lower()

    # Sesuaikan dengan dataset kamu (Liquor, Wine, Spirit, Beer, RTD, Cider, Sake, dll)
    cat_score_map = {
        "spirit": 5,
        "wine": 5,
        "liquer": 5,   # jaga-jaga typo
        "liquor": 5,
        "sake": 4,
        "cider": 3,
        "rtd": 3,
        "beer": 2,
    }
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

    cat_avail_map = {
        "beer": 5,
        "rtd": 5,
        "cider": 4,
        "spirit": 3,
        "liquor": 3,
        "wine": 2,
        "sake": 2,
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

    # NOTE: price_idr di dataset kamu ada "Rp 450,000" → harus dibersihkan
    price_clean = (
        df["price_idr"]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    out["Harga"] = pd.to_numeric(price_clean, errors="coerce")

    out["Brand"] = df.apply(score_brand, axis=1)
    out["Komposisi"] = df.apply(score_komposisi, axis=1)
    out["Estetika Botol"] = df.apply(score_estetika, axis=1)
    out["Ketersediaan"] = df.apply(score_ketersediaan, axis=1)

    out = out.dropna(subset=["Harga"]).reset_index(drop=True)
    return out


# ================== LOAD DATASET ==================
dataset_path = find_dataset_path(DATASET_FILENAME)
if not dataset_path.exists():
    st.error(
        "Dataset Excel tidak ditemukan.\n\n"
        "Pastikan file Excel ada di root repo (sefolder app.py), contoh: Dataset Alkohol.xlsx"
    )
    st.stop()

st.sidebar.caption(f"Dataset dipakai: {dataset_path.name}")
df = read_dataset(dataset_path)

# dataset disembunyikan (expander)
with st.expander("📂 View Full Dataset (All Products)", expanded=False):
    st.dataframe(df, use_container_width=True)

# ================== BUILD MATRIX ==================
spk_df = build_spk_matrix(df)

with st.expander("ℹ️ Aturan pembentukan skor (Brand/Komposisi/Estetika/Ketersediaan)", expanded=False):
    st.markdown(
        """
**Harga (Cost)**: dari kolom `price_idr` (dibersihkan dari 'Rp' & koma).  
**Brand (Benefit)**: proxy dari `category` dan `origin_country`.  
**Komposisi (Benefit)**: jumlah bahan pada `main_ingredients`.  
**Estetika Botol (Benefit)**: `packaging` Bottle=5, Can=3, lainnya=4.  
**Ketersediaan (Benefit)**: Beer/RTD lebih tinggi; bonus jika origin Indonesia.
"""
    )

# ================== FIXED ALTERNATIVES ==================
st.subheader("Alternatif yang Dibandingkan (Tetap)")
st.markdown(
    """
- Wine Merah  
- Vodka  
- Baileys  
- Tequila  
- Aperol  
"""
)

# Normalisasi nama untuk matching
spk_df["_alt_norm"] = (
    spk_df["Alternatif"].astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
    .str.lower()
)

fixed_norm = [a.strip().lower() for a in FIXED_ALTERNATIVES]
data_df = spk_df[spk_df["_alt_norm"].isin(fixed_norm)].copy()

# Urutkan sesuai list paten
order_map = {name.strip().lower(): i for i, name in enumerate(FIXED_ALTERNATIVES)}
data_df["_order"] = data_df["_alt_norm"].map(order_map)
data_df = data_df.sort_values("_order").drop(columns=["_alt_norm", "_order"]).reset_index(drop=True)

# Validasi missing
if len(data_df) != len(FIXED_ALTERNATIVES):
    found_norm = set(spk_df["_alt_norm"].tolist())
    missing = [a for a in FIXED_ALTERNATIVES if a.strip().lower() not in found_norm]
    st.error(f"Alternatif berikut tidak ditemukan di dataset: {missing}")
    st.stop()

with st.expander("📌 Data yang Dipakai untuk TOPSIS (5 Alternatif)", expanded=False):
    st.dataframe(data_df, use_container_width=True)

# ================== SIDEBAR WEIGHTS ==================
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
top_n = st.sidebar.slider("Jumlah rekomendasi (Top N)", 1, len(FIXED_ALTERNATIVES), 5)

# Tombol calculate (hasil tidak langsung muncul)
calculate = st.sidebar.button("Hitung Rekomendasi", type="primary")

# ================== SHOW WEIGHTS TABLE ==================
with st.expander("⚙️ Bobot (Normalisasi)", expanded=False):
    w_df = pd.DataFrame({
        "Kriteria": list(weights.keys()),
        "Bobot (raw)": [weights_raw[k] for k in weights.keys()],
        "Bobot (normalisasi)": [weights[k] for k in weights.keys()],
        "Tipe": [TYPES[k] for k in weights.keys()],
    })
    st.dataframe(w_df, use_container_width=True)

# ================== CALC TOPSIS ONLY IF BUTTON CLICKED ==================
if not calculate:
    st.info("Atur bobot di sidebar, lalu klik **Hitung Rekomendasi** untuk menampilkan hasil.")
    st.stop()

# ================== TOPSIS COMPUTATION ==================
criteria_cols = CRITERIA
X = data_df[criteria_cols].to_numpy(float)
w = np.array([weights[c] for c in criteria_cols], dtype=float)
is_benefit = np.array([TYPES[c] == "Benefit" for c in criteria_cols], dtype=bool)

result = topsis_rank(X, w, is_benefit)

out = pd.DataFrame({
    "Alternatif": data_df["Alternatif"].astype(str),
    "D_plus": result["D_plus"],
    "D_minus": result["D_minus"],
    "V": result["V"],
})
out["Ranking"] = out["V"].rank(ascending=False, method="min").astype(int)
out = out.sort_values(["Ranking", "Alternatif"]).reset_index(drop=True)

# ================== RESULT SECTION ==================
st.markdown("## 🏆 Hasil Rekomendasi (TOPSIS)")

best = out.iloc[0]
st.success(f"Rekomendasi terbaik (TOPSIS): **{best['Alternatif']}** (V = {best['V']:.4f})")

# tampil top-n (card)
top_out = out.head(top_n).copy()

for i, row in top_out.iterrows():
    alt = row["Alternatif"]
    v = float(row["V"])
    rank = int(row["Ranking"])

    # ambil nilai kriterianya
    src = data_df[data_df["Alternatif"] == alt].iloc[0]

    st.markdown(
        f"""
<div class="card">
  <div class="card-title">#{rank} {alt}</div>
  <div class="card-sub">TOPSIS Score (V): <b>{v:.4f}</b></div>

  <div class="badge-row">
    <div class="badge"><b>Harga</b><div class="small">{src['Harga']:.0f}</div></div>
    <div class="badge"><b>Brand</b><div class="small">{int(src['Brand'])} / 5</div></div>
    <div class="badge"><b>Komposisi</b><div class="small">{int(src['Komposisi'])} / 5</div></div>
    <div class="badge"><b>Estetika</b><div class="small">{int(src['Estetika Botol'])} / 5</div></div>
    <div class="badge"><b>Ketersediaan</b><div class="small">{int(src['Ketersediaan'])} / 5</div></div>
  </div>

  <hr class="soft" />
</div>
""",
        unsafe_allow_html=True,
    )

# ================== FULL TABLE OUTPUT (EXPANDER) ==================
with st.expander("📊 Tabel Hasil TOPSIS (D+, D-, V, Ranking)", expanded=False):
    st.dataframe(out, use_container_width=True)

# ================== BEST PER CRITERIA ==================
st.markdown("## ✅ Rekomendasi Terbaik per Kriteria")

rows = []
for c in criteria_cols:
    if TYPES[c] == "Cost":
        idx = data_df[c].astype(float).idxmin()
    else:
        idx = data_df[c].astype(float).idxmax()
    best_row = data_df.loc[idx]
    rows.append({
        "Kriteria": c,
        "Tipe": TYPES[c],
        "Alternatif Terbaik": best_row["Alternatif"],
        "Nilai": float(best_row[c]),
    })

best_per_criterion = pd.DataFrame(rows)
st.dataframe(best_per_criterion, use_container_width=True)

for _, r in best_per_criterion.iterrows():
    st.info(f"**{r['Kriteria']} ({r['Tipe']})** → **{r['Alternatif Terbaik']}** (Nilai = {r['Nilai']})")
