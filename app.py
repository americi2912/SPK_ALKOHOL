import streamlit as st
import pandas as pd
import numpy as np

# Konfigurasi Halaman
st.set_page_config(page_title="SPK Rekomendasi Alkohol", layout="wide")

st.title("🍷 Sistem Pendukung Keputusan Pemilihan Alkohol")
st.markdown("Pilih preferensi Anda di samping kiri untuk mendapatkan rekomendasi terbaik menggunakan metode **TOPSIS**.")

# --- 1. DATASET STATIS (Berdasarkan dokumen Anda) ---
# Matriks Keputusan (X) 
data = {
    'Kode': ['A1', 'A2', 'A3', 'A4', 'A5'],
    'Nama Produk': ['Wine Merah', 'Vodka', 'Baileys', 'Tequila', 'Aperol'],
    'C1 (Harga)': [4, 1, 3, 2, 5],
    'C2 (Brand)': [4, 5, 2, 4, 1],
    'C3 (Komposisi)': [2, 1, 5, 1, 1],
    'C4 (Estetika)': [3, 1, 1, 4, 5],
    'C5 (Ketersediaan)': [3, 4, 1, 5, 3]
}
df_dataset = pd.DataFrame(data)

# --- 2. SIDEBAR: INPUT BOBOT USER ---
st.sidebar.header("⚖️ Atur Prioritas Anda")
st.sidebar.write("Skala 1 (Tidak Penting) sampai 5 (Sangat Penting)")

w1 = st.sidebar.slider("Harga (C1)", 1, 5, 5) # Default 5 [cite: 18]
w2 = st.sidebar.slider("Brand (C2)", 1, 5, 4) # Default 4 [cite: 19]
w3 = st.sidebar.slider("Komposisi (C3)", 1, 5, 3) # Default 3 [cite: 20]
w4 = st.sidebar.slider("Estetika Botol (C4)", 1, 5, 2) # Default 2 [cite: 21]
w5 = st.sidebar.slider("Ketersediaan (C5)", 1, 5, 1) # Default 1 [cite: 22]

# Kalkulasi Bobot Relatif (W) [cite: 24, 25]
total_w = w1 + w2 + w3 + w4 + w5
weights = np.array([w1/total_w, w2/total_w, w3/total_w, w4/total_w, w5/total_w])

# --- 3. PERHITUNGAN TOPSIS ---

# Ambil hanya kolom kriteria untuk perhitungan
matrix_x = df_dataset.iloc[:, 2:].values 

# a. Normalisasi (R) [cite: 29]
divider = np.sqrt(np.sum(matrix_x**2, axis=0))
matrix_r = matrix_x / divider

# b. Matriks Terbobot (Y) [cite: 38, 39]
matrix_y = matrix_r * weights

# c. Solusi Ideal Positif (A+) dan Negatif (A-) [cite: 42]
a_plus = np.max(matrix_y, axis=0)
a_minus = np.min(matrix_y, axis=0)

# d. Jarak Solusi (D+ dan D-) [cite: 46, 47]
d_plus = np.sqrt(np.sum((matrix_y - a_plus)**2, axis=1))
d_minus = np.sqrt(np.sum((matrix_y - a_minus)**2, axis=1))

# e. Nilai Preferensi (V) [cite: 53]
v_score = d_minus / (d_minus + d_plus)

# --- 4. TAMPILAN HASIL ---
df_dataset['Skor Kedekatan (V)'] = v_score
hasil_ranking = df_dataset[['Kode', 'Nama Produk', 'Skor Kedekatan (V)']].sort_values(by='Skor Kedekatan (V)', ascending=False)

# Menentukan Rekomendasi Utama
rekomendasi_utama = hasil_ranking.iloc[0]['Nama Produk']

st.success(f"### 🏆 Rekomendasi Terbaik: **{rekomendasi_utama}**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tabel Ranking")
    # Menampilkan tabel dengan gaya highlight untuk ranking 1
    st.dataframe(hasil_ranking.style.highlight_max(subset=['Skor Kedekatan (V)'], color='#2E7D32'))

with col2:
    st.subheader("Visualisasi Skor")
    st.bar_chart(hasil_ranking.set_index('Nama Produk')['Skor Kedekatan (V)'])

with st.expander("Lihat Dataset Acuan (Tetap)"):
    st.write("Data ini digunakan sebagai dasar perhitungan dan tidak dapat diubah.")
    st.table(df_dataset.iloc[:, :7])