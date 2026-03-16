import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import re

# --- 1. KONFIGURASI ---
st.set_page_config(page_title="WebGIS UMKM | Smart Search", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_umkm.csv', sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['lat', 'lon', 'nama'])
        
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
        else:
            df['cluster'] = 0
        return df
    except Exception as e:
        st.error(f"Eror Dataset: {e}"); st.stop()

df = load_data()

# --- 3. UI PENCARIAN ---
st.markdown("### 📍 WebGIS UMKM Malawele")
query_user = st.text_input("🔍 Cari (Contoh: 'jl gambas' atau 'pencucian terong'):", 
                          placeholder="Masukkan nama UMKM atau nama jalan...")

# --- 4. LOGIKA FILTER CERDAS (AND LOGIC) ---
def smart_search(query, data):  # SUDAH DIPERBAIKI: ddef jadi def
    if not query:
        return data
    
    # 1. Daftar kata yang akan dibuang otomatis (Stopwords)
    stopwords = ['di', 'dan', 'yang', 'ada', 'ke', 'dari', 'pada', 'dalam']
    
    # 2. Pecah query dan buang kata-kata sampah
    raw_keywords = re.findall(r'\w+', query.lower())
    keywords = [k for k in raw_keywords if k not in stopwords]
    
    if not keywords:
        return data

    def filter_logic(row):
        full_text = f"{row['nama']} {row['alamat']} {row.get('kategori', '')}".lower()
        full_text = re.sub(r'[^\w\s]', '', full_text)
        
        # Cek apakah SEMUA kata kunci inti ada di dalam teks baris tersebut
        return all(k in full_text for k in keywords)

    return data[data.apply(filter_logic, axis=1)]

filtered_df = smart_search(query_user, df)

# --- 5. TAMPILAN ---
col_map, col_info = st.columns([2, 1])

with col_map:
    if not filtered_df.empty:
        start_loc = [filtered_df['lat'].mean(), filtered_df['lon'].mean()]
        zoom = 17 if len(filtered_df) < 5 else 16
    else:
        start_loc = [-0.9648, 131.3059]
        zoom = 15

    m = folium.Map(location=start_loc, zoom_start=zoom)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for _, row in filtered_df.iterrows():
        popup_html = f"<b>{row['nama'].upper()}</b><br>{row['alamat']}"
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=colors.get(row['cluster'], 'gray'), icon='shopping-cart', prefix='fa')
        ).add_to(m)
    st_folium(m, width="100%", height=500, key="map")

with col_info:
    st.subheader("📋 Hasil Pencarian")
    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan semua kata kunci tersebut.")
    else:
        st.success(f"Ditemukan {len(filtered_df)} UMKM.")
        st.dataframe(filtered_df[['nama', 'alamat']], use_container_width=True, hide_index=True)

st.caption("Logika: Multi-keyword Intersection (Anti-Stopwords)")
