import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import google.generativeai as genai
import time

# --- 1. KONFIGURASI TAMPILAN ---
st.set_page_config(page_title="WebGIS UMKM | Omnibox AI", layout="wide")

st.markdown("""
    <style>
    .header-box { background-color: #0e1117; padding: 1.5rem; border-radius: 10px; text-align: center; color: white; font-family: 'Segoe UI', sans-serif; font-size: 2.2rem; font-weight: bold; margin-bottom: 1.5rem; border: 1px solid #30363d; }
    .sub-header { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 0.5rem; margin-bottom: 1rem; font-weight: bold; font-size: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi AI")
    api_key = st.text_input("Masukkan Google Gemini API Key", type="password")
    st.markdown("[Ambil API Key Gratis di sini](https://aistudio.google.com/app/apikey)")

# --- 2. LOAD DATASET ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_umkm.csv', sep=None, engine='python')
        df.columns = df.columns.str.strip().str.lower()
        df['lat'] = pd.to_numeric(df['lat'].astype(str).str.replace(',', '.'), errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['lat', 'lon', 'nama', 'alamat'])
        
        # Klastering K-Means
        if len(df) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
        else: 
            df['cluster'] = 0
        return df
    except Exception as e: 
        st.error(f"Gagal memuat dataset: {e}")
        st.stop()

df = load_data()

# --- 3. MEMORY CARD (SESSION STATE ANTI-PIKUN) ---
if 'saved_query' not in st.session_state:
    st.session_state.saved_query = ""
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""

# --- 4. OMNIBOX (KOLOM PENCARIAN) ---
st.markdown('<div class="header-box"> WebGIS K-Means & AI Malawele</div>', unsafe_allow_html=True)

with st.form(key='search_form'):
    query_input = st.text_input("🔍 Tanya AI & Cari Lokasi UMKM Sekaligus:", placeholder="Contoh: umkm apa saja yg ada di jl. gambas")
    submit_button = st.form_submit_button(label='Cari Lokasi & Tanya AI')

if submit_button and query_input:
    st.session_state.saved_query = query_input
    st.session_state.ai_response = "" 

query = st.session_state.saved_query
filtered_df = df.copy()

# --- LOGIKA FILTER PETA ---
if query:
    user_input = query.lower().replace(',', ' ').replace('?', '').replace('.', ' ').split()
    
    # Kata-kata sampah yang akan diabaikan oleh peta (termasuk 'umkm')
    ignore = ['dimana', 'lokasi', 'cari', 'cariin', 'carikan', 'ada', 'di', 'mana', 'tahu', 'apa', 'tunjukkan', 'ke', 'tolong', 'dong', 'info', 'tempat', 'buatkan', 'puisi', 'siapa', 'jalan', 'jl', 'jln', 'yg', 'yang', 'dari', 'saja', 'sebutkan', 'umkm', 'daftar', 'semua', 'toko', 'warung']
    keywords = [t for t in user_input if t not in ignore and len(t) > 2]

    if keywords:
        df_pool = df['nama'].astype(str).str.lower() + " " + df['alamat'].astype(str).str.lower()
        
        # Logika Strict (AND)
        mask_strict = df_pool.apply(lambda x: all(t in x for t in keywords))
        match = df[mask_strict]
        
        # Fallback (OR) hanya jika kata kunci cuma 1
        if match.empty and len(keywords) == 1:
            mask_flexible = df_pool.apply(lambda x: any(t in x for t in keywords))
            match = df[mask_flexible]
            
        filtered_df = match.copy() if not match.empty else pd.DataFrame()

# --- 5. TATA LETAK: KIRI (PETA) & KANAN (RESPON AI) ---
col_map, col_ai = st.columns([2, 1.2])

with col_map:
    start_loc = [-0.9648, 131.3059]
    if not filtered_df.empty: 
        start_loc = [filtered_df.iloc[0]['lat'], filtered_df.iloc[0]['lon']]
        
    m = folium.Map(location=start_loc, zoom_start=16 if (query and not filtered_df.empty) else 15)
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for _, row in filtered_df.iterrows():
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['lat']},{row['lon']}"
        popup_html = f"""
        <div style="font-family: sans-serif; width: 160px;">
            <h5 style="margin:0; color:#0e1117;">{row['nama'].upper()}</h5>
            <p style="margin:5px 0; font-size:11px;">{row['alamat']}</p>
            <a href="{gmaps_url}" target="_blank" style="display: block; text-align: center; background-color: #4285F4; color: white; padding: 8px; border-radius: 5px; text-decoration: none; font-weight: bold; font-size: 10px; margin-top: 10px;">🚗 PETUNJUK ARAH</a>
        </div>
        """
        folium.Marker([row['lat'], row['lon']], popup=folium.Popup(popup_html, max_width=200),
                      icon=folium.Icon(color=colors.get(row['cluster'], 'gray'), icon="info-sign")).add_to(m)
    
    st_folium(m, width="100%", height=500)

with col_ai:
    st.markdown('<div class="sub-header">🤖 Asisten AI Gemini</div>', unsafe_allow_html=True)
    
    if query:
        if not st.session_state.ai_response:
            if not api_key: 
                st.warning("⚠️ Masukkan API Key di sidebar kiri terlebih dahulu.")
            else:
                with st.spinner("AI berpikir..."):
                    try:
                        time.sleep(1) # Jeda aman anti limit 429
                        genai.configure(api_key=api_key)
                        
                        # AI Pintar: Beda konteks Spesifik vs Umum
                        if not filtered_df.empty and len(filtered_df) < len(df):
                            data_context = filtered_df[['nama', 'kategori', 'alamat', 'cluster']].to_csv(index=False)
                            sys_prompt = f"""Anda Asisten UMKM Malawele. User mencari: '{query}'. 
                            Ini data SPESIFIK dari peta: 
                            {data_context}
                            Jawab ramah, sebut lokasinya, dan arahkan user lihat peta di kiri."""
                        else:
                            data_full = df[['nama', 'kategori', 'alamat', 'cluster']].to_csv(index=False)
                            sys_prompt = f"""Anda Asisten UMKM. User bertanya: '{query}'. 
                            Ini SELURUH DATABASE UMKM Malawele: 
                            {data_full}
                            TUGAS: 
                            1. Jika bertanya info umum (daftar jalan, kategori), rangkum dari data ini.
                            2. JIKA user mencari nama spesifik yang memang tidak ada, katakan "Mohon maaf, tidak ditemukan".
                            3. Tolak pertanyaan di luar konteks UMKM."""

                        # Auto-Discovery Model (Anti Error 404)
                        tersedia = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        model_pilihan = next((m for m in tersedia if 'flash' in m.lower()), tersedia[0] if tersedia else 'gemini-1.5-flash')
                        
                        model = genai.GenerativeModel(model_pilihan, system_instruction=sys_prompt)
                        response = model.generate_content(query)
                        
                        st.session_state.ai_response = response.text
                        
                    except Exception as e: 
                        st.error(f"Error AI: {e}")
        
        if st.session_state.ai_response:
            st.info(st.session_state.ai_response)
            
    else:
        st.write("Silakan ketik pertanyaan di atas lalu tekan Enter.")
        st.markdown("**Direktori UMKM (Top 10):**")
        disp = df[['nama', 'kategori', 'cluster']].copy().head(10)
        disp['cluster'] = disp['cluster'] + 1
        st.dataframe(disp, use_container_width=True, hide_index=True)