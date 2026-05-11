import streamlit as st
import requests

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Buzzer Detection",
    page_icon="🔍",
    layout="centered"
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            color: #1a1a2e;
        }
        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 2rem;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 1rem;
        }
        .buzzer {
            background-color: #ffe0e0;
            color: #c0392b;
            border: 2px solid #e74c3c;
        }
        .normal {
            background-color: #e0f7ea;
            color: #1e8449;
            border: 2px solid #2ecc71;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="title">🔍 Buzzer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Model Hybrid Machine Learning untuk Deteksi Perilaku Buzzer di Media Sosial</div>', unsafe_allow_html=True)
st.divider()

# ============================================================
# FORM INPUT
# ============================================================
st.subheader("📋 Input Data Akun")

col1, col2 = st.columns(2)

with col1:
    score = st.number_input("Score Komentar", value=10, help="Skor komentar pengguna")
    user_comment_karma = st.number_input("Comment Karma", value=500, help="Total karma komentar")
    user_total_karma = st.number_input("Total Karma", value=600, help="Total karma keseluruhan")
    account_age_days = st.number_input("Umur Akun (hari)", value=365, help="Umur akun dalam hari")

with col2:
    controversiality = st.selectbox("Controversiality", options=[0, 1],
                                     format_func=lambda x: "Tidak Kontroversial (0)" if x == 0 else "Kontroversial (1)")
    user_link_karma = st.number_input("Link Karma", value=100, help="Total karma link")
    comment_length = st.number_input("Panjang Komentar (kata)", value=17, help="Jumlah kata dalam komentar")

st.divider()

# ============================================================
# CONTOH PRESET
# ============================================================
st.subheader("⚡ Contoh Preset")
col_a, col_b = st.columns(2)

with col_a:
    if st.button("👤 Isi Contoh Normal", use_container_width=True):
        st.session_state.preset = "normal"

with col_b:
    if st.button("🚨 Isi Contoh Buzzer", use_container_width=True):
        st.session_state.preset = "buzzer"

st.divider()

# ============================================================
# TOMBOL PREDIKSI
# ============================================================
if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
    with st.spinner("Menganalisis akun..."):
        try:
            payload = {
                "score": score,
                "controversiality": controversiality,
                "user_comment_karma": user_comment_karma,
                "user_link_karma": user_link_karma,
                "user_total_karma": user_total_karma,
                "account_age_days": account_age_days,
                "comment_length": comment_length
            }

            response = requests.post(
                "https://skripsi1-production-9d2e.up.railway.app/predict",
                json=payload,
                timeout=10
            )
            result = response.json()

            label      = result["label"]
            confidence = result["confidence"]
            prediction = result["prediction"]

            # Tampilkan hasil
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-box buzzer">
                        🚨 {label}<br>
                        <span style="font-size:1rem;">Confidence: {confidence*100:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-box normal">
                        ✅ {label}<br>
                        <span style="font-size:1rem;">Confidence: {confidence*100:.2f}%</span>
                    </div>
                """, unsafe_allow_html=True)

            # Detail
            st.divider()
            st.subheader("📊 Detail Hasil")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediksi", label)
            c2.metric("Confidence", f"{confidence*100:.2f}%")
            c3.metric("Kode", prediction)

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
    <div style="text-align:center; color:#aaa; font-size:0.8rem;">
        © 2025 Albert Jonathan · Skripsi · Buzzer Detection System
    </div>
""", unsafe_allow_html=True)
