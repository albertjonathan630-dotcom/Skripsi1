import streamlit as st
import requests

st.set_page_config(
    page_title="Buzzer Detection",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        .stApp, .stMarkdown, label, .stSelectbox, p {
            color: #e0e0e0 !important;
        }
        .stNumberInput input, .stSelectbox select {
            background-color: #1e2d45 !important;
            color: #e0e0e0 !important;
            border: 1px solid #3a7bd5 !important;
            border-radius: 8px !important;
        }
        hr { border-color: #3a7bd5 !important; opacity: 0.3; }

        /* Header Box */
        .header-box {
            background: linear-gradient(135deg, #0f3460, #1a1a2e);
            border: 2px solid #3a7bd5;
            border-radius: 20px;
            padding: 2rem 2.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 30px rgba(58,123,213,0.3);
        }
        .logo-circle {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #3a7bd5, #e0c068);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            margin: 0 auto 1rem auto;
            box-shadow: 0 0 20px rgba(58,123,213,0.5);
        }
        .title {
            font-size: 2.8rem;
            font-weight: 900;
            background: linear-gradient(90deg, #3a7bd5, #e0c068);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #a0b4c8 !important;
            font-size: 1rem;
        }
        .badge {
            display: inline-block;
            background: rgba(58,123,213,0.2);
            border: 1px solid #3a7bd5;
            border-radius: 20px;
            padding: 0.3rem 1rem;
            font-size: 0.8rem;
            color: #a0c4e8 !important;
            margin-top: 0.8rem;
        }

        .result-box {
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.6rem;
            font-weight: bold;
            margin-top: 1rem;
        }
        .buzzer {
            background: linear-gradient(135deg, #3d0000, #7b1a1a);
            color: #ff9090;
            border: 2px solid #e74c3c;
            box-shadow: 0 0 20px rgba(231,76,60,0.3);
        }
        .normal {
            background: linear-gradient(135deg, #003d1a, #1a7b3d);
            color: #90ffb8;
            border: 2px solid #2ecc71;
            box-shadow: 0 0 20px rgba(46,204,113,0.3);
        }
        [data-testid="stMetricValue"] {
            color: #e0c068 !important;
            font-size: 1.2rem !important;
        }
        [data-testid="stMetricLabel"] { color: #a0b4c8 !important; }
        .stMarkdown h3 { color: #e0c068 !important; }
        .stButton button {
            background: linear-gradient(90deg, #3a7bd5, #1a5aad) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #e0c068, #c8a030) !important;
            color: #1a1a2e !important;
        }
        .footer {
            text-align: center;
            color: #5a7a9a !important;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER BOX DENGAN LOGO
# ============================================================
st.markdown("""
    <div class="header-box">
        <div class="logo-circle">🔍</div>
        <div class="title">Buzzer Detection</div>
        <div class="subtitle">Model Hybrid Machine Learning untuk Deteksi Perilaku Buzzer di Media Sosial</div>
        <div class="badge">🤖 XGBoost · StandardScaler · FastAPI · Railway</div>
    </div>
""", unsafe_allow_html=True)

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
# TOMBOL PREDIKSI
# ============================================================
if st.button("🔍 Deteksi Sekarang", type="primary", use_container_width=True):
    with st.spinner("Menganalisis akun..."):
        try:
            payload = {
                "score": score, "controversiality": controversiality,
                "user_comment_karma": user_comment_karma, "user_link_karma": user_link_karma,
                "user_total_karma": user_total_karma, "account_age_days": account_age_days,
                "comment_length": comment_length
            }
            response = requests.post(
                "https://skripsi1-production-9d2e.up.railway.app/predict",
                json=payload, timeout=10
            )
            result = response.json()
            label      = result["label"]
            confidence = result["confidence"]
            prediction = result["prediction"]

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
st.markdown('<div class="footer">© 2025 Albert Jonathan · Skripsi · Buzzer Detection System</div>', unsafe_allow_html=True)
