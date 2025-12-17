import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Prediksi Penjualan Produk Nike",
    page_icon="👟",
    layout="wide"
)

# =====================================================
# GLOBAL STYLE (LIGHT DASHBOARD)
# =====================================================
st.markdown("""
<style>
body { background-color: #f8fafc; }
.card {
    background: white;
    padding: 26px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}
.section-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 12px;
}
.pred-box {
    background: linear-gradient(135deg, #e0e7ff, #f8fafc);
    border: 2px solid #6366f1;
    padding: 32px;
    border-radius: 22px;
    text-align: center;
}
.analysis-box {
    background: #f1f5f9;
    border-left: 6px solid #6366f1;
    padding: 22px;
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER (LOGO AMAN)
# =====================================================
col1, col2 = st.columns([1, 6])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg",
        width=120
    )
with col2:
    st.markdown("""
    <h1>Prediksi Penjualan Produk Nike</h1>
    <p>
    Menggunakan Algoritma <b>Random Forest</b> dan
    <b>Long Short-Term Memory (LSTM)</b>
    </p>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/model_lstm.h5", compile=False)
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Nike Dataset.csv")

    rename_map = {}
    for col in df.columns:
        c = col.lower()
        if "date" in c:
            rename_map[col] = "date"
        elif "state" in c:
            rename_map[col] = "state"
        elif "product" in c:
            rename_map[col] = "product_name"
        elif ("unit" in c or "quantity" in c) and "price" not in c:
            rename_map[col] = "quantity_sold"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna()

df = load_data()

# =====================================================
# SIDEBAR FILTER
# =====================================================
st.sidebar.header("🔎 Filter Data")
state = st.sidebar.selectbox("Wilayah", sorted(df["state"].unique()))
product = st.sidebar.selectbox("Produk", sorted(df["product_name"].unique()))
year = st.sidebar.selectbox("Tahun", sorted(df["date"].dt.year.unique()))

filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["date"].dt.year == year)
].sort_values("date")

# =====================================================
# KPI
# =====================================================
st.markdown('<div class="section-title">📊 Ringkasan Penjualan</div>', unsafe_allow_html=True)
total_unit = int(filtered_df["quantity_sold"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("Wilayah", state)
c2.metric("Produk", product)
c3.metric("Total Unit Terjual", f"{total_unit:,} unit")

# =====================================================
# DATA
# =====================================================
with st.expander("📄 Data Penjualan Historis"):
    st.dataframe(filtered_df, use_container_width=True)

# =====================================================
# PREDIKSI
# =====================================================
st.markdown('<div class="section-title">🔮 Prediksi Penjualan</div>', unsafe_allow_html=True)

WINDOW_SIZE = 10
history = filtered_df["quantity_sold"].tolist()

if len(history) < WINDOW_SIZE:
    st.warning("Data historis belum cukup untuk prediksi.")
else:
    if st.button("🚀 Jalankan Prediksi"):
        seq = np.array(history[-WINDOW_SIZE:]).reshape(-1, 1)
        seq_scaled = scaler.transform(seq).reshape(1, WINDOW_SIZE, 1)
        pred = model.predict(seq_scaled)
        pred_value = int(scaler.inverse_transform(pred)[0][0])

        st.markdown(f"""
        <div class="pred-box">
            <h2>Prediksi Penjualan Periode Berikutnya</h2>
            <h1>{pred_value:,} Unit</h1>
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        values = history + [pred_value]
        ax.plot(values[:-1], linewidth=3, label="Historis")
        ax.scatter(len(values)-1, values[-1], color="red", s=120, label="Prediksi")
        ax.plot(values, linestyle="--", alpha=0.4)
        ax.set_ylabel("Unit Terjual")
        ax.set_title("Tren Penjualan Produk Nike")
        ax.legend()
        st.pyplot(fig)

        trend = "meningkat" if pred_value > history[-1] else "menurun" if pred_value < history[-1] else "stabil"

        st.markdown(f"""
        <div class="analysis-box">
        <b>Interpretasi Grafik:</b><br><br>
        Grafik menunjukkan pola penjualan produk <b>{product}</b>
        di wilayah <b>{state}</b> pada tahun <b>{year}</b>.
        Model <b>LSTM</b> memprediksi tren penjualan <b>{trend}</b>
        dengan estimasi <b>{pred_value:,} unit</b> pada periode berikutnya.
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# PENJELASAN ALGORITMA (BALIK LAGI)
# =====================================================
with st.expander("📚 Penjelasan Algoritma yang Digunakan"):
    st.markdown("""
    **1. Random Forest**  
    Digunakan pada tahap eksperimen sebagai model pembanding.
    Random Forest mampu menangkap hubungan non-linear antar fitur,
    namun kurang optimal untuk data deret waktu.

    **2. Long Short-Term Memory (LSTM)**  
    Digunakan sebagai model utama pada deployment karena
    mampu mempelajari pola berurutan (time series) penjualan.
    Model LSTM memanfaatkan 10 periode terakhir untuk
    memprediksi penjualan periode selanjutnya.
    """)

# =====================================================
# FOOTER
# =====================================================
st.divider()
st.caption(
    "© 2025 | Dashboard Prediksi Penjualan Produk Nike | Streamlit Cloud"
)

