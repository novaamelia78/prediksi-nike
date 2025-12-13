import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Nike Sales Prediction",
    page_icon="👟",
    layout="wide"
)

# =========================
# GLOBAL STYLE (BENERAN BERASA)
# =========================
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.hero {
    background: linear-gradient(90deg, #000000, #1f2937);
    padding: 40px;
    border-radius: 20px;
    color: white;
}
.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}
.hero p {
    font-size: 18px;
    color: #d1d5db;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.kpi {
    text-align: center;
}
.kpi h2 {
    font-size: 32px;
    margin: 0;
}
.kpi p {
    color: #6b7280;
    margin: 0;
}
.pred-box {
    background: linear-gradient(135deg, #111827, #2563eb);
    color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
}
.info-box {
    background: #eef2ff;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #4f46e5;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <h1>👟 Prediksi Penjualan Produk Nike</h1>
    <p>Analisis & Prediksi Penjualan Menggunakan Random Forest dan LSTM</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/model_lstm.h5", compile=False)
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =========================
# LOAD DATA
# =========================
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

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.title("🔎 Filter Data")

state = st.sidebar.selectbox("Wilayah", sorted(df["state"].unique()))
product = st.sidebar.selectbox("Produk", sorted(df["product_name"].unique()))
year = st.sidebar.selectbox("Tahun", sorted(df["date"].dt.year.unique()))

filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["date"].dt.year == year)
].sort_values("date")

# =========================
# KPI CARDS
# =========================
total_unit = int(filtered_df["quantity_sold"].sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""
    <div class="card kpi">
        <p>Wilayah</p>
        <h2>{state}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card kpi">
        <p>Produk</p>
        <h2>{product}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card kpi">
        <p>Total Terjual</p>
        <h2>{total_unit:,}</h2>
    </div>
    """, unsafe_allow_html=True)

# =========================
# HISTORICAL DATA
# =========================
with st.expander("📄 Data Penjualan Historis"):
    st.dataframe(filtered_df, use_container_width=True)

# =========================
# PREDICTION
# =========================
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("🔮 Prediksi Penjualan")

WINDOW_SIZE = 10
history = filtered_df["quantity_sold"].tolist()

if len(history) < WINDOW_SIZE:
    st.warning("Data tidak cukup untuk prediksi (minimal 10 periode).")
else:
    if st.button("🚀 Jalankan Prediksi"):
        seq = np.array(history[-WINDOW_SIZE:]).reshape(-1, 1)
        seq_scaled = scaler.transform(seq).reshape(1, WINDOW_SIZE, 1)

        pred = model.predict(seq_scaled)
        pred_value = int(scaler.inverse_transform(pred)[0][0])

        # =========================
        # PREDICTION BOX
        # =========================
        st.markdown(f"""
        <div class="pred-box">
            <h1>{pred_value:,} Unit</h1>
            <p>Prediksi Penjualan Periode Berikutnya</p>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # TREND GRAPH
        # =========================
        fig, ax = plt.subplots(figsize=(10, 4))
        values = history + [pred_value]

        ax.plot(values[:-1], label="Historis", linewidth=3)
        ax.scatter(len(values)-1, values[-1], color="red", s=120, label="Prediksi")
        ax.plot(values, linestyle="--", alpha=0.4)

        ax.set_title("Tren Penjualan")
        ax.set_ylabel("Unit Terjual")
        ax.legend()

        st.pyplot(fig)

        # =========================
        # EXPLANATION BOX
        # =========================
        diff = pred_value - history[-1]
        trend = "meningkat" if diff > 0 else "menurun" if diff < 0 else "stabil"

        st.markdown(f"""
        <div class="info-box">
            <b>Interpretasi Hasil:</b><br><br>
            Model <b>LSTM</b> mempelajari pola penjualan historis produk
            <b>{product}</b> di wilayah <b>{state}</b> pada tahun <b>{year}</b>.
            <br><br>
            Hasil prediksi menunjukkan tren penjualan <b>{trend}</b> dengan
            estimasi <b>{pred_value:,} unit</b> pada periode berikutnya.
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("© 2025 | Nike Sales Prediction Dashboard | Streamlit Deployment")
