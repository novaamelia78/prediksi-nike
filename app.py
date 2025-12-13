import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
import joblib
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Prediksi Penjualan Nike",
    page_icon="👟",
    layout="wide"
)

# ===============================
# CSS CUSTOM
# ===============================
st.markdown("""
<style>
body {font-family: 'Segoe UI', sans-serif;}
h1, h2, h3 {color:#111;}
.metric-box {
    background-color:#ffffff;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Nike Dataset.csv")
MODEL_LSTM_PATH = os.path.join(BASE_DIR, "model", "model_lstm.h5")
MODEL_RF_PATH = os.path.join(BASE_DIR, "model", "model_rf.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    date_col = next(c for c in df.columns if "date" in c.lower())
    df = df.rename(columns={date_col: "date"})

    for c in df.columns:
        if c.lower() == "state":
            df = df.rename(columns={c: "state"})
        if "product" in c.lower():
            df = df.rename(columns={c: "product_name"})
        if ("unit" in c.lower() or "quantity" in c.lower()) and "price" not in c.lower():
            df = df.rename(columns={c: "quantity_sold"})

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["year"] = df["date"].dt.year
    return df

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    lstm = tf.keras.models.load_model(MODEL_LSTM_PATH, compile=False)
    rf = joblib.load(MODEL_RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    return lstm, rf, scaler

df = load_data()
lstm_model, rf_model, scaler = load_models()

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style="text-align:center;">
<img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" width="120">
<h1>Prediksi Penjualan Produk Nike</h1>
<p>Menggunakan Random Forest & Long Short-Term Memory (LSTM)</p>
</div>
<hr>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("🔎 Filter Data")
    state = st.selectbox("State / Wilayah", sorted(df["state"].unique()))
    product = st.selectbox("Produk", sorted(df["product_name"].unique()))
    year = st.selectbox("Tahun", sorted(df["year"].unique()))

# ===============================
# FILTER
# ===============================
filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["year"] == year)
].sort_values("date")

total_sales = int(filtered_df["quantity_sold"].sum())

# ===============================
# METRICS
# ===============================
col1, col2, col3 = st.columns(3)
col1.metric("📍 Wilayah", state)
col2.metric("📦 Produk", product)
col3.metric("🧮 Total Unit Terjual", f"{total_sales} unit")

# ===============================
# DATA
# ===============================
with st.expander("📄 Lihat Data Penjualan Historis"):
    st.dataframe(
        filtered_df[["date", "state", "product_name", "quantity_sold"]],
        use_container_width=True
    )

# ===============================
# GRAFIK TREN
# ===============================
st.subheader("📈 Tren Penjualan")

fig_trend = px.line(
    filtered_df,
    x="date",
    y="quantity_sold",
    markers=True,
    title="Tren Penjualan Produk"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ===============================
# PREDIKSI
# ===============================
st.subheader("🔮 Prediksi Penjualan")

if len(filtered_df) >= 10:
    if st.button("🚀 Jalankan Prediksi"):
        series = filtered_df["quantity_sold"].values.reshape(-1, 1)
        series_scaled = scaler.transform(series)

        X = []
        for i in range(len(series_scaled) - 10):
            X.append(series_scaled[i:i + 10])
        X = np.array(X)

        lstm_pred = scaler.inverse_transform(lstm_model.predict(X))
        rf_pred = rf_model.predict(np.arange(len(series)).reshape(-1, 1))

        fig_pred = px.line(title="Perbandingan Prediksi Penjualan")
        fig_pred.add_scatter(
            x=filtered_df["date"],
            y=series.flatten(),
            name="Data Aktual"
        )
        fig_pred.add_scatter(
            x=filtered_df["date"][10:],
            y=lstm_pred.flatten(),
            name="Prediksi LSTM"
        )
        fig_pred.add_scatter(
            x=filtered_df["date"],
            y=rf_pred,
            name="Prediksi Random Forest"
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.info(
            f"""
            **Interpretasi Hasil**  
            Pada tahun **{year}**, produk **{product}** di wilayah **{state}**
            menunjukkan total penjualan **{total_sales} unit**.
            Model **LSTM** mampu mengikuti pola tren waktu,
            sedangkan **Random Forest** digunakan sebagai pembanding pola umum data.
            """
        )
else:
    st.warning("Data tidak cukup untuk prediksi (minimal 10 periode).")

# ===============================
# FOOTER
# ===============================
st.markdown(
    "<hr><p style='text-align:center;'>© 2025 | Prediksi Penjualan Produk Nike</p>",
    unsafe_allow_html=True
)
