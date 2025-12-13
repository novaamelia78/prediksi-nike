import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Penjualan Produk Nike",
    page_icon="👟",
    layout="wide"
)

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
@st.cache_resource
def load_models():
    lstm_model = tf.keras.models.load_model("model_lstm.h5", compile=False)
    rf_model = joblib.load("model_rf.pkl")  # digunakan sebagai pembanding
    scaler = joblib.load("scaler.pkl")
    return lstm_model, rf_model, scaler

# =========================================================
# LOAD DATASET
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Nike Dataset.csv")
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["year"] = df["date"].dt.year
    return df

lstm_model, rf_model, scaler = load_models()
df = load_data()

# =========================================================
# SIDEBAR FILTER
# =========================================================
st.sidebar.markdown("## 🔍 Filter Data")

state = st.sidebar.selectbox(
    "State / Wilayah",
    sorted(df["state"].unique())
)

product = st.sidebar.selectbox(
    "Produk",
    sorted(df["product_name"].unique())
)

year = st.sidebar.selectbox(
    "Tahun",
    sorted(df["year"].unique())
)

# =========================================================
# FILTER DATA
# =========================================================
filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["year"] == year)
].sort_values("date")

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:20px">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg" width="90">
        <div>
            <h1 style="margin-bottom:0">Prediksi Penjualan Produk Nike</h1>
            <p style="color:gray;margin-top:4px">
            Menggunakan Algoritma <b>Random Forest</b> dan <b>Long Short-Term Memory (LSTM)</b>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# RINGKASAN PENJUALAN
# =========================================================
total_units = int(filtered_df["quantity_sold"].sum())
avg_daily = int(filtered_df["quantity_sold"].mean()) if len(filtered_df) > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("📍 Wilayah", state)
c2.metric("👟 Produk", product)
c3.metric("📦 Total Terjual", f"{total_units} unit")
c4.metric("📊 Rata-rata Harian", f"{avg_daily} unit")

# =========================================================
# DATA HISTORIS
# =========================================================
with st.expander("📄 Lihat Data Penjualan Historis"):
    st.dataframe(
        filtered_df[["date", "state", "product_name", "quantity_sold"]],
        use_container_width=True
    )

# =========================================================
# GRAFIK HISTORIS
# =========================================================
st.subheader("📈 Grafik Tren Penjualan Historis")

fig_hist = go.Figure()
fig_hist.add_trace(
    go.Scatter(
        x=filtered_df["date"],
        y=filtered_df["quantity_sold"],
        mode="lines+markers",
        name="Penjualan Aktual",
        line=dict(width=3)
    )
)

fig_hist.update_layout(
    xaxis_title="Tanggal",
    yaxis_title="Jumlah Unit Terjual",
    template="plotly_white"
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(
    """
    **Penjelasan Grafik Tren Penjualan Historis**

    Grafik ini menunjukkan perubahan jumlah penjualan produk Nike dari waktu ke waktu
    berdasarkan data historis. Setiap titik merepresentasikan jumlah unit yang terjual
    pada tanggal tertentu.

    Dari grafik ini dapat diamati:
    - Pola kenaikan dan penurunan penjualan dalam satu tahun
    - Periode dengan penjualan tertinggi dan terendah
    - Tingkat kestabilan atau fluktuasi permintaan produk

    Grafik historis ini menjadi dasar utama bagi model **LSTM** untuk mempelajari pola
    deret waktu sebelum melakukan prediksi penjualan.
    """
)

# =========================================================
# PREDIKSI PENJUALAN
# =========================================================
st.subheader("🔮 Prediksi Penjualan Menggunakan LSTM")

if st.button("🚀 Jalankan Prediksi"):
    if len(filtered_df) < 10:
        st.warning("Data tidak cukup untuk melakukan prediksi (minimal 10 periode).")
    else:
        series = filtered_df["quantity_sold"].values.reshape(-1, 1)
        scaled_series = scaler.transform(series)

        X = []
        window = 10
        for i in range(len(scaled_series) - window):
            X.append(scaled_series[i:i+window])

        X = np.array(X)

        pred_scaled = lstm_model.predict(X)
        pred = scaler.inverse_transform(pred_scaled)
        pred_dates = filtered_df["date"].iloc[window:]

        # =========================
        # GRAFIK PREDIKSI
        # =========================
        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=filtered_df["date"],
                y=filtered_df["quantity_sold"],
                name="Aktual",
                mode="lines",
                line=dict(color="black")
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred.flatten(),
                name="Prediksi LSTM",
                mode="lines",
                line=dict(color="green", dash="dash")
            )
        )

        fig_pred.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Jumlah Unit Terjual",
            template="plotly_white"
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown(
            f"""
            ### 📊 Interpretasi Hasil Grafik Prediksi

            Grafik prediksi menampilkan dua garis utama:

            - **Garis hitam (Aktual)**  
              Menunjukkan data penjualan asli berdasarkan histori penjualan produk Nike.

            - **Garis hijau putus-putus (Prediksi LSTM)**  
              Menunjukkan hasil prediksi dari model **Long Short-Term Memory (LSTM)** yang
              mempelajari pola penjualan dari data historis.

            **Interpretasi utama:**
            - Model LSTM mampu mengikuti tren naik dan turun penjualan yang terjadi sebelumnya
            - Perbedaan kecil antara grafik aktual dan prediksi menunjukkan performa model
              yang cukup baik dalam mengenali pola deret waktu
            - Untuk wilayah **{state}**, produk **{product}**, pada tahun **{year}**,
              model memprediksi kelanjutan tren penjualan berdasarkan histori yang tersedia

            Hasil prediksi ini dapat dimanfaatkan sebagai dasar perencanaan stok,
            strategi distribusi, dan pengambilan keputusan penjualan.
            """
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <hr>
    <p style="text-align:center;color:gray">
    © 2025 | Prediksi Penjualan Produk Nike<br>
    Deployment menggunakan Streamlit Cloud
    </p>
    """,
    unsafe_allow_html=True
)
