import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import os

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Prediksi Penjualan Produk Nike",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# CUSTOM CSS (UI ENHANCEMENT)
# ======================================================
st.markdown("""
<style>
.main { background-color: #f7f9fc; }

.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.pred-box {
    background: linear-gradient(90deg, #111827, #1f2937);
    color: white;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    background-color: #e5e7eb;
    margin-top: 8px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER + LOGO
# ======================================================
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg",
        width=110
    )

with col_title:
    st.title("Prediksi Penjualan Produk Nike")
    st.markdown(
        "Menggunakan Algoritma **Random Forest** dan "
        "**Long Short-Term Memory (LSTM)**"
    )

st.markdown("---")

# ======================================================
# LOAD MODEL & SCALER (SAFE FOR STREAMLIT CLOUD)
# ======================================================
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "model_lstm.h5")
    scaler_path = os.path.join("model", "scaler.pkl")

    model = tf.keras.models.load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    return model, scaler

model, scaler = load_model()

# ======================================================
# LOAD & NORMALIZE DATASET
# ======================================================
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

    return df.dropna(subset=["date", "state", "product_name", "quantity_sold"])

df = load_data()

# ======================================================
# SIDEBAR FILTER
# ======================================================
st.sidebar.header("🔎 Filter Data")

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
    sorted(df["date"].dt.year.unique())
)

# ======================================================
# FILTER DATA
# ======================================================
filtered_df = df[
    (df["state"] == state) &
    (df["product_name"] == product) &
    (df["date"].dt.year == year)
].sort_values("date")

# ======================================================
# RINGKASAN PENJUALAN
# ======================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📦 Ringkasan Penjualan</div>', unsafe_allow_html=True)

total_unit = int(filtered_df["quantity_sold"].sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Wilayah", state)
with c2:
    st.metric("Produk", product)
with c3:
    st.metric("Total Unit Terjual", f"{total_unit:,} unit")

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# DATA HISTORIS
# ======================================================
with st.expander("📄 Lihat Data Penjualan Historis"):
    st.dataframe(
        filtered_df[["date", "state", "product_name", "quantity_sold"]],
        use_container_width=True
    )

# ======================================================
# PREDIKSI PENJUALAN (LSTM)
# ======================================================
st.markdown("---")
st.subheader("🔮 Prediksi Penjualan Periode Berikutnya")

history = filtered_df["quantity_sold"].tolist()
WINDOW_SIZE = 10

if len(history) < WINDOW_SIZE:
    st.warning(f"Data historis minimal {WINDOW_SIZE} periode untuk prediksi.")
else:
    if st.button("🚀 Jalankan Prediksi"):
        seq = np.array(history[-WINDOW_SIZE:]).reshape(-1, 1)
        seq_scaled = scaler.transform(seq).reshape(1, WINDOW_SIZE, 1)

        prediction = model.predict(seq_scaled)
        pred_value = int(scaler.inverse_transform(prediction)[0][0])

        # ===============================
        # HIGHLIGHT RESULT
        # ===============================
        st.markdown(f"""
        <div class="pred-box">
            <h2>📈 Prediksi Penjualan Berikutnya</h2>
            <h1>{pred_value:,} unit</h1>
            <div class="badge">
                Berdasarkan {WINDOW_SIZE} periode terakhir
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===============================
        # GRAFIK
        # ===============================
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📉 Grafik Tren Penjualan")

        trend_values = history + [pred_value]
        period = list(range(1, len(trend_values) + 1))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(period[:-1], trend_values[:-1], marker="o", label="Data Historis")
        ax.scatter(period[-1], trend_values[-1], color="red", s=120, label="Prediksi")
        ax.plot(period, trend_values, linestyle="--", alpha=0.4)

        ax.set_xlabel("Periode")
        ax.set_ylabel("Unit Terjual")
        ax.set_title("Tren Penjualan Produk Nike")
        ax.legend()

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # ===============================
        # PENJELASAN HASIL
        # ===============================
        diff = pred_value - history[-1]
        trend = "meningkat 📈" if diff > 0 else "menurun 📉" if diff < 0 else "stabil ➖"

        st.subheader("📝 Penjelasan Hasil Prediksi")
        st.write(f"""
        Grafik menunjukkan pola penjualan **{product}**
        di wilayah **{state}** pada tahun **{year}**.

        Berdasarkan pola historis tersebut, model **LSTM**
        memprediksi bahwa penjualan periode berikutnya
        akan **{trend}** dengan estimasi **{pred_value:,} unit**.

        Titik merah menandakan hasil prediksi,
        sedangkan garis biru adalah data historis.
        """)

# ======================================================
# METODOLOGI
# ======================================================
with st.expander("📚 Penjelasan Metodologi"):
    st.markdown("""
    - Dataset penjualan Nike periode 2020–2021  
    - *Random Forest* digunakan sebagai pembanding eksperimen  
    - *LSTM* digunakan sebagai model utama time series  
    - Window size: **10 periode terakhir**  
    - Output: prediksi penjualan periode berikutnya  
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "© 2025 | Deployment Model Deep Learning "
    "Prediksi Penjualan Produk Nike menggunakan Streamlit"
)
