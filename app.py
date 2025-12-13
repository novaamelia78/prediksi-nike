import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Prediksi Penjualan Produk Nike",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# HEADER + LOGO
# ======================================================
col_logo, col_title = st.columns([1, 4])

with col_logo:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg",
        width=120
    )

with col_title:
    st.title("Prediksi Penjualan Produk Nike")
    st.markdown(
        "Menggunakan Algoritma **Random Forest** dan "
        "**Long Short-Term Memory (LSTM)**"
    )

st.markdown("---")

# ======================================================
# LOAD MODEL & SCALER (LSTM)
# ======================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/model_lstm.h5",
        compile=False
    )
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ======================================================
# LOAD & NORMALISASI DATASET
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
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    return df

df = load_data()

# ======================================================
# SIDEBAR FILTER
# ======================================================
st.sidebar.header("🔎 Filter Data")

state = st.sidebar.selectbox(
    "State (Wilayah Penjualan)",
    sorted(df["state"].dropna().unique())
)

product = st.sidebar.selectbox(
    "Produk",
    sorted(df["product_name"].dropna().unique())
)

year = st.sidebar.selectbox(
    "Tahun",
    sorted(df["date"].dt.year.dropna().unique())
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
st.subheader("📦 Ringkasan Penjualan")

total_unit = int(filtered_df["quantity_sold"].sum())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Wilayah", state)
with c2:
    st.metric("Produk", product)
with c3:
    st.metric("Total Unit Terjual", f"{total_unit:,} unit")

# ======================================================
# DATA HISTORIS
# ======================================================
with st.expander("📄 Lihat Data Penjualan Historis"):
    st.dataframe(
        filtered_df[
            ["date", "state", "product_name", "quantity_sold"]
        ],
        use_container_width=True
    )

# ======================================================
# PREDIKSI PENJUALAN (LSTM)
# ======================================================
st.markdown("---")
st.subheader("🔮 Prediksi Penjualan Periode Berikutnya")

history_values = filtered_df["quantity_sold"].tolist()
WINDOW_SIZE = 10
data_cukup = len(history_values) >= WINDOW_SIZE

if not data_cukup:
    st.warning(
        f"Data historis belum mencukupi "
        f"(minimal {WINDOW_SIZE} periode)"
    )

if st.button(
    "🔮 Prediksi Penjualan",
    disabled=not data_cukup
):
    # Ambil sequence terakhir
    sequence = history_values[-WINDOW_SIZE:]

    seq_scaled = scaler.transform(
        np.array(sequence).reshape(-1, 1)
    )
    seq_scaled = seq_scaled.reshape(1, WINDOW_SIZE, 1)

    prediction = model.predict(seq_scaled)
    pred_value = int(
        scaler.inverse_transform(prediction)[0][0]
    )

    # ===============================
    # HASIL PREDIKSI (DITONJOLKAN)
    # ===============================
    st.success(
        f"📈 Prediksi penjualan periode berikutnya: "
        f"**{pred_value:,} unit**"
    )

    # ===============================
    # GRAFIK TREN + PREDIKSI
    # ===============================
    st.subheader("📉 Grafik Tren Penjualan")

    trend_values = history_values + [pred_value]
    periode = list(range(1, len(trend_values) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        periode[:-1],
        trend_values[:-1],
        marker="o",
        label="Data Historis"
    )

    ax.scatter(
        periode[-1],
        trend_values[-1],
        color="red",
        s=120,
        label="Prediksi"
    )

    ax.plot(
        periode,
        trend_values,
        linestyle="--",
        alpha=0.4
    )

    ax.set_xlabel("Periode")
    ax.set_ylabel("Jumlah Unit Terjual")
    ax.set_title("Tren Penjualan Produk Nike")
    ax.legend()

    st.pyplot(fig)

    # ===============================
    # PENJELASAN HASIL GRAFIK
    # ===============================
    st.subheader("📝 Penjelasan Hasil Prediksi")

    trend_diff = pred_value - history_values[-1]

    if trend_diff > 0:
        trend_text = "meningkat"
    elif trend_diff < 0:
        trend_text = "menurun"
    else:
        trend_text = "cenderung stabil"

    st.write(f"""
    Grafik di atas menunjukkan pola penjualan produk **{product}**
    di wilayah **{state}** pada tahun **{year}** berdasarkan data historis.

    Berdasarkan pola tersebut, model **LSTM** memprediksi bahwa
    penjualan pada periode berikutnya akan **{trend_text}**
    dengan estimasi sebesar **{pred_value:,} unit**.

    Titik merah pada grafik merepresentasikan hasil prediksi,
    sedangkan garis biru menunjukkan data penjualan historis.
    """)

# ======================================================
# METODOLOGI
# ======================================================
with st.expander("📚 Penjelasan Metodologi"):
    st.write("""
    - Dataset penjualan produk Nike periode 2020–2021 digunakan
      sebagai data historis.
    - Algoritma **Random Forest** digunakan sebagai model pembanding
      pada tahap eksperimen.
    - Algoritma **LSTM** digunakan sebagai model utama karena mampu
      menangkap pola time series penjualan.
    - Prediksi dilakukan berdasarkan 10 periode penjualan terakhir.
    """)

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.caption(
    "Aplikasi ini merupakan hasil deployment model Deep Learning "
    "untuk prediksi penjualan produk Nike menggunakan Streamlit."
)