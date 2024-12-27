import streamlit as st
from PreProcess import PreProcess
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import numpy as np
import io
import wave

# Fungsi untuk memproses file WAV yang diunggah pengguna
def process_uploaded_audio(uploaded_file):
    # Simpan file yang diunggah sebagai file WAV sementara
    with open("uploaded_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Mengembalikan nama file yang disimpan untuk pemrosesan lebih lanjut
    return "uploaded_audio.wav"

st.title("OrdinaryVoice")
st.subheader("Ayo kenali suara mu!")
st.write("Unggah file WAV Anda untuk dianalisis.")
st.write("### Unggah Audio Anda")

# Memungkinkan pengguna untuk mengunggah file WAV
uploaded_file = st.file_uploader("Pilih file WAV", type=["wav"])

if uploaded_file is not None:
    # Proses file WAV yang diunggah
    file_name = process_uploaded_audio(uploaded_file)
    st.success("Audio berhasil diunggah dan disimpan sementara.")
    st.audio(uploaded_file, format='audio/wav')  # Menyediakan preview audio yang diunggah

    pp = PreProcess(file_name)

    if st.button("Analisis Suara"):
        try:
            # Extract features dari audio yang diunggah
            new_fhe = pp.extract_fhe()
            new_sc = pp.extract_sc()
            new_sb = pp.extract_sb()
            new_avg_f0 = pp.extract_average_f0()
            new_mfcc = pp.mfcc()
            new_dmfccs1_ = pp.delta_mfcc1()
            new_dmfccs2_ = pp.delta_mfcc2()
            new_chroma = pp.extract_chroma()
            new_scon_ = pp.spectral_contrast_range()
            new_sflat_ = pp.spectral_flatness_range()
            new_srolof_ = pp.spectral_rolloff_range()

            # Gabungkan semua fitur menjadi satu list
            new_features = (
                [new_fhe, new_sc, new_avg_f0, new_sb]
                + list(new_mfcc)
                + list(new_chroma)
                + list(new_dmfccs1_)
                + list(new_dmfccs2_)
                + list(new_scon_)
                + [new_sflat_, new_srolof_]
            )
            new_features = [f[0] if isinstance(f, np.ndarray) else f for f in new_features]

            # Standardisasi fitur
            scaler = StandardScaler()
            file = pd.read_csv("Features.csv")
            file = file.drop(columns=["Unnamed: 0"])

            drop_features = [
                "Jenis_suara",
                "Path_lengkap",
                "avg_f0",
                "mfcc_",
                "chroma_",
                "dmfccs1_",
                "dmfccs2_",
                "scon_",
            ] 
            X = file.drop(columns=drop_features)
            y = file["Jenis_suara"]

            # Encode label
            y_encoded = []
            for i in y:
                if i == "sopran":
                    i = 0
                elif i == "mezzo_sopran":
                    i = 1
                elif i == "tenor":
                    i = 2
                elif i == "baritone":
                    i = 3
                elif i == "bass":
                    i = 4
                y_encoded.append(i)

            # Split data untuk pelatihan
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            X_train = np.array(X_train)
            scaler.fit(X_train)

            # Skala fitur baru
            new_features_scaled = scaler.transform([new_features])

            # Muat model yang sudah dilatih
            loaded_model = load("model.joblib")
            result = loaded_model.predict(new_features_scaled)

            # Peta hasil ke label yang sesuai
            label_map = {0: "Sopran", 1: "Mezzo Sopran", 2: "Tenor", 3: "Baritone", 4: "Bass"}
            predicted_label = label_map.get(result[0], "Tidak Diketahui")

            st.success(f"Prediksi suara Anda adalah: **{predicted_label}**")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menganalisis suara: {e}")
