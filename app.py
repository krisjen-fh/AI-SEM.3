import streamlit as st
import pyaudio
import wave
from PreProcess import PreProcess
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import numpy as np

# Function to record audio using PyAudio
def record_audio_pyaudio(seconds, filename):
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16-bit format
    channels = 1  # Mono
    rate = 44100  # Sample rate
    frames = []

    audio = pyaudio.PyAudio()

    # Start recording
    st.write("Merekam...")
    stream = audio.open(format=format, channels=channels,
                        rate=rate, input=True,
                        frames_per_buffer=chunk)

    for i in range(0, int(rate / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    st.write("Rekaman selesai.")

    # Save audio as .wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

st.title("OrdinaryVoice")
st.subheader("Ayo kenali suara mu!")
st.write("Klik tombol di bawah untuk merekam suara Anda. Setelah selesai, suara Anda akan dianalisis.")
st.write("### Rekam Suara Anda")

file_name = "output.wav"
if st.button("Rekam Audio (PyAudio)"):
    record_audio_pyaudio(10, file_name)  # Record for 10 seconds
    st.success("Audio telah direkam dan disimpan sebagai output.wav")

st.write("Setelah Anda selesai merekam, kami akan mengolah suara Anda untuk mendeteksi karakteristik unik suara Anda!")

pp = PreProcess(file_name)

if st.button("Analisis Suara"):
    try:
        # Extract features from the recorded audio
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

        # Combine all features into a single list
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

        # Standardize features
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

        # Encode labels
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

        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        X_train = np.array(X_train)
        scaler.fit(X_train)

        # Scale new features
        new_features_scaled = scaler.transform([new_features])

        # Load the pre-trained model
        loaded_model = load("model.joblib")
        result = loaded_model.predict(new_features_scaled)

        # Map result to corresponding label
        label_map = {0: "Sopran", 1: "Mezzo Sopran", 2: "Tenor", 3: "Baritone", 4: "Bass"}
        predicted_label = label_map.get(result[0], "Tidak Diketahui")

        st.success(f"Prediksi suara Anda adalah: **{predicted_label}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menganalisis suara: {e}")
