import streamlit as st
import sounddevice as sd
import numpy as np
import wave
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from PreProcess import PreProcess
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load

def record_audio(seconds):
    fs = 44100  # Sample rate
    st.write("Merekam...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait() 
    st.write("Rekaman selesai.")
    return audio

def save_audio(audio, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(44100)
        wf.writeframes(audio.tobytes())

st.title("OrdinaryVoice")
st.subheader("Ayo kenali suara mu!")
st.write("Klik tombol di bawah untuk merekam suara Anda. Setelah selesai, suara Anda akan dianalisis.")
st.write("### Rekam Suara Anda")

file_name = "output.wav"
if st.button("Rekam Audio (SoundDevice)"):
    frames = record_audio(10)  # Record for 10 seconds
    save_audio(frames, file_name)
    st.success("Audio telah direkam dan disimpan sebagai output.wav")

webrtc_streamer(
    key="voice-recorder",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_receiver_size=1024,  
    video_transformer_factory=None,  
)

st.write("Setelah Anda selesai merekam, kami akan mengolah suara Anda untuk mendeteksi karakteristik unik suara Anda!")

pp = PreProcess(file_name)

if st.button("Analisis Suara"):
    try:
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

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        X_train = np.array(X_train)
        scaler.fit(X_train)

        new_features_scaled = scaler.transform([new_features])
        print(new_features_scaled)
        print(len(*new_features_scaled))

        # Load the model
        loaded_model = load("model.joblib")
        result = loaded_model.predict(new_features_scaled)

        label_map = {0: "Sopran", 1: "Mezzo Sopran", 2: "Tenor", 3: "Baritone", 4: "Bass"}
        predicted_label = label_map.get(result[0], "Tidak Diketahui")
        print(predicted_label)

        st.success(f"Prediksi suara Anda adalah: **{predicted_label}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menganalisis suara: {e}")
