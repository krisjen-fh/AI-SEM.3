import librosa
import numpy as np

class PreProcess:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_fhe(self, B1=2000, B2=4500):
        y, sr = librosa.load(self.file_path, sr=None)
        D = np.abs(librosa.stft(y))
        S = np.mean(D, axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        idx_b1 = np.searchsorted(freqs, B1)
        idx_b2 = np.searchsorted(freqs, B2)

        spectrum_band = S[idx_b1:idx_b2]
        freqs_band = freqs[idx_b1:idx_b2]

        total_energy = np.sum(spectrum_band**2)
        if total_energy == 0:
            return 0  # Avoid division by zero
        energy_density = (spectrum_band**2) / total_energy

        cumulative_energy = np.cumsum(energy_density)
        fhe_idx = np.argmax(cumulative_energy >= 0.5)
        fhe = freqs_band[fhe_idx]

        return fhe

    def extract_sc(self, B1=2000, B2=4500):
        y, sr = librosa.load(self.file_path, sr=None)
        D = np.abs(librosa.stft(y))
        S = np.mean(D, axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        idx_b1 = np.searchsorted(freqs, B1)
        idx_b2 = np.searchsorted(freqs, B2)

        spectrum_band = S[idx_b1:idx_b2]
        freqs_band = freqs[idx_b1:idx_b2]

        total_energy = np.sum(spectrum_band**2)
        if total_energy == 0:
            return 0
        energy_density = (spectrum_band**2) / total_energy

        spectral_centroid = np.sum(freqs_band * energy_density)
        return spectral_centroid

    def extract_average_f0(self, fmin_="E2", fmax_="C6"):
        y, sr = librosa.load(self.file_path, sr=None)
        fmin = librosa.note_to_hz(fmin_)
        fmax = librosa.note_to_hz(fmax_)

        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        f0_avg = np.mean(f0_clean) if len(f0_clean) > 0 else 0
        return f0_avg

    def extract_sb(self, B1=2000, B2=4500):
        y, sr = librosa.load(self.file_path, sr=None)
        D = np.abs(librosa.stft(y))
        S = np.mean(D, axis=1)
        freqs = librosa.fft_frequencies(sr=sr)

        idx_b1 = np.searchsorted(freqs, B1)
        idx_b2 = np.searchsorted(freqs, B2)

        spectrum_band = S[idx_b1:idx_b2]
        freqs_band = freqs[idx_b1:idx_b2]

        total_energy = np.sum(spectrum_band**2)
        if total_energy == 0:
            return 0
        energy_density = (spectrum_band**2) / total_energy

        mean_freq = np.sum(freqs_band * energy_density)
        sb = np.sqrt(np.sum(energy_density * (freqs_band - mean_freq)**2))
        return sb

    def mfcc(self, n_mfcc=20):
        y, sr = librosa.load(self.file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = np.mean(mfccs.T, axis=0)
        return mfccs

    def extract_chroma(self, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        mean_chroma = np.mean(chroma, axis=1)
        return mean_chroma

    def delta_mfcc1(self, n_mfcc=20, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        delta_mfccs1 = librosa.feature.delta(mfccs, order=1)
        delta_mfccs1 = np.mean(delta_mfccs1.T, axis=0)
        return delta_mfccs1

    def delta_mfcc2(self, n_mfcc=20, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        delta_mfccs2 = librosa.feature.delta(mfccs, order=2)
        delta_mfccs2 = np.mean(delta_mfccs2.T, axis=0)
        return delta_mfccs2

    def spectral_contrast_range(self, B1=2000, B2=4500, n_bands=6, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, hop_length=hop_length, n_fft=n_fft)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        idx_b1 = np.searchsorted(freqs, B1)
        idx_b2 = np.searchsorted(freqs, B2)

        contrast_band = contrast[:, idx_b1:idx_b2]
        contrast_mean = np.mean(contrast_band, axis=1)
        return contrast_mean

    def spectral_flatness_range(self, B1=2000, B2=4500, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length, n_fft=n_fft)
        flatness_mean = np.mean(flatness, axis=1)
        return flatness_mean

    def spectral_rolloff_range(self, threshold=0.85, hop_length=512, n_fft=2048):
        y, sr = librosa.load(self.file_path, sr=None)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, n_fft=n_fft)
        rolloff_mean = np.mean(rolloff, axis=1)
        return rolloff_mean
