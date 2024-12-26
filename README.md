OrdinaryVoice adalah aplikasi berbasis web yang dikembangkan menggunakan Streamlit untuk membantu pengguna mengenali karakteristik suara mereka. Pengguna dapat merekam suara melalui aplikasi, dan setelah perekaman selesai, suara mereka dianalisis untuk mendeteksi jenis suara seperti sopran, tenor, bass, dan lainnya.
Aplikasi ini menggunakan model pembelajaran mesin untuk memprediksi jenis suara berdasarkan ekstraksi fitur dari rekaman suara, termasuk MFCC, chroma, spectral contrast, dan lainnya.

Fitur
Rekaman Suara Langsung: Menggunakan sounddevice untuk merekam suara pengguna.
Analisis Suara: Mengekstraksi fitur dari suara pengguna seperti MFCC, chroma, spectral contrast, dan lainnya.
Prediksi Jenis Suara: Menggunakan model machine learning untuk memprediksi jenis suara (Sopran, Tenor, Bass, dll.).

Prasyarat
Pastikan Anda telah menginstal semua dependensi yang diperlukan:
pip install -r requirements.txt

Penggunaan
Jalankan aplikasi dengan perintah:
streamlit run app.py

Klik tombol "Rekam Audio" untuk merekam suara Anda.
Setelah perekaman selesai, klik tombol "Analisis Suara", kemudian suara Anda akan dianalisis dan prediksi jenis suara akan ditampilkan.

SUMBER REFERENSI:
Model pembelajaran mesin yang digunakan dalam aplikasi ini mengacu pada algoritma Random Forest dan Decision Tree yang diimplementasikan dalam Machine Learning From Scratch - Random Forest.

Kontribusi
Jika Anda ingin berkontribusi pada proyek ini, silakan buat pull request dengan perubahan atau perbaikan yang diinginkan.
