# Sistem Penghitung Pengunjung (People Detection)

Sistem penghitung pengunjung otomatis menggunakan computer vision dan deep learning untuk mendeteksi dan menghitung pengunjung secara real-time dari video feed.

## Fitur

- ✅ Deteksi orang real-time menggunakan YOLOv5
- ✅ Object tracking untuk menghitung pengunjung unik
- ✅ Line crossing detection (menghitung saat pengunjung melewati garis virtual)
- ✅ Mendukung arah spesifik (kanan ke kiri)
- ✅ Penyimpanan data otomatis ke database SQLite
- ✅ Export data otomatis ke Excel dengan timestamp
- ✅ Interface visual dengan informasi jumlah pengunjung

## Persyaratan Sistem

- Python 3.8+
- GPU opsional (untuk performa lebih baik)
- Webcam atau video input

## Instalasi

1. **Clone repository:**
   ```bash
   git clone https://github.com/DIMASMario/people_detection.git
   cd people_detection
