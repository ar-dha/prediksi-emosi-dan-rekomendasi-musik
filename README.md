# Prediksi Emosi dan Rekomendasi Musik

## Ringkasan

Proyek ini berfokus pada pengembangan sistem yang menggabungkan prediksi emosi dan rekomendasi musik berdasarkan emosi yang terdeteksi. Aplikasi memungkinkan pengguna untuk mengunggah gambar atau mengambil gambar menggunakan kamera, menganalisis emosi yang tergambar dalam gambar, dan menyarankan musik yang sesuai berdasarkan emosi yang diidentifikasi.

## Komponen Proyek

1. **Model Prediksi Emosi:**
   - Menggunakan model deep learning yang telah dilatih sebelumnya untuk memprediksi emosi dalam gambar.
   - Model ini memproses gambar yang diunggah/diambil dan memprediksi emosi dari 7 kategori: anger, sad, disgust, fear, neutral, happy, surprise.

2. **Rekomendasi Musik:**
   - Berdasarkan emosi yang diprediksi, sistem merekomendasikan musik secara acak yang sesuai dengan keadaan emosional pengguna.
   - Rekomendasi musik berasal dari kumpulan data yang dikategorikan berdasarkan suasana hati.

3. **Antarmuka Pengguna:**
   - Memberikan opsi bagi pengguna untuk mengunggah gambar atau mengambil gambar menggunakan kamera.
   - Menampilkan gambar yang diunggah/diambil dan emosi yang diprediksi.
   - Menyediakan antarmuka yang ramah pengguna untuk memilih dan mendengarkan musik yang direkomendasikan.

## Cara Kerja

1. **Interaksi Pengguna:**
   - Pengguna dapat memilih untuk mengunggah gambar atau mengambil gambar menggunakan kamera.
   - Mereka diminta untuk memilih opsi yang diinginkan.

2. **Pemrosesan Gambar:**
   - Untuk gambar yang diunggah, aplikasi memproses gambar menggunakan teknik computer vision.
   - Untuk gambar yang diambil, sistem memproses bingkai untuk prediksi emosi.

3. **Prediksi Emosi:**
   - Menggunakan model yang telah dilatih sebelumnya untuk memprediksi emosi berdasarkan gambar yang diproses.
   - Berbagai emosi diprediksi, seperti anger, sad, disgust, fear, neutral, happy, surprise.

4. **Rekomendasi Musik:**
   - Memetakan emosi yang diprediksi ke suasana hati yang sesuai (misalnya: bahagia, sedih, energik).
   - Merekomendasikan musik yang sesuai berdasarkan suasana hati yang terdeteksi dari kumpulan data musik yang telah ditentukan sebelumnya.

5. **Menampilkan Hasil:**
   - Menampilkan gambar yang diunggah/diambil dan emosi yang diprediksi kepada pengguna.
   - Menampilkan lagu-lagu yang direkomendasikan berdasarkan emosi yang diprediksi untuk pengguna untuk menjelajahi.

6. **Interaksi Pengguna (Lanjutan):**
   - Pengguna dapat mendengarkan musik yang direkomendasikan, membuka tautan Spotify, dan memperbarui rekomendasi musik jika diinginkan.

## Tujuan

- Mengembangkan model deteksi emosi yang akurat dan efisien menggunakan teknik deep learning.
- Menciptakan antarmuka pengguna yang lancar dan intuitif untuk pengalaman pengguna yang lebih baik.
- Memberikan rekomendasi musik yang menarik dan personal berdasarkan emosi yang terdeteksi.
- Memungkinkan pengguna untuk dengan mudah menjelajahi dan menikmati musik yang direkomendasikan.

