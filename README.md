# Session 03: GIS Smart City Settlement Suitability Model with Deep Learning

Sistem otomatisasi penentuan kesesuaian lahan pemukiman *Smart City* berbasis PyTorch untuk membantu *GIS Developer* mengklasifikasikan wilayah layak huni berdasarkan variabel geografis dan spasial.

## 1. Dari Materi Pemateri vs Pengembangan Mandiri
* **Materi Pemateri (Widi Prima Deva):** Membahas arsitektur Cloud GIS, tantangan *GIS Developer*, serta peluang inovasi skripsi berbasis teknologi spasial dan AI.
* **Pengembangan Mandiri:** Mengembangkan model *GeoAI* (PyTorch MLP) untuk mengklasifikasikan kesesuaian zona pemukiman perkotaan secara otomatis berdasarkan integrasi multivariat spasial.

## 2. Masalah dan Tujuan
Analisis kesesuaian lahan manual pada GIS memakan waktu dan sering kali kaku terhadap hubungan variabel non-linear. Penelitian ini bertujuan membangun model klasifikasi cerdas untuk memetakan kesesuaian wilayah pemukiman berbasis atribut spasial secara presisi.

## 3. Spesifikasi Dataset
* **Sumber:** Dataset Sintetik (`dataset/gis_smartcity_synthetic.csv`) yang disintesis melalui skrip `dataset/generate_dataset.py`.
* **Ukuran:** 2.000 sampel spasial.
* **Fitur Input:** `jarak_pusat_kota`, `elevasi_tanah`, `tutupan_vegetasi`, `indeks_risiko_banjir`, `akses_jalan`.
* **Target:** `layak_huni` (Binary: `0` = Berisiko Tinggi, `1` = Layak/Aman).

## 4. Metodologi & Pencegahan Leakage
1. **Pencegahan Data Leakage:** Splitting data Train/Val/Test (70/15/15) dilakukan terlebih dahulu. `StandardScaler` di-fit **hanya** pada data Train, lalu ditransformasikan ke Val dan Test.
2. **Penanganan Imbalanced Data:** Memakai `pos_weight` pada `BCEWithLogitsLoss` serta penyesuaian threshold keputusan optimal menggunakan data Validation.
3. **Early Stopping:** Diatur berdasarkan evaluasi Validation Loss (patience=10) untuk menghindari *overfitting*.
4. **Reproducibility:** Menjalankan eksperimen pada 3 Seed acak (`42`, `123`, `999`).

## 5. Arsitektur / Alur Sistem
[Atribut Spasial GIS (Elevasi, Banjir, Jalan, dll)]->[StandardScaler (Fit Train)]->[PyTorch MLP: 5 -> 32 (BatchNorm+ReLU+Dropout) -> 16 -> 1]->[Threshold Tuning (Validation Set)]->[Prediksi Spasial: Berisiko / Layak Huni]

## 6. Cara Menjalankan Perintah
**Di Vscode (PowerShell / Command Prompt):**
```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python dataset/generate_dataset.py

# 4. Jalankan eksperimen Baseline (Random Forest)
python src/01_baseline.py

# 5. Jalankan eksperimen Deep Learning (PyTorch)
python src/02_train_dl.py