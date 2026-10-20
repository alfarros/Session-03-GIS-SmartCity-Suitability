import os
import numpy as np
import pandas as pd

# Menyetel seed agar sintesis data spasial dapat diulang dengan hasil sama
np.random.seed(42)

def generate_gis_smartcity_data(n_samples=2000):
    """
    Membuat dataset sintetik variabel spasial GIS untuk kesesuaian lahan pemukiman.
    Target (y): 1 jika Zona Layak Huni (Aman), 0 jika Berisiko Tinggi/Tidak Layak.
    """
    print(f"Membuat {n_samples} sampel data sintetik atribut GIS Smart City...")
    
    # Fitur-fitur Spasial
    jarak_pusat_kota = np.random.uniform(0.5, 25.0, size=n_samples)      # Jarak ke CBD (km)
    elevasi_tanah = np.random.uniform(2.0, 150.0, size=n_samples)        # Ketinggian tanah (meter)
    tutupan_vegetasi = np.random.uniform(0.0, 1.0, size=n_samples)      # Rasio tutupan hijau (NDVI: 0-1)
    indeks_risiko_banjir = np.random.uniform(0.0, 1.0, size=n_samples)  # Skor rawan banjir (0 aman - 1 bahaya)
    akses_jalan = np.random.randint(1, 6, size=n_samples)                # Kerapatan & kualitas jalan (skala 1-5)

    # Logika Asumsi Perencanaan Wilayah (Fungsi Non-Linear dengan Noise)
    # Skor kelayakan tinggi jika: Risiko banjir rendah + Akses jalan bagus + Elevasi aman (tidak terlalu rendah)
    skor_kelayakan = (
        (1.0 - indeks_risiko_banjir) * 3.0 +
        (akses_jalan / 5.0) * 2.5 +
        (np.where(elevasi_tanah > 10.0, 1.0, 0.2)) * 2.0 -
        (jarak_pusat_kota / 25.0) * 1.5 +
        (tutupan_vegetasi * 1.0)
    )
    
    # Menambahkan noise acak variasi kondisi geografis nyata
    noise = np.random.normal(0, 0.6, size=n_samples)
    final_score = skor_kelayakan + noise
    
    # Tentukan threshold agar sekitar 35% wilayah tergolong Layak Huni (imbalanced)
    threshold = np.percentile(final_score, 65)
    layak_huni = (final_score >= threshold).astype(int)

    df = pd.DataFrame({
        'jarak_pusat_kota': jarak_pusat_kota,
        'elevasi_tanah': elevasi_tanah,
        'tutupan_vegetasi': tutupan_vegetasi,
        'indeks_risiko_banjir': indeks_risiko_banjir,
        'akses_jalan': akses_jalan,
        'layak_huni': layak_huni
    })

    os.makedirs('dataset', exist_ok=True)
    filepath = os.path.join('dataset', 'gis_smartcity_synthetic.csv')
    df.to_csv(filepath, index=False)
    print(f"Dataset spasial sintetik berhasil disimpan di: {filepath}")
    print(f"Distribusi Kelas:\n{df['layak_huni'].value_counts(normalize=True)}\n")

if __name__ == '__main__':
    generate_gis_smartcity_data()