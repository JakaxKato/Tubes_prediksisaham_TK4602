import pandas as pd

# Membaca dataset (ganti 'diabetes.csv' dengan lokasi file dataset kamu)
data = pd.read_csv('diabetes.csv')

# Menampilkan beberapa baris awal dataset
print("Data Sample:")
print(data.head())

# Menampilkan statistik deskriptif
print("\nStatistik Deskriptif:")
print(data.describe())
