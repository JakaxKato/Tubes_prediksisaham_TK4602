import pandas as pd

# Membaca dataset dari file CSV
file_path = "diabetes_dataset.csv"  
df = pd.read_csv(file_path)

# Menampilkan informasi dataset
print("Informasi Dataset:")
print(df.info())

# Menampilkan data awal
print("\nData Awal:")
print(df.head())

# Contoh analisis sederhana
print("\nStatistik Deskriptif:")
print(df.describe())

# Menyimpan DataFrame yang telah diproses (jika diperlukan)
df.to_csv("processed_diabetes_dataset.csv", index=False)
print("\nDataset telah disimpan sebagai 'processed_diabetes_dataset.csv'")
