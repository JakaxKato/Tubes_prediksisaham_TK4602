import tkinter as tk
from tkinter import messagebox
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Inisialisasi GUI Tkinter
root = tk.Tk()
root.title("Stock Price Predictor")
root.geometry("800x600")

# Frame Tampilan Awal
frame_start = tk.Frame(root)
frame_start.pack(fill="both", expand=True)

# Frame Tampilan Grafik
frame_chart = tk.Frame(root)

# Variabel untuk menyimpan hasil prediksi
result_text = tk.StringVar()

# Fungsi untuk berpindah ke frame grafik
def show_chart_frame():
    frame_start.pack_forget()
    frame_chart.pack(fill="both", expand=True)

# Fungsi untuk memprediksi harga saham
def predict_stock_price():
    ticker = entry_ticker.get().upper()
    if not ticker:
        messagebox.showwarning("Input Error", "Please enter a valid stock ticker symbol.")
        return
    
    # Ambil data saham dari yfinance
    try:
        data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
        if data.empty:
            messagebox.showerror("Error", f"No data found for ticker symbol '{ticker}'")
            return
        else:
            print(data.head())  # Tampilkan beberapa baris pertama di console untuk verifikasi
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data for ticker symbol '{ticker}'.\nError: {e}")
        return
    
    # Persiapan data
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data.dropna(inplace=True)
    
    X = data[['SMA_10', 'SMA_30']]
    y = data['Close']
    
    # Pisahkan data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Buat model regresi linear
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediksi data testing
    predictions = model.predict(X_test)
    
    # Hitung Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    
    # Prediksi harga penutupan berikutnya
    last_10 = data['SMA_10'].iloc[-1]
    last_30 = data['SMA_30'].iloc[-1]
    next_day_pred = model.predict([[last_10, last_30]])
    
    # Tampilkan hasil prediksi
    result_text.set(f"Predicted Closing Price: ${next_day_pred[0]:.2f}\nMean Squared Error: {mse:.2f}")
    
    # Tampilkan grafik di frame
    plot_chart(data)
    
    # Pindah ke frame chart
    show_chart_frame()

# Fungsi untuk menampilkan chart harga penutupan
def plot_chart(data):
    plt.clf()
    
    # Buat grafik baru
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data['Close'], label="Closing Price", color="blue")
    ax.plot(data.index, data['SMA_10'], label="10-Day SMA", color="orange")
    ax.plot(data.index, data['SMA_30'], label="30-Day SMA", color="green")
    ax.set_title("Stock Price and Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    # Hapus chart sebelumnya (jika ada)
    for widget in frame_chart.winfo_children():
        widget.destroy()
    
    # Tampilkan chart di Tkinter menggunakan FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=frame_chart)
    canvas.draw()
    canvas.get_tk_widget().pack()

# ==================== Tampilan Awal (Frame) ====================

# Label dan Entry untuk ticker saham
tk.Label(frame_start, text="Stock Ticker Symbol:", font=("Arial", 14)).pack(pady=20)
entry_ticker = tk.Entry(frame_start, font=("Arial", 14))
entry_ticker.pack(pady=10)

# Tombol untuk memulai prediksi
btn_predict = tk.Button(frame_start, text="Predict Stock Price", command=predict_stock_price, font=("Arial", 14))
btn_predict.pack(pady=20)

# ==================== Tampilan Grafik (Frame) ====================

# Label untuk menampilkan hasil prediksi di frame grafik
result_label = tk.Label(frame_chart, textvariable=result_text, font=("Arial", 14), fg="blue")
result_label.pack(pady=10)

# Tombol untuk kembali ke tampilan awal
def back_to_start_frame():
    frame_chart.pack_forget()
    frame_start.pack(fill="both", expand=True)

btn_back = tk.Button(frame_chart, text="Back", command=back_to_start_frame, font=("Arial", 12))
btn_back.pack(pady=10)

# Jalankan aplikasi
root.mainloop()
