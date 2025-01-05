import os
import tkinter as tk
from tkinter import messagebox
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime

# Ganti direktori kerja ke D:\web
new_directory = "/Applications/learnpython/fix"  # Ganti dengan direktori yang sesuai
os.chdir(new_directory)
print("Direktori saat ini:", os.getcwd())  # Pastikan direktori telah berubah

# Inisialisasi GUI Tkinter
root = tk.Tk()
root.title("Stock Price Predictor")
root.geometry("800x600")  # Ukuran jendela aplikasi

# Frame Tampilan Awal
frame_start = tk.Frame(root)
frame_start.pack(fill="both", expand=True)

# Frame untuk memasukkan input
frame_inputs = tk.Frame(frame_start, bd=2, relief="solid", padx=20, pady=20)  # Tambahkan border pada frame
frame_inputs.place(relx=0.5, rely=0.5, anchor="center")  # Posisi frame di tengah

# Label dan Entry untuk ticker saham
tk.Label(frame_inputs, text="Stock Ticker Symbol:", font=("Arial", 14)).pack(pady=5, anchor="w")
entry_ticker = tk.Entry(frame_inputs, font=("Arial", 14), width=20)
entry_ticker.pack(pady=5)

# Label dan Entry untuk tanggal mulai
tk.Label(frame_inputs, text="Start Date (YYYY-MM-DD):", font=("Arial", 14)).pack(pady=5, anchor="w")
entry_start_date = tk.Entry(frame_inputs, font=("Arial", 14), width=20)
entry_start_date.pack(pady=5)

# Label dan Entry untuk tanggal akhir
tk.Label(frame_inputs, text="End Date (YYYY-MM-DD):", font=("Arial", 14)).pack(pady=5, anchor="w")
entry_end_date = tk.Entry(frame_inputs, font=("Arial", 14), width=20)
entry_end_date.pack(pady=5)

# Tombol untuk memulai prediksi
btn_predict = tk.Button(frame_inputs, text="Predict Stock Price", command=lambda: predict_stock_price(), font=("Arial", 14))
btn_predict.pack(pady=10)

# Frame Tampilan Grafik
frame_chart = tk.Frame(root)

# Variabel untuk menyimpan hasil prediksi
result_text = tk.StringVar()

# Fungsi untuk berpindah ke frame grafik
def show_chart_frame():
    frame_start.pack_forget()
    frame_chart.pack(fill="both", expand=True)

    # Tambahkan tombol "Back" untuk kembali ke frame awal
    btn_back = tk.Button(frame_chart, text="Back", command=back_to_start_frame, font=("Arial", 12))
    btn_back.pack(side="bottom", pady=50)

# Fungsi untuk kembali ke tampilan awal
def back_to_start_frame():
    # Hapus semua widget di frame_chart untuk membersihkan layar
    for widget in frame_chart.winfo_children():
        widget.destroy()
    
    # Tampilkan frame awal
    frame_chart.pack_forget()
    frame_start.pack(fill="both", expand=True)

# Fungsi untuk mendapatkan mata uang berdasarkan ticker saham dari yfinance
def get_currency_by_ticker(ticker):
    try:
        stock_info = yf.Ticker(ticker).info  # Ambil informasi saham dari yfinance
        currency = stock_info.get("currency", "USD")  # Ambil mata uang atau default ke USD
        return currency
    except Exception as e:
        print(f"Error fetching currency for ticker {ticker}: {e}")
        return "USD"  # Default ke USD jika terjadi error

# Fungsi untuk menyimpan hasil prediksi ke file CSV (dengan tab sebagai pemisah)
def save_results_to_csv(data, next_day_pred, mse):
    file_name = os.path.join(os.getcwd(), "predicted_stock_price.tsv")  # Menggunakan ekstensi .tsv
    ticker = entry_ticker.get().upper()
    currency = get_currency_by_ticker(ticker)
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)  # Tambahkan 1 hari untuk prediksi keesokan harinya
    if next_day.weekday() == 5:  # Sabtu
        next_day += pd.Timedelta(days=2)
    elif next_day.weekday() == 6:  # Minggu
        next_day += pd.Timedelta(days=1)
    
    # Mendapatkan waktu percobaan
    experiment_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Membuat DataFrame dengan header yang sesuai
    results = pd.DataFrame({
        "DATE": [next_day.strftime('%Y-%m-%d')],
        "STOCK": [ticker],
        "CURRENCY": [currency],
        "PREDICTED": [next_day_pred.item()],
        "EXPRERIMENT TIME": [experiment_time]  # Menambahkan waktu percobaan
    })
    
    # Memeriksa apakah file TSV sudah ada
    if not os.path.exists(file_name):
        # Jika file belum ada, buat file dan tambahkan header
        results.to_csv(file_name, mode="w", sep="\t", header=True, index=False)
        print(f"File '{file_name}' created with header.")
    else:
        # Jika file sudah ada, tambahkan data tanpa header
        results.to_csv(file_name, mode="a", sep="\t", header=False, index=False)
        print(f"Results appended to '{file_name}'.")

# Fungsi untuk memprediksi harga saham
def predict_stock_price():
    ticker = entry_ticker.get().upper()
    start_date = entry_start_date.get()
    end_date = entry_end_date.get()
    if not ticker or not start_date or not end_date:
        messagebox.showwarning("Input Error", "Please enter a valid stock ticker and date range.")
        return
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            messagebox.showerror("Error", f"No data found for ticker symbol '{ticker}' in the given date range.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data for ticker symbol '{ticker}'.\nError: {e}")
        return
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean() 
    data.dropna(inplace=True)
    X = data[['SMA_10', 'SMA_30']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    last_10 = data['SMA_10'].iloc[-1]
    last_30 = data['SMA_30'].iloc[-1]
    next_day_pred = model.predict([[last_10, last_30]])
    currency = get_currency_by_ticker(ticker)
    last_date = data.index[-1]
    next_day = last_date + pd.Timedelta(days=1)
    if next_day.weekday() == 5:  # Sabtu
        next_day += pd.Timedelta(days=2)
    elif next_day.weekday() == 6:  # Minggu
        next_day += pd.Timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    result_text.set(
        f"Day Prediction: {next_day_str}\n"
        f"Predicted Closing Price: {currency} {next_day_pred.item():,.2f}\n"
        f"Mean Squared Error: {mse:,.2f}"
    )
    plot_chart(data, next_day_pred, mse, next_day_str, currency)
    save_results_to_csv(data, next_day_pred, mse)
    show_chart_frame()

# Fungsi untuk menampilkan chart harga penutupan
def plot_chart(data, next_day_pred, mse, next_day_str, currency):
    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data.index, data['Close'], label="Closing Price", color="blue")
    ax.plot(data.index, data['SMA_10'], label="10-Day SMA", color="orange")
    ax.plot(data.index, data['SMA_30'], label="30-Day SMA", color="green")
    ax.set_title("Stock Price and Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    
    # Menambahkan nama saham di atas grafik
    stock_ticker = entry_ticker.get().upper()
    ax.text(
        0.5, 1.05, f"Stock: {stock_ticker}", ha='center', va='bottom', fontsize=14,
        color="black", transform=ax.transAxes
    )
    
    prediction_text = (
        f"Next Day Prediction: {next_day_str}\n"
        f"Predicted Closing Price: {currency} {next_day_pred.item():,.2f}\n"
        f"Mean Squared Error: {mse:,.2f}"
    )
    plt.subplots_adjust(bottom=0.25)
    ax.text(
        0.5, -0.15, prediction_text, ha='center', va='top', fontsize=12, 
        color="blue", transform=ax.transAxes
    )
    
    # Membuat direktori CHART jika belum ada
    chart_directory = r"D:\web\CHART"
    if not os.path.exists(chart_directory):
        os.makedirs(chart_directory)

    # Menyimpan grafik ke file JPG dengan nama stock_tanggal_predict
    image_filename = os.path.join(chart_directory, f"{stock_ticker}_{next_day_str}_predict.jpg")
    fig.savefig(image_filename, format="jpg")
    print(f"Chart saved as {image_filename}")

    # Tambahkan tombol untuk mengunduh gambar
    download_button = tk.Button(frame_chart, text="Download Chart", command=lambda: download_chart(image_filename), font=("Arial", 12))
    download_button.pack(side="bottom", pady=10)
    
    # Display the chart in the Tkinter GUI
    for widget in frame_chart.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=frame_chart)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Fungsi untuk mendownload gambar
def download_chart(image_filename):
    try:
        # Untuk mendownload gambar, kita hanya mengasumsikan pengguna dapat mengakses file langsung
        # pada file yang telah disimpan.
        messagebox.showinfo("Download", f"The chart has been saved as '{image_filename}'. You can access the file.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to download chart.\nError: {e}")


root.mainloop()
