import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar menggunakan matplotlib
def show_image(title, image, cmap_type):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Fungsi untuk memproses gambar
def process_image(image_path, output_path):
    # Baca gambar dari file
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Tidak dapat membaca gambar dari {image_path}")
        return

    # Konversi gambar dari BGR (format OpenCV) ke RGB (format umum)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Konversi gambar ke grayscale
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Lakukan histogram equalization pada gambar grayscale
    image_eq = cv2.equalizeHist(image_gray)

    # Simpan gambar hasil histogram equalization
    cv2.imwrite(output_path, image_eq)

# Direktori dataset
base_dir = r'D:\Kuliah Semester 6\Jurnal\Materi\CNN\Dataset'

# Loop melalui setiap folder dalam direktori dataset
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Buat folder output jika belum ada
        output_folder_path = os.path.join(base_dir, 'processed', folder)
        os.makedirs(output_folder_path, exist_ok=True)

        # Loop melalui setiap file dalam folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                # Path untuk menyimpan hasil
                output_file_path = os.path.join(output_folder_path, filename)
                # Proses gambar dan simpan hasilnya
                process_image(file_path, output_file_path)

print("Proses selesai. Hasil disimpan di folder 'processed' dalam direktori dataset.")
