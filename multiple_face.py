import os
import time
import csv
from collections import defaultdict

# Direktori dataset
crop_dir = "/Users/taufiq/workspace/infran-spark/data_inference"
csv_file_path = "log_multiple_faces.csv"

# Inisialisasi log untuk mencatat kelas dan path file dengan hasil crop lebih dari satu wajah
log_multiple_faces = []

# Mendapatkan daftar kelas (folder) dalam direktori crop
class_names = sorted([d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d))])
total_classes = len(class_names)

# Mulai pemrosesan
start_time = time.time()
print("Proses dimulai...")

for idx, class_name in enumerate(class_names, start=1):
    class_path = os.path.join(crop_dir, class_name)
    face_count = defaultdict(list)  # Simpan path file untuk setiap gambar dasar
    
    # Memproses setiap file dalam folder kelas
    for file_name in os.listdir(class_path):
        if "_face_" in file_name and file_name.endswith(".jpg"):
            base_name = "_".join(file_name.split("_")[:-2])  # Mengambil nama dasar gambar
            face_count[base_name].append(os.path.join(class_path, file_name))
    
    # Cek apakah ada lebih dari satu wajah dalam satu gambar
    for base_name, paths in face_count.items():
        if len(paths) > 1:
            for path in paths:
                log_multiple_faces.append([class_name, path])

    # Cetak progres ke terminal
    elapsed_time = time.time() - start_time
    progress = (idx / total_classes) * 100
    estimated_total_time = (elapsed_time / idx) * total_classes
    remaining_time = estimated_total_time - elapsed_time
    
    print(f"[{progress:.2f}%] Class {class_name} selesai. "
          f"Estimasi selesai dalam {remaining_time:.2f} detik.")

# Tulis log ke file CSV
with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Class Name", "File Path"])  # Header
    csv_writer.writerows(log_multiple_faces)

print("Proses selesai!")
print(f"Log disimpan di {csv_file_path}")
