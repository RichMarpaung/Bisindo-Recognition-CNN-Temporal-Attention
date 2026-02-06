import os
import cv2
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# KONFIGURASI PATH (SESUAIKAN DI SINI)
# ==========================================
SOURCE_PATH = r"C:\Users\robot\OneDrive\Documents\Rich\raw_video"       
TARGET_PATH = r"C:\Users\robot\OneDrive\Documents\Rich\processed_data_uint8" 

# Config Gambar
IMG_SIZE = 112
MAX_FRAMES = 20

def process_single_video(args):
    """Fungsi ini berjalan paralel di banyak core CPU"""
    src_path, tgt_path = args
    
    # Skip jika sudah ada (Resume capability)
    if os.path.exists(tgt_path):
        return 0 

    cap = cv2.VideoCapture(src_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            # Resize
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            # Simpan mentah 0-255 (uint8) biar hemat storage & RAM
            frames.append(frame) 
    finally:
        cap.release()
    
    frames = np.array(frames, dtype=np.uint8)
    
    # Sampling / Padding Frames ke 20
    total_frames = len(frames)
    if total_frames == 0:
        final_data = np.zeros((MAX_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    elif total_frames >= MAX_FRAMES:
        indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int)
        final_data = frames[indices]
    else:
        padding = np.zeros((MAX_FRAMES - total_frames, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        final_data = np.concatenate([frames, padding], axis=0)
        
    # Simpan ke SSD/HDD
    np.save(tgt_path, final_data)
    return 1 

def main_conversion():
    if not os.path.exists(TARGET_PATH):
        os.makedirs(TARGET_PATH)
        
    # 1. Scanning Folder
    print(f"Scanning Folder: {SOURCE_PATH}")
    tasks = []
    
    if not os.path.exists(SOURCE_PATH):
        print("ERROR: Path Source tidak ditemukan!")
        return

    classes = [d for d in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, d))]
    
    for label in classes:
        src_folder = os.path.join(SOURCE_PATH, label)
        tgt_folder = os.path.join(TARGET_PATH, label)
        os.makedirs(tgt_folder, exist_ok=True)
        
        videos = [f for f in os.listdir(src_folder) if f.lower().endswith('.mp4')]
        for vid in videos:
            s_path = os.path.join(src_folder, vid)
            # Ubah nama file .mp4 -> .npy
            t_path = os.path.join(tgt_folder, os.path.splitext(vid)[0] + ".npy")
            tasks.append((s_path, t_path))
            
    print(f"Total video terdeteksi: {len(tasks)}")
    
    # 2. EKSEKUSI PARALEL (Memanfaatkan i7-12700)
    print("Mulai konversi multi-core... (Kipas CPU akan kencang)")
    start_time = time.time()
    
    # Gunakan max_workers=16 agar sisa thread untuk OS aman
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_single_video, tasks))
        
    duration = time.time() - start_time
    print(f"\nSELESAI! {sum(results)} video baru diproses dalam {duration:.2f} detik.")
    print(f"Lokasi data siap pakai: {TARGET_PATH}")

if __name__ == '__main__':
    main_conversion()