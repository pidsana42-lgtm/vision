#!/usr/bin/env python3
"""
clean_multiface.py — สแกนวิดีโอทั้งหมดใน DATASET/videos/
ถ้าเจอวิดีโอที่มีคนมากกว่า 1 คน จะทำการลบทิ้งทั้งไฟล์วิดีโอ, ฟีเจอร์ (.npy) และเอาออกจาก labels.csv
"""

import os
import cv2
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "DATASET", "videos")
FEATURES_DIR = os.path.join(BASE_DIR, "DATASET", "features")
DATASET_CSV = os.path.join(BASE_DIR, "DATASET", "labels.csv")

def check_multiple_faces(video_path: str, face_mesh) -> bool:
    """เช็คว่ามีคนมากกว่า 1 คนในเฟรมไหนสักเฟรมหรือไม่"""
    cap = cv2.VideoCapture(video_path)
    has_multiple_faces = False
    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 1:
            has_multiple_faces = True
            break
            
    cap.release()
    return has_multiple_faces

def main():
    if not os.path.exists(VIDEOS_DIR):
        print("❌ ไม่พบโฟลเดอร์ DATASET/videos/")
        return

    videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(('.mp4', '.mov', '.avi'))]
    if not videos:
        print("✅ ไม่มีวิดีโอในระบบ")
        return

    print(f"\n🔍 กำลังสแกนวิดีโอทั้งหมด {len(videos)} ไฟล์เพื่อหาคลิปที่มีคนมากกว่า 1 คน...")
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=5, refine_landmarks=False,
        min_detection_confidence=0.4, min_tracking_confidence=0.4
    )

    to_delete = []
    
    # วนลูปเช็คทีละคลิป
    for vname in tqdm(videos, desc="Scanning"):
        vpath = os.path.join(VIDEOS_DIR, vname)
        if check_multiple_faces(vpath, face_mesh):
            to_delete.append(vname)

    face_mesh.close()

    if not to_delete:
        print("\n✅ ยินดีด้วย! Dataset ของคุณสะอาดกริ๊บ ไม่มีคลิปไหนที่มีคนเกิน 1 คนเลยครับ")
        return

    print(f"\n🗑️ พบวิดีโอที่มีคนมากกว่า 1 คน จำนวน {len(to_delete)} ไฟล์")
    print("กำลังทำการลบไฟล์วิดีโอ, .npy และอัปเดต labels.csv...")

    # ลบไฟล์
    for vname in to_delete:
        vpath = os.path.join(VIDEOS_DIR, vname)
        fpath = os.path.join(FEATURES_DIR, os.path.splitext(vname)[0] + ".npy")
        
        if os.path.exists(vpath): os.remove(vpath)
        if os.path.exists(fpath): os.remove(fpath)
        print(f"  - ลบ {vname}")

    # อัปเดต CSV
    if os.path.exists(DATASET_CSV):
        df = pd.read_csv(DATASET_CSV)
        original_count = len(df)
        df = df[~df.iloc[:, 0].isin(to_delete)]  # เอาแถวที่มีชื่อวิดีโอที่โดนลบออก
        df.to_csv(DATASET_CSV, index=False)
        print(f"\n✅ อัปเดต labels.csv เสร็จสิ้น (ลดจาก {original_count} เหลือ {len(df)} แถว)")

    print("🎉 ทำความสะอาด Dataset เสร็จสมบูรณ์!")

if __name__ == "__main__":
    main()
