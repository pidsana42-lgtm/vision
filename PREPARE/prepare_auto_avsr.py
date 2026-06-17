#!/usr/bin/env python3
"""
เตรียมไฟล์ labels.csv ให้เข้ากันได้กับ Auto-AVSR format
Format ของ Auto-AVSR: dataset,basename,video_length,token_id_str
"""
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, "auto_avsr"))
import cv2
import pandas as pd

from vocabulary import ThaiTokenizer

def get_video_length(video_path):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ready_csv = os.path.join(base_dir, "DATASET", "ready", "labels.csv")
    ready_vid_dir = os.path.join(base_dir, "DATASET", "ready", "videos")
    out_csv = os.path.join(base_dir, "DATASET", "ready", "auto_avsr_train.csv")

    if not os.path.exists(ready_csv):
        print(f"❌ ไม่พบ {ready_csv} กรุณารัน extract.py ก่อน")
        return

    df = pd.read_csv(ready_csv)
    tokenizer = ThaiTokenizer()

    out_data = []
    for _, row in df.iterrows():
        vid_name = row["video"]
        caption = str(row["caption"]).strip()
        
        vid_path = os.path.join(ready_vid_dir, vid_name)
        if not os.path.exists(vid_path):
            continue
            
        n_frames = get_video_length(vid_path)
        
        # Tokenize (ใช้ระดับตัวอักษรภาษาไทยจาก vocabulary.py)
        tokens = tokenizer.encode(caption)
        token_id_str = " ".join(map(str, tokens))
        
        # auto-avsr format
        dataset_name = "thai_vsr"
        basename = os.path.join("videos", vid_name) # relative to DATASET/ready
        
        out_data.append({
            "dataset": dataset_name,
            "basename": basename,
            "video_length": n_frames,
            "token_id_str": token_id_str
        })
        
    out_df = pd.DataFrame(out_data)
    out_df.to_csv(out_csv, index=False, header=False) # Auto-AVSR ไม่ใช้ header
    
    print(f"✅ แปลงไฟล์สำเร็จ: {out_csv}")
    print(f"  ข้อมูลทั้งหมด: {len(out_df)} รายการ")
    print(f"  Vocabulary Size: {tokenizer.get_vocab_size()}")
    print("\n👉 คุณสามารถเริ่มเทรนกับ Auto-AVSR ได้ด้วยคำสั่ง:")
    print(f"   python train.py --root-dir {os.path.join(base_dir, 'DATASET', 'ready')} \\")
    print(f"                   --train-file auto_avsr_train.csv \\")
    print(f"                   --modality video")

if __name__ == "__main__":
    main()
