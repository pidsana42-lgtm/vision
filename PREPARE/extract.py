#!/usr/bin/env python3
"""
extract.py — Phase 2: สกัดฟีเจอร์ Mouth ROI (Auto-AVSR format) + กรองคุณภาพ
"""

import os, sys, shutil, time, argparse
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from huggingface_hub import HfApi

# ─────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "Phonsiri/Thai-Lip-Reading-Dataset"

if not HF_TOKEN:
    print('⚠️  ไม่พบ HF_TOKEN — พุช HF จะถูกข้าม')

PREPARE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(PREPARE_DIR)

RAW_VIDEOS = os.path.join(PROJECT_DIR, "DATASET", "raw", "videos")
RAW_CSV    = os.path.join(PROJECT_DIR, "DATASET", "raw", "labels.csv")

READY_DIR      = os.path.join(PROJECT_DIR, "DATASET", "ready")
READY_VIDEOS   = os.path.join(READY_DIR, "videos")        # สำหรับเก็บ .mp4 ที่ crop แล้ว
READY_CSV      = os.path.join(READY_DIR, "labels.csv")

# ─────────────────────────────────────────────────────────
#  AUTO-AVSR CROPPER
# ─────────────────────────────────────────────────────────
def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks

class MouthCropper:
    def __init__(self, mean_face_path, crop_width=96, crop_height=96, window_margin=12):
        self.reference = np.load(mean_face_path)
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_margin = window_margin
        
        # 4 จุดอ้างอิงจาก 20words_mean_face.npy (68 points): ตาขวา, ตาซ้าย, จมูก, ปาก
        self.stable_reference = np.vstack([
            np.mean(self.reference[36:42], axis=0),
            np.mean(self.reference[42:48], axis=0),
            np.mean(self.reference[31:36], axis=0),
            np.mean(self.reference[48:68], axis=0),
        ])

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
        if not valid_frames_idx: return None
        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        return landmarks

    def crop_video(self, video_frames, landmarks):
        landmarks = self.interpolate_landmarks(landmarks)
        if not landmarks: return None
        
        sequence = []
        for frame_idx, frame in enumerate(video_frames):
            margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - margin, frame_idx + margin + 1)],
                axis=0
            )
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            
            # Affine Transform ให้ใบหน้าตรงตาม reference
            transform = cv2.estimateAffinePartial2D(
                smoothed_landmarks, self.stable_reference, method=cv2.LMEDS
            )[0]
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            transformed_frame = cv2.warpAffine(
                gray, transform, dsize=(256, 256),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            
            # หาจุดศูนย์กลางปากใหม่หลัง transform (จุดที่ 3 คือปาก)
            transformed_landmarks = np.matmul(smoothed_landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
            center_x, center_y = transformed_landmarks[3]
            
            height, width = self.crop_height // 2, self.crop_width // 2
            y_min = int(round(np.clip(center_y - height, 0, 256)))
            y_max = int(round(np.clip(center_y + height, 0, 256)))
            x_min = int(round(np.clip(center_x - width, 0, 256)))
            x_max = int(round(np.clip(center_x + width, 0, 256)))
            
            patch = transformed_frame[y_min:y_max, x_min:x_max]
            
            # ป้องกัน patch เล็กกว่าที่กำหนด (ติดขอบภาพ)
            if patch.shape != (self.crop_height, self.crop_width):
                patch = cv2.resize(patch, (self.crop_width, self.crop_height))
                
            sequence.append(patch)
        return sequence

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def log(emoji, msg):  print(f"\n{emoji}  {msg}")
def banner(title):    print("\n" + "═"*60 + f"\n  {title}\n" + "═"*60)

def extract_one(video_path: str, detector, cropper: MouthCropper):
    cap = cv2.VideoCapture(video_path)
    frames = []
    landmarks = []
    multi_face_count = 0
    lip_openings = []

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame)
        
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            landmarks.append(None)
        else:
            if len(results.detections) > 1:
                multi_face_count += 1
                
            ih, iw, _ = frame.shape
            max_id, max_size = 0, 0
            
            # เลือกหน้าที่มีขนาดใหญ่ที่สุด
            for idx, face in enumerate(results.detections):
                bboxC = face.location_data.relative_bounding_box
                bbox_size = bboxC.width * iw + bboxC.height * ih
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
            
            face = results.detections[max_id]
            kpts = face.location_data.relative_keypoints
            # MediaPipe FaceDetection keypoints: 0=RightEye, 1=LeftEye, 2=NoseTip, 3=MouthCenter
            lmx = np.array([
                [kpts[0].x * iw, kpts[0].y * ih],
                [kpts[1].x * iw, kpts[1].y * ih],
                [kpts[2].x * iw, kpts[2].y * ih],
                [kpts[3].x * iw, kpts[3].y * ih]
            ])
            landmarks.append(lmx)
            
            # ประมาณการเปิดปากจาก bbox (เนื่องจากไม่มีจุดขอบปากชัดเจน)
            # เราใช้ confidence ว่าพูดหรือไม่พูดคร่าวๆ จากการเปลี่ยนแปลงของ MouthCenter.y เทียบกับ NoseTip.y
            lip_openings.append(abs((kpts[3].y * ih) - (kpts[2].y * ih)))

    cap.release()
    
    if len(frames) == 0: return None, 0.0, 0
    
    lip_var = float(np.var(lip_openings)) if lip_openings else 0.0
    sequence = cropper.crop_video(frames, landmarks)
    
    return sequence, lip_var, multi_face_count

def check_caption(caption: str, duration: float) -> str | None:
    words = caption.strip().split()
    n_words = len(words)
    if n_words < 2: return f"caption สั้นเกิน ({n_words} คำ)"
    if duration > 0:
        wps = n_words / duration
        if wps > 12: return f"WPS สูงเกิน ({wps:.1f} คำ/วิ)"
        if wps < 0.3 and n_words > 2: return f"WPS ต่ำเกิน ({wps:.1f} คำ/วิ)"
    return None

def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps if fps > 0 else 0.0

# ─────────────────────────────────────────────────────────
#  MAIN: สกัดฟีเจอร์ทั้งหมด
# ─────────────────────────────────────────────────────────
def run_extract(no_push: bool = False):
    if not os.path.exists(RAW_CSV):
        log("❌", f"ไม่พบ {RAW_CSV}\n   → กรุณารัน Phase 1 ก่อน")
        return

    os.makedirs(READY_VIDEOS, exist_ok=True)
    mean_face = os.path.join(PREPARE_DIR, "20words_mean_face.npy")
    if not os.path.exists(mean_face):
        log("❌", "ไม่พบ 20words_mean_face.npy ในโฟลเดอร์ PREPARE/")
        return

    df_raw = pd.read_csv(RAW_CSV)
    df_raw.columns = ["video", "caption"]
    
    done = {f for f in os.listdir(READY_VIDEOS) if f.endswith(".mp4")}
    todo = [row for _, row in df_raw.iterrows() if row["video"] not in done]

    banner("🧠 Phase 2: Extract Mouth ROI (96x96) → DATASET/ready/")
    print(f"  📦 Raw Dataset : {len(df_raw)} clips")
    print(f"  ✅ ทำแล้ว      : {len(done)} clips (resume)")
    print(f"  🔄 จะประมวลผล  : {len(todo)} clips")
    print("═" * 60)

    detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
    cropper = MouthCropper(mean_face_path=mean_face)

    accepted = []
    rejected_counts = {"no_face": 0, "caption": 0}

    for i, row in enumerate(todo, 1):
        vname = row["video"]
        caption = str(row["caption"]).strip()
        vpath = os.path.join(RAW_VIDEOS, vname)
        out_path = os.path.join(READY_VIDEOS, vname)

        print(f"\n  [{i:>3}/{len(todo)}] {vname}")
        if not os.path.exists(vpath): continue

        duration = get_video_duration(vpath)
        cap_err = check_caption(caption, duration)
        if cap_err:
            print(f"         ❌ Caption: {cap_err}")
            rejected_counts["caption"] += 1
            continue

        sequence, lip_var, mf_count = extract_one(vpath, detector, cropper)

        if sequence is None or len(sequence) == 0:
            print(f"         ❌ ไม่พบใบหน้า / ไม่สามารถ crop ได้")
            rejected_counts["no_face"] += 1
            continue

        # Save MP4 96x96 grayscale
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, 25.0, (96, 96), isColor=False)
        for frame in sequence:
            out.write(frame)
        out.release()

        mf_note = f" | multi-face={mf_count}" if mf_count > 0 else ""
        print(f"         ✅ shape=({len(sequence)}, 96, 96){mf_note}")
        accepted.append({"video": vname, "caption": caption})

    detector.close()

    # ── บันทึก labels.csv ──
    if accepted:
        new_df = pd.DataFrame(accepted)
        if os.path.exists(READY_CSV) and os.path.getsize(READY_CSV) > 0:
            old_df = pd.read_csv(READY_CSV)
            old_df.columns = ["video", "caption"]
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["video"], keep="last")
        else:
            combined = new_df
        combined.to_csv(READY_CSV, index=False)
    else:
        combined = pd.read_csv(READY_CSV) if os.path.exists(READY_CSV) else pd.DataFrame()

    banner("🏁 Phase 2 เสร็จสมบูรณ์!")
    print(f"  ✅ ผ่านการกรอง     : {len(accepted)} clips ใหม่")
    print(f"  ❌ ถูกกรองออก      : {sum(rejected_counts.values())} clips")
    print(f"  📊 DATASET/ready   : {len(combined)} clips พร้อมนำไปเข้า Auto-AVSR")

    if not no_push and HF_TOKEN:
        print()
        _push_to_hf()

def _push_to_hf():
    log("🚀", f"กำลังพุชขึ้น HuggingFace: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_folder(
            folder_path=READY_DIR,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo="dataset",
            commit_message=f"Phase 2: อัปเดต Dataset (Mouth ROI 96x96)",
            allow_patterns=["videos/*", "labels.csv"],
        )
        log("✅", f"พุชสำเร็จ → https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        log("❌", f"พุชล้มเหลว: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Extract Mouth ROI (96x96) → DATASET/ready/")
    parser.add_argument("--no-push",  action="store_true", help="ข้ามการพุชขึ้น HuggingFace")
    args = parser.parse_args()
    run_extract(no_push=args.no_push)
