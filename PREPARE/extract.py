#!/usr/bin/env python3
"""
extract.py — Phase 2: สกัดฟีเจอร์ + กรองคุณภาพ → DATASET/ready/ → HuggingFace

อ่านข้อมูลจาก DATASET/raw/ แล้วผ่านตัวกรอง 4 ชั้น:
  1. ปากไม่ขยับ (off-screen speaker)  → ลบทิ้ง
  2. เฟรมมีคนมากกว่า 1 คน            → ข้ามเฟรมนั้น
  3. Caption สั้นเกิน (< 2 คำ)         → ลบทิ้ง
  4. WPS ผิดปกติ (> 12 หรือ < 0.3)    → ลบทิ้ง

ผลลัพธ์ที่ผ่านทุกตัวกรองจะถูกบันทึกลง DATASET/ready/ และพุชขึ้น HuggingFace

วิธีใช้:
  python3 PREPARE/extract.py                  # รันปกติ
  python3 PREPARE/extract.py --no-push        # ข้ามการพุช HF
  python3 PREPARE/extract.py --validate       # ตรวจ Dataset ที่ ready/ เท่านั้น
  python3 PREPARE/extract.py --validate --delete  # ตรวจ + ลบจริง

ตั้งค่า Token ก่อนรัน:
  export HF_TOKEN="hf_xxx..."
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
    print('⚠️  ไม่พบ HF_TOKEN — พุช HF จะถูกข้าม (ใช้ --no-push เพื่อปิด warning นี้)')

PREPARE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR   = os.path.dirname(PREPARE_DIR)

# Input: Phase 1 output
RAW_VIDEOS = os.path.join(PROJECT_DIR, "DATASET", "raw", "videos")
RAW_CSV    = os.path.join(PROJECT_DIR, "DATASET", "raw", "labels.csv")

# Output: Phase 2 output (clean, ready to train)
READY_DIR      = os.path.join(PROJECT_DIR, "DATASET", "ready")
READY_VIDEOS   = os.path.join(READY_DIR, "videos")
READY_FEATURES = os.path.join(READY_DIR, "features")
READY_CSV      = os.path.join(READY_DIR, "labels.csv")

# ─────────────────────────────────────────────────────────
#  MEDIAPIPE CONFIG
# ─────────────────────────────────────────────────────────
# 60 Landmarks (ปาก 40 + ขากรรไกร 15 + จมูก 5) × 3 แกน = 180 มิติ
LIPS_INDICES = list(dict.fromkeys([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    152, 148, 176, 149, 150, 136, 172, 58, 288, 397, 365, 379, 378, 400, 377,
    1, 4, 168, 197, 5
]))

# Threshold ตรวจปากขยับ: variance < นี้ = คนหน้ากล้องไม่ได้พูด
LIP_VAR_THRESHOLD = 2e-6

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def log(emoji, msg):  print(f"\n{emoji}  {msg}")
def banner(title):    print("\n" + "═"*60 + f"\n  {title}\n" + "═"*60)

# ─────────────────────────────────────────────────────────
#  CORE: สกัดฟีเจอร์ 1 วิดีโอ
# ─────────────────────────────────────────────────────────
def extract_one(video_path: str, face_mesh) -> tuple[np.ndarray, float, int]:
    """
    คืนค่า (feature_array, lip_variance, multi_face_frame_count)
    - feature_array: shape (n_frames, 180)
    - lip_variance: ความแปรปรวนของการเปิดปากตลอดคลิป
    - multi_face_frame_count: จำนวนเฟรมที่เจอหน้า > 1 คน (ถูกข้ามไป)
    """
    feats, lip_openings = [], []
    multi_face_count = 0

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ok, img = cap.read()
        if not ok: break
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if res.multi_face_landmarks:
            # เฟรมนี้มีหน้าคนมากกว่า 1 → ข้ามเฟรมนี้ ใส่ค่า 0
            if len(res.multi_face_landmarks) > 1:
                multi_face_count += 1
                feats.append([0.0] * (len(LIPS_INDICES) * 3))
            else:
                lm = res.multi_face_landmarks[0]
                lip_openings.append(abs(lm.landmark[13].y - lm.landmark[14].y))
                pts = [c for idx in LIPS_INDICES
                       for c in [lm.landmark[idx].x, lm.landmark[idx].y, lm.landmark[idx].z]]
                feats.append(pts)
        else:
            feats.append([0.0] * (len(LIPS_INDICES) * 3))

    cap.release()
    arr = np.array(feats, dtype=np.float32)
    lip_var = float(np.var(lip_openings)) if lip_openings else 0.0
    return arr, lip_var, multi_face_count


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return n_frames / fps if fps > 0 else 0.0

# ─────────────────────────────────────────────────────────
#  FILTER: ตรวจสอบ caption คุณภาพ
# ─────────────────────────────────────────────────────────
def check_caption(caption: str, duration: float) -> str | None:
    """คืนค่า None ถ้า OK, คืนสาเหตุถ้ามีปัญหา"""
    words = caption.strip().split()
    n_words = len(words)

    if n_words < 2:
        return f"caption สั้นเกิน ({n_words} คำ)"

    if duration > 0:
        wps = n_words / duration
        if wps > 12:
            return f"WPS สูงเกิน ({wps:.1f} คำ/วิ) — {n_words} คำ / {duration:.1f}s"
        if wps < 0.3 and n_words > 2:
            return f"WPS ต่ำเกิน ({wps:.1f} คำ/วิ) — {n_words} คำ / {duration:.1f}s"

    return None

# ─────────────────────────────────────────────────────────
#  MAIN: สกัดฟีเจอร์ทั้งหมด + กรอง + บันทึก
# ─────────────────────────────────────────────────────────
def run_extract(no_push: bool = False):
    # ตรวจสอบ input
    if not os.path.exists(RAW_CSV):
        log("❌", f"ไม่พบ {RAW_CSV}\n   → กรุณารัน Phase 1 ก่อน: python3 PREPARE/collect.py --batch PREPARE/url.txt")
        return
    if not os.path.exists(RAW_VIDEOS):
        log("❌", f"ไม่พบโฟลเดอร์ {RAW_VIDEOS}")
        return

    os.makedirs(READY_VIDEOS, exist_ok=True)
    os.makedirs(READY_FEATURES, exist_ok=True)

    df_raw = pd.read_csv(RAW_CSV)
    df_raw.columns = ["video", "caption"]
    total = len(df_raw)

    # หา .npy ที่ทำเสร็จแล้ว (resume)
    done = {os.path.splitext(f)[0] for f in os.listdir(READY_FEATURES) if f.endswith(".npy")}
    todo = [row for _, row in df_raw.iterrows()
            if os.path.splitext(row["video"])[0] not in done]

    banner("🧠 Phase 2: Extract + Filter → DATASET/ready/")
    print(f"  📦 Raw Dataset : {total} clips")
    print(f"  ✅ ทำแล้ว      : {len(done)} clips (resume)")
    print(f"  🔄 จะประมวลผล  : {len(todo)} clips")
    print("═" * 60)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=2, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    accepted = []
    rejected_counts = {"lip_static": 0, "caption": 0, "no_face": 0}

    for i, row in enumerate(todo, 1):
        vname   = row["video"]
        caption = str(row["caption"]).strip()
        vpath   = os.path.join(RAW_VIDEOS, vname)
        fpath   = os.path.join(READY_FEATURES, os.path.splitext(vname)[0] + ".npy")
        v_ready = os.path.join(READY_VIDEOS, vname)

        print(f"\n  [{i:>3}/{len(todo)}] {vname}")

        if not os.path.exists(vpath):
            print(f"         ⚠️  ไม่พบไฟล์วิดีโอ — ข้าม")
            continue

        # ── ตัวกรอง 1: Caption ──
        duration = get_video_duration(vpath)
        cap_err  = check_caption(caption, duration)
        if cap_err:
            print(f"         ❌ Caption: {cap_err}")
            print(f"            '{caption[:70]}{'...' if len(caption)>70 else ''}'")
            rejected_counts["caption"] += 1
            continue

        # ── สกัดฟีเจอร์ ──
        arr, lip_var, mf_count = extract_one(vpath, face_mesh)

        # ── ตัวกรอง 2: ไม่พบใบหน้าเลย ──
        if len(arr) == 0 or arr.max() == 0:
            print(f"         ❌ ไม่พบใบหน้าในวิดีโอ")
            rejected_counts["no_face"] += 1
            continue

        # ── ตัวกรอง 3: ปากไม่ขยับ (off-screen speaker) ──
        if lip_var < LIP_VAR_THRESHOLD:
            print(f"         ❌ ปากไม่ขยับ (lip_var={lip_var:.2e}) — คนหน้ากล้องไม่ได้พูด")
            rejected_counts["lip_static"] += 1
            continue

        # ── ผ่านทุกตัวกรอง ──
        np.save(fpath, arr)
        if not os.path.exists(v_ready):
            shutil.copy2(vpath, v_ready)

        mf_note = f" | multi-face frames={mf_count}" if mf_count > 0 else ""
        print(f"         ✅ shape={arr.shape} | lip_var={lip_var:.2e}{mf_note}")
        accepted.append({"video": vname, "caption": caption})

    face_mesh.close()

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

    ready_total = len(combined) if not combined.empty else 0

    # ── สรุป ──
    total_rejected = sum(rejected_counts.values())
    banner("🏁 Phase 2 เสร็จสมบูรณ์!")
    print(f"  ✅ ผ่านการกรอง     : {len(accepted)} clips ใหม่")
    print(f"  ❌ ถูกกรองออก      : {total_rejected} clips")
    print(f"     ├── ปากไม่ขยับ  : {rejected_counts['lip_static']}")
    print(f"     ├── Caption      : {rejected_counts['caption']}")
    print(f"     └── ไม่พบหน้า   : {rejected_counts['no_face']}")
    print(f"  📊 DATASET/ready   : {ready_total} clips พร้อมเทรน")

    if not no_push and HF_TOKEN:
        print()
        _push_to_hf()
    elif not no_push and not HF_TOKEN:
        log("⚠️", "ข้ามการพุช HF เนื่องจากไม่พบ HF_TOKEN")

    print(f"\n  👉 เริ่มเทรนได้เลย: python3 train.py --local --epochs 100")
    print("═" * 60 + "\n")

# ─────────────────────────────────────────────────────────
#  VALIDATE: ตรวจ Dataset/ready ที่มีอยู่แล้ว
# ─────────────────────────────────────────────────────────
def validate(delete: bool = False):
    """ตรวจสอบ DATASET/ready/ ที่มีอยู่แล้ว"""
    if not os.path.exists(READY_CSV):
        log("❌", "ไม่พบ DATASET/ready/labels.csv"); return

    df = pd.read_csv(READY_CSV)
    df.columns = ["video", "caption"]
    total = len(df)
    bad_rows = []

    banner(f"🔍 Validate DATASET/ready/ ({total} clips)")

    for i, row in df.iterrows():
        vname   = row["video"]
        caption = str(row["caption"]).strip()
        reason  = None

        npy = os.path.join(READY_FEATURES, os.path.splitext(vname)[0] + ".npy")
        if not os.path.exists(npy):
            reason = "ไม่มีไฟล์ .npy"
        else:
            vpath = os.path.join(READY_VIDEOS, vname)
            duration = get_video_duration(vpath) if os.path.exists(vpath) else 0
            reason = check_caption(caption, duration)

        if reason:
            bad_rows.append(i)
            print(f"  ❌ [{i+1:>3}] {vname}")
            print(f"         {reason}")
            print(f"         '{caption[:70]}{'...' if len(caption)>70 else ''}'")

    print(f"\n  📊 ผลการตรวจ: พบปัญหา {len(bad_rows)} / {total} clips")

    if not bad_rows:
        print("  ✅ Dataset สะอาด พร้อมเทรน!")
        print("═" * 60 + "\n")
        return

    if delete:
        for i in bad_rows:
            vname = df.at[i, "video"]
            for path in [
                os.path.join(READY_VIDEOS, vname),
                os.path.join(READY_FEATURES, os.path.splitext(vname)[0] + ".npy"),
            ]:
                if os.path.exists(path): os.remove(path)
        clean = df.drop(index=bad_rows)
        clean.to_csv(READY_CSV, index=False)
        print(f"  🧹 ลบออกแล้ว {len(bad_rows)} clips → เหลือ {len(clean)} clips")
    else:
        print("  ⚠️  ใช้ --validate --delete เพื่อลบออกจริง")

    print("═" * 60 + "\n")

# ─────────────────────────────────────────────────────────
#  PUSH TO HUGGINGFACE
# ─────────────────────────────────────────────────────────
def _push_to_hf():
    log("🚀", f"กำลังพุชขึ้น HuggingFace: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_folder(
            folder_path=READY_DIR,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo="dataset",
            commit_message=f"Phase 2: อัปเดต Dataset ที่กรองแล้ว ({time.strftime('%Y-%m-%d %H:%M')})",
            allow_patterns=["videos/*", "features/*", "labels.csv"],
        )
        log("✅", f"พุชสำเร็จ → https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        log("❌", f"พุชล้มเหลว: {e}")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract lip features + Quality Filter → DATASET/ready/"
    )
    parser.add_argument("--no-push",  action="store_true", help="ข้ามการพุชขึ้น HuggingFace")
    parser.add_argument("--validate", action="store_true", help="ตรวจสอบคุณภาพ DATASET/ready/")
    parser.add_argument("--delete",   action="store_true", help="ใช้คู่กับ --validate เพื่อลบจริง")
    parser.add_argument("--push-only",action="store_true", help="พุช HF อย่างเดียว")
    args = parser.parse_args()

    if args.push_only:
        _push_to_hf(); return

    if args.validate:
        validate(delete=args.delete); return

    run_extract(no_push=args.no_push)


if __name__ == "__main__":
    main()
