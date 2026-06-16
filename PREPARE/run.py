#!/usr/bin/env python3
"""
PREPARE/run.py — Data Preparation Pipeline (URL / Batch / Playlist → DATASET → HuggingFace)

🎯 สคริปต์หลักสำหรับเตรียมข้อมูล ทำงานจากโฟลเดอร์ PREPARE/
   ผลลัพธ์ทั้งหมดจะถูกบันทึกลงใน DATASET/ ที่ root ของ project

วิธีใช้ (URL เดียว):
  python3 PREPARE/run.py "https://youtube.com/watch?v=..."
  python3 PREPARE/run.py "https://youtube.com/watch?v=..." 13           # เริ่มที่วินาทีที่ 13
  python3 PREPARE/run.py "https://youtube.com/watch?v=..." 0 --no-push  # ข้ามการพุช HF

วิธีใช้ (Batch Mode — ป้อน URL หลายอัน):
  python3 PREPARE/run.py --batch PREPARE/urls.txt
  python3 PREPARE/run.py --batch PREPARE/urls.txt --no-push
  python3 PREPARE/run.py --batch PREPARE/urls.txt --push-every 5

วิธีใช้ (Playlist Mode — ดึง URL ทั้ง Playlist อัตโนมัติ):
  python3 PREPARE/run.py --yt-playlist "https://youtube.com/playlist?list=..."
  python3 PREPARE/run.py --yt-playlist "URL" --push-every 10 --no-extract

อาร์กิวเมนต์เสริม:
  --no-push       ข้ามการพุชขึ้น Hugging Face ทั้งหมด
  --push-every N  พุช HF ทุก N URLs (ป้องกันการสูญข้อมูลหาก crash กลางคัน)
  --no-extract    ข้ามการสกัดฟีเจอร์ .npy (ประหยัดเวลาเมื่อเน้นเก็บข้อมูลก่อน)
  --label-only    เฉพาะถอดเสียงแล้วบันทึก label (ไม่ดาวน์โหลดจาก YouTube)

ตั้งค่า Token ก่อนรัน:
  export TYPHOON_API_KEY="sk-xxx..."
  export HF_TOKEN="hf_xxx..."
"""

import os, sys, shutil, json, time, argparse
import cv2, numpy as np
import pandas as pd
import yt_dlp
import mediapipe as mp
from openai import OpenAI
from moviepy.editor import VideoFileClip
from huggingface_hub import HfApi

# ─────────────────────────────────────────────────────────
#  CONFIG — ตั้งค่าผ่าน environment variable
# ─────────────────────────────────────────────────────────
TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY", "")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HF_REPO_ID      = "Phonsiri/Thai-Lip-Reading-Dataset"

_missing = []
if not TYPHOON_API_KEY: _missing.append('export TYPHOON_API_KEY="sk-xxx..."')
if not HF_TOKEN:        _missing.append('export HF_TOKEN="hf_xxx..."')
if _missing:
    print("❌ ไม่พบ environment variable — กรุณารัน:")
    for cmd in _missing: print(f"   {cmd}")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────
#  PATHS — PREPARE/ อยู่ใน subfolder; ทุก dataset อยู่ที่ root
# ─────────────────────────────────────────────────────────
PREPARE_DIR  = os.path.dirname(os.path.abspath(__file__))   # ม.5/PREPARE/
PROJECT_DIR  = os.path.dirname(PREPARE_DIR)                  # ม.5/

RAW_DIR      = os.path.join(PREPARE_DIR, "raw_data")
VIDEOS_DIR   = os.path.join(PROJECT_DIR, "DATASET", "videos")
FEATURES_DIR = os.path.join(PROJECT_DIR, "DATASET", "features")
DATASET_CSV  = os.path.join(PROJECT_DIR, "DATASET", "labels.csv")
DATASET_JSON = os.path.join(PROJECT_DIR, "DATASET", "labels.json")
DATASET_DIR  = os.path.join(PROJECT_DIR, "DATASET")

# Resume logs (เก็บไว้ใน PREPARE/ เพราะเป็น data prep state)
BATCH_DONE_LOG   = os.path.join(PREPARE_DIR, "_batch_done.txt")
BATCH_FAILED_LOG = os.path.join(PREPARE_DIR, "_batch_failed.txt")

MAX_CLIP_SEC   = 45.0
GAP_TOLERANCE  = 0.8
MIN_SEGMENT_SEC = 1.5
DETECTION_CONF = 0.4

# MediaPipe landmarks: 60 จุด × 3 (x,y,z) = 180 มิติ
LIPS_INDICES = list(dict.fromkeys([
    # Lips 40 จุด
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Jaw 15 จุด
    152, 148, 176, 149, 150, 136, 172, 58, 288, 397, 365, 379, 378, 400, 377,
    # Nose anchor 5 จุด
    1, 4, 168, 197, 5
]))

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def log(emoji, msg):    print(f"\n{emoji}  {msg}")
def step(n, t, msg):    print(f"\n{'─'*55}\n  ขั้นตอน {n}/{t}: {msg}\n{'─'*55}")
def banner(title):      print("\n" + "═"*60 + f"\n  {title}\n" + "═"*60)

# ─────────────────────────────────────────────────────────
#  STEP 1: Download
# ─────────────────────────────────────────────────────────
def download_youtube(url: str, out_path: str) -> str:
    log("⬇", f"กำลังดาวน์โหลด: {url}")
    ydl_opts = {
        "format": "best", 
        "outtmpl": out_path, 
        "quiet": True, 
        "no_warnings": True,
        "noplaylist": True  # บังคับโหลดแค่วิดีโอเดียว (ป้องกันบัคถ้ายัดลิ้งก์เพลย์ลิสต์เข้ามาใน batch)
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    log("✅", f"ดาวน์โหลดเสร็จ → {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────
#  STEP 2: Detect speaking segments
# ─────────────────────────────────────────────────────────
def get_speaking_segments(video_path: str):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=DETECTION_CONF, min_tracking_confidence=DETECTION_CONF
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    gap_frames = int(GAP_TOLERANCE * fps)
    min_frames = int(MIN_SEGMENT_SEC * fps)

    segments, face_hist = [], {}
    start_frame = last_valid = None
    frame_idx = 0

    log("🔍", f"วิเคราะห์วิดีโอ (FPS={fps:.0f})...")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        talking = False

        if results.multi_face_landmarks:
            # ถ้าเจอหน้าคนมากกว่า 1 หน้าในเฟรม ให้ถือว่าเฟรมนี้ไม่มีคนพูด (กลายเป็น Gap ทันที)
            if len(results.multi_face_landmarks) > 1:
                pass 
            else:
                for i, lm in enumerate(results.multi_face_landmarks):
                    dist = abs(lm.landmark[13].y - lm.landmark[14].y)
                    face_hist.setdefault(i, []).append(dist)
                if len(face_hist[i]) > 15: face_hist[i].pop(0)
                if len(face_hist[i]) >= 10:
                    avg = sum(face_hist[i]) / len(face_hist[i])
                    var = sum((x - avg)**2 for x in face_hist[i]) / len(face_hist[i])
                    if var > 0.0000015: talking = True

        if talking:
            if start_frame is None: start_frame = frame_idx
            last_valid = frame_idx
        elif start_frame is not None and (frame_idx - last_valid) > gap_frames:
            if (last_valid - start_frame) >= min_frames:
                segments.append((start_frame / fps, last_valid / fps))
            start_frame = last_valid = None

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f"   {frame_idx} frames | {len(segments)} segments พบแล้ว...", end="\r")

    if start_frame and last_valid and (last_valid - start_frame) >= min_frames:
        segments.append((start_frame / fps, last_valid / fps))
    cap.release(); face_mesh.close()
    return segments

# ─────────────────────────────────────────────────────────
#  STEP 3: Chunk + Transcribe + Save clips
# ─────────────────────────────────────────────────────────
def chunk_and_transcribe(video_path: str, segments: list) -> list:
    client = OpenAI(api_key=TYPHOON_API_KEY, base_url=TYPHOON_BASE_URL)
    os.makedirs(RAW_DIR, exist_ok=True)
    video = VideoFileClip(video_path)
    results = []

    final_segs = []
    for s, e in segments:
        if (e - s) > MAX_CLIP_SEC:
            cur = s
            while cur < e:
                nxt = min(cur + MAX_CLIP_SEC, e)
                if (nxt - cur) >= 2.0: final_segs.append((cur, nxt))
                cur = nxt
        else:
            final_segs.append((s, e))

    log("🎙", f"พบ {len(final_segs)} clips — กำลัง transcribe...")
    for i, (s, e) in enumerate(final_segs):
        ts = int(time.time())
        clip_name  = f"yt_clip_{ts}_{i:03d}.mp4"
        clip_path  = os.path.join(RAW_DIR, clip_name)
        audio_tmp  = os.path.join(PREPARE_DIR, f"_tmp_audio_{i}.wav")

        print(f"   [{i+1}/{len(final_segs)}] {s:.1f}s–{e:.1f}s  →  {clip_name}")
        try:
            clip = video.subclip(s, e)
            clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            clip.audio.write_audiofile(audio_tmp, logger=None)

            with open(audio_tmp, "rb") as f:
                tx = client.audio.transcriptions.create(file=f, model="typhoon-asr-realtime")
            text = tx.text.strip()
            if text:
                print(f"      📝 {text[:80]}{'...' if len(text) > 80 else ''}")
                results.append({"video": clip_name, "caption": text})
        except Exception as ex:
            print(f"      ⚠️  ข้าม clip นี้: {ex}")
        finally:
            if os.path.exists(audio_tmp): os.remove(audio_tmp)

    video.close()
    return results

# ─────────────────────────────────────────────────────────
#  STEP 4: Sync raw_data → DATASET
# ─────────────────────────────────────────────────────────
def sync_to_dataset(new_records: list):
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    moved = 0
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith((".mp4", ".mov", ".avi")): continue
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(VIDEOS_DIR, fname)
        if not os.path.exists(dst):
            shutil.move(src, dst); moved += 1

    log("📂", f"ย้ายวิดีโอ {moved} ไฟล์ไป DATASET/videos/")

    # Merge CSV
    new_df = pd.DataFrame(new_records)
    if os.path.exists(DATASET_CSV) and os.path.getsize(DATASET_CSV) > 0:
        old_df = pd.read_csv(DATASET_CSV)
        old_df.columns = ["video", "caption"]
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["video"], keep="last")
    else:
        combined = new_df
    combined.to_csv(DATASET_CSV, index=False)

    # Merge JSON
    old_json = []
    if os.path.exists(DATASET_JSON):
        try:
            with open(DATASET_JSON, "r", encoding="utf-8") as f: old_json = json.load(f)
        except: pass
    existing = {r["video"]: r["caption"] for r in old_json}
    for r in new_records: existing[r["video"]] = r["caption"]
    with open(DATASET_JSON, "w", encoding="utf-8") as f:
        json.dump([{"video": k, "caption": v} for k, v in existing.items()], f, ensure_ascii=False, indent=2)

    log("✅", f"labels.csv ตอนนี้มี {len(combined)} แถว")

# ─────────────────────────────────────────────────────────
#  STEP 5: Extract lip features (.npy)
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
#  ค่า threshold สำหรับตรวจ lip movement
#  variance ของ lip opening ต่ำกว่านี้ = คนหน้ากล้องไม่ได้พูด
# ─────────────────────────────────────────────────────────
LIP_VAR_THRESHOLD = 2e-6

def _lip_variance(cap, face_mesh) -> tuple[np.ndarray, float]:
    """คืนค่า (feature array, lip_variance) สำหรับวิดีโอหนึ่งคลิป"""
    feats, lip_openings = [], []
    while cap.isOpened():
        ok, img = cap.read()
        if not ok: break
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0]
            # จุด 13 = ริมฝีปากบน, จุด 14 = ริมฝีปากล่าง (แกน Y)
            lip_openings.append(abs(lm.landmark[13].y - lm.landmark[14].y))
            pts = [c for idx in LIPS_INDICES for c in [lm.landmark[idx].x, lm.landmark[idx].y, lm.landmark[idx].z]]
            feats.append(pts)
        else:
            feats.append([0.0] * (len(LIPS_INDICES) * 3))
    var = float(np.var(lip_openings)) if lip_openings else 0.0
    return np.array(feats), var


def extract_features():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    done  = {os.path.splitext(f)[0] for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")}
    todos = [v for v in os.listdir(VIDEOS_DIR)
             if v.endswith((".mp4", ".mov", ".avi")) and os.path.splitext(v)[0] not in done]

    if not todos:
        log("✅", "ไม่มีวิดีโอใหม่ที่ต้องสกัดฟีเจอร์"); face_mesh.close(); return

    log("🧠", f"สกัดฟีเจอร์ {len(todos)} วิดีโอใหม่ (180 มิติ/เฟรม)")
    rejected = []
    for vname in todos:
        vpath = os.path.join(VIDEOS_DIR, vname)
        fpath = os.path.join(FEATURES_DIR, os.path.splitext(vname)[0] + ".npy")
        cap = cv2.VideoCapture(vpath)
        arr, lip_var = _lip_variance(cap, face_mesh)
        cap.release()

        # ❌ ปากไม่ขยับเลย → น่าจะเป็นคนหลังกล้องพูด
        if lip_var < LIP_VAR_THRESHOLD:
            print(f"   ❌ ข้าม {vname}  (lip_var={lip_var:.2e} < {LIP_VAR_THRESHOLD:.0e}) — คนหน้ากล้องไม่ได้พูด")
            rejected.append(vname)
            if os.path.exists(vpath): os.remove(vpath)
            continue

        if len(arr) > 0:
            np.save(fpath, arr)
            print(f"   💾 {vname}  shape={arr.shape}  lip_var={lip_var:.2e}")
        else:
            print(f"   ⚠️  ข้าม {vname} (ไม่พบใบหน้า)")
            rejected.append(vname)

    face_mesh.close()

    # ลบ label ของคลิปที่ถูกกรองออกจาก CSV
    if rejected and os.path.exists(DATASET_CSV):
        df = pd.read_csv(DATASET_CSV)
        before = len(df)
        df = df[~df.iloc[:, 0].isin(rejected)]
        df.to_csv(DATASET_CSV, index=False)
        log("🧹", f"ลบ label คลิปปากไม่ขยับ {len(rejected)} ออก ({before} → {len(df)} แถว)")

# ─────────────────────────────────────────────────────────
#  STEP 5.5: ตรวจสอบคุณภาพ Dataset
# ─────────────────────────────────────────────────────────
def validate_dataset(delete: bool = False):
    """
    ตรวจสอบและกรอง label ออกตาม 3 เงื่อนไข:
      1. ไม่มีไฟล์ .npy คู่กัน (orphan label)
      2. Caption สั้นเกินไป (< 2 คำ)
      3. อัตรา WPM เป็นไปไม่ได้ (> 12 คำ/วิ หรือ < 0.3 คำ/วิ)
    """
    if not os.path.exists(DATASET_CSV):
        log("❌", "ไม่พบ labels.csv"); return

    df = pd.read_csv(DATASET_CSV)
    df.columns = ["video", "caption"]
    total = len(df)
    bad_rows = []

    print(f"\n{'═'*60}")
    print(f"  🔍 ตรวจสอบ Dataset ({total} แถว)")
    print(f"{'═'*60}")

    for i, row in df.iterrows():
        vname   = row["video"]
        caption = str(row["caption"]).strip()
        reason  = None

        # 1. ไม่มีไฟล์ .npy
        npy = os.path.join(FEATURES_DIR, os.path.splitext(vname)[0] + ".npy")
        if not os.path.exists(npy):
            reason = "ไม่มีไฟล์ .npy"

        # 2. Caption สั้นเกินไป (< 2 คำ)
        elif len(caption.split()) < 2:
            reason = f"caption สั้นเกิน ({len(caption.split())} คำ)"

        # 3. ตรวจ WPM ถ้าวิดีโออยู่ในเครื่อง
        else:
            vpath = os.path.join(VIDEOS_DIR, vname)
            if os.path.exists(vpath):
                cap = cv2.VideoCapture(vpath)
                fps      = cap.get(cv2.CAP_PROP_FPS) or 30
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                duration = n_frames / fps if fps > 0 else 0
                n_words  = len(caption.split())
                wps      = n_words / duration if duration > 0 else 0
                if wps > 12 or (wps < 0.3 and n_words > 2):
                    reason = f"WPS={wps:.1f} ผิดปกติ ({n_words} คำ / {duration:.1f}s)"

        if reason:
            bad_rows.append(i)
            print(f"   ❌ [{i+1}] {vname}  →  {reason}")
            print(f"         caption: '{caption[:60]}{'...' if len(caption)>60 else ''}'")

    print(f"\n  📊 ผลการตรวจ: พบปัญหา {len(bad_rows)} / {total} แถว")

    if not bad_rows:
        print("  ✅ Dataset สะอาด ไม่มีปัญหา!")
        print(f"{'═'*60}\n")
        return

    if delete:
        # ลบ video + npy + label
        for i in bad_rows:
            vname = df.at[i, "video"]
            for path in [
                os.path.join(VIDEOS_DIR, vname),
                os.path.join(FEATURES_DIR, os.path.splitext(vname)[0] + ".npy")
            ]:
                if os.path.exists(path): os.remove(path)
        clean = df.drop(index=bad_rows)
        clean.to_csv(DATASET_CSV, index=False)
        print(f"  🧹 ลบออกแล้ว {len(bad_rows)} แถว → เหลือ {len(clean)} แถวใน labels.csv")
    else:
        print("  ⚠️  ใช้ python3 PREPARE/run.py --validate --delete เพื่อลบออกจริง")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────
#  STEP 6: Push to HuggingFace
# ─────────────────────────────────────────────────────────
def push_to_hf():
    log("🚀", f"กำลังพุชขึ้น HuggingFace: {HF_REPO_ID}")
    api = HfApi(token=HF_TOKEN)
    try:
        api.upload_folder(
            folder_path=DATASET_DIR,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            path_in_repo="dataset",
            commit_message=f"Auto-pipeline: เพิ่มข้อมูลใหม่ ({time.strftime('%Y-%m-%d %H:%M')})",
            allow_patterns=["videos/*", "features/*", "labels.csv", "labels.json"]
        )
        log("✅", f"พุชสำเร็จ → https://huggingface.co/datasets/{HF_REPO_ID}")
    except Exception as e:
        log("❌", f"พุชล้มเหลว: {e}")

# ─────────────────────────────────────────────────────────
#  CORE: ประมวลผล 1 URL
# ─────────────────────────────────────────────────────────
def process_one(url: str, start_time: float = 0.0,
                do_extract: bool = True, do_push: bool = False) -> int:
    """คืนค่าจำนวน clips ที่เพิ่มได้ หรือ -1 ถ้าล้มเหลว"""
    tmp_full = os.path.join(PREPARE_DIR, "_tmp_full.mp4")
    tmp_trim = os.path.join(PREPARE_DIR, "_tmp_trim.mp4")
    try:
        step(1, 5 + (1 if do_push else 0), "ดาวน์โหลดวิดีโอจาก YouTube")
        download_youtube(url, tmp_full)

        analyze_path = tmp_full
        if start_time > 0:
            step(2, 5, f"ตัดวิดีโอเริ่มที่ {start_time}s")
            v = VideoFileClip(tmp_full)
            v.subclip(start_time).write_videofile(tmp_trim, codec="libx264", audio_codec="aac", logger=None)
            v.close(); analyze_path = tmp_trim
        else:
            step(2, 5, "ข้ามการตัดวิดีโอ (start=0)")

        step(3, 5, "วิเคราะห์รูปปาก + ถอดเสียง (Typhoon ASR)")
        segs = get_speaking_segments(analyze_path)
        log("📊", f"พบ {len(segs)} speaking segments")
        if not segs:
            log("⚠️", "ไม่พบใบหน้าพูดในวิดีโอ — ข้ามไป URL ถัดไป"); return 0
        new_records = chunk_and_transcribe(analyze_path, segs)
        if not new_records:
            log("⚠️", "ไม่มี clip ที่ transcribe ได้ — ข้ามไป URL ถัดไป"); return 0

        step(4, 5, f"รวม {len(new_records)} clips เข้า DATASET/")
        sync_to_dataset(new_records)

        if do_extract:
            step(5, 5, "สกัดฟีเจอร์ปาก 60 จุด × 3 = 180 มิติ (.npy)")
            extract_features()

        if do_push:
            step(6, 6, "พุช concat ขึ้น HuggingFace")
            push_to_hf()

        return len(new_records)

    except Exception as e:
        log("❌", f"เกิดข้อผิดพลาด: {e}"); return -1

    finally:
        for tmp in [tmp_full, tmp_trim]:
            if os.path.exists(tmp): os.remove(tmp)
        for f in os.listdir(PREPARE_DIR):
            if f.startswith("_tmp_audio_"): os.remove(os.path.join(PREPARE_DIR, f))

# ─────────────────────────────────────────────────────────
#  BATCH: อ่าน URL หลายอันจากไฟล์ / Playlist
# ─────────────────────────────────────────────────────────
def fetch_playlist_urls(playlist_url: str) -> list:
    log("📋", "กำลังดึงรายการ URL จาก Playlist...")
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info.get("entries", []) if e.get("id")]
    log("✅", f"พบ {len(urls)} วิดีโอใน Playlist")
    return urls


def process_batch(urls: list, no_push: bool = False,
                  push_every: int = 0, do_extract: bool = True):
    total_urls = len(urls)
    total_clips = 0
    failed_urls = []
    skipped_urls = []

    # Resume log
    done_set = set()
    if os.path.exists(BATCH_DONE_LOG):
        with open(BATCH_DONE_LOG, "r", encoding="utf-8") as f:
            done_set = {l.strip() for l in f if l.strip()}
        log("🔄", f"พบ Resume log — ข้าม {len(done_set)} URLs ที่ทำแล้ว")

    banner("🎬 Thai Lip Reading — Batch Data Pipeline")
    print(f"  📦 URL ทั้งหมด     : {total_urls}")
    print(f"  ✅ ทำแล้ว (Resume): {len(done_set)}")
    print(f"  🚀 จะประมวลผล      : {total_urls - len(done_set)} URLs")
    print(f"  ☁️  Push HF       : {'ปิด' if no_push else f'ทุก {push_every} URLs' if push_every else 'ท้ายสุด'}")
    print(f"  🧠 Extract .npy   : {'✅' if do_extract else '❌ ข้าม'}")
    print("═" * 60)

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or url.startswith("#"): continue

        if url in done_set:
            print(f"  [{i:>3}/{total_urls}] ⏭  ข้ามแล้ว: {url[:65]}")
            skipped_urls.append(url); continue

        print(f"\n{'─'*60}\n  [{i:>3}/{total_urls}] 🎬 {url[:65]}\n{'─'*60}")

        should_push = not no_push and push_every > 0 and i % push_every == 0
        result = process_one(url, do_extract=do_extract, do_push=should_push)

        if result >= 0:
            total_clips += result
            with open(BATCH_DONE_LOG, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            done_set.add(url)
            print(f"  ✅ เพิ่ม {result} clips | รวม: {total_clips} clips")
        else:
            failed_urls.append(url)
            print(f"  ❌ ข้ามไป URL ถัดไป ({len(failed_urls)} ล้มเหลวสะสม)")

    # Extract รอบสุดท้าย
    if do_extract:
        log("🧠", "สกัดฟีเจอร์ .npy รอบสุดท้าย...")
        extract_features()

    # Push รอบสุดท้าย
    if not no_push:
        log("🚀", "พุชขึ้น HuggingFace รอบสุดท้าย...")
        push_to_hf()

    # สรุป
    total_done = len(pd.read_csv(DATASET_CSV)) if os.path.exists(DATASET_CSV) else 0
    banner("🏁 Batch Pipeline เสร็จสมบูรณ์!")
    print(f"  📦 URL ทั้งหมด    : {total_urls}")
    print(f"  ✅ สำเร็จ          : {total_urls - len(failed_urls) - len(skipped_urls)} URLs")
    print(f"  ⏭  ข้าม (Resume)  : {len(skipped_urls)} URLs")
    print(f"  ❌ ล้มเหลว         : {len(failed_urls)} URLs")
    print(f"  🎬 เพิ่ม clips รวม : {total_clips} clips")
    print(f"  📊 Dataset รวม    : {total_done} clips")
    if failed_urls:
        with open(BATCH_FAILED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
        print(f"  📋 รายการล้มเหลว  : PREPARE/_batch_failed.txt")
    if not no_push:
        print(f"  🤗 HF             : https://huggingface.co/datasets/{HF_REPO_ID}")
    print("═" * 60 + "\n")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Thai Lip Reading — Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # โหมด URL เดียว
    parser.add_argument("url",        nargs="?", default=None, help="YouTube URL (โหมดเดียว)")
    parser.add_argument("start_time", nargs="?", type=float, default=0.0,
                        help="เริ่มตั้งแต่วินาทีที่ (โหมดเดียว)")

    # โหมด Batch / Playlist
    parser.add_argument("--batch",       metavar="FILE", help="ไฟล์รายการ URL (ทีละบรรทัด)")
    parser.add_argument("--yt-playlist", metavar="URL",  help="ดึง URL จาก YouTube Playlist อัตโนมัติ")

    # ตัวเลือกเสริม
    parser.add_argument("--no-push",    action="store_true", help="ข้ามการพุชขึ้น Hugging Face")
    parser.add_argument("--push-every", type=int, default=0, metavar="N",
                        help="พุช HF ทุก N URLs (0 = พุชท้ายสุดครั้งเดียว)")
    parser.add_argument("--no-extract", action="store_true",
                        help="ข้ามการสกัดฟีเจอร์ .npy (เก็บวิดีโอก่อน สกัดทีหลัง)")
    parser.add_argument("--extract-only", action="store_true",
                        help="สกัดฟีเจอร์ .npy อย่างเดียว (ไม่ดาวน์โหลด)")
    parser.add_argument("--push-only",   action="store_true",
                        help="พุชขึ้น HF อย่างเดียว (ไม่ดาวน์โหลด)")
    parser.add_argument("--clear-resume", action="store_true",
                        help="ล้าง Resume log เพื่อเริ่มใหม่ทั้งหมด")
    parser.add_argument("--validate", action="store_true",
                        help="ตรวจสอบคุณภาพ Dataset (orphan/caption/WPM)")
    parser.add_argument("--delete",   action="store_true",
                        help="ใช้คู่กับ --validate เพื่อลบ label ที่มีปัญหาออกจริง")

    args = parser.parse_args()

    # ── ล้าง Resume log ──
    if args.clear_resume:
        if os.path.exists(BATCH_DONE_LOG):
            os.remove(BATCH_DONE_LOG)
            print("✅ ล้าง Resume log แล้ว")
        else:
            print("ℹ️  ไม่มี Resume log")
        return

    # ── Validate Dataset ──
    if args.validate:
        banner("🔍 Validate Dataset Quality")
        validate_dataset(delete=args.delete)
        return

    # ── Extract อย่างเดียว ──
    if args.extract_only:
        banner("🧠 Extract Features Only")
        extract_features()
        return

    # ── Push อย่างเดียว ──
    if args.push_only:
        banner("🚀 Push to HuggingFace Only")
        push_to_hf()
        return

    # ── โหมด Playlist ──
    if args.yt_playlist:
        urls = fetch_playlist_urls(args.yt_playlist)
        if not urls: print("❌ ไม่พบ URL ใน Playlist"); sys.exit(1)
        process_batch(urls, no_push=args.no_push,
                      push_every=args.push_every, do_extract=not args.no_extract)
        return

    # ── โหมด Batch File ──
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ ไม่พบไฟล์: {args.batch}"); sys.exit(1)
        with open(args.batch, "r", encoding="utf-8") as f:
            urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        if not urls: print("❌ ไฟล์ไม่มี URL"); sys.exit(1)
        process_batch(urls, no_push=args.no_push,
                      push_every=args.push_every, do_extract=not args.no_extract)
        return

    # ── โหมด URL เดียว ──
    if not args.url:
        parser.print_help(); sys.exit(0)

    url        = args.url
    start_time = args.start_time
    no_push    = args.no_push
    do_extract = not args.no_extract

    banner("🎬 Thai Lip Reading — Single URL Pipeline")
    print(f"  URL    : {url}")
    print(f"  Start  : {start_time}s")
    print(f"  Push   : {'❌ ข้าม' if no_push else '✅ หลังเสร็จ'}")
    print(f"  Extract: {'✅' if do_extract else '❌ ข้าม'}")
    print("═" * 60)

    n = process_one(url, start_time, do_extract=do_extract, do_push=not no_push)

    total = len(pd.read_csv(DATASET_CSV)) if os.path.exists(DATASET_CSV) else 0
    banner("🏁 Pipeline เสร็จสมบูรณ์!")
    print(f"  ✅ เพิ่ม {max(n, 0)} clips ใหม่")
    print(f"  📊 Dataset รวม: {total} clips")
    if not no_push:
        print(f"  🤗 HF: https://huggingface.co/datasets/{HF_REPO_ID}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
