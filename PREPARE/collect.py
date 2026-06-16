#!/usr/bin/env python3
"""
collect.py — Phase 1: ดาวน์โหลด + ถอดเสียง → DATASET/raw/

วิธีใช้:
  python3 PREPARE/collect.py "https://youtube.com/watch?v=..."
  python3 PREPARE/collect.py --batch PREPARE/url.txt
  python3 PREPARE/collect.py --batch PREPARE/url.txt --push-every 5
  python3 PREPARE/collect.py --yt-playlist "https://youtube.com/playlist?list=..."

ผลลัพธ์:
  DATASET/raw/videos/  ← คลิปวิดีโอที่ตัดแล้ว
  DATASET/raw/labels.csv ← caption จาก Typhoon ASR (ยังไม่กรองคุณภาพ)

ตั้งค่า Token ก่อนรัน:
  export TYPHOON_API_KEY="sk-xxx..."
"""

import os, sys, shutil, json, time, argparse
import cv2, numpy as np
import pandas as pd
import yt_dlp
import mediapipe as mp
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────
#  CONFIG & MODEL INITIALIZATION
# ─────────────────────────────────────────────────────────
GLOBAL_ASR_PIPE = None

def init_asr_pipe():
    global GLOBAL_ASR_PIPE
    if GLOBAL_ASR_PIPE is not None:
        return GLOBAL_ASR_PIPE

    print("\n[INIT] 🚀 กำลังโหลดโมเดล Typhoon-Whisper Large v3 (อาจใช้เวลาสักครู่ในครั้งแรก)...")
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.bfloat16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    model_id = "typhoon-ai/typhoon-whisper-large-v3"
    
    # Load .env to get HF_TOKEN
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(env_path)
    hf_token = os.environ.get("HF_TOKEN")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, token=hf_token
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    GLOBAL_ASR_PIPE = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=400,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("[INIT] ✅ โหลดโมเดลสำเร็จ!")
    return GLOBAL_ASR_PIPE

PREPARE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PREPARE_DIR)
RAW_DIR     = os.path.join(PREPARE_DIR, "raw_data")        # temp working dir
RAW_VIDEOS  = os.path.join(PROJECT_DIR, "DATASET", "raw", "videos")
RAW_CSV     = os.path.join(PROJECT_DIR, "DATASET", "raw", "labels.csv")
RAW_JSON    = os.path.join(PROJECT_DIR, "DATASET", "raw", "labels.json")

BATCH_DONE_LOG   = os.path.join(PREPARE_DIR, "_batch_done.txt")
BATCH_FAILED_LOG = os.path.join(PREPARE_DIR, "_batch_failed.txt")

MAX_CLIP_SEC    = 45.0
GAP_TOLERANCE   = 0.8
MIN_SEGMENT_SEC = 1.5
DETECTION_CONF  = 0.4

# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────
def log(emoji, msg):  print(f"\n{emoji}  {msg}")
def step(n, t, msg):  print(f"\n{'─'*55}\n  ขั้นตอน {n}/{t}: {msg}\n{'─'*55}")
def banner(title):    print("\n" + "═"*60 + f"\n  {title}\n" + "═"*60)

# ─────────────────────────────────────────────────────────
#  STEP 1: Download
# ─────────────────────────────────────────────────────────
def download_youtube(url: str, out_path: str) -> str:
    log("⬇", f"กำลังดาวน์โหลด: {url}")
    ydl_opts = {
        "format": "best", "outtmpl": out_path,
        "quiet": True, "no_warnings": True, "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    log("✅", f"ดาวน์โหลดเสร็จ → {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────
#  STEP 2: Detect speaking segments (rough lip movement)
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
            # ถ้าเฟรมนี้มีคนมากกว่า 1 คน → ถือว่าเฟรมนี้ไม่มีคนพูด
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
#  STEP 3: Chunk + Transcribe
# ─────────────────────────────────────────────────────────
def chunk_and_transcribe(video_path: str, segments: list) -> list:
    asr_pipe = init_asr_pipe()
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
        clip_name = f"yt_clip_{ts}_{i:03d}.mp4"
        clip_path = os.path.join(RAW_DIR, clip_name)
        audio_tmp = os.path.join(PREPARE_DIR, f"_tmp_audio_{i}.wav")

        print(f"   [{i+1}/{len(final_segs)}] {s:.1f}s–{e:.1f}s  →  {clip_name}")
        try:
            clip = video.subclip(s, e)
            clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            clip.audio.write_audiofile(audio_tmp, logger=None)
            tx = asr_pipe(audio_tmp, generate_kwargs={"language": "thai"})
            text = tx["text"].strip()
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
#  STEP 4: Save to DATASET/raw/
# ─────────────────────────────────────────────────────────
def save_to_raw(new_records: list):
    os.makedirs(RAW_VIDEOS, exist_ok=True)
    moved = 0
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith((".mp4", ".mov", ".avi")): continue
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(RAW_VIDEOS, fname)
        if not os.path.exists(dst):
            shutil.move(src, dst); moved += 1

    log("📂", f"ย้ายวิดีโอ {moved} ไฟล์ไป DATASET/raw/videos/")

    new_df = pd.DataFrame(new_records)
    if os.path.exists(RAW_CSV) and os.path.getsize(RAW_CSV) > 0:
        old_df = pd.read_csv(RAW_CSV)
        old_df.columns = ["video", "caption"]
        combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["video"], keep="last")
    else:
        combined = new_df
    combined.to_csv(RAW_CSV, index=False)

    # JSON
    old_json = []
    if os.path.exists(RAW_JSON):
        try:
            with open(RAW_JSON, "r", encoding="utf-8") as f: old_json = json.load(f)
        except: pass
    existing = {r["video"]: r["caption"] for r in old_json}
    for r in new_records: existing[r["video"]] = r["caption"]
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump([{"video": k, "caption": v} for k, v in existing.items()], f, ensure_ascii=False, indent=2)

    log("✅", f"DATASET/raw/labels.csv ตอนนี้มี {len(combined)} แถว")

# ─────────────────────────────────────────────────────────
#  CORE: Process 1 URL
# ─────────────────────────────────────────────────────────
def process_one(url: str, start_time: float = 0.0) -> int:
    tmp_full = os.path.join(PREPARE_DIR, "_tmp_full.mp4")
    tmp_trim = os.path.join(PREPARE_DIR, "_tmp_trim.mp4")
    try:
        step(1, 4, "ดาวน์โหลดวิดีโอจาก YouTube")
        download_youtube(url, tmp_full)

        analyze_path = tmp_full
        if start_time > 0:
            step(2, 4, f"ตัดวิดีโอเริ่มที่ {start_time}s")
            v = VideoFileClip(tmp_full)
            v.subclip(start_time).write_videofile(tmp_trim, codec="libx264", audio_codec="aac", logger=None)
            v.close(); analyze_path = tmp_trim
        else:
            step(2, 4, "ข้ามการตัดวิดีโอ (start=0)")

        step(3, 4, "วิเคราะห์รูปปาก + ถอดเสียง (Typhoon ASR)")
        segs = get_speaking_segments(analyze_path)
        log("📊", f"พบ {len(segs)} speaking segments")
        if not segs:
            log("⚠️", "ไม่พบใบหน้าพูดในวิดีโอ — ข้ามไป URL ถัดไป"); return 0
        new_records = chunk_and_transcribe(analyze_path, segs)
        if not new_records:
            log("⚠️", "ไม่มี clip ที่ transcribe ได้"); return 0

        step(4, 4, f"บันทึก {len(new_records)} clips → DATASET/raw/")
        save_to_raw(new_records)
        return len(new_records)

    except Exception as e:
        log("❌", f"เกิดข้อผิดพลาด: {e}"); return -1
    finally:
        for tmp in [tmp_full, tmp_trim]:
            if os.path.exists(tmp): os.remove(tmp)
        for f in os.listdir(PREPARE_DIR):
            if f.startswith("_tmp_audio_"): os.remove(os.path.join(PREPARE_DIR, f))

# ─────────────────────────────────────────────────────────
#  BATCH
# ─────────────────────────────────────────────────────────
def fetch_playlist_urls(playlist_url: str) -> list:
    log("📋", "กำลังดึงรายการ URL จาก Playlist...")
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info.get("entries", []) if e.get("id")]
    log("✅", f"พบ {len(urls)} วิดีโอใน Playlist")
    return urls


def process_batch(urls: list):
    total_urls = len(urls)
    total_clips = 0
    failed_urls = []
    skipped_urls = []

    done_set = set()
    if os.path.exists(BATCH_DONE_LOG):
        with open(BATCH_DONE_LOG, "r", encoding="utf-8") as f:
            done_set = {l.strip() for l in f if l.strip()}
        log("🔄", f"Resume log — ข้าม {len(done_set)} URLs ที่ทำแล้ว")

    banner("🎬 Phase 1: Collect — Download + ASR")
    print(f"  📦 URL ทั้งหมด     : {total_urls}")
    print(f"  ✅ ทำแล้ว (Resume): {len(done_set)}")
    print(f"  🚀 จะประมวลผล      : {total_urls - len(done_set)} URLs")
    print("═" * 60)

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url or url.startswith("#"): continue
        if url in done_set:
            print(f"  [{i:>3}/{total_urls}] ⏭  ข้ามแล้ว: {url[:65]}")
            skipped_urls.append(url); continue

        print(f"\n{'─'*60}\n  [{i:>3}/{total_urls}] 🎬 {url[:65]}\n{'─'*60}")
        result = process_one(url)

        if result >= 0:
            total_clips += result
            with open(BATCH_DONE_LOG, "a", encoding="utf-8") as f:
                f.write(url + "\n")
            done_set.add(url)
        else:
            failed_urls.append(url)

    total_done = len(pd.read_csv(RAW_CSV)) if os.path.exists(RAW_CSV) else 0
    banner("🏁 Phase 1 เสร็จสมบูรณ์!")
    print(f"  🎬 เพิ่ม clips รวม : {total_clips}")
    print(f"  📊 DATASET/raw     : {total_done} clips")
    print(f"  ❌ ล้มเหลว          : {len(failed_urls)} URLs")
    if failed_urls:
        with open(BATCH_FAILED_LOG, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
    print(f"\n  👉 รันขั้นตอนต่อไป: python3 PREPARE/extract.py")
    print("═" * 60 + "\n")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: ดาวน์โหลด YouTube + Transcribe → DATASET/raw/"
    )
    parser.add_argument("url",        nargs="?", default=None)
    parser.add_argument("start_time", nargs="?", type=float, default=0.0)
    parser.add_argument("--batch",       metavar="FILE")
    parser.add_argument("--yt-playlist", metavar="URL")
    parser.add_argument("--clear-resume", action="store_true")
    args = parser.parse_args()

    if args.clear_resume:
        if os.path.exists(BATCH_DONE_LOG): os.remove(BATCH_DONE_LOG)
        print("✅ ล้าง Resume log แล้ว"); return

    if args.yt_playlist:
        urls = fetch_playlist_urls(args.yt_playlist)
        process_batch(urls); return

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        process_batch(urls); return

    if not args.url:
        parser.print_help(); sys.exit(0)

    banner("🎬 Phase 1: Single URL")
    n = process_one(args.url, args.start_time)
    total = len(pd.read_csv(RAW_CSV)) if os.path.exists(RAW_CSV) else 0
    print(f"\n  ✅ เพิ่ม {max(n,0)} clips | DATASET/raw รวม: {total} clips")
    print(f"  👉 รันต่อ: python3 PREPARE/extract.py")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
