"""
train.py — ดาวน์โหลด Dataset จาก HuggingFace → สกัดฟีเจอร์ → เทรน

วิธีใช้:
  python3 train.py               # ดาวน์โหลด HF อัตโนมัติ แล้วเทรน
  python3 train.py --sync        # บังคับ sync ใหม่จาก HF ก่อนเทรน
  python3 train.py --local       # ข้ามการ sync ใช้ข้อมูลในเครื่อง
  python3 train.py --epochs 50   # กำหนด epoch เอง
"""

import os, sys, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import mediapipe as mp
import pandas as pd

from model import LipReadingTransformer, count_parameters
from vocabulary import ThaiTokenizer
from dataset import LipReadingDataset, collate_fn

# ─────────────────────────────────────────────────────────
#  CONFIG — ตั้งค่าผ่าน environment variable:
#    export HF_TOKEN="hf_xxx..."
# ─────────────────────────────────────────────────────────
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "Phonsiri/Thai-Lip-Reading-Dataset"

if not HF_TOKEN:
    print("⚠️  ไม่พบ HF_TOKEN — กรุณารัน: export HF_TOKEN=\"hf_xxx...\"")
    print("   หรือใช้ --local เพื่อเทรนจากข้อมูลในเครื่อง")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "DATASET")
VIDEOS_DIR   = os.path.join(DATASET_DIR, "videos")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")
DATASET_CSV  = os.path.join(DATASET_DIR, "labels.csv")

# Landmarks: 60 จุด × 3 = 180 มิติ
LIPS_INDICES = list(dict.fromkeys([
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    152, 148, 176, 149, 150, 136, 172, 58, 288, 397, 365, 379, 378, 400, 377,
    1, 4, 168, 197, 5
]))

# ─────────────────────────────────────────────────────────
#  STEP A: ดาวน์โหลดจาก HuggingFace
# ─────────────────────────────────────────────────────────
def sync_from_hf(force: bool = False):
    """ดาวน์โหลด videos + labels จาก HuggingFace ลง DATASET/"""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    os.makedirs(VIDEOS_DIR, exist_ok=True)

    # ตรวจว่ามีข้อมูลในเครื่องแล้วหรือยัง
    local_videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".mov"))]
    if local_videos and not force:
        print(f"✅ พบวิดีโอในเครื่อง {len(local_videos)} ไฟล์ — ข้ามการ sync (ใช้ --sync บังคับ)")
        return

    print(f"\n{'─'*50}")
    print(f"  🤗 ดาวน์โหลด Dataset จาก HuggingFace")
    print(f"  repo: {HF_REPO_ID}")
    print(f"{'─'*50}")

    try:
        # Download ทั้งโฟลเดอร์ dataset/ จาก HF มาไว้ที่ local DATASET/
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=DATASET_DIR,
            allow_patterns=["dataset/videos/*", "dataset/labels.csv", "dataset/labels.json"],
            local_dir_use_symlinks=False,
        )

        # HF snapshot จะดาวน์โหลดมาเป็น DATASET/dataset/videos/ → ย้ายขึ้นมาชั้นเดียว
        hf_sub = os.path.join(DATASET_DIR, "dataset")
        if os.path.exists(hf_sub):
            for item in os.listdir(hf_sub):
                src = os.path.join(hf_sub, item)
                dst = os.path.join(DATASET_DIR, item)
                if os.path.isdir(src):
                    os.makedirs(dst, exist_ok=True)
                    for f in os.listdir(src):
                        fsrc = os.path.join(src, f)
                        fdst = os.path.join(dst, f)
                        if not os.path.exists(fdst):
                            os.rename(fsrc, fdst)
                else:
                    if not os.path.exists(dst):
                        os.rename(src, dst)
            # ลบ subfolder ที่ว่างแล้ว
            try:
                import shutil
                shutil.rmtree(hf_sub)
            except: pass

        downloaded = len([f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".mov"))])
        print(f"✅ ดาวน์โหลดวิดีโอ {downloaded} ไฟล์เรียบร้อย")

    except Exception as e:
        print(f"❌ sync_from_hf ล้มเหลว: {e}")
        print("   → ลองใช้ข้อมูลในเครื่องแทน")

# ─────────────────────────────────────────────────────────
#  STEP B: สกัดฟีเจอร์ (.npy) สำหรับวิดีโอที่ยังไม่มี
# ─────────────────────────────────────────────────────────
def ensure_features():
    """สร้างไฟล์ .npy สำหรับวิดีโอที่ยังไม่ได้สกัดฟีเจอร์"""
    os.makedirs(FEATURES_DIR, exist_ok=True)
    done = {os.path.splitext(f)[0] for f in os.listdir(FEATURES_DIR) if f.endswith(".npy")}
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.endswith((".mp4", ".mov", ".avi"))]
    todo = [v for v in videos if os.path.splitext(v)[0] not in done]

    if not todo:
        print(f"✅ ฟีเจอร์ครบทุกไฟล์ ({len(done)} ไฟล์)")
        return

    print(f"\n{'─'*50}")
    print(f"  🧠 สกัดฟีเจอร์ {len(todo)} วิดีโอใหม่ (180 มิติ/เฟรม)")
    print(f"{'─'*50}")

    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    for i, vname in enumerate(todo):
        vpath = os.path.join(VIDEOS_DIR, vname)
        fpath = os.path.join(FEATURES_DIR, os.path.splitext(vname)[0] + ".npy")
        cap = cv2.VideoCapture(vpath)
        feats = []
        while cap.isOpened():
            ok, img = cap.read()
            if not ok: break
            res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = [c for idx in LIPS_INDICES for c in [lm.landmark[idx].x, lm.landmark[idx].y, lm.landmark[idx].z]]
                feats.append(pts)
            else:
                feats.append([0.0] * (len(LIPS_INDICES) * 3))
        cap.release()
        arr = np.array(feats)
        if len(arr) > 0:
            np.save(fpath, arr)
            print(f"  [{i+1}/{len(todo)}] {vname}  shape={arr.shape}")
        else:
            print(f"  [{i+1}/{len(todo)}] ⚠️  ข้าม {vname}")
    face_mesh.close()
    print("✅ สกัดฟีเจอร์เสร็จ")

# ─────────────────────────────────────────────────────────
#  STEP C: Training
# ─────────────────────────────────────────────────────────
def train(epochs: int = 200, batch_size: int = 2, lr: float = 5e-5,
          save_path: str = "lip_reading_model.pth"):

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n{'─'*50}")
    print(f"  🚀 เริ่มเทรน | device={device} | epochs={epochs} | batch={batch_size} | lr={lr}")
    print(f"{'─'*50}")

    # ตรวจ dataset
    if not os.path.exists(DATASET_CSV):
        print("❌ ไม่พบ DATASET/labels.csv — รัน python3 run.py <URL> ก่อนครับ")
        return

    df = pd.read_csv(DATASET_CSV)
    # กรองเฉพาะ row ที่มีไฟล์ .npy จริง
    valid = [
        i for i in range(len(df))
        if os.path.exists(os.path.join(FEATURES_DIR, os.path.splitext(df.iloc[i, 0])[0] + ".npy"))
    ]
    print(f"  📊 Dataset: {len(df)} แถวใน CSV | {len(valid)} ไฟล์มี features พร้อมเทรน")
    if not valid:
        print("❌ ไม่มีไฟล์ .npy — ตรวจสอบ DATASET/features/")
        return

    # Model & Tokenizer
    tokenizer   = ThaiTokenizer()
    num_classes = tokenizer.get_vocab_size()
    model       = LipReadingTransformer(num_classes=num_classes).to(device)
    optimizer   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion   = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

    # Resume
    start_epoch = 0
    if os.path.exists(save_path):
        print(f"  🔄 พบ checkpoint: {save_path} — กำลังตรวจสอบสถาปัตยกรรม...")
        try:
            ckpt = torch.load(save_path, map_location=device)
            # ตรวจสอบขนาดเพื่อกัน mismatch
            ckpt_dim = ckpt["model_state_dict"]["embedding.weight"].shape[1]
            model_dim = model.embedding.in_features
            if ckpt_dim != model_dim:
                print(f"  ⚠️ Checkpoint มิติไม่ตรงกัน (Checkpoint: {ckpt_dim} มิติ vs โมเดลปัจจุบัน: {model_dim} มิติ)")
                print("  → จะเริ่มฝึกสอนใหม่จากศูนย์ (Scratch)")
            else:
                model.load_state_dict(ckpt["model_state_dict"])
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except:
                    print("  ⚠️ ไม่สามารถโหลด state ของ optimizer ได้ — จะตั้งค่าใหม่")
                start_epoch = ckpt["epoch"] + 1
                print(f"  ↩️  โหลด checkpoint สำเร็จ เริ่มต้นที่ epoch {start_epoch}")
        except Exception as e:
            print(f"  ⚠️ ไม่สามารถโหลด checkpoint ได้: {e}")
            print("  → จะเริ่มฝึกสอนใหม่จากศูนย์ (Scratch)")

    print(f"  🏗️  Parameters: {count_parameters(model):,}")

    dataset     = LipReadingDataset(DATASET_CSV, FEATURES_DIR, tokenizer)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    # Training loop
    best_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch_idx, (features, targets, input_lengths, target_lengths) in enumerate(train_loader):
            features = features.to(device)
            targets  = targets.to(device)

            outputs = model(features)                                          # (Seq, Batch, Classes)
            loss    = criterion(outputs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"  Epoch [{epoch+1:>4}/{epochs}]  Loss: {avg_loss:.4f}")

        # Save ทุก 10 epoch หรือถ้า loss ดีขึ้น
        if (epoch + 1) % 10 == 0 or avg_loss < best_loss:
            best_loss = min(avg_loss, best_loss)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, save_path)
            print(f"  💾 บันทึก checkpoint → {save_path}")

    print(f"\n✅ เทรนเสร็จ | Best Loss: {best_loss:.4f}")

# ─────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thai Lip Reading — Train with HF sync")
    parser.add_argument("--sync",       action="store_true", help="บังคับ sync ข้อมูลจาก HF ก่อนเทรน")
    parser.add_argument("--local",      action="store_true", help="ข้ามการ sync ใช้ข้อมูลในเครื่อง")
    parser.add_argument("--epochs",     type=int,   default=200,   help="จำนวน epoch")
    parser.add_argument("--batch_size", type=int,   default=2,     help="batch size")
    parser.add_argument("--lr",         type=float, default=5e-5,  help="learning rate")
    parser.add_argument("--save",       type=str,   default="lip_reading_model.pth", help="ชื่อไฟล์ checkpoint")
    args = parser.parse_args()

    # A. Sync จาก HF (ถ้าไม่ได้บอก --local)
    if not args.local:
        sync_from_hf(force=args.sync)

    # B. สกัดฟีเจอร์ที่ยังขาด
    ensure_features()

    # C. เทรน
    train(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        save_path  = args.save,
    )
