# 🎬 Thai Lip Reading (Visual Speech Recognition) — The Ultimate Master Guide

โปรเจกต์ระบบอ่านริมฝีปากภาษาไทยระดับสูง (State-of-the-Art) ด้วยสถาปัตยกรรม **Hybrid Transformer ขนาด 0.44 Billion Parameters** (32 Layers) มาพร้อมกับระบบสกัดพิกัดริมฝีปาก 180 มิติ, ท่อส่งข้อมูลอัตโนมัติ (Master Pipeline) ถอดเสียงภาษาไทยด้วย Typhoon ASR และบันทึกข้อมูลสำรองบน Hugging Face Dataset

---

## 🔐 ความปลอดภัยและการตั้งค่า (Security & Environment Setup)

เพื่อความปลอดภัยของข้อมูล โปรเจกต์นี้จะ**ไม่มีการบันทึก (Hardcode) API Key หรือ Token ไว้ในไฟล์สคริปต์** แต่จะดึงข้อมูลผ่าน Environment Variables ก่อนรันโปรแกรม กรุณากำหนดค่าคีย์ดังต่อไปนี้ในระบบของคุณ:

```bash
# กำหนดค่าสำหรับเซสชันปัจจุบัน
export TYPHOON_API_KEY="sk-xxx..." # คีย์จาก opentyphoon.ai
export HF_TOKEN="hf_xxx..."        # Token สิทธิ์ Write จาก huggingface.co

# หรือหากต้องการกำหนดค่าแบบถาวร (สำหรับ macOS / zsh)
echo 'export TYPHOON_API_KEY="sk-xxx..."' >> ~/.zshrc
echo 'export HF_TOKEN="hf_xxx..."' >> ~/.zshrc
source ~/.zshrc
```

---

## 🔧 การติดตั้ง (Installation)

### 🖥️ ความต้องการของระบบ (Requirements)
- **Python:** 3.9 หรือใหม่กว่า
- **OS:** macOS (รองรับ Apple Silicon M1/M2/M3 เป็นพิเศษผ่าน MPS) หรือ Linux/Windows (รองรับการเร่งความเร็วด้วย CUDA GPU)
- **RAM:** ขั้นต่ำ 16 GB

### 📦 ติดตั้ง Libraries
รันคำสั่งต่อไปนี้เพื่อติดตั้งไลบรารีที่จำเป็นทั้งหมด:
```bash
pip install torch torchvision torchaudio
pip install mediapipe yt-dlp datasets huggingface_hub
pip install openai-whisper requests tqdm pandas opencv-python moviepy
```

---

## 📂 โครงสร้างไดเรกทอรี (Project Structure)

```
ม.5/
├── train.py               # 🏋️ สคริปต์ฝึกสอนโมเดล (ดาวน์โหลดอัตโนมัติ -> สกัดฟีเจอร์ -> เทรน)
├── inference.py           # 🔮 สคริปต์ทดสอบถอดความหมายจากการเคลื่อนไหวปากบนคลิปจริง
├── cleanup.py             # 🧹 สคริปต์ทำความสะอาดและลบไฟล์ชั่วคราวซ้ำซ้อน
├── extract_lips.py        # 👄 สคริปต์หลักสำหรับสกัดฟีเจอร์พิกัดปาก 180 มิติ (แบบเดี่ยว)
├── model.py               # 🏗️ สถาปัตยกรรมโมเดล Hybrid Transformer
├── vocabulary.py          # 🔤 ตัวแปลงรหัสพยัญชนะ/สระภาษาไทย (Tokenizer)
├── dataset.py             # 📊 ตัวดึงข้อมูลพร้อม Dynamic Padding ใน PyTorch
├── .env.example           # 📄 ไฟล์เทมเพลตสำหรับกำหนด Environment Variables
├── .gitignore             # 🚫 ไฟล์ระบุสิ่งที่ไม่ต้องการพุชขึ้น Git
├── README.md              # 📖 คู่มือการใช้งานโปรเจกต์ (ไฟล์นี้)
├── DATASET/               # 💾 แหล่งเก็บ Dataset หลัก
│   ├── labels.csv         # ไฟล์จับคู่วิดีโอกับประโยคถอดเสียง (รูปแบบหลัก)
│   ├── labels.json        # ไฟล์สำรองข้อมูลคู่เสียงและคำถอดความ
│   ├── videos/            # ไดเรกทอรีเก็บคลิปวิดีโอทั้งหมด (.mp4 / .mov)
│   ├── features/          # ไดเรกทอรีเก็บฟีเจอร์ปากที่สกัดแล้ว (.npy)
│   └── test/              # ไดเรกทอรีสำหรับวิดีโอที่รอทดสอบ/ประมวลผล
└── PREPARE/               # 🛠️ เครื่องมือหลักในสายการผลิตข้อมูล
    ├── run.py             # 🚀 Master Pipeline (ดาวน์โหลด -> ถอดเสียง -> สกัดฟีเจอร์ -> HF)
    └── raw_data/          # ไดเรกทอรีเก็บข้อมูลดิบชั่วคราว
```

---

## 🚀 คู่มือการรันทุกสคริปต์อย่างละเอียด (Execution Guide)

### 1. สายพานหลัก (Root Scripts)

#### 🎬 1.1 Master Data Pipeline (`PREPARE/run.py`)
เป็นจุดเริ่มต้นแบบ **One-Stop Service** สำหรับเตรียมข้อมูล รันคำสั่งเดียวจะประมวลผลตั้งแต่ต้นจนจบ:
1. ดาวน์โหลดวิดีโอจาก YouTube (รองรับ URL เดียว, โหมด Batch แบบไฟล์, และโหมด Playlist)
2. วิเคราะห์หาใบหน้าและแบ่งคลิปตามประโยคพูดของ Main Speaker (ทนต่อการหันหน้าหนีได้ 0.8 วินาที)
3. ส่งเฉพาะคลิปเสียงไปถอดข้อความเป็นภาษาไทยผ่าน Typhoon ASR
4. ย้ายคลิปและบันทึกป้ายกำกับลงโฟลเดอร์ `DATASET/`
5. สกัดพิกัดปาก 180 มิติในรูปแบบไฟล์ `.npy`
6. อัปโหลดข้อมูลทั้งหมดขึ้น Hugging Face Dataset (`Phonsiri/Thai-Lip-Reading-Dataset`) อัตโนมัติ

**วิธีใช้โหมดต่างๆ:**
```bash
# โหมด URL เดียว (ดาวน์โหลดและรันทั้งระบบตั้งแต่เริ่มคลิป)
python3 PREPARE/run.py "https://www.youtube.com/watch?v=..."

# โหมด Batch (อ่าน URL หลายอันจากไฟล์ urls.txt และพุชขึ้น HF ทุกๆ 5 คลิปเพื่อกันข้อมูลหาย)
python3 PREPARE/run.py --batch PREPARE/urls.txt --push-every 5

# โหมด Playlist (ดึงวิดีโอทั้ง Playlist อัตโนมัติ และข้ามการพุชขึ้น HF)
python3 PREPARE/run.py --yt-playlist "https://youtube.com/playlist?list=..." --no-push
```

#### 🏋️ 1.2 Training (`train.py`)
ใช้ฝึกสอนโมเดล AI อ่านริมฝีปาก รองรับการเรียกข้อมูลจากคลาวด์และสกัดฟีเจอร์ในตัว:
1. ซิงค์ Dataset ตัวล่าสุดจาก Hugging Face ลงเครื่อง (ข้ามอัตโนมัติหากตรวจพบว่ามีไฟล์วิดีโออยู่แล้ว)
2. ตรวจสอบไฟล์ `.npy` ใน `DATASET/features/` ถ้าพบวิดีโอใหม่ที่ยังไม่ได้ถูกประมวลผล จะเรียกใช้ MediaPipe เพื่อสกัดฟีเจอร์ทันที
3. โหลดชุดข้อมูลแบบ Dynamic Padding ขึ้น GPU (CUDA) หรือ Mac (MPS) และเริ่มฝึกฝน (พร้อมระบบ **Resume Training** อัตโนมัติจากโมเดลเดิม)

**วิธีใช้:**
```bash
# เทรนโดยดาวน์โหลดไฟล์จาก Hugging Face และสกัดฟีเจอร์อัตโนมัติ
python3 train.py

# บังคับอัปเดตและดาวน์โหลดข้อมูลใหม่จาก Hugging Face ทับข้อมูลเดิม
python3 train.py --sync

# รันโหมดออฟไลน์ ใช้ข้อมูลเฉพาะในเครื่อง (ไม่เช็ค/ดาวน์โหลดจากคลาวด์)
python3 train.py --local

# กำหนด hyperparameters และเซฟโมเดลในชื่ออื่น
python3 train.py --epochs 100 --batch_size 2 --lr 5e-5 --save my_custom_model.pth
```

#### 🔮 1.3 Inference (`inference.py`)
ใช้ทดสอบโมเดลที่เทรนแล้วเพื่ออ่านคำพูดจากริมฝีปากในวิดีโอตัวอย่าง:

**วิธีใช้:**
```bash
# ระบุไฟล์วิดีโอที่ต้องการให้อ่านปาก
python3 inference.py "DATASET/test/my_test_video.mp4"

# หากไม่ระบุอาร์กิวเมนต์ โปรแกรมจะใช้วิดีโอทดสอบเริ่มต้นใน DATASET/test/ อัตโนมัติ
python3 inference.py
```

#### 🧹 1.4 Cleanup (`cleanup.py`)
เนื่องจากสายพานข้อมูลจะสร้างไฟล์วิดีโอย่อย สคริปต์นี้จะทำความสะอาดลบไฟล์ชั่วคราวซ้ำซ้อนใน `PREPARE/raw_data/` ที่ถูกซิงค์ย้ายเข้า `DATASET/videos/` แล้วเพื่อคืนพื้นที่จัดเก็บข้อมูล พร้อมลบโฟลเดอร์แคชและแจ้งเตือนไฟล์คงค้างในเครื่อง

**วิธีใช้:**
```bash
# จำลองการทำงาน (Dry Run) เพื่อเช็คก่อนลบและดูขนาดพื้นที่ที่จะได้คืน
python3 cleanup.py

# สั่งให้ลบไฟล์ซ้ำออกจริงอย่างปลอดภัย
python3 cleanup.py --delete
```

#### 👄 1.5 Extract Lip Features (`extract_lips.py`)
สกัดพิกัดพอยต์ริมฝีปากจากวิดีโอทั้งหมดใน `DATASET/videos/` แปลงเป็นไฟล์ `.npy` ขนาด 180 มิติต่อเฟรม (รองรับการข้ามไฟล์ที่มีฟีเจอร์อยู่แล้ว)

**วิธีใช้:**
```bash
python3 extract_lips.py
```

#### 🏗️ 1.6 Check Architecture (`model.py`)
รันไฟล์สถาปัตยกรรมหลักเดี่ยวๆ เพื่อคำนวณโครงสร้าง พารามิเตอร์ และตรวจสอบความลึกของเลเยอร์:

**วิธีใช้:**
```bash
python3 model.py
```

---

### 2. เครื่องมือจัดการข้อมูลย่อย (Advanced Data Tools)

หากต้องการรันกระบวนการย่อยแยกส่วน สามารถใช้คำสั่งเสริมผ่าน `PREPARE/run.py` ได้ทั้งหมด โดยไม่จำเป็นต้องใช้สคริปต์แยก:

```bash
# สกัดฟีเจอร์ .npy อย่างเดียวจากวิดีโอที่มีอยู่ใน DATASET/videos (ไม่ดาวน์โหลดใหม่)
python3 PREPARE/run.py --extract-only

# พุชโฟลเดอร์ DATASET/ ขึ้น Hugging Face อย่างเดียว
python3 PREPARE/run.py --push-only

# ล้างประวัติการดาวน์โหลดแบบ Batch (Resume logs) เพื่อเริ่มรัน Batch ใหม่ทั้งหมด
python3 PREPARE/run.py --clear-resume
```

---

## 🛠️ รายละเอียดสเปกของระบบ (Model & Data Specs)

| องค์ประกอบ (Component) | รายละเอียดทางเทคนิค (Technical Details) |
| :--- | :--- |
| **Model Size** | 0.44 Billion Parameters (โมเดลขนาดใหญ่ รองรับประโยคภาษาไทยยาว) |
| **Model Structure** | 32-Layer Transformer (16 Layers Local / 16 Layers Global Attention) |
| **Parameters Spec** | `d_model=1024`, `nhead=16`, `dim_feedforward=4096`, `activation='gelu'` |
| **Local Attention Window** | มองกรอบประวิงเวลาทีละ 7 เฟรม เพื่อจับ Viseme (รูปปาก) และ Phoneme (ฟอนิม) |
| **Feature Dimension** | **180 มิติ** ต่อเฟรม (60 Landmarks x 3 แกน x,y,z ของพิกัด 3 มิติ) |
| **Landmark breakdown** | จุดริมฝีปาก 40 จุด + ขากรรไกรล่าง 15 จุด + แกนกลางพยุงจมูก 5 จุด (ลดการสั่นไหว) |
| **Feature Extraction** | MediaPipe Face Mesh (Confidence 0.4 ช่วยเพิ่ม Recall ในมุมเอียง/ก้ม) |
| **Confidence Verification** | ใช้ค่าความแปรปรวนของการขยับริมฝีปาก (Lip Motion Variance > 0.0000015) ยืนยันใบหน้าที่พูดจริง |
| **Thai ASR Engine** | Typhoon ASR (โดย SCB 10X) — ถอดรหัสข้อความเสียงภาษาไทยที่ถูกต้องสูง |
| **Data Format** | คอลัมน์ JSON/CSV: `video` (ชื่อไฟล์คลิป), `caption` (คำแปลภาษาไทย) |
| **Hardware Accelerator** | รองรับ Native PyTorch MPS สำหรับ Apple Silicon และ CUDA สำหรับ NVIDIA GPU |
| **Deduplication** | มีระบบข้ามไฟล์เมื่อตรวจพบข้อมูลที่ถูกประมวลผลแล้ว (Resumable Pipeline) |

---
*โปรเจกต์นี้ถูกออกแบบมาเพื่อเป็นรากฐานที่มั่นคงและยืดหยุ่นสำหรับการทำวิจัยระบบอ่านริมฝีปากภาษาไทย (Thai Lip Reading) ในระดับอุตสาหกรรม โดยมีระบบการผลิตข้อมูลและตรวจสอบความถูกต้องที่มีเสถียรภาพสูงสุด*
