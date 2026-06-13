# 🎬 Thai Lip Reading — Visual Speech Recognition

ระบบอ่านริมฝีปากภาษาไทยด้วย **Hybrid Local-Global Transformer (353M Parameters)** พร้อมระบบเตรียมข้อมูลแบบ 2 เฟส ที่กรองคุณภาพอัตโนมัติ

---

## 🔐 ตั้งค่า API Keys ก่อนใช้งาน

```bash
export TYPHOON_API_KEY="sk-xxx..."   # จาก opentyphoon.ai
export HF_TOKEN="hf_xxx..."          # จาก huggingface.co (สิทธิ์ Write)
```

---

## 📦 ติดตั้ง

```bash
pip install -r requirements.txt
```

---

## 📂 โครงสร้างโปรเจกต์

```
ม.5/
├── PREPARE/
│   ├── collect.py       ← Phase 1: ดาวน์โหลด + ถอดเสียง → DATASET/raw/
│   ├── extract.py       ← Phase 2: วัดปาก + กรองคุณภาพ + พุช HF → DATASET/ready/
│   ├── run.py           ← Pipeline เดิม (รวม Phase 1+2 ในไฟล์เดียว)
│   └── url.txt          ← รายการ YouTube URL สำหรับ Batch Mode
│
├── DATASET/
│   ├── raw/             ← Phase 1 output (ยังไม่กรอง)
│   │   ├── videos/
│   │   └── labels.csv
│   └── ready/           ← Phase 2 output (สะอาด พร้อมเทรน)
│       ├── videos/
│       ├── features/    ← ฟีเจอร์ปาก 180 มิติ (.npy)
│       └── labels.csv
│
├── model.py             ← สถาปัตยกรรม Hybrid Transformer
├── train.py             ← ฝึกโมเดล (รองรับ CUDA / MPS / CPU)
├── dataset.py           ← PyTorch Dataset + collate_fn
├── vocabulary.py        ← Thai Tokenizer
└── requirements.txt
```

---

## 🚀 วิธีใช้งาน

### Phase 1 — เก็บข้อมูล (`collect.py`)

ดาวน์โหลดวิดีโอ → ตรวจจับช่วงที่ปากขยับ → ถอดเสียงด้วย Typhoon ASR → บันทึกลง `DATASET/raw/`

```bash
# URL เดียว
python3 PREPARE/collect.py "https://youtube.com/watch?v=..."

# Batch จากไฟล์ (มีระบบ Resume อัตโนมัติ)
python3 PREPARE/collect.py --batch PREPARE/url.txt

# ดึงทั้ง Playlist
python3 PREPARE/collect.py --yt-playlist "https://youtube.com/playlist?list=..."

# ล้าง Resume log (เริ่มใหม่ทั้งหมด)
python3 PREPARE/collect.py --clear-resume
```

---

### Phase 2 — เตรียมพร้อมเทรน (`extract.py`)

อ่านจาก `DATASET/raw/` → สกัดฟีเจอร์ปาก 180 มิติ → กรองคลิปคุณภาพต่ำ → บันทึกลง `DATASET/ready/` → พุชขึ้น Hugging Face

```bash
# รันปกติ (สกัดฟีเจอร์ + กรอง + พุช HF)
python3 PREPARE/extract.py

# ข้ามการพุช HF
python3 PREPARE/extract.py --no-push

# ตรวจสอบคุณภาพ Dataset ที่มีอยู่ (ดู log เท่านั้น)
python3 PREPARE/extract.py --validate

# ตรวจสอบและลบคลิปคุณภาพต่ำออกจริง
python3 PREPARE/extract.py --validate --delete
```

---

### เทรนโมเดล (`train.py`)

```bash
# ดาวน์โหลดจาก HF + เทรน (Colab แนะนำ)
python3 train.py --epochs 100 --batch_size 8

# บังคับดาวน์โหลดใหม่จาก HF
python3 train.py --sync --epochs 100

# ใช้ข้อมูลในเครื่อง (ไม่ดาวน์โหลด)
python3 train.py --local --epochs 100
```

---

## 🔍 ระบบกรองคุณภาพ Dataset

Phase 2 มีตัวกรองอัตโนมัติ 3 ชั้น:

| ตัวกรอง | เงื่อนไข | ผลลัพธ์ |
|---|---|---|
| **Off-screen Speaker** | Lip Variance < 2e-6 ตลอดคลิป | ลบทิ้ง — ปากในกล้องไม่ขยับ (คนอื่นพูดแทน) |
| **Caption สั้นเกิน** | น้อยกว่า 2 คำ | ลบทิ้ง — ASR จับได้แค่เสียงสั้น |
| **WPS ผิดปกติ** | > 12 คำ/วิ หรือ < 0.3 คำ/วิ | ลบทิ้ง — caption ไม่สัมพันธ์กับความยาวคลิป |
| **Multi-face Frame** | เจอหน้า > 1 คนในเฟรม | ข้ามเฟรมนั้น — ไม่นับเป็นช่วงพูด |

---

## 🏗️ สเปกโมเดล

| องค์ประกอบ | รายละเอียด |
|---|---|
| **Parameters** | 353M |
| **Architecture** | 32-Layer Hybrid Transformer |
| **Attention** | สลับ Local (7-frame window) / Global |
| **Input** | 180 มิติ/เฟรม (60 Landmarks × 3 แกน x,y,z) |
| **Landmarks** | ปาก 40 + ขากรรไกร 15 + จมูก 5 จุด |
| **Loss** | CTC Loss (รองรับภาษาที่ไม่มีการแบ่งคำชัดเจน) |
| **ASR** | Typhoon ASR (ภาษาไทย) |
| **Hardware** | CUDA / Apple MPS / CPU |

---

## 📊 Dataset บน Hugging Face

🤗 [Phonsiri/Thai-Lip-Reading-Dataset](https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset)

---

*โปรเจกต์นี้เป็นงานวิจัยระบบอ่านริมฝีปากภาษาไทย (Thai Visual Speech Recognition) พัฒนาโดยนักเรียนชั้น ม.5*
