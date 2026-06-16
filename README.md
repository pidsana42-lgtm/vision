# 🎬 Thai Lip Reading — Visual Speech Recognition (Auto-AVSR)

ระบบอ่านริมฝีปากภาษาไทย สร้างขึ้นโดยมีเป้าหมายเพื่อ Fine-tune กับสถาปัตยกรรมระดับ State-of-the-Art อย่าง **[Auto-AVSR](https://github.com/mpc001/auto_avsr)** 
โปรเจกต์นี้มาพร้อมกับระบบเตรียมข้อมูลแบบ 2 เฟส ที่ดาวน์โหลด, สกัดวิดีโอส่วนปาก (Mouth ROI), กรองคุณภาพอัตโนมัติ และปรับโครงสร้างให้พร้อมเทรนกับ Auto-AVSR ในทันที

---

## 🔐 การตั้งค่าสภาพแวดล้อม (Environment & Keys)

1. **ติดตั้งไลบรารีที่จำเป็นสำหรับการถอดเสียงด้วย AI บนเครื่อง (Offline ASR):**
```bash
pip install -r requirements.txt
pip install opencv-python transformers torch accelerate python-dotenv
```

2. **ตั้งค่าคีย์ (ถ้าจำเป็น):**
สร้างไฟล์ `.env` ไว้ในโฟลเดอร์หลัก และใส่รหัสต่างๆ ลงไป (ระบบจะอ่านอัตโนมัติ)
```bash
# สำหรับอัปโหลด Dataset ขึ้น Hugging Face หรือโหลดโมเดลที่มีการล็อคสิทธิ์
HF_TOKEN="hf_xxx..."
```
*(หมายเหตุ: โปรเจกต์ไม่ได้ใช้ API ของ OpenTyphoon อีกต่อไป เนื่องจากเราปรับมาใช้โมเดลแปลภาษาแบบ Offline เต็มรูปแบบบนเครื่องของคุณแล้ว เพื่อความรวดเร็วและฟรี 100%)*

---

## 📂 โครงสร้างโปรเจกต์

```
ม.5/
├── PREPARE/
│   ├── collect.py             ← Phase 1: ดาวน์โหลดวิดีโอ + ถอดเสียงด้วย Typhoon → DATASET/raw/
│   ├── extract.py             ← Phase 2: ทำ Affine Alignment + Crop ปาก 96x96 (.mp4) → DATASET/ready/
│   ├── prepare_auto_avsr.py   ← สร้างไฟล์ train.csv ฟอร์แมตเดียวกับ Auto-AVSR
│   ├── 20words_mean_face.npy  ← อ้างอิง 68 จุด สำหรับการทำ Affine Transform ให้หน้าตรง
│   └── url.txt                ← รายการ YouTube URL สำหรับ Batch Mode
│
├── DATASET/
│   ├── raw/                   ← Phase 1 output (วิดีโอเต็มหน้า ยังไม่กรอง)
│   │   ├── videos/
│   │   └── labels.csv
│   └── ready/                 ← Phase 2 output (วิดีโอปาก 96x96 พร้อมเทรน)
│       ├── videos/            ← ไฟล์ .mp4 เฉพาะส่วนปาก
│       ├── labels.csv         ← ข้อมูลดิบที่ผ่านการกรอง
│       └── auto_avsr_train.csv ← ไฟล์ Label ที่พร้อมส่งเข้าโมเดล Auto-AVSR
│
├── dataset.py                 ← เครื่องมือเสริมสำหรับจัดการ Dataset
├── vocabulary.py              ← Thai Tokenizer สำหรับแปลงภาษาไทย+อังกฤษเป็น Token ID
└── requirements.txt
```

---

## 🔠 ระบบ Tokenizer (รองรับไทย + อังกฤษ)

เพื่อป้องกันโมเดลอ่านค่าผิดพลาดกรณีมีคำทับศัพท์ภาษาอังกฤษผสมในคลิป ไฟล์ `vocabulary.py` ได้ถูกออกแบบมาให้รองรับ:
- ตัวอักษรพยัญชนะและสระภาษาไทย (ก-ฮ, สระต่างๆ)
- ตัวเลขไทยและอารบิก (๐-๙, 0-9)
- **ตัวอักษรภาษาอังกฤษพิมพ์เล็กและพิมพ์ใหญ่ (a-z, A-Z)**
- เครื่องหมายวรรคตอนพื้นฐาน
*ระบบนี้ถูกเชื่อมเข้ากับสถาปัตยกรรมของ Auto-AVSR โดยอัตโนมัติแล้ว!*

---

## 🚀 คู่มือการสร้าง Dataset และเทรนโมเดล (แบบละเอียด)

โปรเจกต์นี้แบ่งการทำงานออกเป็น 3 เฟสหลัก (ดึงข้อมูลดิบ -> สกัดภาพปาก -> เตรียมไฟล์เทรน) และจบด้วยการเทรนผ่าน Auto-AVSR

### Phase 1 — เก็บข้อมูลดิบ (Raw Data) ด้วย `collect.py`
ขั้นตอนนี้จะดาวน์โหลดวิดีโอจาก YouTube, ตรวจจับช่วงที่มีการพูด (VAD), และถอดเสียงพูดภาษาไทยโดยใช้โมเดล **Typhoon-Whisper Large v3 (Offline Mode)** ซึ่งเป็นโมเดลที่แม่นยำที่สุดในปัจจุบัน โดยรันผ่านการ์ดจอในเครื่อง (เช่น ชิป Apple M4 `mps`) ทำให้ทำงานได้รวดเร็ว ปลอดภัย และฟรี 100%

**1. เตรียมรายการวิดีโอ:** 
นำ URL ของ YouTube ไปวางไว้ในไฟล์ `PREPARE/url.txt` (บรรทัดละ 1 ลิงก์)

**2. เริ่มการดึงข้อมูล:**
```bash
# รันเพื่อประมวลผลวิดีโอทั้งหมดใน url.txt (แนะนำใช้ 'python' เฉยๆ หากอยู่ใน Anaconda)
python PREPARE/collect.py --batch PREPARE/url.txt
```
*💡 เกร็ดความรู้:* 
- **ครั้งแรกที่รัน:** โปรแกรมจะดาวน์โหลดก้อนสมองโมเดล AI ขนาด ~3GB มาเก็บไว้ในเครื่องอัตโนมัติ (อาจใช้เวลาสักพัก) หลังจากนั้นจะประมวลผลได้รวดเร็วทันที
- **ระบบ Auto-Resume:** หากหยุดโปรแกรมกลางคัน (Ctrl+C) หรือเกิดปัญหา คุณสามารถรันคำสั่งเดิมซ้ำได้เลย โปรแกรมจะข้ามวิดีโอที่ทำเสร็จแล้วใน `labels.csv` ให้อัตโนมัติ!
- ข้อมูลที่ได้จะไปอยู่ในโฟลเดอร์ `DATASET/raw/videos/` พร้อมกับไฟล์คำบรรยายที่ `DATASET/raw/labels.csv`

---

### Phase 2 — ตัดเฉพาะบริเวณปาก (Mouth ROI) ด้วย `extract.py`
โมเดล Auto-AVSR ต้องการภาพวิดีโอขนาด 96x96 แบบขาวดำที่โฟกัสเฉพาะปาก ขั้นตอนนี้จะทำการ "Crop ปาก" และกรองคลิปที่คุณภาพต่ำทิ้ง (เช่น หน้าหัน, มีหลายคน, ปากไม่ขยับ)

**1. เริ่มการสกัดวิดีโอ:**
```bash
# แนะนำให้ใช้ --no-push หากยังไม่อยากอัปโหลดขึ้น Hugging Face ทันที
python3 PREPARE/extract.py --no-push
```
*💡 เกร็ดความรู้:*
- ระบบจะใช้ MediaPipe หา Landmark บนใบหน้า และทำ Affine Transform เพื่อหมุนหน้าให้ตั้งตรงเสมอ
- วิดีโอที่เป็น "ผลลัพธ์พร้อมใช้" จะถูกเซฟไว้ที่ `DATASET/ready/videos/`

---

### Phase 3 — เตรียมไฟล์ Label ให้โมเดลเข้าใจ (`prepare_auto_avsr.py`)
การเทรนด้วย Auto-AVSR ต้องใช้ไฟล์ CSV ที่มีโครงสร้างเฉพาะ (`dataset,basename,video_length,token_id_str`)

**1. รันสคริปต์แปลงไฟล์:**
```bash
python3 PREPARE/prepare_auto_avsr.py
```
- คุณจะได้ไฟล์ `DATASET/ready/auto_avsr_train.csv` ที่แปลงคำภาษาไทยเป็นรหัสตัวเลข (Token IDs) เรียบร้อยแล้ว

---

## 🤖 วิธีเทรนโมเดล (Cross-lingual Fine-tuning)

เราแนะนำให้เทรนโมเดลบน **Google Colab** (หรือเซิร์ฟเวอร์ที่มีการ์ดจอ NVIDIA) เพื่อความรวดเร็วและหลีกเลี่ยงปัญหาความเข้ากันไม่ได้ของ PyTorch DDP บน Mac

**1. เปิด Google Colab และเปิดใช้งาน GPU (T4 หรือ A100)**

**2. ติดตั้งไลบรารีและดาวน์โหลดโค้ด:**
```bash
!git clone https://github.com/mpc001/auto_avsr.git
%cd auto_avsr
!pip install -r requirements.txt
!pip install pytorch-lightning wandb
```

**3. โหลด Dataset และ Checkpoint จาก Hugging Face:**
นำ Dataset ที่คุณพุชไว้ และดาวน์โหลด Pre-trained Model (`vsr_trlrs2lrs3vox2avsp_base.pth`) เตรียมไว้
```python
from huggingface_hub import snapshot_download
import os

# โหลด Dataset
dataset_dir = snapshot_download(repo_id="Phonsiri/Thai-Lip-Reading-Dataset", repo_type="dataset")
# โหลด Checkpoint (เปลี่ยน repo_id ให้ตรงกับที่คุณพุชโมเดลไว้)
checkpoint_dir = snapshot_download(repo_id="Phonsiri/Thai-Lip-Reading-Checkpoints", repo_type="model")
```

**4. รันสคริปต์เตรียมไฟล์ CSV:**
```bash
!python ../PREPARE/prepare_auto_avsr.py --input {dataset_dir} --output ./
```

**5. เริ่มรันคำสั่งเทรน:**
> **ข้อควรระวัง:** ต้องใส่ `--transfer-encoder` เพื่อให้โหลดเฉพาะความรู้ด้านการมองเห็น และสุ่มการถอดรหัสภาษา (Decoder) ใหม่ให้เป็นภาษาไทย

```bash
!python train.py \
    --exp-dir ./exp \
    --exp-name thai_lip_reading \
    --modality video \
    --root-dir {dataset_dir} \
    --train-file train.csv \
    --val-file val.csv \
    --num-nodes 1 \
    --gpus 1 \
    --pretrained-model-path {checkpoint_dir}/vsr_trlrs2lrs3vox2avsp_base.pth \
    --transfer-frontend \
    --transfer-encoder \
    --max-epochs 50 \
    --max-frames 800 \
    --lr 1e-4
```

*💡 คำแนะนำ:* หากขึ้น Out of Memory บน Colab ให้ปรับลด `--max-frames` ลง (เช่น 600 หรือ 400)

---

## 📊 Dataset บน Hugging Face

🤗 [Phonsiri/Thai-Lip-Reading-Dataset](https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset)

---

*โปรเจกต์นี้เป็นระบบเตรียมข้อมูลและท่อส่ง (Pipeline) สำหรับงานวิจัยระบบอ่านริมฝีปากภาษาไทย (Thai Visual Speech Recognition) พัฒนาโดยนักเรียนชั้น ม.5 เพื่อใช้งานร่วมกับโมเดล Auto-AVSR*
