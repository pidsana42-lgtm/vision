# 🚀 คู่มือแบบสมบูรณ์: ตั้งแต่เตรียมข้อมูล จนถึงเทรนบน Cloud

คู่มือนี้จะแบ่งออกเป็น **2 ส่วนหลัก** คือ:
1. **รันบนเครื่อง Mac ของคุณ:** เพื่อดึงข้อมูล, ตัดหน้า, และเตรียมไฟล์ (เพราะทำบนเครื่องตัวเองฟรีและจัดการง่าย)
2. **รันบน Cloud / Colab:** เพื่อใช้การ์ดจอแรงๆ (GPU) ในการเทรน AI

---

## 🖥️ ส่วนที่ 1: เตรียมข้อมูลบนเครื่อง Mac ของคุณ

ขั้นตอนเหล่านี้ให้รันใน Terminal บนเครื่อง Mac ของคุณในโฟลเดอร์โปรเจกต์ (ม.5) ครับ

### เฟส 1: ดึงคลิปจาก YouTube และแกะสับไตเติ้ล
1. นำลิงก์ YouTube ไปวางเรียงกันในไฟล์ `PREPARE/url.txt` (บรรทัดละ 1 ลิงก์)
2. รันคำสั่งดึงข้อมูล (โมเดล Whisper จะถอดเสียงให้):
```bash
python3 PREPARE/collect.py --batch PREPARE/url.txt
```
> วิดีโอดิบจะถูกเก็บไว้ที่ `DATASET/raw/videos` และมีไฟล์ตารางที่ `DATASET/raw/labels.csv`

### เฟส 2: คัดกรองและสกัดเฉพาะภาพปาก (Mouth ROI)
คำสั่งนี้จะครอปเฉพาะปากเป็นภาพขาวดำ 96x96 และกรองคลิปที่คุณภาพต่ำหรือประโยคสั้นเกินไปทิ้ง:
```bash
python3 PREPARE/extract.py
```
> วิดีโอที่ใช้ได้จริงจะไปอยู่ที่ `DATASET/ready/videos`

### เฟส 3: แปลงคำบรรยายเป็นรหัสตัวเลข (Tokenization)
รันคำสั่งนี้เพื่อสร้างตารางให้ AI อ่านเข้าใจ:
```bash
python3 PREPARE/prepare_auto_avsr.py
```
> จะได้ไฟล์ `DATASET/ready/auto_avsr_train.csv` (ตารางพร้อมเทรน)

### เฟส 4: พุชข้อมูลขึ้น Hugging Face
ถ้าทำเฟส 1-3 เสร็จแล้ว มีคลิปใหม่เพิ่มเข้ามา ให้อัปโหลดขึ้น Hugging Face เพื่อเอาไปเทรนบน Cloud:
```bash
# ถ้าตอนรัน extract.py ไม่ได้พุช หรืออยากพุชแบบบังคับ:
python3 PREPARE/extract.py
```
*(ถ้ารันโค้ดผมเมื่อกี้ มันพุชให้ครบหมดแล้วครับ!)*

---

## ☁️ ส่วนที่ 2: การเทรนบน Google Colab / Cloud GPU

เนื่องจากโค้ดบนเครื่อง Mac ของคุณถูกปรับแต่งเพื่อภาษาไทยไว้สมบูรณ์แบบ 100% แล้ว เราจะดึงโค้ดทั้งหมดผ่าน GitHub ของคุณเองเลยครับ ทำให้การเทรนบน Cloud ง่ายและคลีนสุดๆ 

### 1. ดาวน์โหลดโปรเจกต์ของคุณเองและติดตั้งไลบรารี
```bash
!git clone https://github.com/pidsana42-lgtm/vision.git
%cd vision/auto_avsr
!sudo apt-get update && sudo apt-get install -y ffmpeg
!pip install torch torchvision torchaudio torchcodec pytorch-lightning sentencepiece av opencv-python wandb huggingface_hub "numpy<2"
```

### 2. ล็อคอินเข้า Hugging Face (ป้องกันการโดนจำกัดความเร็วเน็ต) 🔑
> *เพื่อไม่ให้โหลด Dataset ค้าง ให้เอา Token (ที่ขึ้นต้นด้วย `hf_...`) มาใส่ในนี้แล้วรันครับ*
```python
from huggingface_hub import login

# ✏️ เปลี่ยนตรงนี้เป็น Token ของคุณ
HF_TOKEN = "ใส่_TOKEN_ของคุณที่นี่"

login(token=HF_TOKEN)
print("✅ เข้าสู่ระบบ Hugging Face สำเร็จ! พร้อมดาวน์โหลดด้วยความเร็วสูงสุด 🚀")
```

### 3. ดาวน์โหลด Dataset และ Checkpoint
> *เพื่อแก้ปัญหาจุกจิกเรื่องเน็ตและ Rate Limit เราจะโหลดก้อน ZIP มาแตกไฟล์เลย เร็วและชัวร์ที่สุดครับ*
```bash
# 1. สร้างโฟลเดอร์สำหรับเก็บ Dataset
!mkdir -p DATASET

# 2. โหลดไฟล์ทั้งหมดจาก Hugging Face
!wget -nc https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset/resolve/main/videos.zip
!rm -f DATASET/auto_avsr_train.csv DATASET/auto_avsr_val.csv
!wget https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset/resolve/main/auto_avsr_train.csv -O DATASET/auto_avsr_train.csv
!wget https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset/resolve/main/auto_avsr_val.csv -O DATASET/auto_avsr_val.csv

# 3. แตกไฟล์วิดีโอ (ใช้ -q เพื่อซ่อนข้อความยาวๆ)
# ต้องเอาไปไว้ในโฟลเดอร์ thai_vsr เพื่อให้ตรงกับโครงสร้างในไฟล์ CSV
!unzip -q -o videos.zip -d DATASET/thai_vsr/


```

```python
import os

print("🧹 กำลังเคลียร์รายชื่อไฟล์ผี (ที่ไม่มีวิดีโออยู่จริง) ออกจาก CSV...")
def clean_csv(file_path):
    if not os.path.exists(file_path): return
    with open(file_path, "r") as f: lines = f.readlines()
    
    valid = []
    for line in lines:
        video_path = line.split(",")[1]
        full_path = os.path.join("DATASET/thai_vsr", video_path)
        if os.path.exists(full_path):
            valid.append(line)
            
    with open(file_path, "w") as f: f.writelines(valid)
    print(f"✅ คลีน {file_path}: จาก {len(lines)} เหลือ {len(valid)} คลิป")

clean_csv("DATASET/auto_avsr_train.csv")
clean_csv("DATASET/auto_avsr_val.csv")
print("✅ พร้อมเทรนแล้ว!")
```

### 3.5. ปรับจูนรหัส (Freeze Encoder) เพื่อความเร็วและความแม่นยำ ⚡
> *เนื่องจากการเทรนทั้งโมเดล 250 ล้านพารามิเตอร์ตั้งแต่ต้นจะใช้เวลานานและทำลายค่าน้ำหนักที่เคยเรียนรู้การอ่านปากมา ให้รันโค้ดนี้เพื่อแช่แข็ง (Freeze) ส่วนสายตาไว้ แล้วเทรนเฉพาะส่วนแปลผลภาษาไทย*
```python
import os

# หาระบุตำแหน่งไฟล์ lightning.py ให้เจอ
filepath = "lightning.py"
if not os.path.exists(filepath):
    filepath = "auto_avsr/lightning.py"
if not os.path.exists(filepath):
    filepath = "/content/vision/auto_avsr/lightning.py"

if os.path.exists(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    # แทรกโค้ด Freeze หลังจากโหลด Weights
    old_block = """                self.model.encoder.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
            else:"""
                
    new_block = """                self.model.encoder.load_state_dict(tmp_ckpt)
                print("Pretrained weights of the frontend, proj_encoder and encoder component are loaded successfully.")
                
                # Freeze the pretrained layers so we only train CTC and Decoder
                for param in self.model.frontend.parameters():
                    param.requires_grad = False
                for param in self.model.proj_encoder.parameters():
                    param.requires_grad = False
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                print("🔥 แช่แข็ง (Freeze) Frontend และ Encoder แล้ว! โมเดลจะฝึกแค่ CTC กับ Decoder")
            else:"""

    if "param.requires_grad = False" not in code:
        code = code.replace(old_block, new_block)

    # ให้ Optimizer อัปเดตเฉพาะส่วนที่ไม่ได้แช่แข็ง
    old_opt = """    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))"""
            
    new_opt = """    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))"""

    code = code.replace(old_opt, new_opt)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)

    print("✅ อัปเดต lightning.py เปิดระบบ Freeze สำเร็จ! ลุยเทรนต่อได้เลยครับ")
else:
    print("❌ หาไฟล์ไม่เจอ ลองเช็คดูว่ารัน git clone แล้วหรือยังครับ")
```

### 4. ล็อกอิน Hugging Face และเริ่มทำการเทรน (Training) 🚀
> *ระบบจะอัพโหลด Checkpoint ขึ้น Hugging Face อัตโนมัติทุกๆ 50 Epoch*
> *และถ้า Colab หลุด พอรันเซลล์นี้ใหม่ มันจะดาวน์โหลด Checkpoint ล่าสุดจาก HF มาเทรนต่อให้เอง!*

**4.1 รันเซลล์นี้เพื่อเช็ค/ดาวน์โหลด Checkpoint ล่าสุดจาก Hugging Face**
```python
# ใส่ Token ของ Hugging Face ที่มีสิทธิ์ Write (เพื่อให้มันอัพโหลดโมเดลกลับไปได้)
HF_TOKEN = "ใส่_TOKEN_ของพี่ตรงนี้"

import os
from huggingface_hub import login, HfApi, hf_hub_download

# ล็อกอิน
login(token=HF_TOKEN)

# ดาวน์โหลดไฟล์ Base Weights
if not os.path.exists("vsr_trlrs2lrs3vox2avsp_base.pth"):
    print("กำลังดาวน์โหลด Base Weights...")
    hf_hub_download(repo_id='Phonsiri/Thai-Lip-Reading-Checkpoints', filename='vsr_trlrs2lrs3vox2avsp_base.pth', local_dir='.')

repo_id = "Phonsiri/Thai-Lip-Reading-Checkpoints"
api = HfApi()

try:
    files = api.list_repo_files(repo_id=repo_id)
    # หาไฟล์ที่ชื่อขึ้นต้นด้วย epoch_ 
    epoch_files = [f for f in files if f.startswith("epoch_") and f.endswith(".ckpt")]
    
    if epoch_files:
        # หา epoch ที่ตัวเลขมากที่สุด
        latest_file = max(epoch_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        print(f"🔥 พบ Checkpoint ล่าสุดบน HF: {latest_file}")
        print("⏳ กำลังดาวน์โหลดเพื่อนำมาเทรนต่อ (Resume)...")
        local_path = hf_hub_download(repo_id=repo_id, filename=latest_file, local_dir=".")
        
        # ส่งพารามิเตอร์ให้เซลล์ถัดไปรันต่อ
        with open("resume_args.txt", "w") as f:
            f.write(f"--ckpt-path {latest_file}")
        print("✅ พร้อมเทรนต่อแล้ว!")
    else:
        print("💡 ไม่พบ Checkpoint เดิมบน HF (ระบบจะเริ่มเทรนใหม่ตั้งแต่ Epoch 0)")
        with open("resume_args.txt", "w") as f:
            f.write("")
            
except Exception as e:
    print(f"⚠️ ไม่สามารถดึงข้อมูลจาก HF ได้: {e}")
    with open("resume_args.txt", "w") as f:
        f.write("")
```

**4.2 รันเซลล์นี้เพื่อเริ่มเทรน (Training)**
```bash
%%bash
# อ่านค่าคำสั่ง Resume (ถ้ามี)
RESUME_ARGS=$(cat resume_args.txt)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled SLURM_JOB_ID=1 python train.py \
    --exp-dir ./exp \
    --exp-name thai_lip_reading \
    --modality video \
    --root-dir DATASET \
    --train-file DATASET/auto_avsr_train.csv \
    --val-file DATASET/auto_avsr_val.csv \
    --num-nodes 1 \
    --gpus 1 \
    --pretrained-model-path vsr_trlrs2lrs3vox2avsp_base.pth \
    --transfer-encoder \
    --max-epochs 50 \
    --max-frames 600 \
    --lr 1e-4 \
    --ctc-weight 0.15 $RESUME_ARGS
```

### 5. ทดสอบการอ่านปาก (Inference) ระหว่าง/หลังเทรน 🎬
> *คุณสามารถเปิดเซลล์ใหม่แล้วรันโค้ดนี้ เพื่อสุ่มวิดีโอจาก Dataset มาให้โมเดลลองอ่านปากดูได้เลยครับ (ดูพัฒนาการของมัน)*

```python
import os
import sys
import glob
import random
import torch

# 1. ตั้งค่า Path สำหรับ Jupyter Notebook
if not os.getcwd().endswith('vision'):
    %cd /teamspace/studios/this_studio/vision/
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'auto_avsr'))

# 2. จำเป็นต้องมีไฟล์ base weights ไว้เพื่อโหลดโครงสร้างโมเดลตอนแรก
if not os.path.exists("vsr_trlrs2lrs3vox2avsp_base.pth"):
    print("กำลังคัดลอกไฟล์ Base Weights...")
    !cp ../vsr_trlrs2lrs3vox2avsp_base.pth . || wget https://huggingface.co/Phonsiri/Thai-Lip-Reading-Checkpoints/resolve/main/vsr_trlrs2lrs3vox2avsp_base.pth

from auto_avsr.datamodule.transforms import VideoTransform
from auto_avsr.lightning import ModelModule

# 3. ค้นหาไฟล์ Checkpoint ล่าสุดที่เพิ่งเทรน
ckpt_files = glob.glob("auto_avsr/exp/thai_lip_reading/*.ckpt")
if not ckpt_files:
    print("❌ ไม่พบไฟล์โมเดล! กรุณาเช็คว่าได้รันการเทรนสำเร็จหรือไม่")
    sys.exit(1)
latest_ckpt = max(ckpt_files, key=os.path.getmtime)
print(f"🧠 กำลังโหลดโมเดลล่าสุด: {os.path.basename(latest_ckpt)}")

# โหลดโมเดล
modelmodule = ModelModule.load_from_checkpoint(latest_ckpt)
modelmodule.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelmodule.to(device)
print("✅ โหลดโมเดลเสร็จสิ้น!")

# 4. สุ่มวิดีโอจากชุดทดสอบ (Validation/Test Set)
val_csv = None
for path in ["DATASET/auto_avsr_val.csv", "auto_avsr/DATASET/auto_avsr_val.csv", "DATASET/ready/auto_avsr_val.csv"]:
    if os.path.exists(path):
        val_csv = path
        break

videos = []
if val_csv:
    dataset_root = os.path.dirname(val_csv) # เช่น 'DATASET' หรือ 'auto_avsr/DATASET'
    import csv
    with open(val_csv, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) > 1:
                # row[0] คือ "thai_vsr", row[1] คือ "videos/xxx.mp4"
                videos.append(os.path.join(dataset_root, row[0], row[1]))

if not videos:
    print("❌ ไม่พบรายการวิดีโอในชุดทดสอบ (กำลังสุ่มจากทั้งหมดแทน...)")
    videos = glob.glob("**/*.mp4", recursive=True) # ค้นหาทั่วทั้งโปรเจกต์เลย

if not videos:
    print("❌ หาวิดีโอไม่เจอเลยครับ (ลองเช็คว่ารัน unzip สำเร็จหรือไม่)")
else:
    test_video = random.choice(videos)
    print(f"🎬 วิดีโอที่สุ่มมาทดสอบ: {os.path.basename(test_video)}")
    
    # 5. โหลดและแปลงวิดีโอ (Crop 88x88 และ Normalize) ด้วย OpenCV
    import cv2
    import numpy as np
    
    cap = cv2.VideoCapture(test_video)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # แปลงเป็นภาพขาวดำ (1 Channel)
        frames.append(frame)
    cap.release()
    
    video_tensor = torch.tensor(np.array(frames)).float().unsqueeze(1) # เพิ่มมิติ Channel เข้าไป -> Shape: (T, 1, H, W)
    
    # ดึง VideoTransform มาใช้ครอบ
    transform = VideoTransform(subset="test")
    video_tensor = transform(video_tensor) # Shape: (T, 1, 88, 88)
    video_tensor = video_tensor.to(device)
    
    print(f"Frame Count: {video_tensor.shape[0]}")
    
    # 6. ให้โมเดลอ่านปาก (Inference)
    with torch.no_grad():
        x = modelmodule.model.frontend(video_tensor.unsqueeze(0))
        x = modelmodule.model.proj_encoder(x)
        enc_feat, _ = modelmodule.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        
        # ใช้ Beam Search ดึงคำตอบที่น่าจะเป็นไปได้มากสุด
        from auto_avsr.lightning import get_beam_search_decoder
        beam_search = get_beam_search_decoder(modelmodule.model, modelmodule.token_list, ctc_weight=0.3)
        nbest_hyps = beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        prediction = modelmodule.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        
    # ดึงเฉลยจากไฟล์ auto_avsr_val.csv มาเทียบ (แกะรหัส Token กลับมาเป็นข้อความ)
    actual_label = "ไม่พบเฉลย"
    if os.path.exists(val_csv):
        with open(val_csv, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 4 and os.path.basename(test_video) in row[1]:
                    # Token ID จะอยู่ในคอลัมน์ที่ 4 (index 3) เช่น "66 38 52 11"
                    actual_token_ids = torch.tensor([int(i) for i in row[3].split() if i.isdigit()])
                    actual_label = modelmodule.text_transform.post_process(actual_token_ids)
                    break
        
    print("\n" + "🔥" * 20)
    print(f"🎯 เฉลยที่แท้จริง: {actual_label}")
    print(f"🤖 ผลลัพธ์ที่ AI อ่านได้: {prediction}")
    print("🔥" * 20 + "\n")
```
