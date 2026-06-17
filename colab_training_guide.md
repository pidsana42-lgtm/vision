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
!wget -nc https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset/resolve/main/auto_avsr_train.csv -O DATASET/auto_avsr_train.csv
!wget -nc https://huggingface.co/datasets/Phonsiri/Thai-Lip-Reading-Dataset/resolve/main/auto_avsr_val.csv -O DATASET/auto_avsr_val.csv

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

### 4. เริ่มทำการเทรน (Training) 🚀
> *โค้ดนี้สามารถรันได้เลยโดยไม่ต้องกังวลเรื่องตัวแปรแล้วครับ*
```bash
# โหลดโมเดลตั้งต้นด้วย Python (ใช้ได้ทุกระบบ 100% แน่นอน)
!python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Phonsiri/Thai-Lip-Reading-Checkpoints', filename='vsr_trlrs2lrs3vox2avsp_base.pth', local_dir='.')"

# รันเทรนโมเดล
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled SLURM_JOB_ID=1 python train.py \
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
    --lr 1e-3 \
    --ctc-weight 0.9
```

### 5. ทดสอบการอ่านปาก (Inference) ระหว่าง/หลังเทรน 🎬
> *คุณสามารถเปิดเซลล์ใหม่แล้วรันโค้ดนี้ เพื่อสุ่มวิดีโอจาก Dataset มาให้โมเดลลองอ่านปากดูได้เลยครับ (ดูพัฒนาการของมัน)*

```python
import os
import glob
import torch
import torchvision
import sys
import argparse
import random

# ดึงคลาสจากโปรเจกต์
if os.path.exists("auto_avsr"):
    os.chdir("auto_avsr")
elif os.path.exists("vision/auto_avsr"):
    os.chdir("vision/auto_avsr")
sys.path.append(os.getcwd())

try:
    from lightning import ModelModule
except ModuleNotFoundError:
    print("❌ ไม่พบไฟล์ lightning.py! กรุณาตรวจสอบว่าคุณอยู่ในโฟลเดอร์ auto_avsr แล้ว")
    sys.exit(1)

# 1. หาไฟล์ Checkpoint (.ckpt) ล่าสุดที่เซฟไว้
ckpt_files = glob.glob("./exp/thai_lip_reading/*.ckpt")
if not ckpt_files:
    print("❌ ไม่พบไฟล์โมเดล! กรุณาเช็คว่าได้รันการเทรนสำเร็จหรือไม่")
    sys.exit(1)

# เรียงตามเวลาแก้ไขล่าสุด (เพื่อให้ได้โมเดลรอบล่าสุดจริงๆ)
latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        
print(f"🧠 กำลังโหลดโมเดลล่าสุด: {os.path.basename(latest_ckpt)}")

# 2. ชี้เป้าไปที่โฟลเดอร์ Dataset ที่เราเพิ่งโหลดและแตกไฟล์ไว้
dataset_dir = "../DATASET"

# ค้นหาวิดีโอทั้งหมดใน Dataset
videos = glob.glob(f"{dataset_dir}/**/*.mp4", recursive=True)

if not videos:
    print(f"❌ หาวิดีโอไม่เจอใน {dataset_dir} เลยครับ (ลองเช็คว่ารัน unzip สำเร็จหรือไม่)")
else:
    test_video = random.choice(videos)
    print(f"🎬 วิดีโอที่สุ่มมาทดสอบ: {os.path.basename(test_video)}")
        
        # 3. โหลดและแปลงวิดีโอ (Crop 88x88 และ Normalize) ด้วย OpenCV
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
        
        vid_np = np.stack(frames, axis=0) # (T, H, W)
        vid_np = np.expand_dims(vid_np, axis=-1) # (T, H, W, 1)
        vid = torch.from_numpy(vid_np).permute((0, 3, 1, 2))  # (T, 1, H, W)
        
        T, C, H, W = vid.shape
        th, tw = 88, 88
        i = int(round((H - th) / 2.))
        j = int(round((W - tw) / 2.))
        vid = vid[:, :, i:i+th, j:j+tw] # Crop ตรงกลาง
        
        vid = vid.float() / 255.0
        vid = (vid - 0.421) / 0.165     # Normalize
        
        # เตรียม Device (ย้ายไปรันบนการ์ดจอ GPU ถ้ามี)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ กำลังรันบน: {device}")
        
        vid = vid.to(device)
        
        # 4. โหลดโมเดล PyTorch Lightning
        args = argparse.Namespace(modality="video", pretrained_model_path=None)
        modelmodule = ModelModule(args)
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        
        # ดึงเฉพาะ state_dict ส่วนของ model
        if "state_dict" in ckpt:
            states = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
            modelmodule.model.load_state_dict(states)
            
        modelmodule.to(device)
        modelmodule.eval()
        
        # 5. ทำนายผล (Inference)
        print("🗣️ กำลังให้ AI อ่านปาก...")
        with torch.no_grad():
            from lightning import get_beam_search_decoder
            # เปิดใช้งาน Decoder ที่มีความแม่นยำ 24% (ใช้ ctc_weight=0.1 เพื่อให้ Decoder มีน้ำหนัก 0.9)
            beam_search = get_beam_search_decoder(modelmodule.model, modelmodule.token_list, ctc_weight=0.1)
            
            x = modelmodule.model.frontend(vid.unsqueeze(0))
            x = modelmodule.model.proj_encoder(x)
            enc_feat, _ = modelmodule.model.encoder(x, None)
            enc_feat = enc_feat.squeeze(0)
            
            nbest_hyps = beam_search(enc_feat)
            nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
            predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
            prediction = modelmodule.text_transform.post_process(predicted_token_id).replace("<eos>", "")
            
        print("\n" + "🔥" * 20)
        print(f"ผลลัพธ์ที่ AI อ่านได้: {prediction}")
        print("🔥" * 20 + "\n")
```
