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

เมื่อข้อมูลบน Hugging Face ครบแล้ว ให้คุณเปิด Google Colab หรือเซิร์ฟเวอร์ Cloud **เปิด GPU (T4 หรือ A100)** แล้วรันโค้ดต่อไปนี้ทีละช่องครับ:

### 1. ดาวน์โหลดโปรเจกต์และติดตั้งไลบรารี
```bash
!git clone https://github.com/mpc001/auto_avsr.git
%cd auto_avsr
!sudo apt-get update && sudo apt-get install -y ffmpeg
!pip install torch torchvision torchaudio pytorch-lightning sentencepiece av opencv-python wandb huggingface_hub
```

### 1.5 ล็อคอินเข้า Hugging Face (ป้องกันการโดนจำกัดความเร็วเน็ต) 🔑
> *เพื่อไม่ให้โหลด Dataset ค้าง ให้เอา Token (ที่ขึ้นต้นด้วย `hf_...`) มาใส่ในนี้แล้วรันครับ*
```python
from huggingface_hub import login

# ✏️ เปลี่ยนตรงนี้เป็น Token ของคุณ
HF_TOKEN = "ใส่_TOKEN_ของคุณที่นี่"

login(token=HF_TOKEN)
print("✅ เข้าสู่ระบบ Hugging Face สำเร็จ! พร้อมดาวน์โหลดด้วยความเร็วสูงสุด 🚀")
```

### 2. ดาวน์โหลด Dataset ของคุณจาก Hugging Face
```python
from huggingface_hub import snapshot_download
import os

# โหลด Dataset 
dataset_dir = snapshot_download(repo_id="Phonsiri/Thai-Lip-Reading-Dataset", repo_type="dataset")
print(f"Dataset โหลดมาไว้ที่: {dataset_dir}")
```

### 3. ดาวน์โหลด Checkpoint (โมเดลตั้งต้น)
```python
# โหลด Checkpoint
checkpoint_dir = snapshot_download(repo_id="Phonsiri/Thai-Lip-Reading-Checkpoints", repo_type="model")
print(f"Checkpoint โหลดมาไว้ที่: {checkpoint_dir}")
```

### 4. Patch โค้ดป้องกัน Error ทั้งหมด 🛠️
> *รันเซลล์นี้เพียงครั้งเดียวเพื่อแก้ไข 2 จุดที่ไม่รองรับ PyTorch เวอร์ชันใหม่บน Colab*

```python
import os

# --- ตรวจสอบพื้นที่ทำงาน (Working Directory Resolution) ---
if os.path.exists("auto_avsr") and not os.path.exists("datamodule"):
    os.chdir("auto_avsr")
    print(f"🔄 ตรวจพบโฟลเดอร์โครงการ: เปลี่ยนไปรันที่ -> {os.getcwd()}")
elif not os.path.exists("datamodule"):
    raise FileNotFoundError("ไม่พบโฟลเดอร์ datamodule/ ในไดเรกทอรีปัจจุบัน กรุณาตรวจสอบการรัน Git Clone ใน Step 1")

# --- แก้ไขที่ 1: แก้ AssertionError และปรับปรุง Dataloader ใน data_module.py ---
# (ขยายขีดจำกัดเฟรม Validation, ลดจำนวน worker และกรองคลิปยาวเพื่อป้องกัน CUDA OOM)
with open("datamodule/data_module.py", "r") as f:
    code = f.read()

# 1. ปลดล็อก validation limit และลด num_workers, พร้อมเซ็ต batch_size = 4
code = code.replace(
    "self.batch_size = batch_size",
    "self.batch_size = 4"
)

code = code.replace(
    "dataset = CustomBucketDataset(\n            dataset, dataset.input_lengths, 1000, 1, batch_size=self.batch_size\n        )",
    "dataset = CustomBucketDataset(\n            dataset, dataset.input_lengths, max(1000, max(dataset.input_lengths)), 1, batch_size=self.batch_size\n        )"
)

code = code.replace(
    "        num_workers=10,",
    "        num_workers=2,"
)

# 2. ปรับปรุง CustomBucketDataset ให้กรองวิดีโอที่ยาวเกิน max_frames ออกอัตโนมัติ เพื่อเลี่ยงการ AssertionError และ CUDA OOM
target_class = """class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, lengths, max_frames, num_buckets, shuffle=False, batch_size=None
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        self.dataset = dataset

        max_length = max(lengths)
        min_length = min(lengths)

        assert max_frames >= max_length

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths = torch.tensor(lengths)
        bucket_assignments = torch.bucketize(lengths, buckets)

        idx_length_buckets = [
            (idx, length, bucket_assignments[idx]) for idx, length in enumerate(lengths)
        ]
        if shuffle:
            idx_length_buckets = random.sample(
                idx_length_buckets, len(idx_length_buckets)
            )
        else:
            idx_length_buckets = sorted(
                idx_length_buckets, key=lambda x: x[1], reverse=True
            )
        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)"""

replacement_class = """class CustomBucketDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset, lengths, max_frames, num_buckets, shuffle=False, batch_size=None
    ):
        super().__init__()

        assert len(dataset) == len(lengths)

        # Filter out samples longer than max_frames
        valid_indices = [i for i, l in enumerate(lengths) if l <= max_frames]
        if not valid_indices:
            raise ValueError(f"No samples have length <= max_frames ({max_frames})")

        self.dataset = dataset
        self.lengths = [lengths[i] for i in valid_indices]

        max_length = max(self.lengths)
        min_length = min(self.lengths)

        buckets = torch.linspace(min_length, max_length, num_buckets)
        lengths_tensor = torch.tensor(self.lengths)
        bucket_assignments = torch.bucketize(lengths_tensor, buckets)

        idx_length_buckets = [
            (idx, length, bucket_assignments[i]) for i, (idx, length) in enumerate(zip(valid_indices, self.lengths))
        ]
        if shuffle:
            import random
            idx_length_buckets = random.sample(
                idx_length_buckets, len(idx_length_buckets)
            )
        else:
            idx_length_buckets = sorted(
                idx_length_buckets, key=lambda x: x[1], reverse=True
            )
        sorted_idx_length_buckets = sorted(idx_length_buckets, key=lambda x: x[2])
        self.batches = _batch_by_token_count(
            [(idx, length) for idx, length, _ in sorted_idx_length_buckets],
            max_frames,
            batch_size=batch_size,
        )

    def __getitem__(self, idx):
        return [self.dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)"""

code = code.replace(target_class, replacement_class)

with open("datamodule/data_module.py", "w") as f:
    f.write(code)
print("✅ Patch 1: data_module.py สำเร็จ!")


# --- แก้ไขที่ 2: แก้ TypeError ใน cosine.py ---
# (ลบ argument 'verbose' ที่ถูกลบออกจาก PyTorch >= 2.2 แล้ว)
with open("cosine.py", "r") as f:
    code = f.read()

code = code.replace(
    "        verbose=False,\n    ):\n        self.warmup_steps = warmup_epochs * steps_per_epoch\n        self.total_steps = total_epochs * steps_per_epoch\n        super().__init__(optimizer, last_epoch=last_epoch, verbose=verbose)",
    "    ):\n        self.warmup_steps = warmup_epochs * steps_per_epoch\n        self.total_steps = total_epochs * steps_per_epoch\n        super().__init__(optimizer, last_epoch=last_epoch)"
)

with open("cosine.py", "w") as f:
    f.write(code)
print("✅ Patch 2: cosine.py สำเร็จ!")

# --- แก้ไขที่ 3: แก้ AttributeError และเพิ่ม Path Resolution/LFS checker ใน av_dataset.py ---
# (ใช้ OpenCV เป็น fallback แทน torchvision.io.read_video และช่วยกู้คืนพาธกรณีโฟลเดอร์ไม่ตรง รวมถึงเช็คไฟล์ LFS pointer)
with open("datamodule/av_dataset.py", "r") as f:
    code = f.read()

target_load = """def load_video(path):
    \"\"\"
    rtype: torch, T x C x H x W
    \"\"\"
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid"""

replacement_load = """def load_video(path):
    \"\"\"
    rtype: torch, T x C x H x W
    \"\"\"
    try:
        vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
        vid = vid.permute((0, 3, 1, 2))
        return vid
    except (AttributeError, RuntimeError):
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return torch.zeros((0, 3, 112, 112), dtype=torch.uint8)
        vid = np.stack(frames, axis=0)
        vid = torch.from_numpy(vid)
        vid = vid.permute((0, 3, 1, 2))
        return vid"""

code = code.replace(target_load, replacement_load)

target_getitem = """    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}"""

replacement_getitem = """    def __getitem__(self, idx):
        dataset_name, rel_path, input_length, token_id = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        
        if not os.path.exists(path):
            parts = path.split(os.sep)
            try:
                target_subfolder = rel_path.split("/")[0]
                idx_sub = parts.index(target_subfolder)
                rel_path_from_sub = os.sep.join(parts[idx_sub:])
                root_parts = parts[:idx_sub - 1]
                root_dir = os.sep.join(root_parts)
                if not root_dir and path.startswith(os.sep):
                    root_dir = os.sep + root_dir
                
                fallbacks = [
                    os.path.join(root_dir, rel_path_from_sub),
                    os.path.join(root_dir, "dataset", rel_path_from_sub),
                    os.path.join(root_dir, "ready", rel_path_from_sub)
                ]
                for fb in fallbacks:
                    if os.path.exists(fb):
                        path = fb
                        break
            except ValueError:
                pass
                
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at: {path}")
            
        if os.path.getsize(path) < 1000:
            raise ValueError(
                f"File at {path} is too small ({os.path.getsize(path)} bytes). "
                f"It is likely a Git LFS pointer file. Please check your dataset download on Colab."
            )

        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            return {"input": video, "target": token_id}
        elif self.modality == "audio":
            audio = load_audio(path)
            audio = self.audio_transform(audio)
            return {"input": audio, "target": token_id}"""

code = code.replace(target_getitem, replacement_getitem)

with open("datamodule/av_dataset.py", "w") as f:
    f.write(code)
print("✅ Patch 3: av_dataset.py สำเร็จ!")

# --- แก้ไขที่ 4: แสดงผล Loss และความแม่นยำในทุก Epoch ใน lightning.py ---
# (เพิ่ม prog_bar=True และตัว Print ตอนจบรอบ Epoch)
with open("lightning.py", "r") as f:
    code = f.read()

target_step = """        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss"""

replacement_step = """        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=False)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True, prog_bar=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True, prog_bar=False)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True, prog_bar=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True, prog_bar=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_end(self):
        metrics = self.trainer.logged_metrics
        loss = metrics.get("loss_epoch") or metrics.get("loss")
        if loss is not None:
            print(f"\\n🔥 Epoch {self.current_epoch} Training Loss: {loss:.4f}")

    def on_validation_epoch_end(self):
        metrics = self.trainer.logged_metrics
        loss_val = metrics.get("loss_val")
        acc_val = metrics.get("decoder_acc_val")
        if loss_val is not None:
            acc_str = f" | Accuracy: {acc_val:.4f}" if acc_val is not None else ""
            print(f"📊 Epoch {self.current_epoch} Validation Loss: {loss_val:.4f}{acc_str}\\n")"""

code = code.replace(target_step, replacement_step)

with open("lightning.py", "w") as f:
    f.write(code)
print("✅ Patch 4: lightning.py สำเร็จ!")

# --- แก้ไขที่ 5: ปรับความถี่ในการทำ Validation เป็นทุกๆ 10 Epoch ใน train.py ---
with open("train.py", "r") as f:
    code = f.read()

code = code.replace(
    "        reload_dataloaders_every_n_epochs=1,",
    "        reload_dataloaders_every_n_epochs=1,\n        check_val_every_n_epoch=10,"
)

with open("train.py", "w") as f:
    f.write(code)
print("✅ Patch 5: train.py (Validation frequency) สำเร็จ!")

# --- แก้ไขที่ 6: เปลี่ยนไปใช้พจนานุกรมภาษาไทย (ThaiTokenizer) แทนของภาษาอังกฤษ 🇹🇭 ---
# ⚠️ สำคัญมาก: คุณต้องอัปโหลดไฟล์ vocabulary.py จากโฟลเดอร์โครงการของคุณ ขึ้นไปไว้ในโฟลเดอร์เดียวกับโปรเจกต์บน Cloud ด้วย!
with open("datamodule/transforms.py", "r") as f:
    code = f.read()

target_text_transform = """class TextTransform:
    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        self.token_list = [line.split()[0] for line in open(dict_path, "r")]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.token_list.index(t) for t in tokens]
        return torch.tensor(token_ids)

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self.spm.DecodePieces([self.token_list[t] for t in token_ids])
        return text

    def _ids_to_str(self, token_ids, char_list):
        return self.post_process(torch.tensor(token_ids))"""

replacement_text_transform = """class TextTransform:
    \"\"\"Mapping Dictionary Class for Thai tokenization.\"\"\"

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):
        import sys
        import os
        # Add parent folder to sys.path to import vocabulary.py
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from vocabulary import ThaiTokenizer

        self.tokenizer = ThaiTokenizer()
        self.token_list = self.tokenizer.vocab
        self.ignore_id = -1

    def tokenize(self, text):
        token_ids = self.tokenizer.encode(text)
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self.tokenizer.decode(token_ids.tolist())
        return text

    def _ids_to_str(self, token_ids, char_list):
        return self.post_process(torch.tensor(token_ids))"""

code = code.replace(target_text_transform, replacement_text_transform)

with open("datamodule/transforms.py", "w") as f:
    f.write(code)
print("✅ Patch 6: transforms.py (ThaiTokenizer) สำเร็จ!")
```



**Patch 7: เพิ่มระบบกลับด้านวิดีโอ (Horizontal Flip Augmentation) เพื่อเบิ้ลจำนวนข้อมูล**
```python
with open("datamodule/transforms.py", "r") as f:
    code = f.read()

target_video_transform = """class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                torchvision.transforms.RandomCrop(88),"""

replacement_video_transform = """class RandomHorizontalFlipVideo(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if random.random() < self.p:
            return torch.flip(x, [-1])
        return x

class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(lambda x: x / 255.0),
                RandomHorizontalFlipVideo(p=0.5),
                torchvision.transforms.RandomCrop(88),"""

if "RandomHorizontalFlipVideo" not in code:
    code = code.replace(target_video_transform, replacement_video_transform)
    with open("datamodule/transforms.py", "w") as f:
        f.write(code)
    print("✅ Patch 7: เพิ่ม Horizontal Flip Augmentation สำเร็จ!")
```

### 4.5 เตรียมไฟล์ Labels และแบ่งข้อมูลสำหรับ Train/Val 📊
> *รันสคริปต์นี้เพื่อแปลง `labels.csv` ให้ตรงกับรูปแบบที่ Auto-AVSR ต้องการ (ต้องนับเฟรมวิดีโอและแปลงคำเป็นตัวเลข) และแบ่ง 80/20*

```python
import os
import random
import cv2
import sys

# ดึงตัวตัดคำภาษาไทยมาใช้
sys.path.append(os.getcwd())
from vocabulary import ThaiTokenizer
tokenizer = ThaiTokenizer()

input_csv = os.path.join(dataset_dir, "labels.csv")
train_csv = os.path.join(dataset_dir, "train_labels.csv")
val_csv = os.path.join(dataset_dir, "val_labels.csv")

with open(input_csv, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

# ข้ามบรรทัดแรกถ้าเป็น Header
if len(lines) > 0 and lines[0].startswith("video"):
    lines = lines[1:]

random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * 0.8)
train_lines_raw = lines[:split_idx]
val_lines_raw = lines[split_idx:]

def process_lines(raw_lines):
    out_lines = []
    for line in raw_lines:
        if "," not in line: continue
        parts = line.split(",", 1)
        if len(parts) != 2: continue
        vid_name, caption = parts
        vid_name = vid_name.strip()
        caption = caption.strip()
        
        vid_path = os.path.join(dataset_dir, "videos", vid_name)
        if not os.path.exists(vid_path): continue
        
        # นับเฟรมวิดีโอ
        cap = cv2.VideoCapture(vid_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n_frames <= 0: continue
        
        # แปลงข้อความเป็น Token
        tokens = tokenizer.encode(caption)
        token_id_str = " ".join(map(str, tokens))
        
        # แปลงให้อยู่ใน Format: dataset_name, rel_path, input_length, token_id
        dataset_name = "thai_vsr"
        rel_path = f"videos/{vid_name}"
        out_lines.append(f"{dataset_name},{rel_path},{n_frames},{token_id_str}")
    return out_lines

print("⏳ กำลังประมวลผล Train Set (อาจใช้เวลาสักครู่)...")
train_ready = process_lines(train_lines_raw)
print("⏳ กำลังประมวลผล Val Set (อาจใช้เวลาสักครู่)...")
val_ready = process_lines(val_lines_raw)

with open(train_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(train_ready) + "\n")
with open(val_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(val_ready) + "\n")

print(f"✅ เตรียมไฟล์ Train เสร็จสิ้น: {len(train_ready)} คลิป")
print(f"✅ เตรียมไฟล์ Val เสร็จสิ้น: {len(val_ready)} คลิป")
```

### 5. เริ่มทำการเทรน (Training) 🚀
> *คำสั่งนี้กำหนด `--max-frames 600` เพื่อหลีกเลี่ยงปัญหาหน่วยความจำการ์ดจอเต็ม (CUDA OOM), ปิดการใช้งาน WandB (`WANDB_MODE=disabled`) และตั้งค่าป้องกันหน่วยความจำแยกส่วน (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)*
```bash
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled SLURM_JOB_ID=1 python train.py \
    --exp-dir ./exp \
    --exp-name thai_lip_reading \
    --modality video \
    --root-dir {dataset_dir} \
    --train-file {dataset_dir}/train_labels.csv \
    --val-file {dataset_dir}/val_labels.csv \
    --num-nodes 1 \
    --gpus 1 \
    --pretrained-model-path {checkpoint_dir}/vsr_trlrs2lrs3vox2avsp_base.pth \
    --transfer-encoder \
    --max-epochs 50 \
    --max-frames 600 \
    --lr 1e-5 \
    --ctc-weight 0.9
```

### 6. ทดสอบการอ่านปาก (Inference) ระหว่าง/หลังเทรน 🎬
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
sys.path.append(os.getcwd())

try:
    from lightning import ModelModule
except ModuleNotFoundError:
    print("❌ ไม่พบไฟล์ lightning.py! กรุณาตรวจสอบว่าคุณอยู่ในโฟลเดอร์ auto_avsr แล้ว")
    sys.exit(1)

# 1. หาไฟล์ Checkpoint (.ckpt) ล่าสุดที่เซฟไว้
ckpt_files = glob.glob("./exp/thai_lip_reading/*.ckpt")
if not ckpt_files:
    print("❌ ยังไม่มีไฟล์โมเดล (.ckpt) ถูกสร้างขึ้น กรุณารอให้เทรนจบอย่างน้อย 1 Epoch ก่อนครับ")
else:
    # กรองเอาเฉพาะไฟล์ที่มีคำว่า epoch (ป้องกันการไปหยิบ last.ckpt ที่โดนขัดจังหวะ)
    epoch_ckpts = [f for f in ckpt_files if "epoch" in f]
    if epoch_ckpts:
        latest_ckpt = max(epoch_ckpts, key=os.path.getmtime)
    else:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        
    print(f"🧠 กำลังโหลดโมเดลล่าสุด: {os.path.basename(latest_ckpt)}")

    # 2. หาโฟลเดอร์ Dataset อัตโนมัติด้วย snapshot_download
    from huggingface_hub import snapshot_download
    dataset_dir = snapshot_download(repo_id="Phonsiri/Thai-Lip-Reading-Dataset", repo_type="dataset")
    video_dir = os.path.join(dataset_dir, "videos")
    
    videos = glob.glob(f"{video_dir}/*.mp4")
    
    if not videos:
        print(f"❌ หาวิดีโอไม่เจอใน {video_dir} กรุณาตรวจสอบว่ามีไฟล์หรือไม่")
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
            # 🔥 เคล็ดลับ: ปิดการทำงานของ Decoder (ที่มักจะพังถ้าข้อมูลน้อย) แล้วใช้ความแม่นยำจาก CTC 100%
            beam_search = get_beam_search_decoder(modelmodule.model, modelmodule.token_list, ctc_weight=1.0)
            
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
