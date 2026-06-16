# 🚀 คู่มือการเทรนบน Google Colab (ฉบับปรับปรุงล่าสุด)

เมื่อคุณย้ายไปใช้งานบน Google Colab และเปิด **GPU (T4 หรือ A100)** แล้ว ให้ก๊อปปี้โค้ดด้านล่างนี้ไปวางในแต่ละเซลล์ (Cell) แล้วกดรันทีละอันได้เลยครับ

### 1. ดาวน์โหลดโปรเจกต์และติดตั้งไลบรารี
```bash
!git clone https://github.com/mpc001/auto_avsr.git
%cd auto_avsr
!sudo apt-get update && sudo apt-get install -y ffmpeg
!pip install torch torchvision torchaudio pytorch-lightning sentencepiece av opencv-python wandb
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

# 1. ปลดล็อก validation limit และลด num_workers
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
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size, prog_bar=True)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True, prog_bar=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True, prog_bar=True)
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



### 4.5 แบ่งข้อมูลสำหรับสอน (Train) และสอบย่อย (Val) 📊
> *รันสคริปต์นี้เพื่อแบ่ง `labels.csv` เป็น 80/20 ป้องกันไม่ให้โมเดลจำข้อสอบ (Overfitting)*

```python
import os
import random

input_csv = os.path.join(dataset_dir, "labels.csv")
train_csv = os.path.join(dataset_dir, "train_labels.csv")
val_csv = os.path.join(dataset_dir, "val_labels.csv")

with open(input_csv, "r", encoding="utf-8") as f:
    lines = f.read().splitlines()

random.seed(42)
random.shuffle(lines)

split_idx = int(len(lines) * 0.8)
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open(train_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines) + "\n")
with open(val_csv, "w", encoding="utf-8") as f:
    f.write("\n".join(val_lines) + "\n")

print(f"✅ สร้างไฟล์ Train: {len(train_lines)} คลิป")
print(f"✅ สร้างไฟล์ Val: {len(val_lines)} คลิป")
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
    --lr 1e-4
```
