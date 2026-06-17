import os
import subprocess
from huggingface_hub import hf_hub_download

def main():
    print("🧹 1. ลบไฟล์โมเดลเก่าที่อาจจะพังทิ้ง...")
    if os.path.exists("vsr_trlrs2lrs3vox2avsp_base.pth"):
        os.remove("vsr_trlrs2lrs3vox2avsp_base.pth")

    print("📥 2. โหลดโมเดลตั้งต้น...")
    hf_hub_download(
        repo_id="Phonsiri/Thai-Lip-Reading-Checkpoints",
        filename="vsr_trlrs2lrs3vox2avsp_base.pth",
        local_dir="."
    )
    print("✅ โหลดโมเดลตั้งต้นสำเร็จ!")

    print("🚀 3. เริ่มทำการเทรน...")
    command = [
        "python", "train.py",
        "--exp-dir", "./exp",
        "--exp-name", "thai_lip_reading",
        "--modality", "video",
        "--root-dir", "DATASET",
        "--train-file", "DATASET/auto_avsr_train.csv",
        "--val-file", "DATASET/auto_avsr_val.csv",
        "--num-nodes", "1",
        "--gpus", "1",
        "--pretrained-model-path", "vsr_trlrs2lrs3vox2avsp_base.pth",
        "--transfer-encoder",
        "--max-epochs", "50",
        "--max-frames", "600",
        "--lr", "1e-3",
        "--ctc-weight", "0.9"
    ]
    
    # เพิ่ม Environment Variables ให้ Pytorch
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_MODE"] = "disabled"
    env["SLURM_JOB_ID"] = "1"

    subprocess.run(command, env=env)

if __name__ == "__main__":
    main()
