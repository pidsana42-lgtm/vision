import logging
import os
from argparse import ArgumentParser

from average_checkpoints import ensemble
from datamodule.data_module import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from huggingface_hub import HfApi

class HuggingFacePushCallback(Callback):
    def __init__(self, repo_id="Phonsiri/Thai-Lip-Reading-Checkpoints", every_n_epochs=50):
        super().__init__()
        self.repo_id = repo_id
        self.every_n_epochs = every_n_epochs
        self.api = HfApi()

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            ckpt_path = trainer.checkpoint_callback.last_model_path
            if not ckpt_path or not os.path.exists(ckpt_path):
                # Fallback
                ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, "last.ckpt")
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, "last-v1.ckpt")
            
            if ckpt_path and os.path.exists(ckpt_path):
                filename = f"epoch_{trainer.current_epoch + 1}.ckpt"
                try:
                    print(f"\n🚀 [HF Push] กำลังอัพโหลด Checkpoint ไปที่ {self.repo_id} (Epoch {trainer.current_epoch + 1})...")
                    self.api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=filename,
                        repo_id=self.repo_id,
                        repo_type="model"
                    )
                    print("✅ อัพโหลดขึ้น Hugging Face สำเร็จ!\n")
                except Exception as e:
                    print(f"❌ อัพโหลดล้มเหลว (กรุณาเช็ค HF_TOKEN): {e}\n")# Set environment variables and logger level
# logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    seed_everything(42, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    hf_push_callback = HuggingFacePushCallback(every_n_epochs=10)
    callbacks = [checkpoint, lr_monitor, TQDMProgressBar(refresh_rate=50), hf_push_callback]

    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.max_epochs,
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=10,
        logger=WandbLogger(name=args.exp_name, project="auto_avsr_lipreader", group=args.group_name),
        gradient_clip_val=10.0,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        default="./exp",
        type=str,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
        required=True,
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--group-name",
        type=str,
        help="Group name of the task (wandb API)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Type of input modality",
        required=True,
        choices=["audio", "video"],
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        help="Root directory of preprocessed dataset",
        required=True,
    )
    parser.add_argument(
        "--train-file",
        type=str,
        help="Filename of training label list",
        required=True,
    )
    parser.add_argument(
        "--val-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of validation label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--test-file",
        default="lrs3_test_transcript_lengths_seg16s.csv",
        type=str,
        help="Filename of testing label list. (Default: lrs3_test_transcript_lengths_seg16s.csv)",
    )
    parser.add_argument(
        "--num-nodes",
        default=4,
        type=int,
        help="Number of machines used. (Default: 4)",
        required=True,
    )
    parser.add_argument(
        "--gpus",
        default=8,
        type=int,
        help="Number of gpus in each machine. (Default: 8)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--transfer-frontend",
        action="store_true",
        help="Flag to load the front-end only, works with `pretrained-model`",
    )
    parser.add_argument(
        "--transfer-encoder",
        action="store_true",
        help="Flag to load the weights of encoder, works with `pretrained-model`",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs for warmup. (Default: 5)",
    )
    parser.add_argument(
        "--max-epochs",
        default=75,
        type=int,
        help="Number of epochs. (Default: 75)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1600,
        help="Maximal number of frames in a batch. (Default: 1600)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate. (Default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.1,
        help="CTC weight",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=None,
        help="Path of the checkpoint from which training is resumed.",
    )
    parser.add_argument(
        "--slurm-job-id",
        type=float,
        default=0,
        help="Slurm job id",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    #init_logger(args.debug)
    args.slurm_job_id = os.environ["SLURM_JOB_ID"]
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args, train_num_buckets=args.train_num_buckets)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule, ckpt_path=args.ckpt_path)
    ensemble(args)


if __name__ == "__main__":
    cli_main()
