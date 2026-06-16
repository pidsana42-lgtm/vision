#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import torchaudio
import torchvision


def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
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
        return vid



def load_audio(path):
    """
    rtype: torch, T x 1
    """
    waveform, sample_rate = torchaudio.load(path[:-4] + ".wav", normalize=True)
    return waveform.transpose(1, 0)


class AVDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio

        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]

        self.audio_transform = audio_transform
        self.video_transform = video_transform

    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            dataset_name, rel_path, input_length, token_id = path_count_label.split(",")
            paths_counts_labels.append((dataset_name, rel_path, int(input_length), torch.tensor([int(_) for _ in token_id.split()])))
        return paths_counts_labels

    def __getitem__(self, idx):
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
            return {"input": audio, "target": token_id}


    def __len__(self):
        return len(self.list)
