import os
import sys
import torch
import cv2
import numpy as np
import argparse

# บังคับ Import แบบเจาะจงเพื่อแก้บัค mp.solutions บน Mac
import mediapipe as mp
import mediapipe.python.solutions.face_detection as mp_face_detection

# เพิ่ม path ของ auto_avsr เข้าไปในระบบ
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_DIR, "auto_avsr"))

from auto_avsr.lightning import ModelModule

def load_model(pth_path):
    print(f"🧠 กำลังโหลดโมเดลจาก: {pth_path}")
    args = argparse.Namespace(modality="video", pretrained_model_path=None)
    modelmodule = ModelModule(args)
    ckpt = torch.load(pth_path, map_location="cpu")
    if "state_dict" in ckpt:
        states = {k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        modelmodule.model.load_state_dict(states)
    else:
        modelmodule.model.load_state_dict(ckpt)
    modelmodule.eval()
    return modelmodule

def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks

def crop_video_standalone(video_frames, landmarks):
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    if not valid_frames_idx: return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
        
    stable_reference = np.array([
        [102.07394306,  94.27230352],
        [156.36130542,  93.57815605],
        [129.00373787, 135.90343029],
        [129.31337323, 157.82299635]
    ])
    
    sequence = []
    for frame_idx, frame in enumerate(video_frames):
        margin = min(6, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = np.mean(
            [landmarks[x] for x in range(frame_idx - margin, frame_idx + margin + 1)], axis=0
        )
        smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
        
        transform = cv2.estimateAffinePartial2D(smoothed_landmarks, stable_reference, method=cv2.LMEDS)[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frame = cv2.warpAffine(gray, transform, dsize=(256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        transformed_landmarks = np.matmul(smoothed_landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        center_x, center_y = transformed_landmarks[3]
        
        height, width = 48, 48
        y_min = int(round(np.clip(center_y - height, 0, 256)))
        y_max = int(round(np.clip(center_y + height, 0, 256)))
        x_min = int(round(np.clip(center_x - width, 0, 256)))
        x_max = int(round(np.clip(center_x + width, 0, 256)))
        
        patch = transformed_frame[y_min:y_max, x_min:x_max]
        if patch.shape != (96, 96):
            patch = cv2.resize(patch, (96, 96))
        sequence.append(patch)
    return sequence

def main(video_path, model_path):
    if not os.path.exists(video_path):
        print(f"❌ ไม่พบไฟล์วิดีโอ: {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
        return

    print("🎬 กำลังสแกนหาใบหน้าและดึงเฉพาะรูปปากจากวิดีโอ...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    landmarks = []
    
    # 📌 ใช้ mp_face_detection ที่ Import มาเจาะจงเลยแก้บัคบน Mac
    detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frames.append(frame)
        
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            landmarks.append(None)
        else:
            ih, iw, _ = frame.shape
            max_id, max_size = 0, 0
            for idx, face in enumerate(results.detections):
                bboxC = face.location_data.relative_bounding_box
                bbox_size = bboxC.width * iw + bboxC.height * ih
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
            
            face = results.detections[max_id]
            kpts = face.location_data.relative_keypoints
            lmx = np.array([
                [kpts[0].x * iw, kpts[0].y * ih],
                [kpts[1].x * iw, kpts[1].y * ih],
                [kpts[2].x * iw, kpts[2].y * ih],
                [kpts[3].x * iw, kpts[3].y * ih]
            ])
            landmarks.append(lmx)
            
    cap.release()
    detector.close()
    
    if len(frames) == 0:
        print("❌ อ่านวิดีโอไม่ได้ครับ")
        return
        
    sequence = crop_video_standalone(frames, landmarks)
    if sequence is None or len(sequence) == 0:
        print("❌ ไม่พบใบหน้าในวิดีโอ หรือสกัดรูปปากไม่ได้ครับ")
        return
        
    processed_frames = []
    for frame in sequence:
        h, w = frame.shape
        th, tw = 88, 88
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        cropped = frame[i:i+th, j:j+tw] 
        
        tensor = torch.from_numpy(cropped).float() / 255.0
        tensor = (tensor - 0.421) / 0.165
        processed_frames.append(tensor)
        
    input_tensor = torch.stack(processed_frames).unsqueeze(1)
    print(f"✅ ดึงรูปปากสำเร็จ ได้จำนวน {input_tensor.shape[0]} เฟรม")
    
    model = load_model(model_path)
    
    print("🗣️ กำลังอ่านปาก...")
    with torch.no_grad():
        prediction = model.forward(input_tensor)
        
    print("\n" + "🔥" * 20)
    print(f"ผลลัพธ์ที่โมเดลอ่านปากได้: {prediction}")
    print("🔥" * 20 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="พาธไฟล์วิดีโอ (.mov หรือ .mp4)")
    parser.add_argument("--model", type=str, default="model_final.pth", help="พาธไฟล์โมเดล (.pth)")
    args = parser.parse_args()
    main(args.video, args.model)
