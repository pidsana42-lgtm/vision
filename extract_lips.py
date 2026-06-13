import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Lip, Lower Face, and Anchor landmark indices (MediaPipe Face Mesh)
# Total 60 points -> 180 features (x, y, z)
LIPS_INDICES = [
    # --- Lips (40 points) ---
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    191, 80, 81, 82, 13, 312, 311, 310, 415, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # --- Lower Face / Jaw (15 points) ---
    152, 148, 176, 149, 150, 136, 172, 58, 288, 397, 365, 379, 378, 400, 377,
    # --- Nose / Anchor Points (5 points) ---
    1, 4, 168, 197, 5
]
LIPS_INDICES = list(dict.fromkeys(LIPS_INDICES)) # Ensure 60 unique points

def extract_lip_features(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence_features = []

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract only lip landmarks
            lip_points = []
            for idx in LIPS_INDICES:
                landmark = face_landmarks.landmark[idx]
                # We store x, y, z coordinates
                lip_points.extend([landmark.x, landmark.y, landmark.z])
            
            sequence_features.append(lip_points)
        else:
            # If no face detected, we can put zeros or skip (skipping might desync with audio)
            # For Lip Reading, we usually skip frames with no face or pad with zeros
            sequence_features.append([0.0] * (len(LIPS_INDICES) * 3))

    cap.release()
    return np.array(sequence_features)

def process_dataset():
    video_dir = 'DATASET/videos'
    feature_dir = 'DATASET/features'

    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # Check for already processed features
    processed_features = {os.path.splitext(f)[0] for f in os.listdir(feature_dir) if f.endswith('.npy')}
    
    videos_to_process = [v for v in videos if os.path.splitext(v)[0] not in processed_features]
    
    if not videos_to_process:
        print("Everything is up to date. No new videos to process.")
        return

    print(f"Found {len(videos_to_process)} new videos to process out of {len(videos)} total.")

    for video_name in videos_to_process:
        video_path = os.path.join(video_dir, video_name)
        feature_name = os.path.splitext(video_name)[0] + '.npy'
        feature_path = os.path.join(feature_dir, feature_name)

        print(f"Processing: {video_name}...")
        features = extract_lip_features(video_path)
        
        if len(features) > 0:
            np.save(feature_path, features)
            print(f"Saved: {feature_path} (Shape: {features.shape})")
        else:
            print(f"Warning: No features extracted for {video_name}")

if __name__ == "__main__":
    process_dataset()
