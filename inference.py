import torch
import cv2
import numpy as np
from model import LipReadingTransformer
from vocabulary import ThaiTokenizer
from extract_lips import extract_lip_features
import os

class LipReadingInference:
    def __init__(self, model_path, device='cpu'):
        self.tokenizer = ThaiTokenizer()
        self.device = torch.device(device)
        
        # Load Model
        self.model = LipReadingTransformer(
            num_classes=self.tokenizer.get_vocab_size()
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # ตรวจสอบ dimension ก่อนโหลด
            ckpt_dim = checkpoint["model_state_dict"]["embedding.weight"].shape[1]
            model_dim = self.model.embedding.in_features
            if ckpt_dim != model_dim:
                raise ValueError(f"Checkpoint dimension mismatch ({ckpt_dim} vs {model_dim})")
                
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ โหลดโมเดลล้มเหลว: {e}")
            print("💡 คำแนะนำ: โมเดลหลักของโปรเจกต์นี้ได้รับการอัปเดตเป็นสเปก 180 มิติ (0.44B Parameters)")
            print("   แต่ไฟล์ checkpoint 'lip_reading_model.pth' ในเครื่องของคุณอาจเป็นรุ่นเก่า (120 มิติ)")
            print("   กรุณาฝึกสอนโมเดลใหม่ด้วยคำสั่ง: python3 train.py")
            raise SystemExit(1)

    def ctc_decode(self, log_probs):
        """
        Greedy Decoder for CTC. 
        In a real app, you might use Beam Search + Language Model here.
        """
        # log_probs shape: (T, 1, num_classes)
        arg_maxes = torch.argmax(log_probs, dim=-1).squeeze(1) # (T)
        
        # 1. Remove repeated tokens
        # 2. Remove blank tokens (Index 0)
        decoded_ids = []
        for i in range(len(arg_maxes)):
            if arg_maxes[i] != 0: # Not blank
                if i == 0 or arg_maxes[i] != arg_maxes[i-1]: # Not repeated
                    decoded_ids.append(arg_maxes[i].item())
        
        return self.tokenizer.decode(decoded_ids)

    def predict(self, video_path):
        # 1. Extract Features
        print(f"Extracting features from {video_path}...")
        features = extract_lip_features(video_path)
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device) # (1, T, 120)
        
        # 2. Forward Pass
        with torch.no_grad():
            log_probs = self.model(features) # (T, 1, num_classes)
            
        # 3. Decode
        raw_text = self.ctc_decode(log_probs)
        
        # 4. Post-processing (The 'Correction' step)
        # This is where you'd plug in a Language Model or Dictionary check
        refined_text = self.refine_text(raw_text)
        
        return refined_text

    def refine_text(self, text):
        """
        Simple Thai post-processing.
        You can expand this to use a Thai BERT or a Dictionary.
        """
        # Example: Basic cleaning
        text = text.strip()
        # You could use something like 'pythainlp' here for better correction
        return text

import sys

if __name__ == "__main__":
    # Path to your trained model
    model_file = 'lip_reading_model.pth'
    
    if os.path.exists(model_file):
        inference = LipReadingInference(model_file)
        
        # Check for command line argument
        if len(sys.argv) > 1:
            test_video = sys.argv[1]
        else:
            test_video = 'DATASET/test/การบันทึกหน้าจอ 2569-06-12 เวลา 18.44.52.mov'
            
        if os.path.exists(test_video):
            result = inference.predict(test_video)
            print("\n" + "="*30)
            print(f"RESULT: {result}")
            print("="*30)
        else:
            print(f"Video file {test_video} not found.")
    else:
        print(f"Model file {model_file} not found.")
