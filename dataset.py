import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from vocabulary import ThaiTokenizer

class LipReadingDataset(Dataset):
    def __init__(self, csv_file, feature_dir, tokenizer, max_seq_len=200):
        """
        Args:
            csv_file: Path to CSV with columns [video_name, sentence]
            feature_dir: Path to directory containing .npy files
            tokenizer: ThaiTokenizer instance
        """
        self.data = pd.read_csv(csv_file)
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Load Feature (Lip Landmarks)
        video_name = self.data.iloc[idx, 0]
        feature_name = os.path.splitext(video_name)[0] + '.npy'
        feature_path = os.path.join(self.feature_dir, feature_name)
        
        # Load .npy and convert to tensor
        features = np.load(feature_path)
        features = torch.FloatTensor(features)
        
        # 2. Load Label (Sentence)
        sentence = self.data.iloc[idx, 1]
        tokens = self.tokenizer.encode(sentence)
        tokens = torch.LongTensor(tokens)
        
        return features, tokens

def collate_fn(batch):
    """
    Since each video and sentence has different length, we must pad them.
    """
    features, tokens = zip(*batch)
    
    # Get lengths for CTC loss
    input_lengths = torch.LongTensor([len(f) for f in features])
    target_lengths = torch.LongTensor([len(t) for t in tokens])
    
    # Pad Features (Batch, Max_T, 120)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    # Pad Tokens (Batch, Max_L)
    tokens_padded = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=1) # 1 is <pad>
    
    return features_padded, tokens_padded, input_lengths, target_lengths

if __name__ == "__main__":
    # Create a dummy labels.csv for testing
    if not os.path.exists('DATASET/labels.csv'):
        df = pd.DataFrame([
            ['test_video.mp4', 'สวัสดีครับ'],
            ['test_video_2.mp4', 'กินข้าวหรือยัง']
        ], columns=['video_name', 'sentence'])
        df.to_csv('DATASET/labels.csv', index=False)
        print("Created dummy DATASET/labels.csv")

    # Example Usage
    tokenizer = ThaiTokenizer()
    # dataset = LipReadingDataset('DATASET/labels.csv', 'DATASET/features', tokenizer)
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    # print("Dataset Loader Ready!")
