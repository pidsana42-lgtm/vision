import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class LipReadingTransformer(nn.Module):
    def __init__(self, input_dim=180, num_classes=120, d_model=1024, nhead=16, num_layers=28, dim_feedforward=4096, dropout=0.1, window_size=7):
        """
        Input Dim: 180 (60 points * 3 axis)
        Supports: Lips + Lower Face + Anchor Points
        """
        super(LipReadingTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.window_size = window_size
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def get_local_mask(self, sz):
        mask = torch.full((sz, sz), float('-inf'))
        for i in range(sz):
            start = max(0, i - self.window_size)
            end = min(sz, i + self.window_size + 1)
            mask[i, start:end] = 0.0
        return mask

    def forward(self, src, src_key_padding_mask=None):
        x = self.embedding(src)
        x = self.pos_encoder(x)
        
        seq_len = x.size(1)
        # Avoid re-creating mask if possible for performance
        local_mask = self.get_local_mask(seq_len).to(x.device)
        
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x = layer(x, src_mask=local_mask, src_key_padding_mask=src_key_padding_mask)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        logits = self.fc_out(x)
        logits = logits.transpose(0, 1)
        return torch.log_softmax(logits, dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = LipReadingTransformer(window_size=5)
    total_params = count_parameters(model)
    print(f"Hybrid Transformer Parameters: {total_params:,}")
    print(f"Model Size: {total_params / 1e9:.4f}B")
    
    # Test with a sequence of 100 frames
    dummy_input = torch.randn(1, 100, 180)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
