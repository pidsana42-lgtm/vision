import torch

class ThaiTokenizer:
    def __init__(self):
        # 1. Special Tokens (Necessary for CTC and Padding)
        # Index 0 is traditionally the CTC 'Blank' token
        self.blank_token = "<blank>"
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        
        self.special_tokens = [self.blank_token, self.pad_token, self.sos_token, self.eos_token]
        
        # 2. Thai Characters
        # Consonants: ก-ฮ
        self.consonants = [chr(x) for x in range(ord('ก'), ord('ฮ') + 1)]
        # Vowels, Tones, and other symbols
        self.vowels_tones = [chr(x) for x in range(ord('ะ'), ord('๎') + 1)]
        # Thai Digits ๐-๙
        self.thai_digits = [chr(x) for x in range(ord('๐'), ord('๙') + 1)]
        # Arabic Digits 0-9
        self.digits = [str(i) for i in range(10)]
        # Basic punctuation and space
        self.extras = [" ", ".", ",", "!", "?", "(", ")", "-"]

        # Combine all to create vocabulary
        self.vocab = self.special_tokens + self.consonants + self.vowels_tones + self.thai_digits + self.digits + self.extras
        
        # Mapping: Char to ID and ID to Char
        self.char2id = {char: i for i, char in enumerate(self.vocab)}
        self.id2char = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text):
        """Convert Thai string to list of IDs"""
        # Note: You might want to normalize text here (e.g., remove some accents)
        return [self.char2id[char] for char in text if char in self.char2id]

    def decode(self, ids):
        """Convert list of IDs back to Thai string"""
        res = []
        for i in ids:
            if i in self.id2char:
                char = self.id2char[i]
                if char not in self.special_tokens:
                    res.append(char)
        return "".join(res)

    def get_vocab_size(self):
        return len(self.vocab)

if __name__ == "__main__":
    tokenizer = ThaiTokenizer()
    print(f"Vocabulary Size: {tokenizer.get_vocab_size()}")
    
    test_text = "สวัสดีครับ 123"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Check if Blank token is at index 0
    print(f"Index 0: {tokenizer.id2char[0]}")
