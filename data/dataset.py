# dataset.py
import os
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import Dataset

class XACLEDataset(Dataset):
    def __init__(self, csv_path, wav_dir, processor, max_sec=10, target_sr=48000):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.wav_dir = wav_dir
        self.processor = processor
        self.target_sr = target_sr
        self.max_pts = int(max_sec * target_sr)
        self.resampler = {}
        
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = os.path.join(self.wav_dir, row["wav_file_name"].lstrip("/"))
        try: wav, sr = torchaudio.load(wav_path)
        except: wav, sr = torch.zeros(1, self.target_sr), self.target_sr
            
        if sr != self.target_sr:
            if sr not in self.resampler: self.resampler[sr] = AT.Resample(sr, self.target_sr)
            wav = self.resampler[sr](wav)
            
        if wav.shape[1] > self.max_pts: wav = wav[:, :self.max_pts]
        else: wav = F.pad(wav, (0, self.max_pts - wav.shape[1]))
        
        caption = str(row["text"]).strip()
        if len(caption) < 3: caption = "Audio clip"
        
        mos = float(row["average_score"]) if "average_score" in row else -1.0
        return {"wav": wav, "text": caption, "mos": mos, "path": wav_path}

    def collate_fn(self, batch):
        wavs = [b["wav"].squeeze(0).numpy() for b in batch]
        texts = [b["text"] for b in batch]
        mos = torch.tensor([b["mos"] for b in batch], dtype=torch.float)
        paths = [b["path"] for b in batch]
        inputs = self.processor(text=texts, audios=wavs, return_tensors="pt", 
                                padding=True, truncation=True, sampling_rate=48000)
        return inputs, mos, paths
