# config.py
import torch

CONFIG = {
    "train_list": "./datasets/XACLE_dataset/meta_data/train_average.csv",
    "validation_list": "./datasets/XACLE_dataset/meta_data/validation_average.csv",
    "test_list": "./datasets/XACLE_test_data/meta_data/test.csv",
    "wav_dir": "./datasets/XACLE_dataset/wav",
    "output_dir": "./chkpt_attention_focal",
    "checkpoint_name": "best_model_v8.3",
    
    "batch_size": 16,
    "val_batch_size": 16,
    "lr": 8e-5,
    "epochs": 30,
    "early_stop_patience": 6,
    "num_workers": 2,
    "max_len": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    "use_lora": True,
    "lora_r": 16,
    "attn_heads": 8,
    "attn_dropout": 0.1,

    "focal_gamma": 2.0,
    "focal_alpha": 0.25,
}
