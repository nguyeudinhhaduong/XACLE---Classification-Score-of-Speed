# model.py
import torch
import torch.nn as nn
from transformers import ClapModel
from peft import LoraConfig, get_peft_model

class OrdinalFocalLoss(nn.Module):
    def __init__(self, num_classes=11, gamma=2.0, alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer('levels', torch.arange(num_classes - 1).float())

    def forward(self, logits, targets):
        device = logits.device
        targets = targets.to(device).clamp(0, self.num_classes - 1)
        target_labels = targets.long()
        cumulative_labels = (target_labels.unsqueeze(1) > self.levels.unsqueeze(0)).float()
        probs = torch.sigmoid(logits)
        pt = torch.where(cumulative_labels == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_weight = torch.where(cumulative_labels == 1, self.alpha, 1 - self.alpha)
            focal_weight = focal_weight * alpha_weight
        bce = nn.functional.binary_cross_entropy_with_logits(logits, cumulative_labels, reduction='none')
        loss = (focal_weight * bce).sum(dim=1).mean()
        return loss

class XACLEAttentionModel(nn.Module):
    def __init__(self, cfg, weights_path):
        super().__init__()
        self.clap = ClapModel.from_pretrained(weights_path)
        if cfg['use_lora']:
            peft_config = LoraConfig(r=cfg['lora_r'], lora_alpha=32, lora_dropout=0.05,
                                     bias="none", target_modules=["q_proj","v_proj","k_proj","out_proj","dense"])
            self.clap = get_peft_model(self.clap, peft_config)
        embed_dim = 512
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=cfg['attn_heads'],
                                                 dropout=cfg['attn_dropout'], batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(embed_dim*3, 1024), nn.LayerNorm(1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 10)
        )
        self.fusion_attn.apply(self._init_weights)
        self.attn_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == 10: nn.init.normal_(module.weight, mean=0.0, std=0.01)
            else: nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.constant_(module.out_proj.bias, 0.)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, inputs):
        base = self.clap.base_model if hasattr(self.clap, "base_model") else self.clap
        valid_keys = ['input_ids','attention_mask','pixel_values','input_features']
        valid_inputs = {k:v for k,v in inputs.items() if k in valid_keys}
        outputs = base(**valid_inputs)
        audio_emb = outputs.audio_embeds
        text_emb = outputs.text_embeds
        diff_emb = torch.abs(audio_emb - text_emb)
        seq = torch.stack([audio_emb, text_emb, diff_emb], dim=1)
        attn_out,_ = self.fusion_attn(seq, seq, seq)
        seq_fused = self.attn_norm(seq + attn_out)
        flat = seq_fused.reshape(seq_fused.size(0), -1)
        logits = self.head(flat)
        return logits
