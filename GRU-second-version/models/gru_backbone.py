import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticFeatureExtractor(nn.Module):
    def __init__(self, obj_dim, int_dim, img_dim, mix_dim, dropout=0.3):
        super().__init__()
        self.proj_obj = nn.Linear(obj_dim, mix_dim)
        self.proj_int = nn.Linear(int_dim, mix_dim)
        self.proj_img = nn.Linear(img_dim, mix_dim)
        
        self.ln_obj = nn.LayerNorm(mix_dim)
        self.ln_int = nn.LayerNorm(mix_dim)
        self.ln_img = nn.LayerNorm(mix_dim)
        
        self.mixer = nn.Sequential(
            nn.Linear(mix_dim * 3, mix_dim),
            nn.LayerNorm(mix_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x_obj, x_int, x_img):
        emb_obj = self.ln_obj(self.proj_obj(x_obj))
        emb_int = self.ln_int(self.proj_int(x_int))
        emb_img = self.ln_img(self.proj_img(x_img))
        
        concated = torch.cat([emb_obj, emb_int, emb_img], dim=-1)
        return self.mixer(concated)