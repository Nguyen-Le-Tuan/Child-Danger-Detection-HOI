# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SemanticMixer(nn.Module):
#     def __init__(self, obj_dim, int_dim, img_dim, mix_dim, dropout=0.3):
#         super().__init__()
#         self.proj_obj = nn.Linear(obj_dim, mix_dim)
#         self.proj_int = nn.Linear(int_dim, mix_dim)
#         self.proj_img = nn.Linear(img_dim, mix_dim)
        
#         self.ln_obj = nn.LayerNorm(mix_dim)
#         self.ln_int = nn.LayerNorm(mix_dim)
#         self.ln_img = nn.LayerNorm(mix_dim)
        
#         self.mixer = nn.Sequential(
#             nn.Linear(mix_dim * 3, mix_dim),
#             nn.LayerNorm(mix_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout) # Tăng dropout ở đây
#         )

#     def forward(self, x_obj, x_int, x_img):
#         emb_obj = self.ln_obj(self.proj_obj(x_obj))
#         emb_int = self.ln_int(self.proj_int(x_int))
#         emb_img = self.ln_img(self.proj_img(x_img))
        
#         concated = torch.cat([emb_obj, emb_int, emb_img], dim=-1)
#         return self.mixer(concated)

# class CrossModalFusion(nn.Module):
#     def __init__(self, query_dim, key_dim, fuse_dim, num_heads=4, dropout=0.3):
#         super().__init__()
#         self.proj_q = nn.Linear(query_dim, fuse_dim)
#         self.proj_k = nn.Linear(key_dim, fuse_dim)
        
#         self.attn = nn.MultiheadAttention(embed_dim=fuse_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
#         self.ln = nn.LayerNorm(fuse_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x_num, x_sem_mixed):
#         query = self.proj_q(x_num)
#         key_val = self.proj_k(x_sem_mixed)
#         attn_out, _ = self.attn(query, key_val, key_val)
#         return self.ln(query + self.dropout(attn_out))
# class Fusison(nn.Module):
#     def __init__(self, semantic_dim,numeric_dim,fusion_dim,dropout=0.3):
#         super().__init__()
#         self.proj_semantic = nn.Linear(semantic_dim, fusion_dim)
#         self.proj_numeric = nn.Linear(numeric_dim, fusion_dim)
        
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(fusion_dim * 2, fusion_dim),
#             nn.LayerNorm(fusion_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x_sem, x_num):
#         # CHIẾU CÁC ĐẦU VÀO VỀ CÙNG CHIỀU fusion_dim
#         emb_sem = F.relu(self.proj_semantic(x_sem))
#         emb_num = F.relu(self.proj_numeric(x_num))
        
#         # KẾT HỢP VÀ ĐƯA QUA LỚP FUSION
#         concated = torch.cat([emb_sem, emb_num], dim=-1)
#         return self.fusion_layer(concated)
# class MTFN_Triple_Input(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Sử dụng dropout rate cao hơn từ config hoặc mặc định 0.5
#         drop_rate = config.get('dropout', 0.3)
        
#         self.num_enc = nn.Sequential(
#             nn.Linear(config['num_dim'], config['fuse_dim']),
#             nn.LayerNorm(config['fuse_dim']),
#             nn.ReLU(),
#             nn.Dropout(drop_rate)
#         )
        
#         self.sem_mixer = SemanticMixer(
#             config['obj_dim'], config['int_dim'], config['img_dim'], 
#             config['fuse_dim'], dropout=drop_rate
#         )
        
#         self.fusion = CrossModalFusion(
#             config['fuse_dim'], config['fuse_dim'], config['fuse_dim'], dropout=drop_rate
#         )
        
#         """
#         self.fusion = Fusison(
#             semantic_dim = config['obj_dim'] + config['int_dim'] + config['img_dim'],
#             numeric_dim = config['num_dim'],
#             fusion_dim = config['fuse_dim'],
#             dropout = drop_rate
#         )
#         """
#         self.gru = nn.GRU(
#             input_size=config['fuse_dim'], 
#             hidden_size=config['hidden_dim'], 
#             num_layers=config['gru_layers'], 
#             batch_first=True,
#             dropout=drop_rate if config['gru_layers'] > 1 else 0
#         )
#         self.gru_ln = nn.LayerNorm(config['hidden_dim'])
#         self.gru_dropout = nn.Dropout(drop_rate)
        
#         self.temp_attn = TemporalAttention(config['hidden_dim'])
        
#         # HEADS: Không dùng Sigmoid ở đây vì đã dùng BCEWithLogitsLoss
#         self.main_head = nn.Linear(config['hidden_dim'], 1)
#         self.aux_head = nn.Linear(config['hidden_dim'], 1)

#     def forward(self, x_num, x_obj, x_int, x_img, mask=None):
        
#         x_num_emb = self.num_enc(x_num)
#         semantic_context = self.sem_mixer(x_obj, x_int, x_img)
#         fused = self.fusion(x_num_emb, semantic_context)
#         """
#         fused = self.fusion(
#             torch.cat([x_obj, x_int, x_img], dim=-1),
#             x_num
#         )
#         """
#         gru_out, _ = self.gru(fused)
#         gru_out = self.gru_dropout(self.gru_ln(gru_out))
        
#         # Aux head: dự đoán rủi ro cho từng frame (Logits)
#         aux_out = self.aux_head(gru_out) 
        
#         # Main head: dự đoán rủi ro cho toàn video thông qua Temporal Attention (Logits)
#         ctx_vec, w = self.temp_attn(gru_out, mask)
#         main_out = self.main_head(ctx_vec) 
        
#         return main_out, aux_out, w
    
#     def initialize_weights(model):
#         for m in model.modules():
#             # Khởi tạo cho các lớp Tuyến tính (Linear)
#             if isinstance(m, nn.Linear):
#                 # Nếu lớp ngay sau là ReLU -> dùng Kaiming
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
            
#             # Khởi tạo cho GRU
#             elif isinstance(m, nn.GRU):
#                 for name, param in m.named_parameters():
#                     if 'weight_ih' in name:
#                         nn.init.xavier_uniform_(param.data)
#                     elif 'weight_hh' in name:
#                         nn.init.orthogonal_(param.data) # Trực giao hóa rất tốt cho RNN/GRU
#                     elif 'bias' in name:
#                         nn.init.constant_(param.data, 0)
            
#             # Khởi tạo cho LayerNorm
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)

# # Cách dùng trong Trainer:
# # self.model = MTFN_Triple_Input(self.cfg['model']).to(self.device)
# # initialize_weights(self.model)

# class TemporalAttention(nn.Module):
#     def __init__(self, hidden_dim, dropout=0.3):
#         super().__init__()
#         self.ln = nn.LayerNorm(hidden_dim) # Thêm LayerNorm để ổn định score
#         self.attn_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim // 2, 1)
#         )
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, gru_out, mask=None):
#         # Ổn định hóa đầu vào trước khi tính scores
#         gru_out_norm = self.ln(gru_out) 
#         scores = self.attn_mlp(gru_out_norm).squeeze(-1)
        
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         weights = self.dropout(F.softmax(scores, dim=1))

#         ctx_vec = (weights.unsqueeze(-1) * gru_out).sum(dim=1)
#         return ctx_vec, weights

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticMixer(nn.Module):
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

class CrossModalFusion(nn.Module):
    def __init__(self, query_dim, key_dim, fuse_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.proj_q = nn.Linear(query_dim, fuse_dim)
        self.proj_k = nn.Linear(key_dim, fuse_dim)
        
        self.attn = nn.MultiheadAttention(embed_dim=fuse_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(fuse_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num, x_sem_mixed):
        query = self.proj_q(x_num)
        key_val = self.proj_k(x_sem_mixed)
        attn_out, _ = self.attn(query, key_val, key_val)
        return self.ln(query + self.dropout(attn_out))

class InstantaneousAttention(nn.Module):
    """
    Lớp liên kết đầu vào tức thời (fused) và đầu ra chuỗi (gru_out)
    Giúp Aux Head bắt được các biến động nguy hiểm đột ngột.
    """
    def __init__(self, fuse_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.gate_logic = nn.Sequential(
            nn.Linear(fuse_dim + hidden_dim, (fuse_dim + hidden_dim) // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((fuse_dim + hidden_dim) // 2, 2), # Trả về weights cho [f_t, h_t]
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, f_t, h_t):
        combined = torch.cat([f_t, h_t], dim=-1)
        weights = self.gate_logic(combined)
        
        w_f = weights[..., 0:1]
        w_h = weights[..., 1:2]
        
        # Tạo đặc trưng lai (Hybrid Features)
        hybrid_context = torch.cat([f_t * w_f, h_t * w_h], dim=-1)
        return self.classifier(hybrid_context), weights

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.attn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, gru_out, mask=None):
        gru_out_norm = self.ln(gru_out) 
        scores = self.attn_mlp(gru_out_norm).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = self.dropout(F.softmax(scores, dim=1))
        ctx_vec = (weights.unsqueeze(-1) * gru_out).sum(dim=1)
        return ctx_vec, weights

class MTFN_Triple_Input(nn.Module):
    def __init__(self, config):
        super().__init__()
        drop_rate = config.get('dropout', 0.3)
        
        # 1. Encoder cho Numeric
        self.num_enc = nn.Sequential(
            nn.Linear(config['num_dim'], config['fuse_dim']),
            nn.LayerNorm(config['fuse_dim']),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )
        
        # 2. Mixer cho Semantic (Object, Interaction, Image)
        self.sem_mixer = SemanticMixer(
            config['obj_dim'], config['int_dim'], config['img_dim'], 
            config['fuse_dim'], dropout=drop_rate
        )
        
        # 3. Cross-Modal Fusion (Attention-based)
        self.fusion = CrossModalFusion(
            config['fuse_dim'], config['fuse_dim'], config['fuse_dim'], dropout=drop_rate
        )
        
        # 4. Temporal Memory (GRU)
        self.gru = nn.GRU(
            input_size=config['fuse_dim'], 
            hidden_size=config['hidden_dim'], 
            num_layers=config['gru_layers'], 
            batch_first=True,
            dropout=drop_rate if config['gru_layers'] > 1 else 0
        )
        self.gru_ln = nn.LayerNorm(config['hidden_dim'])
        self.gru_dropout = nn.Dropout(drop_rate)
        
        # 5. Attention Layers
        self.aux_attention = InstantaneousAttention(config['fuse_dim'], config['hidden_dim'], drop_rate)
        self.temp_attn = TemporalAttention(config['hidden_dim'], drop_rate)
        
        # 6. Final Heads
        self.main_head = nn.Linear(config['hidden_dim'], 1)

    def forward(self, x_num, x_obj, x_int, x_img, mask=None):
        # Bước 1: Fusion dữ liệu tức thời (t)
        x_num_emb = self.num_enc(x_num)
        semantic_context = self.sem_mixer(x_obj, x_int, x_img)
        fused = self.fusion(x_num_emb, semantic_context) # [B, S, Fuse_Dim]
        
        # Bước 2: Xử lý chuỗi thời gian
        gru_out, _ = self.gru(fused)
        gru_out = self.gru_dropout(self.gru_ln(gru_out)) # [B, S, Hidden_Dim]
        
        # Bước 3: Aux Head với Instantaneous Attention (Liên kết x_t và h_t)
        # aux_weights trả về để bạn có thể visualize xem mô hình đang tin vào đâu
        aux_out, aux_weights = self.aux_attention(fused, gru_out) 
        
        # Bước 4: Main Head với Temporal Attention (Lọc frame quan trọng nhất)
        ctx_vec, temp_weights = self.temp_attn(gru_out, mask)
        main_out = self.main_head(ctx_vec) 
        
        return main_out, aux_out, temp_weights

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)