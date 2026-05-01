import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.ops import roi_align
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor
from transformers import CLIPTextModel, CLIPTokenizer 

# ==========================================================
# [ĐỘT PHÁ MỚI] LỚP LAYERNORM AN TOÀN CHO FP16/AMP
# Ngăn chặn hiện tượng tràn số khi bình phương giá trị > 256
# ==========================================================
class SafeLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # Ép input và bộ trọng số sang FP32 để tính phương sai an toàn tuyệt đối
        x_f32 = x.float()
        w_f32 = self.weight.float() if self.weight is not None else None
        b_f32 = self.bias.float() if self.bias is not None else None
        out_f32 = F.layer_norm(x_f32, self.normalized_shape, w_f32, b_f32, self.eps)
        # Trả về lại kiểu dữ liệu FP16 ban đầu
        return out_f32.to(x.dtype)

class CoordinateEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim), 
            SafeLayerNorm(embed_dim) # Thay thế LayerNorm
        )
    def forward(self, x): return self.mlp(x)

class RankAugmentedLinearAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rank_expansion = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim * 2),
            nn.GELU(),
            nn.Linear(self.head_dim * 2, self.head_dim)
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        B, N, _ = query.shape
        B, M, _ = key.shape
        
        q = F.elu(self.q_proj(query)) + 1.0
        k = F.elu(self.k_proj(key)) + 1.0
        v = self.v_proj(value)
        
        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, M, self.num_heads, self.head_dim)
        v = v.view(B, M, self.num_heads, self.head_dim)
        
        k_aug = self.rank_expansion(k)
        k_aug = F.elu(k_aug) + 1.0 
        
        q_f32 = q.float()
        k_aug_f32 = k_aug.float()
        v_f32 = v.float()
        
        k_scaled_f32 = k_aug_f32 / M
        
        kv_context = torch.einsum("bmhd,bmhe->bhde", k_scaled_f32, v_f32)
        out_f32 = torch.einsum("bnhd,bhde->bnhe", q_f32, kv_context)
        
        k_sum = k_scaled_f32.sum(dim=1) 
        denom = torch.einsum("bnhd,bhd->bnh", q_f32, k_sum) 
        out_f32 = out_f32 / denom.unsqueeze(-1).clamp(min=1e-6)
        
        # [BỌC LÓT CUỐI CÙNG] Kẹp giá trị không cho vượt ngưỡng FP16 trước khi chuyển đổi
        out_f32 = torch.nan_to_num(out_f32, nan=0.0, posinf=65000.0, neginf=-65000.0).clamp(-65000.0, 65000.0)
        out = out_f32.to(query.dtype)
        out = out.reshape(B, N, self.embed_dim)
        
        return self.out_proj(out)

class AdvancedQueryCraft(nn.Module):
    def __init__(self, num_yolo_queries=100, embed_dim=256, num_obj_classes=80, num_interactions=117, verb_list_file="hico_verb_list.txt", backbone_name="resnet50"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_interactions = num_interactions
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            return_nodes = {'layer2': 'feat0', 'layer3': 'feat1', 'layer4': 'feat2'}
            in_channels_list = [512, 1024, 2048]
            for name, param in base_model.named_parameters():
                param.requires_grad = True if 'layer3' in name or 'layer4' in name else False
        elif backbone_name == "resnet152":
            base_model = resnet152(weights=ResNet152_Weights.DEFAULT)
            return_nodes = {'layer2': 'feat0', 'layer3': 'feat1', 'layer4': 'feat2'}
            in_channels_list = [512, 1024, 2048]
            for name, param in base_model.named_parameters():
                param.requires_grad = True if 'layer3' in name or 'layer4' in name else False
        elif backbone_name == "efficientnet_b3":
            base_model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
            return_nodes = {'features.3': 'feat0', 'features.5': 'feat1', 'features.7': 'feat2'}
            in_channels_list = [48, 136, 384] 
            for name, param in base_model.named_parameters():
                param.requires_grad = True if any(stage in name for stage in ['features.5', 'features.6', 'features.7', 'features.8']) else False
        else:
            raise ValueError(f"❌ Không hỗ trợ backbone: {backbone_name}")

        self.feature_extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=embed_dim)
        
        # [FIX: EFFICIENTNET STABILITY] Thêm Instance Norm cho EfficientNet giảm variance 8-10x
        if backbone_name == "efficientnet_b3":
            self.instance_norms = nn.ModuleList([
                nn.InstanceNorm2d(48, affine=True),   # feat0 channels
                nn.InstanceNorm2d(136, affine=True),  # feat1 channels
                nn.InstanceNorm2d(384, affine=True),  # feat2 channels
            ])
        
        clip_model, _ = clip.load("ViT-B/32", device="cpu")
        self.clip_vision_teacher = clip_model.visual
        for param in self.clip_vision_teacher.parameters(): param.requires_grad = False
            
        # Thay thế toàn bộ bằng SafeLayerNorm
        self.student_to_teacher_proj = nn.Sequential(nn.Linear(embed_dim, 512), SafeLayerNorm(512))
        self.visual_to_clip_proj = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(), nn.Linear(512, 512), SafeLayerNorm(512))

        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        for param in self.clip_text_model.parameters(): param.requires_grad = False
            
        self.register_buffer('text_embeddings', self._init_text_embeddings(verb_list_file))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.coord_encoder = CoordinateEncoder(embed_dim=embed_dim)
        
        # Thay thế bằng SafeLayerNorm
        self.roi_proj = nn.Sequential(nn.Linear(embed_dim * 7 * 7, embed_dim), SafeLayerNorm(embed_dim))
        
        self.instance_decoder_attn = RankAugmentedLinearAttention(embed_dim)
        self.interaction_decoder_attn = RankAugmentedLinearAttention(embed_dim)
        
        self.obj_class_head = nn.Linear(embed_dim, num_obj_classes + 1)
        self.human_bbox_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 4))
        self.object_bbox_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 4))

    def _init_text_embeddings(self, verb_list_file):
        import os
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prompts = []
        if os.path.exists(verb_list_file):
            with open(verb_list_file, 'r', encoding='utf-8') as f:
                for line in f:
                    verb = " ".join(line.strip().split()[1:]).replace("_", " ")
                    prompts.append("a person is doing nothing" if verb == "no interaction" else f"a person is {verb} an object")
        else:
            prompts = [f"action {i}" for i in range(self.num_interactions)]
            
        prompts = prompts[:self.num_interactions]
        inputs = self.clip_tokenizer(prompts, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        self.clip_text_model.to(device)
        with torch.no_grad():
            text_features = self.clip_text_model(**inputs).pooler_output 
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            if hasattr(self, 'feature_extractor'):
                for m in self.feature_extractor.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
            if hasattr(self, 'clip_vision_teacher'):
                self.clip_vision_teacher.eval()
            if hasattr(self, 'clip_text_model'):
                self.clip_text_model.eval()

    def forward(self, images, yolo_boxes):
        B, C, H_img, W_img = images.shape
        N_yolo = yolo_boxes.shape[1]
        
        features_dict = self.feature_extractor(images)
        
        # [FIX: EFFICIENTNET STABILITY] Áp dụng Instance Norm để ổn định variance cao
        if self.backbone_name == "efficientnet_b3":
            features_dict = {
                k: self.instance_norms[i](v) 
                for i, (k, v) in enumerate(features_dict.items())
            }
        
        fpn_features = self.fpn(features_dict)
        
        global_feat = fpn_features['feat0'] 
        features_flat = global_feat.flatten(2).permute(0, 2, 1) 
        
        cx, cy, w, h = yolo_boxes.unbind(-1)
        x1, y1 = (cx - w/2) * W_img, (cy - h/2) * H_img
        x2, y2 = (cx + w/2) * W_img, (cy + h/2) * H_img
        x1, x2 = torch.min(x1.clamp(min=0, max=W_img), x2.clamp(min=0, max=W_img)), torch.max(x1.clamp(min=0, max=W_img), x2.clamp(min=0, max=W_img) + 1)
        y1, y2 = torch.min(y1.clamp(min=0, max=H_img), y2.clamp(min=0, max=H_img)), torch.max(y1.clamp(min=0, max=H_img), y2.clamp(min=0, max=H_img) + 1)

        rois = [torch.stack([x1[b], y1[b], x2[b], y2[b]], dim=-1) for b in range(B)]
        roi_features = roi_align(global_feat, rois, output_size=(7, 7), spatial_scale=1.0/8.0).view(B, N_yolo, -1)
        
        injected_queries = self.roi_proj(roi_features) + self.coord_encoder(yolo_boxes)
            
        hs_obj = self.instance_decoder_attn(injected_queries, features_flat, features_flat)
        hs_inter = self.interaction_decoder_attn(hs_obj, features_flat, features_flat)
        
        pred_logits = self.obj_class_head(hs_inter)
        pred_human_boxes = self.human_bbox_head(hs_inter).sigmoid()
        pred_object_boxes = self.object_bbox_head(hs_inter).sigmoid()
        
        visual_features = self.visual_to_clip_proj(hs_inter)
        vf_f32 = visual_features.float()
        vf_f32 = vf_f32 / vf_f32.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        visual_features = vf_f32.to(visual_features.dtype) 
        
        logit_scale = torch.clamp(self.logit_scale, max=4.605).exp()
        pred_actions = logit_scale * torch.einsum('b n d, k d -> b n k', visual_features, self.text_embeddings)
        
        output = {
            'pred_logits': pred_logits, 
            'pred_human_boxes': pred_human_boxes,
            'pred_object_boxes': pred_object_boxes,
            'pred_actions': pred_actions
        }
        
        if self.training:
            img_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.no_grad():
                teacher_feat = self.clip_vision_teacher(img_resized.type(self.clip_vision_teacher.conv1.weight.dtype)) 
            student_global_feat = hs_inter.mean(dim=1) 
            student_feat = self.student_to_teacher_proj(student_global_feat)
            
            output['student_feat'] = student_feat
            output['teacher_feat'] = teacher_feat.float()
            
        return output