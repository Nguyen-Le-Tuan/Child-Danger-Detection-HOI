import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

class HungarianMatcher(nn.Module):
    """
    Thuật toán Hungarian Matcher thiết kế riêng cho bài toán HOI (Human-Object Interaction).
    Nó tính Cost (Chi phí ghép cặp) dựa trên 4 yếu tố:
    1. Khoảng cách hộp của Người (Human BBox)
    2. Khoảng cách hộp của Vật (Object BBox)
    3. Xác suất phân loại Vật (Object Class)
    4. Xác suất khớp Hành động (Action Multi-hot)
    """
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_action: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_action = cost_action

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # --- 1. Flatten và Ép Kiểu về FP32 để chống tràn số ---
        out_prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)
        out_human_bbox = outputs["pred_human_boxes"].flatten(0, 1).float()
        out_object_bbox = outputs["pred_object_boxes"].flatten(0, 1).float()
        
        # FIX CỰC KỲ QUAN TRỌNG: Lấy thẳng Logits (Không dùng .sigmoid())
        # Để đưa vào binary_cross_entropy_with_logits an toàn trong FP32
        out_actions_logits = outputs["pred_actions"].flatten(0, 1).float() 
        
        indices = []
        for i in range(bs):
            # --- 2. Lấy Ground Truth của ảnh thứ i ---
            tgt_ids = targets[i]["labels"]
            if len(tgt_ids) == 0:
                indices.append((torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)))
                continue
                
            tgt_human_bbox = targets[i]["human_boxes"]
            tgt_object_bbox = targets[i]["object_boxes"]
            tgt_actions = targets[i]["actions"] # Shape: [num_gt_pairs, 117]

            # BỌC LÓT AN TOÀN: Clamp các ID vật thể
            num_classes = outputs["pred_logits"].shape[-1]
            tgt_ids = torch.clamp(tgt_ids, 0, num_classes - 2)

            # --- 3. Tính Ma trận Chi phí (Cost Matrix) ---
            
            # 3.1 Cost cho Phân loại Vật (Classification Cost)
            cost_class = -out_prob[i*num_queries : (i+1)*num_queries][:, tgt_ids]
            
            # 3.2 Cost cho Bounding Boxes (Human + Object L1 Distance)
            out_h_bbox_i = out_human_bbox[i*num_queries : (i+1)*num_queries]
            out_o_bbox_i = out_object_bbox[i*num_queries : (i+1)*num_queries]
            
            cost_h_bbox = torch.cdist(out_h_bbox_i, tgt_human_bbox, p=1)
            cost_o_bbox = torch.cdist(out_o_bbox_i, tgt_object_bbox, p=1)
            cost_bbox = cost_h_bbox + cost_o_bbox

            # 3.3 Cost cho Hành động 
            out_act_logits_i = out_actions_logits[i*num_queries : (i+1)*num_queries]
            
            # Shape của out_act_logits_i_exp: [num_queries, 1, 117]
            # Shape của tgt_actions_exp: [1, num_gt_pairs, 117]
            out_act_logits_i_exp = out_act_logits_i.unsqueeze(1).expand(-1, len(tgt_actions), -1)
            tgt_actions_exp = tgt_actions.unsqueeze(0).expand(num_queries, -1, -1)
            
            # FIX LỖI AMP: Sử dụng BCE_with_logits để an toàn với Autocast (Tính trong FP32)
            cost_action = F.binary_cross_entropy_with_logits(out_act_logits_i_exp, tgt_actions_exp, reduction='none').mean(dim=-1)
            
            # --- 4. Tổng hợp Cost và chạy thuật toán Hungarian ---
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_action * cost_action
            
            # LÁ CHẮN CUỐI CÙNG CHỐNG CRASH: Dọn dẹp toàn bộ NaN/Inf (nếu có) trước khi nạp vào Scipy
            C = torch.nan_to_num(C, nan=1000.0, posinf=1000.0, neginf=-1000.0)
            C = C.cpu()
            
            pred_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(pred_ind, dtype=torch.int64), 
                            torch.as_tensor(tgt_ind, dtype=torch.int64)))
            
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]