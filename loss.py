import torch
import torch.nn as nn
import torch.nn.functional as F

class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef, alpha=0.25, gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.alpha = alpha 
        self.gamma = gamma 
        
    def loss_labels(self, outputs, targets, indices):
        # [FIX FP16] Ép Logits về FP32 trước khi đưa vào CrossEntropy
        src_logits = outputs['pred_logits'].float()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        num_classes = src_logits.shape[-1]
        target_classes_o = torch.clamp(target_classes_o, min=0, max=num_classes - 2) 
        target_classes = torch.full(src_logits.shape[:2], num_classes - 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        return {'loss_ce': loss_ce}

    def loss_interactions(self, outputs, targets, indices):
        src_actions = outputs['pred_actions']
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            return {'loss_action': torch.nan_to_num(src_actions).sum() * 0.0}

        # [FIX FP16] Ép Logits về FP32 để chống tràn số khi tính Focal Loss (có chứa hàm mũ/log)
        src_logits = src_actions[idx].float()
        target_actions = torch.cat([t["actions"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device).float()

        p = torch.sigmoid(src_logits)
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_actions, reduction="none")
        p_t = p * target_actions + (1 - p) * (1 - target_actions)
        loss_action = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target_actions + (1 - self.alpha) * (1 - target_actions)
            loss_action = alpha_t * loss_action
            
        return {'loss_action': loss_action.mean()}

    def loss_boxes(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) == 0:
            h_loss = torch.nan_to_num(outputs['pred_human_boxes']).sum() * 0.0
            o_loss = torch.nan_to_num(outputs['pred_object_boxes']).sum() * 0.0
            return {'loss_human_bbox': h_loss, 'loss_object_bbox': o_loss}
        
        # [FIX FP16] Ép Box về FP32 để L1 Loss không bị thất thoát độ chính xác
        src_human_boxes = outputs['pred_human_boxes'][idx].float()
        src_object_boxes = outputs['pred_object_boxes'][idx].float()
        target_human_boxes = torch.cat([t['human_boxes'][J] for t, (_, J) in zip(targets, indices)]).float()
        target_object_boxes = torch.cat([t['object_boxes'][J] for t, (_, J) in zip(targets, indices)]).float()

        loss_human_bbox = F.l1_loss(src_human_boxes, target_human_boxes, reduction='none')
        loss_object_bbox = F.l1_loss(src_object_boxes, target_object_boxes, reduction='none')
        
        return {'loss_human_bbox': loss_human_bbox.mean(), 'loss_object_bbox': loss_object_bbox.mean()}

    def loss_distillation(self, outputs):
        if 'student_feat' in outputs and 'teacher_feat' in outputs:
            # [FIX CỰC KỲ QUAN TRỌNG] MSE = (A - B)^2. Nếu không có .float(), giá trị bình phương > 65504 sẽ sinh ra NaN!
            student_feat = outputs['student_feat'].float()
            teacher_feat = outputs['teacher_feat'].float()
            loss_distill = F.mse_loss(student_feat, teacher_feat)
            return {'loss_distill': loss_distill}
        return {}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_interactions(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_distillation(outputs))
        return losses