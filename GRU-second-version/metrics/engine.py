import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

class Engine:
    @staticmethod
    def compute_loss(model_outputs, targets, mask, pos_weight):
        main_out, aux_out, _ = model_outputs
        
        # Main Loss (toàn chuỗi)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        main_loss = criterion(main_out, targets)
        
        # Auxiliary Loss (từng khung hình)
        # Lặp lại nhãn cho từng frame: [Batch, 1] -> [Batch, Time, 1]
        target_aux = targets.unsqueeze(1).repeat(1, aux_out.shape[1], 1)
        aux_loss = F.binary_cross_entropy(aux_out, target_aux, weight=mask.unsqueeze(-1).float())
        
        # Total Loss (Tỷ lệ 1:2 như bạn thiết lập)
        return main_loss + 2 * aux_loss

    @staticmethod
    def calculate_metrics(labels, preds, probs):
        return {
            'f1': f1_score(labels, preds, zero_division=0),
            'auc_roc': roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.5
        }