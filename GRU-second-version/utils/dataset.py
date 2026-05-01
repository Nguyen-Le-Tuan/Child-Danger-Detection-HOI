import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RobustVideoDataset(Dataset):
    def __init__(self, video_ids, all_data_dict, config):
        self.window_size = config['window_size']
        self.stride = config.get('stride', self.window_size // 2)
        # 1. Cố định việc chuẩn bị mẫu
        self.samples = self._prepare_windows(video_ids, all_data_dict)

    def _prepare_windows(self, video_ids, all_data_dict):
        samples = []
        for vid in video_ids:
            if vid not in all_data_dict: continue
            for pair_id, frames in all_data_dict[vid].items():
                num_frames = len(frames)
                if num_frames < 5: continue
                
                # Tạo cửa sổ trượt bao phủ toàn bộ video
                for i in range(0, num_frames, self.stride):
                    window = frames[i : i + self.window_size]
                    if len(window) > 0:
                        samples.append(window)
        return samples

    def __getitem__(self, idx):
        window_data = self.samples[idx]
        actual_len = len(window_data)
        
        num_feat = np.zeros((self.window_size, 2))
        obj_emb = np.zeros((self.window_size, 512))
        int_emb = np.zeros((self.window_size, 512))
        img_emb = np.zeros((self.window_size, 512))
        mask = np.zeros(self.window_size, dtype=np.bool_)
        
        # 2. Thu thập nhãn của tất cả các frame trong cửa sổ
        labels = []
        for t in range(actual_len):
            if window_data[t] is not None:
                num_feat[t] = window_data[t]['numeric']
                obj_emb[t] = window_data[t]['obj_emb']
                int_emb[t] = window_data[t]['int_emb']
                img_emb[t] = window_data[t]['img_emb']
                mask[t] = True
                labels.append(window_data[t]['label'])
        
        # 3. Logic nhãn: Nếu có bất kỳ frame nào nguy hiểm -> Cửa sổ nguy hiểm
        final_label = 1.0 if (len(labels) > 0 and max(labels) > 0) else 0.0

        return {
            'numeric': torch.FloatTensor(num_feat),
            'obj_emb': torch.FloatTensor(obj_emb),
            'int_emb': torch.FloatTensor(int_emb),
            'img_emb': torch.FloatTensor(img_emb),
            'mask': torch.BoolTensor(mask),
            'label': torch.FloatTensor([final_label])
        }
    def __len__(self):
        # Trả về tổng số lượng cửa sổ (mẫu) đã tạo được
        return len(self.samples)