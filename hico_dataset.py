import torch
from torch.utils.data import Dataset
import json
import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import v2
from collections import defaultdict

class HICODataset(Dataset):
    def __init__(self, img_dir, json_path, yolo_cache_dir, max_queries=100, num_interactions=117, is_train=False):
        """
        DataLoader chuẩn Q1: Tích hợp Stochastic Augmentation chống Overfitting
        """
        self.img_dir = img_dir
        self.yolo_cache_dir = yolo_cache_dir
        self.max_queries = max_queries
        self.num_interactions = num_interactions
        self.is_train = is_train
        
        print(f"⏳ Đang tải file Annotations: {json_path} (Mode: {'Train' if is_train else 'Val'})...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Base Transform
        self.base_transform = v2.Compose([
            v2.Resize((800, 800)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        # Color Jitter chỉ áp dụng cho Train
        self.color_jitter = v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.data)

    def _normalize_bbox(self, bbox, w_img, h_img):
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        return [cx / w_img, cy / h_img, w / w_img, h / h_img]

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['file_name']
        
        # 1. LOAD IMAGE
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            image = torch.zeros((3, 800, 800))
            w_img, h_img = 1000, 1000
        else:
            image_pil = Image.open(img_path).convert('RGB')
            w_img, h_img = image_pil.size
            image = self.base_transform(image_pil)

        # 2. XÁC SUẤT LẬT ẢNH (HORIZONTAL FLIP) NẾU LÀ TRAIN
        do_flip = self.is_train and random.random() > 0.5
        if do_flip:
            image = v2.functional.hflip(image)
            
        if self.is_train:
            image = self.color_jitter(image)
            
        image = self.normalize(image)

        # 3. LOAD YOLO CACHE
        cache_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
        yolo_path = os.path.join(self.yolo_cache_dir, cache_name)
        
        priors = torch.ones((self.max_queries, 4), dtype=torch.float32) * 0.1 
        
        if os.path.exists(yolo_path):
            yolo_boxes_full = np.load(yolo_path)
            yolo_boxes = yolo_boxes_full[:, :4] 
            
            # Lật BBox của YOLO nếu ảnh bị lật (cx = 1.0 - cx)
            if do_flip and len(yolo_boxes) > 0:
                yolo_boxes[:, 0] = 1.0 - yolo_boxes[:, 0]
                
            num_dets = min(yolo_boxes.shape[0], self.max_queries)
            if num_dets > 0:
                priors[:num_dets] = torch.tensor(yolo_boxes[:num_dets], dtype=torch.float32)

        # 4. PARSE HICO-DET GROUND TRUTH
        gt_human_boxes, gt_object_boxes, gt_object_classes, gt_actions = [], [], [], []
        pair_to_actions = defaultdict(list)
        
        for hoi in item.get('hoi_annotation', []):
            action_index = hoi['category_id'] - 1 if hoi['category_id'] > 0 else 0
            pair_to_actions[(hoi['subject_id'], hoi['object_id'])].append(action_index)

        annotations = item.get('annotations', [])
        for (subj_id, obj_id), actions_list in pair_to_actions.items():
            try:
                human_box = self._normalize_bbox(annotations[subj_id]['bbox'], w_img, h_img)
                object_box = self._normalize_bbox(annotations[obj_id]['bbox'], w_img, h_img)
                
                # Lật BBox của Ground Truth nếu ảnh bị lật
                if do_flip:
                    human_box[0] = 1.0 - human_box[0]
                    object_box[0] = 1.0 - object_box[0]
                
                action_vector = torch.zeros(self.num_interactions, dtype=torch.float32)
                for act_idx in actions_list:
                    if act_idx < self.num_interactions:
                        action_vector[act_idx] = 1.0
                
                gt_human_boxes.append(human_box)
                gt_object_boxes.append(object_box)
                gt_object_classes.append(annotations[obj_id]['category_id'])
                gt_actions.append(action_vector)
            except IndexError:
                continue

        target = {
            'human_boxes': torch.tensor(gt_human_boxes, dtype=torch.float32),
            'object_boxes': torch.tensor(gt_object_boxes, dtype=torch.float32),
            'labels': torch.tensor(gt_object_classes, dtype=torch.long),
            'actions': torch.stack(gt_actions) if len(gt_actions) > 0 else torch.empty((0, self.num_interactions), dtype=torch.float32)
        }
        return image, priors, target