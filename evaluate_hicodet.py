import os
import time
import json
import csv
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as v2
from torchvision.ops import box_iou
from collections import defaultdict
from sklearn.metrics import average_precision_score

from model_advanced import AdvancedQueryCraft

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mAP HOI (Bảng Q1 + Xuất CSV)")
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa ảnh Test")
    parser.add_argument("--test_json", type=str, required=True, help="File JSON tập Test")
    parser.add_argument("--cache_dir", type=str, required=True, help="Thư mục chứa YOLO cache")
    parser.add_argument("--checkpoint", type=str, required=True, help="File trọng số .pth")
    parser.add_argument("--verb_list", type=str, default="hico_verb_list.txt")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet152", "efficientnet_b3"])
    parser.add_argument("--out_csv", type=str, default="evaluation_results.csv", help="Tên file CSV xuất ra")
    return parser.parse_args()

def cxcywh_to_xyxy_tensor(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print(f"🔬 ĐÁNH GIÁ TOÀN NĂNG | BACKBONE: {args.backbone.upper()}")
    print("="*70)

    model = AdvancedQueryCraft(num_obj_classes=80, num_interactions=117, verb_list_file=args.verb_list, backbone_name=args.backbone)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    transform = v2.Compose([
        v2.Resize((800, 800)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    iou_threshold = 0.5
    action_threshold = 0.3 # Dùng riêng cho việc đếm True Positive từng ảnh
    num_actions = 117
    
    # Từ điển lưu trữ nhãn thực tế và điểm dự đoán (Dùng cho tính mAP chuẩn sklearn)
    y_true_per_action = defaultdict(list)
    y_score_per_action = defaultdict(list)
    gt_counts_per_action = defaultdict(int) 

    # Mở file CSV để ghi kết quả theo từng ảnh
    csv_file = open(args.out_csv, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image_Name', 'Inference_Time_ms', 'FPS', 'GT_Interactions', 'True_Positives', 'Image_Recall_Percent'])

    # Warm-up GPU
    print("🔥 Đang Warm-up GPU...")
    dummy_img = torch.randn(1, 3, 800, 800).to(device)
    dummy_priors = torch.rand(1, 100, 4).to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(20): _ = model(dummy_img, dummy_priors)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_inference_time = 0.0
    num_measured_frames = 0

    pbar = tqdm(test_data, desc=f"Tiến trình")
    
    for data in pbar:
        img_name = data['file_name']
        img_path = os.path.join(args.img_dir, img_name)
        npy_path = os.path.join(args.cache_dir, img_name.replace('.jpg', '.npy'))

        if not os.path.exists(img_path) or not os.path.exists(npy_path): continue

        img_pil = Image.open(img_path).convert("RGB")
        w_orig, h_orig = img_pil.size
        img_tensor = transform(img_pil).unsqueeze(0).to(device)

        yolo_boxes_full = np.load(npy_path)
        priors = torch.ones((1, 100, 4), dtype=torch.float32) * 0.1
        if len(yolo_boxes_full) > 0:
            yolo_boxes = yolo_boxes_full[:, :4]
            num_dets = min(yolo_boxes.shape[0], 100)
            priors[0, :num_dets] = torch.tensor(yolo_boxes[:num_dets])
        priors = priors.to(device)

        # ĐO THỜI GIAN
        torch.cuda.synchronize() 
        starter.record()
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model(img_tensor, priors)
            
        ender.record()
        torch.cuda.synchronize() 
        
        infer_time_ms = starter.elapsed_time(ender)
        current_fps = 1000.0 / infer_time_ms if infer_time_ms > 0 else 0
        total_inference_time += infer_time_ms / 1000.0 
        num_measured_frames += 1

        out_actions = outputs['pred_actions'][0].sigmoid().cpu()
        out_h_boxes = outputs['pred_human_boxes'][0].cpu()
        out_o_boxes = outputs['pred_object_boxes'][0].cpu()

        pred_h_boxes_xyxy = cxcywh_to_xyxy_tensor(out_h_boxes)
        pred_o_boxes_xyxy = cxcywh_to_xyxy_tensor(out_o_boxes)

        annotations = data.get('annotations', [])
        pair_to_actions = defaultdict(list)
        
        for hoi in data.get('hoi_annotation', []):
            action_index = hoi['category_id'] - 1 if hoi['category_id'] > 0 else 0
            pair_to_actions[(hoi['subject_id'], hoi['object_id'])].append(action_index)
            gt_counts_per_action[action_index] += 1

        # Dùng cho CSV
        gt_count_in_image = 0
        tp_count_in_image = 0

        # Trích xuất Score cao nhất cho mỗi cặp BBox hợp lệ
        for (subj_id, obj_id), actions_list in pair_to_actions.items():
            try:
                human_bbox = annotations[subj_id]['bbox']
                object_bbox = annotations[obj_id]['bbox']
            except IndexError:
                continue
                
            gt_h_box = torch.tensor([human_bbox[0]/w_orig, human_bbox[1]/h_orig, human_bbox[2]/w_orig, human_bbox[3]/h_orig]).unsqueeze(0)
            gt_o_box = torch.tensor([object_bbox[0]/w_orig, object_bbox[1]/h_orig, object_bbox[2]/w_orig, object_bbox[3]/h_orig]).unsqueeze(0)
            
            iou_h = box_iou(pred_h_boxes_xyxy, gt_h_box).squeeze(-1) 
            iou_o = box_iou(pred_o_boxes_xyxy, gt_o_box).squeeze(-1) 

            valid_queries = torch.where((iou_h > iou_threshold) & (iou_o > iou_threshold))[0]
            
            # 1. Ghi nhận mAP toàn cục
            for act_idx in range(num_actions):
                if act_idx == 57: continue 
                
                max_score = out_actions[valid_queries, act_idx].max().item() if len(valid_queries) > 0 else 0.0
                is_gt = 1 if act_idx in actions_list else 0
                
                y_true_per_action[act_idx].append(is_gt)
                y_score_per_action[act_idx].append(max_score)

            # 2. Đếm số lượng đoán đúng cho CSV (Chỉ tính các hành động thực sự có mặt)
            for act_idx in actions_list:
                if act_idx == 57: continue
                gt_count_in_image += 1
                is_matched = any(out_actions[q_idx, act_idx] > action_threshold for q_idx in valid_queries)
                if is_matched:
                    tp_count_in_image += 1

        # GHI DÒNG CSV CHO ẢNH NÀY
        img_recall = (tp_count_in_image / gt_count_in_image * 100) if gt_count_in_image > 0 else -1 
        csv_writer.writerow([
            img_name, 
            f"{infer_time_ms:.2f}", 
            f"{current_fps:.2f}", 
            gt_count_in_image, 
            tp_count_in_image, 
            f"{img_recall:.2f}"
        ])

    csv_file.close()

    # =========================================================
    # TÍNH TOÁN mAP VÀ PHÂN LOẠI RARE / NON-RARE
    # =========================================================
    ap_scores = {}
    rare_aps = []
    non_rare_aps = []
    
    for act_idx in range(num_actions):
        if act_idx == 57 or len(y_true_per_action[act_idx]) == 0: continue
        
        y_true = np.array(y_true_per_action[act_idx])
        y_score = np.array(y_score_per_action[act_idx])
        
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_score) * 100
            ap_scores[act_idx] = ap
            
            if gt_counts_per_action[act_idx] < 10:
                rare_aps.append(ap)
            else:
                non_rare_aps.append(ap)

    mAP_full = np.mean(list(ap_scores.values())) if ap_scores else 0.0
    mAP_rare = np.mean(rare_aps) if rare_aps else 0.0
    mAP_non_rare = np.mean(non_rare_aps) if non_rare_aps else 0.0
    
    avg_fps = 1.0 / (total_inference_time / num_measured_frames)

    # =========================================================
    # IN KẾT QUẢ THEO ĐÚNG FORMAT BẢNG CỦA BẠN
    # =========================================================
    print("\n")
    print(f"✅ Đã xuất thông số chi tiết từng ảnh vào file: {args.out_csv}")
    print("-" * 65)
    print(f"{'BACKBONE':<20} | {'FULL':<10} | {'RARE':<10} | {'NON_RARE':<10} | {'FPS':<10}")
    print("-" * 65)
    print(f"{args.backbone.upper():<20} | {mAP_full:<10.2f} | {mAP_rare:<10.2f} | {mAP_non_rare:<10.2f} | {avg_fps:<10.2f}")
    print("-" * 65)

if __name__ == "__main__":
    main()