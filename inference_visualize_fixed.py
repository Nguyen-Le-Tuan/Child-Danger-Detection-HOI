import torch
import cv2
import numpy as np
import torchvision.transforms as T
from PIL import Image
import os
import argparse

# Import model của bạn
from model_advanced import AdvancedQueryCraft

def parse_args():
    parser = argparse.ArgumentParser(description="Công cụ kiểm thử trực quan HOI Model (Hard Prior BBox)")
    parser.add_argument("--img_path", type=str, required=True, help="Đường dẫn đến ảnh cần test")
    parser.add_argument("--cache_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa các file YOLO cache .npy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn đến file trọng số (.pth)")
    parser.add_argument("--verb_list", type=str, default="hico_verb_list.txt", help="File danh sách hành động")
    parser.add_argument("--threshold", type=float, default=0.2, help="Ngưỡng tự tin (Nên để thấp khi test Zero-shot trên domain mới)")
    return parser.parse_args()

def load_verb_list(filepath):
    verbs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            verb = " ".join(line.strip().split()[1:])
            verbs.append(verb)
    return verbs

def cxcywh_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

def main():
    args = parse_args()
    print("="*60)
    print("🚀 SAFEGUARD AI - HOI VISUALIZATION (DECOUPLED BBOX MODE)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(args.verb_list):
        print(f"❌ Lỗi: Không tìm thấy file {args.verb_list}")
        return
    verb_list = load_verb_list(args.verb_list)
    
    print(f"⏳ Đang nạp mô hình từ: {args.checkpoint}...")
    model = AdvancedQueryCraft(num_obj_classes=80, num_interactions=len(verb_list), verb_list_file=args.verb_list).to(device)
    
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    img_name = os.path.basename(args.img_path)
    img_pil = Image.open(args.img_path).convert("RGB")
    w_orig, h_orig = img_pil.size
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # ĐỌC YOLO CACHE GỐC
    cache_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
    cache_path = os.path.join(args.cache_dir, cache_name)
    
    priors = torch.ones((100, 4), dtype=torch.float32) * 0.1
    yolo_boxes_raw = [] # Lưu lại box thật để vẽ
    yolo_classes = []
    
    if os.path.exists(cache_path):
        yolo_boxes_full = np.load(cache_path)
        yolo_boxes = yolo_boxes_full[:, :4]
        
        # Lưu class nếu cache có lưu (cột thứ 6)
        if yolo_boxes_full.shape[1] >= 6:
            yolo_classes = yolo_boxes_full[:, 5].astype(int)
        else:
            yolo_classes = [-1] * len(yolo_boxes)

        num_dets = min(yolo_boxes.shape[0], 100)
        if num_dets > 0:
            priors[:num_dets] = torch.tensor(yolo_boxes[:num_dets], dtype=torch.float32)
            yolo_boxes_raw = yolo_boxes[:num_dets]
            yolo_classes = yolo_classes[:num_dets]
    else:
        print("❌ Lỗi: Cần có YOLO Cache (.npy) để chạy Decoupled Mode!")
        return
        
    priors_tensor = priors.unsqueeze(0).to(device)

    print("⚡ Đang suy luận Action...")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = model(img_tensor, priors_tensor)
        # CHỈ LẤY ACTION (Bỏ qua pred_human_boxes và pred_object_boxes)
        pred_actions = outputs['pred_actions'][0].sigmoid() 

    print(f"\n📊 KẾT QUẢ DỰ ĐOÁN (Ngưỡng: {args.threshold})")
    
    # Duyệt qua các box của YOLO
    for i in range(len(yolo_boxes_raw)):
        action_scores = pred_actions[i]
        valid_actions = torch.where(action_scores > args.threshold)[0]
        
        # Lấy tọa độ gốc từ YOLO thay vì Model HOI
        box = yolo_boxes_raw[i]
        x1, y1, x2, y2 = cxcywh_to_xyxy(box, w_orig, h_orig)
        cls_id = yolo_classes[i]

        # Phân biệt màu sắc: Class 0 (Người) màu xanh, vật thể khác màu đỏ
        is_human = (cls_id == 0)
        color = (255, 0, 0) if is_human else (0, 0, 255)
        label_prefix = "Human" if is_human else f"Obj:{cls_id}"
        
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
        
        # Nếu mô hình có dự đoán hành động cho Box này
        if len(valid_actions) > 0:
            action_texts = []
            for act_idx in valid_actions:
                act_name = verb_list[act_idx]
                score = action_scores[act_idx].item()
                action_texts.append(f"{act_name}({score:.2f})")
            
            text_str = f"{label_prefix} " + "|".join(action_texts)
            
            # Vẽ nền đen cho chữ nổi bật
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv2, (x1, max(0, y1 - th - 10)), (x1 + tw, y1), (0,0,0), -1)
            cv2.putText(img_cv2, text_str, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            # Không có hành động thì chỉ in tên Label
            cv2.putText(img_cv2, label_prefix, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_filename = f"fixed_{img_name}"
    cv2.imwrite(output_filename, img_cv2)
    print(f"\n📸 Đã vẽ xong bản FIX! Tọa độ lấy từ YOLO, Hành động lấy từ HOI. Lưu tại: {output_filename}")

if __name__ == "__main__":
    main()