import os
import cv2
import torch
import random
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# Import class mô hình của bạn
from model_advanced import AdvancedQueryCraft

# Danh sách 80 class của COCO (để in ra Terminal cho đẹp thay vì in số)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Trực quan hóa kết quả Mô hình HOI QueryCraft")
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa ảnh HICO-DET")
    parser.add_argument("--cache_dir", type=str, required=True, help="Thư mục chứa YOLO Cache (.npy)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Đường dẫn tới file querycraft_best.pth")
    parser.add_argument("--verb_list", type=str, default="hico_verb_list.txt", help="File danh sách hành động")
    parser.add_argument("--output_dir", type=str, default="visualize_results", help="Thư mục lưu ảnh đầu ra")
    parser.add_argument("--num_images", type=int, default=1000, help="Số lượng ảnh muốn kiểm tra")
    parser.add_argument("--threshold", type=float, default=0.3, help="Ngưỡng tự tin của hành động (Action Confidence)")
    return parser.parse_args()

def load_verb_list(filepath):
    verbs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            verb = " ".join(line.strip().split()[1:])
            verbs.append(verb)
    return verbs

def cxcywh_to_xyxy(box, w, h):
    """Chuyển đổi [cx, cy, w, h] chuẩn hóa về tọa độ Pixel thực tế"""
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

def main():
    args = parse_args()
    print("="*60)
    print("👁️  CÔNG CỤ KIỂM TRA TRỰC QUAN MÔ HÌNH HOI (INFERENCE)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Tải danh sách Hành động
    verbs = load_verb_list(args.verb_list)
    
    # 2. Khởi tạo và Load Model
    print("⏳ Đang tải mô hình...")
    model = AdvancedQueryCraft(num_obj_classes=80, num_interactions=117, verb_list_file=args.verb_list)
    
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"✅ Đã tải trọng số từ {args.checkpoint}")
    else:
        raise FileNotFoundError(f"❌ Không tìm thấy trọng số tại {args.checkpoint}")
    
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Lấy mẫu ngẫu nhiên ảnh
    all_images = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))]
    sample_size = min(args.num_images, len(all_images))
    random.seed(42)
    sampled_images = random.sample(all_images, sample_size)
    print(f"🎲 Đã chọn ngẫu nhiên {sample_size} ảnh để kiểm tra.")

    # 4. Vòng lặp Inference
    for img_name in tqdm(sampled_images, desc="Đang suy luận và vẽ ảnh"):
        img_path = os.path.join(args.img_dir, img_name)
        npy_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
        npy_path = os.path.join(args.cache_dir, npy_name)

        if not os.path.exists(npy_path):
            continue

        # Load ảnh gốc để vẽ
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None: continue
        h_orig, w_orig = img_cv2.shape[:2]

        # Load ảnh cho Model
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(device) # Shape [1, 3, 800, 800]

        # Load YOLO Cache
        yolo_boxes_full = np.load(npy_path)
        priors = torch.ones((1, 100, 4), dtype=torch.float32) * 0.1
        if len(yolo_boxes_full) > 0:
            yolo_boxes = yolo_boxes_full[:, :4]
            num_dets = min(yolo_boxes.shape[0], 100)
            priors[0, :num_dets] = torch.tensor(yolo_boxes[:num_dets])
        priors = priors.to(device)

        # Suy luận (Inference)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(img_tensor, priors)
        
        # Áp dụng hàm kích hoạt để lấy xác suất
        out_actions = outputs['pred_actions'][0].sigmoid() # [100, 117]
        out_h_boxes = outputs['pred_human_boxes'][0]       # [100, 4]
        out_o_boxes = outputs['pred_object_boxes'][0]      # [100, 4]
        out_classes = outputs['pred_logits'][0].softmax(-1)# [100, 81]

        # Lọc các cặp có hành động vượt ngưỡng Threshold
        max_act_probs, _ = out_actions.max(dim=-1)
        keep_idx = torch.where(max_act_probs > args.threshold)[0]

        has_interaction = False

        if len(keep_idx) > 0:
            print(f"\n🖼️ Ảnh: {img_name}")
            
        for idx in keep_idx:
            # Lấy các hành động cụ thể vượt ngưỡng của cặp này
            valid_acts = torch.where(out_actions[idx] > args.threshold)[0]
            if len(valid_acts) == 0: continue
            
            has_interaction = True
            
            # Lấy thông tin Tọa độ và Class Vật thể
            h_box = out_h_boxes[idx].cpu().numpy()
            o_box = out_o_boxes[idx].cpu().numpy()
            obj_cls_id = torch.argmax(out_classes[idx]).item()
            
            # Bỏ qua nếu model phân loại là background (class 80)
            if obj_cls_id == 80: continue 
            
            obj_name = COCO_CLASSES[obj_cls_id] if obj_cls_id < 80 else f"Obj_{obj_cls_id}"

            # Chuyển đổi pixel
            hx1, hy1, hx2, hy2 = cxcywh_to_xyxy(h_box, w_orig, h_orig)
            ox1, oy1, ox2, oy2 = cxcywh_to_xyxy(o_box, w_orig, h_orig)

            # Vẽ Box Người (Xanh dương) và Vật (Đỏ)
            cv2.rectangle(img_cv2, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.rectangle(img_cv2, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)

            # Vẽ đường thẳng nối tâm (Vàng)
            hcx, hcy = int((hx1+hx2)/2), int((hy1+hy2)/2)
            ocx, ocy = int((ox1+ox2)/2), int((oy1+oy2)/2)
            cv2.line(img_cv2, (hcx, hcy), (ocx, ocy), (0, 255, 255), 2)

            # Xử lý In ra Terminal và Vẽ Text lên ảnh
            for act_idx in valid_acts:
                act_name = verbs[act_idx]
                score = out_actions[idx][act_idx].item()
                
                # In Triplet chuẩn ra Terminal
                print(f"   👉 [Human] - [{act_name.upper()} ({score:.2f})] -> [{obj_name}]")
                
                # Vẽ text lên giữa đường nối
                text = f"{act_name} ({score:.2f})"
                text_org = (int((hcx+ocx)/2), int((hcy+ocy)/2) - 10)
                
                # Tạo viền đen cho chữ dễ đọc
                cv2.putText(img_cv2, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(img_cv2, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Lưu ảnh
        if has_interaction:
            output_path = os.path.join(args.output_dir, img_name)
            cv2.imwrite(output_path, img_cv2)

    print("\n" + "="*60)
    print(f"✅ Xong! Vui lòng mở thư mục '{args.output_dir}' để xem các ảnh có tương tác.")

if __name__ == "__main__":
    main()