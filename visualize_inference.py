import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms.v2 as v2
from torchvision.ops import box_iou
from ultralytics import YOLO

# Import model HOI của chúng ta (đảm bảo file model_advanced.py nằm cùng thư mục)
from Child_Danger_Detection_HOI_completed.model_advanced import AdvancedQueryCraft

def parse_args():
    parser = argparse.ArgumentParser(description="Render ảnh Demo độ phân giải cao cho Poster ViSEF")
    parser.add_argument("--img_path", type=str, required=True, help="Đường dẫn tới ảnh raw cần test")
    parser.add_argument("--out_path", type=str, default="poster_result.jpg", help="Tên file ảnh xuất ra")
    parser.add_argument("--hoi_ckpt", type=str, required=True, help="Đường dẫn tới resnet152_best.pth")
    parser.add_argument("--yolo_person", type=str, default="yolov8x.pt", help="Model bắt người")
    parser.add_argument("--yolo_object", type=str, default="yolov8x.pt", help="Model bắt vật")
    parser.add_argument("--verb_list", type=str, default="hico_verb_list.txt")
    parser.add_argument("--conf", type=float, default=0.01, help="Ngưỡng tự tin YOLO")
    parser.add_argument("--score_thresh", type=float, default=0.01, help="Ngưỡng hiển thị hành động")
    return parser.parse_args()

def load_verb_list(filepath):
    with open(filepath, 'r') as f:
        return [" ".join(line.strip().split()[1:]).replace("_", " ") for line in f]

def cxcywh_to_xyxy(boxes, w, h):
    x_c, y_c, bw, bh = boxes.unbind(-1)
    return torch.stack([
        (x_c - 0.5 * bw) * w, 
        (y_c - 0.5 * bh) * h, 
        (x_c + 0.5 * bw) * w, 
        (y_c + 0.5 * bh) * h
    ], dim=-1)

# Hàm vẽ Box có nền chữ chuyên nghiệp (giống các bài báo Q1)
def draw_fancy_box_with_text(img, box, label, box_color, text_color=(255, 255, 255)):
    x1, y1, x2, y2 = map(int, box)
    # Vẽ BBox
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)
    
    # Tính toán kích thước chữ để vẽ nền
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Vẽ hộp nền chữ
    cv2.rectangle(img, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), box_color, -1)
    
    # Ghi chữ
    cv2.putText(img, label, (x1, y1 - baseline - 2), font, font_scale, text_color, thickness)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 Đang khởi tạo mô hình sinh ảnh Poster...")

    # 1. Load Models
    model_person = YOLO(args.yolo_person)
    model_object = YOLO(args.yolo_object)
    
    # Khởi tạo ResNet-152 đúng như yêu cầu
    model_hoi = AdvancedQueryCraft(num_obj_classes=80, num_interactions=117, verb_list_file=args.verb_list, backbone_name="resnet152")
    model_hoi.load_state_dict(torch.load(args.hoi_ckpt, map_location=device))
    model_hoi.to(device)
    model_hoi.eval()
    
    verb_list = load_verb_list(args.verb_list)
    transform = v2.Compose([
        v2.Resize((800, 800)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Đọc ảnh
    if not os.path.exists(args.img_path):
        print(f"❌ Lỗi: Không tìm thấy ảnh tại {args.img_path}")
        return
        
    frame = cv2.imread(args.img_path)
    original_frame = frame.copy()
    h_img, w_img = frame.shape[:2]

    # 3. Chạy YOLO
    res_person = model_person.predict(frame, classes=[0], conf=args.conf, verbose=False)[0] 
    res_object = model_object.predict(frame, conf=args.conf, verbose=False)[0] 

    people_dict = {}
    if len(res_person.boxes) > 0:
        for idx, box in enumerate(res_person.boxes.xyxyn):
            people_dict[idx] = box.tolist()
            
    objects_dict = {}
    if len(res_object.boxes) > 0:
        for idx, (box, cls) in enumerate(zip(res_object.boxes.xyxyn, res_object.boxes.cls)):
            c_id = int(cls.item())
            if c_id == 0: continue # Bỏ qua người
            objects_dict[idx] = {"bbox": box.tolist(), "class_name": model_object.names[c_id]}

    # 4. Chạy HOI Model
    valid_interactions = []
    
    if people_dict and objects_dict:
        img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        
        priors = torch.ones((1, 100, 4), dtype=torch.float32) * 0.1
        all_boxes = []
        for b in people_dict.values(): 
            all_boxes.append([(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]])
        for obj in objects_dict.values(): 
            b = obj['bbox']
            all_boxes.append([(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]])
            
        num_dets = min(len(all_boxes), 100)
        if num_dets > 0:
            priors[0, :num_dets] = torch.tensor(all_boxes[:num_dets])
            
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model_hoi(img_tensor, priors.to(device))
            
        out_actions = outputs['pred_actions'][0].sigmoid().cpu()
        out_h_boxes = cxcywh_to_xyxy(outputs['pred_human_boxes'][0].cpu(), 1.0, 1.0) 
        out_o_boxes = cxcywh_to_xyxy(outputs['pred_object_boxes'][0].cpu(), 1.0, 1.0)

        # Map và Lọc Interaction
        for q_idx in range(100):
            action_probs = out_actions[q_idx]
            sorted_acts = torch.argsort(action_probs, descending=True)
            best_act_idx = sorted_acts[0].item()
            best_score = action_probs[best_act_idx].item()
            act_name = verb_list[best_act_idx]
            
            # Logic chống "no interaction"
            if "no interaction" in act_name or "no_interaction" in act_name:
                top2_idx = sorted_acts[1].item()
                top2_score = action_probs[top2_idx].item()
                if top2_score > args.score_thresh:
                    best_act_idx = top2_idx
                    best_score = top2_score
                    act_name = verb_list[best_act_idx]
            
            # Bỏ qua nếu vẫn là no interaction hoặc điểm quá thấp
            if "no interaction" in act_name or best_score < args.score_thresh:
                continue
                
            pred_h = out_h_boxes[q_idx].unsqueeze(0)
            pred_o = out_o_boxes[q_idx].unsqueeze(0)
            
            best_child_id, best_child_iou = -1, 0
            for cid, cbox in people_dict.items():
                iou = box_iou(pred_h, torch.tensor([cbox]))[0][0].item()
                if iou > best_child_iou: best_child_iou, best_child_id = iou, cid
                    
            best_obj_id, best_obj_iou = -1, 0
            for oid, odata in objects_dict.items():
                iou = box_iou(pred_o, torch.tensor([odata['bbox']]))[0][0].item()
                if iou > best_obj_iou: best_obj_iou, best_obj_id = iou, oid
                    
            if best_child_iou > 0.1 and best_obj_iou > 0.1:
                valid_interactions.append({
                    'h_id': best_child_id, 'o_id': best_obj_id,
                    'act': act_name, 'score': best_score
                })

    # 5. Vẽ hình siêu đẹp (Poster Quality)
    # Lọc bỏ các tương tác trùng lặp, chỉ lấy cái điểm cao nhất
    unique_interactions = {}
    for inter in valid_interactions:
        key = (inter['h_id'], inter['o_id'])
        if key not in unique_interactions or inter['score'] > unique_interactions[key]['score']:
            unique_interactions[key] = inter

    if not unique_interactions:
        print("⚠️ Không tìm thấy tương tác nào đáng kể trong ảnh này.")
    else:
        for key, inter in unique_interactions.items():
            cb = people_dict[inter['h_id']]
            ob = objects_dict[inter['o_id']]['bbox']
            obj_name = objects_dict[inter['o_id']]['class_name']
            act_name = inter['act']
            score = inter['score']

            # Tọa độ pixel
            cb_pix = [int(cb[0]*w_img), int(cb[1]*h_img), int(cb[2]*w_img), int(cb[3]*h_img)]
            ob_pix = [int(ob[0]*w_img), int(ob[1]*h_img), int(ob[2]*w_img), int(ob[3]*h_img)]
            
            # Tâm của box
            c_center = ((cb_pix[0] + cb_pix[2]) // 2, (cb_pix[1] + cb_pix[3]) // 2)
            o_center = ((ob_pix[0] + ob_pix[2]) // 2, (ob_pix[1] + ob_pix[3]) // 2)

            # --- MÀU SẮC CHỦ ĐẠO ---
            color_human = (50, 205, 50)  # Xanh lá mạ
            # color_human = (255, 0, 255) # Hồng tươi
            color_object = (0, 165, 255) # Cam/Vàng
            color_line = (0, 255, 255)   # Vàng chanh
            
            # 1. Vẽ Box Người
            draw_fancy_box_with_text(original_frame, cb_pix, "Child", color_human)
            
            # 2. Vẽ Box Vật
            draw_fancy_box_with_text(original_frame, ob_pix, obj_name.capitalize(), color_object)
            
            # 3. Vẽ Đường Liên Kết (Interaction Line)
            cv2.line(original_frame, c_center, o_center, color_line, 3)
            cv2.circle(original_frame, c_center, 6, color_human, -1)
            cv2.circle(original_frame, o_center, 6, color_object, -1)
            
            # 4. Vẽ Chữ Hành Động ở giữa đường thẳng
            mid_point = ((c_center[0] + o_center[0]) // 2, (c_center[1] + o_center[1]) // 2)
            label_text = f"{act_name} ({score:.2f})"
            
            # Thêm nền đen mờ cho chữ hành động để dễ đọc
            font = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), bl = cv2.getTextSize(label_text, font, 0.8, 2)
            cv2.rectangle(original_frame, (mid_point[0] - tw//2 - 5, mid_point[1] - th//2 - 5), 
                          (mid_point[0] + tw//2 + 5, mid_point[1] + th//2 + 5), (0, 0, 0), -1)
            cv2.putText(original_frame, label_text, (mid_point[0] - tw//2, mid_point[1] + th//2), 
                        font, 0.8, (0, 255, 255), 2)
            
            print(f"🎯 Đã vẽ: Child - [{act_name}] - {obj_name}")

    # Tự động sửa lỗi nếu người dùng truyền vào thư mục thay vì file .jpg
    if os.path.isdir(args.out_path):
        img_basename = os.path.basename(args.img_path)
        args.out_path = os.path.join(args.out_path, f"demo_{img_basename}")

    cv2.imwrite(args.out_path, original_frame)
    print(f"✅ Đã xuất ảnh Poster tuyệt đẹp tại: {args.out_path}")

if __name__ == "__main__":
    main()