import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torchvision.transforms.v2 as v2
from torchvision.ops import box_iou
from PIL import Image
import glob

from model_advanced import AdvancedQueryCraft

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN CỐ ĐỊNH (HARDCODED)
# =====================================================================
VIDEO_DIR = "/home/nguyenletuan/Downloads/NCKH/VIDEOS_new"
OUT_DIR = "/home/nguyenletuan/Downloads/NCKH/Processed_Data"
HOI_CKPT = "/home/nguyenletuan/Downloads/NCKH/SafeGuard_Custom_QueryCraft/tensordock_VM_src/ResNET_152_full_src./checkpoints/resnet152_best.pth"

# Cấu hình Model
YOLO_PERSON = "yolov8x.pt"    # Bắt người bằng v8x
YOLO_OBJECT = "yolo26x.pt"    # Bắt vật thể bằng 26x để vét cạn vật
VERB_LIST = "hico_verb_list.txt"

# Cấu hình Ngưỡng
YOLO_CONF = 0.1   
HOI_THRESH = 0.1  
# =====================================================================

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("🚀 SAFEGUARD AI - CARTESIAN PAIRING (ÉP GHÉP TẤT CẢ OBJECTS)")
    print("="*70)

    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục: {VIDEO_DIR}")
        
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    video_files.sort()
    
    if len(video_files) == 0:
        print(f"⚠️ Không tìm thấy file .mp4 nào trong thư mục {VIDEO_DIR}")
        return
        
    print(f"📦 Đã quét thấy {len(video_files)} video cần xử lý.")
    print("⏳ Đang tải các mô hình AI lên GPU...")

    # 1. Load Models
    model_person = YOLO(YOLO_PERSON)
    model_object = YOLO(YOLO_OBJECT)
    
    model_hoi = AdvancedQueryCraft(num_obj_classes=80, num_interactions=117, verb_list_file=VERB_LIST, backbone_name="resnet152")
    model_hoi.load_state_dict(torch.load(HOI_CKPT, map_location=device))
    model_hoi.to(device)
    model_hoi.eval()
    
    verb_list = load_verb_list(VERB_LIST)
    transform = v2.Compose([
        v2.Resize((800, 800)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Xử lý từng video
    for vid_idx, video_path in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        model_person.predictor = None
        model_object.predictor = None
        
        save_dir = os.path.join(OUT_DIR, video_name)
        frames_dir = os.path.join(save_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"{video_name}.csv")
        
        print(f"\n▶️ [{vid_idx}/{len(video_files)}] Đang xử lý video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        csv_data = []

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            timestamp = round(frame_count / original_fps, 2)
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            h_img, w_img = frame.shape[:2]
            
            # --- TRACKING BẰNG YOLO ---
            res_person = model_person.track(frame, classes=[0], conf=YOLO_CONF, persist=True, verbose=False)[0] 
            res_object = model_object.track(frame, conf=YOLO_CONF, persist=True, verbose=False)[0] 

            people_dict = {}
            if len(res_person.boxes) > 0:
                boxes = res_person.boxes.xyxyn
                if res_person.boxes.id is not None:
                    ids = [str(int(i.item())) for i in res_person.boxes.id]
                else:
                    ids = [f"temp_p{i}" for i in range(len(boxes))] 
                    
                for box, track_id in zip(boxes, ids):
                    people_dict[track_id] = box.tolist() 
                    
            objects_dict = {}
            if len(res_object.boxes) > 0:
                boxes = res_object.boxes.xyxyn
                cls_ids = res_object.boxes.cls
                if res_object.boxes.id is not None:
                    ids = [str(int(i.item())) for i in res_object.boxes.id]
                else:
                    ids = [f"temp_o{i}" for i in range(len(boxes))]
                    
                for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                    c_id = int(cls_id.item())
                    if c_id == 0: continue
                    objects_dict[track_id] = {
                        "bbox": box.tolist(), "class_name": model_object.names[c_id]
                    }

            # --- HOI INFERENCE & CARTESIAN MAPPING ---
            frame_pair_results = {} 
            
            if people_dict and objects_dict:
                # [QUAN TRỌNG NHẤT]: KHỞI TẠO TÍCH ĐỀ-CÁC (CARTESIAN PRODUCT)
                # Đảm bảo CÓ BAO NHIÊU VẬT THÌ ĐỨA TRẺ CÓ BẤY NHIÊU ROW TRONG CSV
                for h_id, h_box in people_dict.items():
                    for o_id, o_data in objects_dict.items():
                        pair_key = (h_id, o_id)
                        
                        cb_pixel = [int(h_box[0]*w_img), int(h_box[1]*h_img), int(h_box[2]*w_img), int(h_box[3]*h_img)]
                        ob_pixel = [int(o_data['bbox'][0]*w_img), int(o_data['bbox'][1]*h_img), int(o_data['bbox'][2]*w_img), int(o_data['bbox'][3]*h_img)]
                        obj_name = o_data['class_name']
                        
                        frame_pair_results[pair_key] = {
                            "Frame_id": frame_count,
                            "Timestamp": timestamp,
                            "Bbox_Human": str(cb_pixel),
                            "Human_ID": h_id,
                            "Human_Label": "person",
                            "Bbox_Object": str(ob_pixel),
                            "Object_ID": f"{obj_name}_{o_id}",
                            "Interaction": "no interaction", # Mặc định là no interaction
                            "Score": 0.0,
                            "Label": 0 
                        }

                # BẮT ĐẦU ĐƯA VÀO HOI ĐỂ SUY LUẬN HÀNH ĐỘNG
                img_tensor = transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                
                priors = torch.ones((1, 100, 4), dtype=torch.float32) * 0.1
                all_boxes = []
                
                for b in people_dict.values(): 
                    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    w, h = b[2] - b[0], b[3] - b[1]
                    all_boxes.append([cx, cy, w, h])
                    
                for obj in objects_dict.values(): 
                    b = obj['bbox']
                    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                    w, h = b[2] - b[0], b[3] - b[1]
                    all_boxes.append([cx, cy, w, h])
                
                num_dets = min(len(all_boxes), 100)
                if num_dets > 0:
                    priors[0, :num_dets] = torch.tensor(all_boxes[:num_dets])
                
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    outputs = model_hoi(img_tensor, priors.to(device))
                
                out_actions = outputs['pred_actions'][0].sigmoid().cpu()
                out_h_boxes = cxcywh_to_xyxy(outputs['pred_human_boxes'][0].cpu(), 1.0, 1.0) 
                out_o_boxes = cxcywh_to_xyxy(outputs['pred_object_boxes'][0].cpu(), 1.0, 1.0)
                
                # --- MAPPING HOI QUERIES ĐỂ GHI ĐÈ VÀO LƯỚI CARTESIAN ---
                for q_idx in range(100):
                    action_probs = out_actions[q_idx]
                    
                    sorted_acts = torch.argsort(action_probs, descending=True)
                    best_act_idx = sorted_acts[0].item()
                    best_score = action_probs[best_act_idx].item()
                    act_name = verb_list[best_act_idx]
                    
                    if "no interaction" in act_name or "no_interaction" in act_name:
                        top2_idx = sorted_acts[1].item()
                        top2_score = action_probs[top2_idx].item()
                        if top2_score > HOI_THRESH: 
                            best_act_idx = top2_idx
                            best_score = top2_score
                            act_name = verb_list[best_act_idx]
                    
                    pred_h = out_h_boxes[q_idx].unsqueeze(0)
                    pred_o = out_o_boxes[q_idx].unsqueeze(0)
                    
                    best_child_id, best_child_iou = None, -1
                    for cid, cbox in people_dict.items():
                        iou = box_iou(pred_h, torch.tensor([cbox]))[0][0].item()
                        if iou > best_child_iou: best_child_iou, best_child_id = iou, cid
                            
                    best_obj_id, best_obj_iou = None, -1
                    for oid, odata in objects_dict.items():
                        iou = box_iou(pred_o, torch.tensor([odata['bbox']]))[0][0].item()
                        if iou > best_obj_iou: best_obj_iou, best_obj_id = iou, oid
                            
                    # Nếu HOI tìm ra được hành động cho cặp (Người, Vật) này
                    if best_child_id is not None and best_obj_id is not None and best_child_iou > 0.01 and best_obj_iou > 0.01:
                        pair_key = (best_child_id, best_obj_id)
                        
                        # Cặp này CHẮC CHẮN tồn tại trong dictionary nhờ Tích Đề-các ở trên
                        if pair_key in frame_pair_results:
                            curr_res = frame_pair_results[pair_key]
                            curr_is_no = ("no interaction" in curr_res['Interaction'] or "no_interaction" in curr_res['Interaction'])
                            new_is_no = ("no interaction" in act_name or "no_interaction" in act_name)
                            
                            should_update = False
                            if curr_is_no and not new_is_no:
                                should_update = True
                            elif curr_is_no == new_is_no:
                                if best_score > curr_res['Score']:
                                    should_update = True
                                    
                            if should_update:
                                # CHỈ ghi đè hành động và điểm, giữ nguyên Tọa độ của YOLO
                                frame_pair_results[pair_key]['Interaction'] = act_name
                                frame_pair_results[pair_key]['Score'] = round(best_score, 2)
            
            # Ghi dữ liệu vào CSV
            if len(frame_pair_results) > 0:
                for result in frame_pair_results.values():
                    csv_data.append(result)
            else:
                csv_data.append({
                    "Frame_id": frame_count,
                    "Timestamp": timestamp,
                    "Bbox_Human": "[]",      
                    "Human_ID": "none",
                    "Human_Label": "none",
                    "Bbox_Object": "[]",
                    "Object_ID": "none",
                    "Interaction": "no interaction",
                    "Score": 0.0,
                    "Label": 0 
                })
            
            frame_count += 1
            if frame_count % 50 == 0: 
                print(f"   ↳ Đã xử lý {frame_count}/{total_frames} frames...")
            
        cap.release()
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"   ✅ Hoàn tất video! Đã quét và lưu {frame_count} Khung hình vào: {csv_path}")

    print("\n" + "="*70)
    print(f"🎉 ĐÃ XỬ LÝ XONG TOÀN BỘ {len(video_files)} VIDEO TRONG THƯ MỤC!")
    print("="*70)

if __name__ == "__main__":
    main()