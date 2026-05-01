import streamlit as st
import cv2
import os
import json
import numpy as np
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from datetime import datetime

# --- 1. CẤU HÌNH HỆ THỐNG ---
SAVE_DIR = r"/home/nguyenletuan/Downloads/NCKH/VIDEOS_new" 
MODEL_OBJECT_PATH = "yolo11x.pt" 
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.set_page_config(layout="wide", page_title="NCKH Tool: FULL AUTOMATIC WORKFLOW")

# --- 2. HÀM HỖ TRỢ (CORE AI & MATH) ---

@st.cache_resource
def load_models():
    models = {}
    try:
        models['obj'] = YOLO(MODEL_OBJECT_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models['clip_proc'] = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        models['clip_model'] = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
        models['device'] = device
        return models
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

def get_clip_embedding(image, bbox, processor, model, device):
    if not bbox or len(bbox) != 4: return None
    h, w, _ = image.shape
    x1, y1, x2, y2 = [int(max(0, c)) for c in bbox]
    x1, x2 = min(x1, w), min(x2, w)
    y1, y2 = min(y1, h), min(y2, h)
    if x2 <= x1 or y2 <= y1: return None 
    try:
        crop = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        inputs = processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        return feats.cpu().numpy().flatten().tolist()
    except Exception:
        return None

def calculate_center(bbox):
    return np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])

def get_distance_normalized(box1, box2, img_w, img_h):
    c1 = calculate_center(box1)
    c2 = calculate_center(box2)
    pixel_dist = np.linalg.norm(c1 - c2)
    diag = np.sqrt(img_w**2 + img_h**2)
    if diag == 0: return 0.0
    return pixel_dist / diag

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0: return 0
    return intersection_area / union_area

# --- 3. HÀM XỬ LÝ VIDEO ---

def extract_keyframes(video_path, out_dir=None, threshold=0.7):
    if out_dir is None: out_dir = SAVE_DIR
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keyframes = []
    metadata_list = [] 

    success, prev_frame = cap.read()
    if not success: return keyframes, fps
    
    real_frame_index = 0
    
    fname_0 = f"frame_{real_frame_index:06d}.jpg"
    path_0 = os.path.join(out_dir, fname_0)
    cv2.imwrite(path_0, prev_frame)
    kf_0 = {"frame_id": real_frame_index, "timestamp": 0.0, "path": path_0, "image": prev_frame}
    keyframes.append(kf_0)
    metadata_list.append({"frame_id": 0, "timestamp": 0.0, "filename": fname_0})

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    total_pixels = prev_gray.size
    
    progress_bar = st.progress(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        real_frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        non_zero_count = cv2.countNonZero(diff_thresh)
        change_percentage = non_zero_count / total_pixels
        
        is_diff = change_percentage > threshold
        is_last = (real_frame_index == total_frames_video - 1)
        
        if is_diff or is_last:
            filename = f"frame_{real_frame_index:06d}.jpg"
            keyframe_path = os.path.join(out_dir, filename)
            ts = round(real_frame_index / fps, 2)
            cv2.imwrite(keyframe_path, frame)
            
            keyframes.append({
                "frame_id": real_frame_index,
                "timestamp": ts,
                "path": keyframe_path,
                "image": frame
            })
            metadata_list.append({"frame_id": real_frame_index, "timestamp": ts, "filename": filename})
            prev_gray = gray 

        if total_frames_video > 0 and real_frame_index % 50 == 0: 
            progress_bar.progress(min(real_frame_index / total_frames_video, 1.0))

    cap.release()
    progress_bar.empty()
    
    with open(os.path.join(out_dir, "project_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)
        
    return keyframes, fps

def load_existing_keyframes(video_dir):
    keyframes = []
    meta_path = os.path.join(video_dir, "project_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f: meta_data = json.load(f)
            meta_data.sort(key=lambda x: x['frame_id'])
            for item in meta_data:
                img_path = os.path.join(video_dir, item['filename'])
                if os.path.exists(img_path):
                    keyframes.append({
                        "frame_id": item['frame_id'], 
                        "timestamp": item['timestamp'],
                        "path": img_path,
                        "image": cv2.imread(img_path)
                    })
            return keyframes, 30
        except: return [], 0
    files = [f for f in os.listdir(video_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    files.sort(key=lambda x: int(x.replace("frame_", "").replace(".jpg", "")))
    for fname in files:
        path = os.path.join(video_dir, fname)
        try: fid = int(fname.replace("frame_", "").replace(".jpg", ""))
        except: fid = 0
        keyframes.append({"frame_id": fid, "timestamp": 0.0, "path": path, "image": cv2.imread(path)})
    return keyframes, 30

# --- 4. GIAO DIỆN CHÍNH ---

st.title("🧬 NCKH Tool: Dog Logic + Fix Review")
st.markdown("---")

all_models = load_models()
if not all_models: st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🗂️ Danh sách Frame")
    if 'keyframes' in st.session_state and st.session_state['keyframes']:
        video_dir = st.session_state.get('video_dir', '')
        status_list = []
        map_idx_to_frame = {}
        for idx, kf in enumerate(st.session_state['keyframes']):
            fid = kf['frame_id']
            # Check for the new nested annotation file
            json_path = os.path.join(video_dir, f"anno_f{fid}", f"anno_f{fid}.json")
            icon = "✅" if os.path.exists(json_path) else "⬜"
            label = f"{icon} Frame {fid} ({kf['timestamp']}s)"
            status_list.append(label)
            map_idx_to_frame[label] = idx
        
        current_idx = st.session_state.get('current_kf_index', 0)
        current_label = status_list[current_idx] if current_idx < len(status_list) else status_list[0]
        selected_str = st.radio("Chuyển đến:", status_list, index=current_idx)
        new_idx = map_idx_to_frame[selected_str]
        if new_idx != st.session_state.get('current_kf_index', 0):
            st.session_state['current_kf_index'] = new_idx
            st.rerun()

# --- SETUP ---
with st.expander("📁 Project Setup", expanded=True):
    c1, c2, c3 = st.columns([1.5, 1, 1])
    uploaded_file = c1.file_uploader("Upload Video", type=["mp4", "avi"])
    threshold_val = c2.slider("Ngưỡng khác biệt (Sensitivity)", 0.01, 0.99, 0.30, 0.01)
    conf_val = c3.slider("YOLO Confidence", 0.1, 0.9, 0.25, key="slider_conf")
    
    if uploaded_file and st.button("🚀 Bắt đầu / Re-Sync"):
        video_id = os.path.splitext(uploaded_file.name)[0]
        video_dir = os.path.join(SAVE_DIR, video_id)
        os.makedirs(video_dir, exist_ok=True)
        meta_path = os.path.join(video_dir, "project_metadata.json")
        existing_frames = [f for f in os.listdir(video_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        
        if len(existing_frames) > 0 and os.path.exists(meta_path):
            st.info(f"📂 Load dự án cũ: {video_id} ({len(existing_frames)} frames)")
            kfs, fps = load_existing_keyframes(video_dir)
        else:
            if len(existing_frames) > 0:
                 for f in os.listdir(video_dir): os.remove(os.path.join(video_dir, f))
            tpath = os.path.join(SAVE_DIR, "temp.mp4")
            with open(tpath, "wb") as f: f.write(uploaded_file.read())
            kfs, fps = extract_keyframes(tpath, out_dir=video_dir, threshold=threshold_val)

        st.session_state.update({
            'keyframes': kfs, 'fps': fps, 'current_kf_index': 0,
            'video_id': video_id, 'video_dir': video_dir,
            'last_processed_kf': -1, 'last_processed_conf': -1
        })
        st.rerun()

# --- MAIN WORKSPACE ---
if 'keyframes' in st.session_state and len(st.session_state['keyframes']) > 0:
    kf_idx = st.session_state['current_kf_index']
    kf_data = st.session_state['keyframes'][kf_idx]
    orig_img = kf_data['image']
    h_img, w_img = orig_img.shape[:2]
    
    REAL_FRAME_ID = kf_data['frame_id']
    
    # State flags
    curr_conf = st.session_state.get('slider_conf', 0.25)
    last_kf = st.session_state.get('last_processed_kf', -1)
    last_conf = st.session_state.get('last_processed_conf', -1)
    
    frame_changed = (kf_idx != last_kf)
    conf_changed = (curr_conf != last_conf)
    
    # Chỉ chạy logic load/detect khi frame đổi hoặc conf đổi
    if frame_changed or conf_changed:
        # Define paths for the new nested annotation structure
        kf_anno_dir = os.path.join(st.session_state['video_dir'], f"anno_f{REAL_FRAME_ID}")
        json_path = os.path.join(kf_anno_dir, f"anno_f{REAL_FRAME_ID}.json")
        people_list = []
        obj_list = []
        
        # LOGIC 1: CHẾ ĐỘ XEM LẠI (REVIEW MODE)
        if os.path.exists(json_path) and (not conf_changed):
            with open(json_path, 'r', encoding='utf-8') as f: saved = json.load(f)
            # Load Children
            for c in saved.get('children', []):
                people_list.append({"id": c['id'], "label": "child", "bbox": str(c['bbox']), "conf": 1.0})
            
            # Load Objects - Load embeddings from .npy files
            for o in saved.get('objects', []):
                embedding_list = []
                embedding_path = o.get('object_embedding')
                if embedding_path:
                    # Path in JSON is relative to the JSON file's directory
                    npy_full_path = os.path.join(kf_anno_dir, embedding_path)
                    if os.path.exists(npy_full_path):
                        try:
                            embedding_list = np.load(npy_full_path).tolist()
                        except Exception as e:
                            st.warning(f"Could not load embedding for {o.get('id')}: {e}")

                obj_list.append({
                    "id": o['id'], 
                    "object_type": o.get('object_type',''), 
                    "bbox": str(o['bbox']), 
                    "interaction": o.get('interaction', {}), # Load dict gốc
                    "label": o.get('label', {}),             # Load dict gốc (hoặc int nếu cũ)
                    "embed": embedding_list
                })
        
        # LOGIC 2: DETECT MỚI
        else:
            # --- 1. CHUẨN BỊ DỮ LIỆU TỪ FRAME TRƯỚC (Tracking) ---
            prev_objects_db = []
            prev_people_db = [] 
            prev_json_path = None
            
            if kf_idx > 0:
                prev_real_id = st.session_state['keyframes'][kf_idx - 1]['frame_id']
                prev_kf_anno_dir = os.path.join(st.session_state['video_dir'], f"anno_f{prev_real_id}")
                prev_json_path = os.path.join(prev_kf_anno_dir, f"anno_f{prev_real_id}.json")

            if prev_json_path and os.path.exists(prev_json_path):
                with open(prev_json_path, 'r') as f: 
                    prev = json.load(f)
                
                # Load Objects cũ
                for o in prev.get('objects', []):
                    embedding_list = []
                    embedding_path = o.get('object_embedding')
                    if embedding_path:
                        npy_full_path = os.path.join(prev_kf_anno_dir, embedding_path)
                        if os.path.exists(npy_full_path):
                            try:
                                embedding_list = np.load(npy_full_path).tolist()
                            except: pass # Silently ignore if loading fails

                    prev_objects_db.append({
                        "id": o['id'], "object_type": o.get('object_type',''), "bbox": o['bbox'],
                        "interaction": o.get('interaction', {}), "label": o.get('label', {}), "embed": embedding_list
                    })
                
                # Load People cũ (Children) để Tracking
                # Lưu ý: Code cũ của bạn save children vào key 'children', ta lấy ra để so khớp
                for p in prev.get('children', []):
                    prev_people_db.append({
                        "id": p['id'],
                        "label": "child", # Mặc định lấy từ danh sách children là child
                        "bbox": p['bbox']
                    })
                # Nếu bạn có lưu Adults vào đâu đó thì load thêm vào prev_people_db ở đây

            # --- 2. CHẠY YOLO ---
            res = all_models['obj'](orig_img, conf=curr_conf, imgsz=960, verbose=False)[0]
            cnt_map = {}
            
            # Danh sách các ID cũ đã được dùng lại (để tránh 2 người mới cùng nhận 1 ID cũ)
            updated_old_obj_ids = []
            updated_old_person_ids = [] 

            for box in res.boxes:
                cls = int(box.cls[0])
                name = res.names[cls]
                cnt_map[name] = cnt_map.get(name, 0) + 1
                xyxy = [int(x) for x in box.xyxy[0].tolist()]
                conf = float(box.conf[0])

                # --- TRACKING CHO PERSON / DOG ---
                if name == 'person' or name == 'dog':
                    # Logic xác định label mặc định
                    default_label = "adult" if name == 'person' else "child"
                    
                    # THUẬT TOÁN TRACKING: Tìm người cũ có IOU cao nhất
                    best_iou = 0
                    match_person = None
                    
                    for old_p in prev_people_db:
                        if old_p['id'] not in updated_old_person_ids: # Chỉ xét người chưa được gán
                            iou = calculate_iou(old_p['bbox'], xyxy)
                            if iou > 0.4: # Ngưỡng chồng lấn (0.4 - 0.5 là ổn)
                                if iou > best_iou:
                                    best_iou = iou
                                    match_person = old_p
                    
                    if match_person:
                        # CASE A: Tìm thấy người cũ -> Dùng lại ID và Label
                        people_list.append({
                            "id": match_person['id'],
                            "label": match_person['label'], # Giữ nguyên label cũ (VD: frame trước đã sửa thành child thì giữ là child)
                            "bbox": str(xyxy),
                            "conf": conf
                        })
                        updated_old_person_ids.append(match_person['id'])
                    else:
                        # CASE B: Người mới -> Tạo ID mới
                        # Xử lý ID cho Dog logic (Dog luôn là child)
                        if name == 'dog':
                            # Đảm bảo ID không trùng
                            new_id = f"person_dog_{cnt_map.get('dog', 0)}_{kf_idx}" 
                            # (Thêm kf_idx vào đuôi để chắc chắn không trùng ID cũ)
                            people_list.append({
                                "id": new_id, 
                                "label": "child",                    
                                "bbox": str(xyxy),
                                "conf": conf
                            })
                        else:
                            # Person thường
                            new_id = f"person_{cnt_map['person']}_{kf_idx}"
                            people_list.append({
                                "id": new_id, 
                                "label": "adult", 
                                "bbox": str(xyxy), 
                                "conf": conf
                            })

                # --- TRACKING CHO OBJECTS (Code cũ của bạn giữ nguyên, có chỉnh nhẹ logic check ID) ---
                else:
                    found_match = False
                    for old_obj in prev_objects_db:
                        try:
                            if old_obj['id'] not in updated_old_obj_ids:
                                iou = calculate_iou(old_obj['bbox'], xyxy)
                                if iou > 0.2:
                                    found_match = True
                                    updated_old_obj_ids.append(old_obj['id'])
                                    new_embed = get_clip_embedding(orig_img, xyxy, all_models['clip_proc'], all_models['clip_model'], all_models['device'])
                                    obj_list.append({
                                        "id": old_obj['id'], "object_type": name, "bbox": str(xyxy), 
                                        "interaction": old_obj.get('interaction', {}), 
                                        "label": old_obj.get('label', {}), 
                                        "embed": new_embed
                                    })
                                    break
                        except: pass
                    
                    if not found_match:
                        new_id = f"{name}_{cnt_map[name]}_new"
                        emb = get_clip_embedding(orig_img, xyxy, all_models['clip_proc'], all_models['clip_model'], all_models['device'])
                        obj_list.append({
                            "id": new_id, "object_type": name, "bbox": str(xyxy), 
                            "interaction": {}, "label": {}, "embed": emb
                        })
            
            # 1. Giữ lại các Objects cũ không detect thấy (Code cũ của bạn)
            for old_obj in prev_objects_db:
                if old_obj['id'] not in updated_old_obj_ids:
                    obj_list.append({
                        "id": old_obj['id'], "object_type": old_obj['object_type'], "bbox": old_obj['bbox'], 
                        "interaction": old_obj.get('interaction', {}), 
                        "label": old_obj.get('label', {}), 
                        "embed": old_obj['embed']
                    })
            # 2. Giữ lại PERSON cũ không tìm thấy (Logic mới thêm)
            for old_p in prev_people_db:
                if old_p['id'] not in updated_old_person_ids:
                    # YOLO không thấy, nhưng frame trước có -> Giữ lại bbox cũ
                    bbox_str = str(old_p['bbox']) # Đảm bảo convert sang string để hiển thị editor
                    people_list.append({
                        "id": old_p['id'],
                        "label": old_p['label'],
                        "bbox": bbox_str,
                        "conf": 0.5 # Gán conf giả định (hoặc thấp) để biết đây là giữ lại
                    })

        st.session_state['people_data'] = people_list
        st.session_state['object_data'] = obj_list
        st.session_state['last_processed_kf'] = kf_idx
        st.session_state['last_processed_conf'] = curr_conf

    # --- LAYOUT HIỂN THỊ & DATA EDITOR ---
    col_vis, col_edit = st.columns([1.8, 1])

    with col_edit:
        st.subheader("📝 Data Editor")
        
        # 1. PEOPLE: Sửa lại có Dropdown cho Label
        st.write("### 1. People")
        edited_people = st.data_editor(
            st.session_state['people_data'], 
            key=f"p_{kf_idx}", 
            use_container_width=True, 
            num_rows="dynamic",
            column_config={
                "label": st.column_config.SelectboxColumn(
                    "Label",
                    options=["child", "adult"],
                    required=True,
                    width="small"
                ),
                "id": st.column_config.TextColumn("ID", required=True)
            }
        )
        
        # Lấy danh sách ID của Child từ bảng People vừa sửa
        children_list = []
        for p in edited_people:
            if p['label'] == 'child':
                try: children_list.append({"id": p['id'], "bbox": json.loads(p['bbox'])})
                except: pass
        child_ids = [c['id'] for c in children_list]

        # 2. OBJECTS: Tự động thêm cột interaction/label cho từng Child
        st.write("### 2. Objects")
        
        # A. Chuẩn bị dữ liệu hiển thị (Flattening Data)
        display_objects = []
        for obj in st.session_state['object_data']:
            # Tạo bản sao phẳng (flat copy) để hiển thị lên bảng
            flat_obj = {
                "id": obj['id'],
                "object_type": obj['object_type'],
                "bbox": obj['bbox'],
                "embed": obj['embed']
            }
            
            # Lấy data cũ (dict)
            curr_interactions = obj.get('interaction', {})
            curr_labels = obj.get('label', {})
            
            # Nếu data cũ là dạng string/int (format cũ), convert sang dict
            if not isinstance(curr_interactions, dict): curr_interactions = {}
            if not isinstance(curr_labels, dict): curr_labels = {}

            # Tạo cột cho từng child
            for cid in child_ids:
                # Interaction: Default "no-interaction"
                flat_obj[f"int_{cid}"] = curr_interactions.get(cid, "no-interaction")
                # Label: Default 0
                flat_obj[f"lbl_{cid}"] = curr_labels.get(cid, 0)
            
            display_objects.append(flat_obj)

        # B. Cấu hình cột động (Dynamic Column Config)
        obj_column_config = {
            "id": st.column_config.TextColumn("ID", required=True),
            "object_type": st.column_config.TextColumn("Type"),
            "bbox": st.column_config.TextColumn("BBox", required=True),
            "embed": None,
        }
        
        # Thêm config cho các cột dynamic
        for cid in child_ids:
            obj_column_config[f"int_{cid}"] = st.column_config.SelectboxColumn(
                f"Interact ({cid})",
                options=["no-interaction", "touching", "holding", "reaching", "climbing", "near"],
                default="no-interaction",
                required=True
            )
            obj_column_config[f"lbl_{cid}"] = st.column_config.NumberColumn(
                f"Label ({cid})",
                default=0,
                min_value=0,
                max_value=1,
                help="0: Safe, 1: Dangerous"
            )

        # C. Hiển thị bảng Objects
        edited_objects = st.data_editor(
            display_objects,
            key=f"o_{kf_idx}_flat",
            use_container_width=True,
            num_rows="dynamic",
            column_config=obj_column_config
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Live Preview")
        # (Phần preview ảnh giữ nguyên)
        for o in edited_objects:
            try:
                final_bbox = json.loads(o.get('bbox', '[0,0,0,0]'))
                if final_bbox[2] > final_bbox[0]:
                    y1, y2 = max(0, final_bbox[1]), min(h_img, final_bbox[3])
                    x1, x2 = max(0, final_bbox[0]), min(w_img, final_bbox[2])
                    crop_img = orig_img[y1:y2, x1:x2]
                    st.sidebar.image(crop_img, caption=o.get('id'), width=100)
            except: pass

        st.markdown("---")
        c_save, c_down = st.columns(2)
        with c_save:
            if st.button("💾 SAVE TO DISK", type="primary", use_container_width=True):
                out_children = [{"id": c['id'], "bbox": c['bbox']} for c in children_list]
                out_objs_json = []
                out_objs_session = []

                # Define new directory structure for saving
                video_id = st.session_state['video_id']
                kf_anno_dir = os.path.join(st.session_state['video_dir'], f"anno_f{REAL_FRAME_ID}")
                embeddings_dir = os.path.join(kf_anno_dir, "embeddings")
                os.makedirs(embeddings_dir, exist_ok=True)

                for row in edited_objects:
                    # Parse BBox
                    try: final_bbox = json.loads(row['bbox'])
                    except: final_bbox = [0,0,0,0]

                    # 1. Tính Distances (Tự động tính toán)
                    dists = {}
                    for child in children_list:
                        d = get_distance_normalized(child['bbox'], final_bbox, w_img, h_img)
                        dists[child['id']] = round(d, 4)
                    
                    # 2. Thu thập Interaction & Label từ các cột dynamic (Unflatten)
                    inter_map = {}
                    label_map = {}
                    for cid in child_ids:
                        inter_map[cid] = row.get(f"int_{cid}", "no-interaction")
                        label_map[cid] = row.get(f"lbl_{cid}", 0)

                    # 3. Tính Embed (nếu chưa có) và lưu file .npy
                    current_embed = row.get('embed', [])
                    if not current_embed or len(current_embed) == 0:
                        if final_bbox[2] > final_bbox[0]:
                            current_embed = get_clip_embedding(orig_img, final_bbox, all_models['clip_proc'], all_models['clip_model'], all_models['device'])
                        else:
                            current_embed = []
                    
                    json_embedding_path = None
                    if current_embed:
                        obj_id = row.get('id', 'unknown_obj')
                        safe_obj_id = "".join(x for x in obj_id if x.isalnum() or x in "-_").strip()
                        npy_filename = f"{video_id}_f{REAL_FRAME_ID}_{safe_obj_id}.npy"
                        npy_save_path = os.path.join(embeddings_dir, npy_filename)
                        
                        np.save(npy_save_path, np.array(current_embed))
                        
                        json_embedding_path = os.path.join("embeddings", npy_filename).replace('\\', '/')

                    # 4. Tạo Object JSON chuẩn
                    obj_base = {
                        "id": row.get('id'), 
                        "object_type": row.get('object_type'),
                        "distances": dists, 
                        "interaction": inter_map, # Key: person_id, Value: status
                        "label": label_map,       # Key: person_id, Value: 0/1
                        "object_embedding": json_embedding_path
                    }
                    
                    obj_json = obj_base.copy()
                    obj_json["bbox"] = final_bbox 
                    out_objs_json.append(obj_json)
                    
                    # Cập nhật lại session state (dạng gốc để lần sau load lại vẫn đúng)
                    obj_sess = obj_base.copy()
                    obj_sess["bbox"] = str(final_bbox)
                    obj_sess["embed"] = current_embed # Keep full list in session
                    out_objs_session.append(obj_sess)

                final_json = {
                    "video_id": st.session_state['video_id'], "fps": st.session_state['fps'],
                    "frame_id": REAL_FRAME_ID, "timestamp": kf_data['timestamp'],
                    "image_width": w_img, "image_height": h_img,
                    "children": out_children, "objects": out_objs_json
                }
                
                # Save to the new nested path
                save_p = os.path.join(kf_anno_dir, f"anno_f{REAL_FRAME_ID}.json")
                with open(save_p, "w", encoding="utf-8") as f: json.dump(final_json, f, indent=2, ensure_ascii=False)
                
                st.session_state['object_data'] = out_objs_session
                st.toast(f"✅ Saved Frame {REAL_FRAME_ID} with detailed interactions!")
                st.rerun()
        
        with c_down:
            if 'final_json' in locals(): dl_data = json.dumps(final_json, indent=2, ensure_ascii=False)
            else: dl_data = "{}"
            st.download_button("⬇️ JSON", data=dl_data, file_name=f"anno_f{REAL_FRAME_ID}.json", mime="application/json", use_container_width=True)

    with col_vis:
        st.header(f"Frame: {REAL_FRAME_ID} | Time: {kf_data['timestamp']}s")
        img_vis = orig_img.copy()
        
        # Vẽ People
        for p in edited_people:
            try:
                b = json.loads(p['bbox'])
                color = (0, 255, 0) if p['label'] == 'child' else (0, 0, 255)
                cv2.rectangle(img_vis, (b[0], b[1]), (b[2], b[3]), color, 2)
                cv2.putText(img_vis, f"{p['id']} ({p['label']})", (b[0], b[1]-10), 0, 0.6, color, 2)
            except: pass
            
        # Vẽ Objects
        for o in edited_objects:
            try:
                b = json.loads(o['bbox'])
                cv2.rectangle(img_vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                # Hiển thị interaction ngắn gọn trên ảnh
                lbl_txt = o['id']
                # Lấy 1 interaction ví dụ để hiển thị
                for k,v in o.items():
                    if k.startswith('int_') and v != 'no-interaction':
                        lbl_txt += f" | {k.replace('int_','')}:{v}"
                cv2.putText(img_vis, lbl_txt, (b[0], b[1]-5), 0, 0.5, (0, 255, 255), 2)
            except: pass
            
        st.image(img_vis, channels="BGR", use_container_width=True)
        
        c_prev, _, c_next = st.columns([1, 2, 1])
        if c_prev.button("⬅️ Previous") and kf_idx > 0:
            st.session_state['current_kf_index'] -= 1
            st.rerun()
        if c_next.button("Next ➡️") and kf_idx < len(st.session_state['keyframes']) - 1:
            st.session_state['current_kf_index'] += 1
            st.rerun()