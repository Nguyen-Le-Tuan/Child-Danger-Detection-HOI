import streamlit as st
import cv2
import numpy as np
import yt_dlp
import os
import tempfile
import math
import time
from collections import deque
from ultralytics import YOLO
import torch

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & CSS (DARK MODE - VIP)
# ==========================================
st.set_page_config(layout="wide", page_title="SafeGuard AI Pipeline", initial_sidebar_state="collapsed")

custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        text-align: center;
        background: -webkit-linear-gradient(#ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900;
        font-size: 2.5rem;
        letter-spacing: 4px;
        margin-bottom: 5px;
    }
    .sub-header {
        text-align: center;
        color: #888888;
        font-family: monospace;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }
    
    .stButton>button {
        background-color: #ff003c !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        border: 2px solid #ff003c !important;
        border-radius: 8px !important;
        width: 100%;
        height: 55px;
        font-size: 16px;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(255, 0, 60, 0.4);
    }
    .stButton>button:hover {
        background-color: transparent !important;
        color: #ff003c !important;
        box-shadow: 0 0 25px rgba(255, 0, 60, 0.6);
    }
    
    /* Bảng Log Hiện Đại */
    .log-table {
        width: 100%; border-collapse: collapse; color: #e0e0e0;
        font-family: 'Courier New', monospace; font-size: 14px;
        background-color: #121212; border-radius: 8px; overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .log-table th {
        background-color: #1f1f1f; color: #00e676; padding: 12px;
        text-align: left; border-bottom: 1px solid #333;
    }
    .log-table td { padding: 10px 12px; border-bottom: 1px solid #222; }
    
    /* Các Trạng thái màu sắc */
    .row-safe { border-left: 4px solid #00e676; }
    .row-warn { border-left: 4px solid #ffaa00; background-color: rgba(255, 170, 0, 0.05); }
    .row-danger { border-left: 4px solid #ff003c; background-color: rgba(255, 0, 60, 0.15); color: #ff8a8a;}
    
    /* Metrics Box */
    .metric-box {
        background-color: #1e1e1e; border-radius: 8px; padding: 15px;
        text-align: center; border-bottom: 3px solid #333;
    }
    .metric-title { font-size: 12px; color: #888; text-transform: uppercase; }
    .metric-val { font-size: 24px; font-weight: bold; color: #fff; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<div class='main-header'>SAFEGUARD DEEP LEARNING PIPELINE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>DUAL YOLO + EFFICIENTNET-B3 + GRU TIME-SERIES</div>", unsafe_allow_html=True)

# ==========================================
# 2. KHỔNG LỒ DATABASE: MA TRẬN 80 LỚP COCO
# ==========================================
# Cấu trúc: { 'class_name': {'type': 'nhóm', 'heavy': Bool, 'risk': 'Mô tả nguy cơ'} }
RISK_MATRIX = {
    # 1. Nhóm Phương tiện (Gara/Ngoài sân)
    'bicycle': {'type': 'vehicle', 'heavy': True, 'risk': 'Traffic hazard/Collision (Va chạm)'},
    'motorcycle': {'type': 'vehicle', 'heavy': True, 'risk': 'Traffic hazard/Burn from exhaust (Va chạm/Bỏng pô)'},
    'car': {'type': 'vehicle', 'heavy': True, 'risk': 'CRITICAL Traffic hazard (Nguy hiểm giao thông)'},
    'truck': {'type': 'vehicle', 'heavy': True, 'risk': 'CRITICAL Traffic hazard (Nguy hiểm giao thông)'},
    'bus': {'type': 'vehicle', 'heavy': True, 'risk': 'CRITICAL Traffic hazard (Nguy hiểm giao thông)'},
    'train': {'type': 'vehicle', 'heavy': True, 'risk': 'FATAL Hazard'},
    
    # 2. Nhóm Động vật / Thú cưng
    'bird': {'type': 'animal', 'heavy': False, 'risk': 'Hygiene/Pecking (Vệ sinh/Mổ)'},
    'cat': {'type': 'animal', 'heavy': False, 'risk': 'Scratching/Allergy (Cào/Dị ứng)'},
    'dog': {'type': 'animal', 'heavy': False, 'risk': 'Biting/Knocking over (Cắn/Xô ngã)'},
    'horse': {'type': 'animal', 'heavy': True, 'risk': 'Kicking/Trampling (Đá/Giẫm đạp)'},
    'sheep': {'type': 'animal', 'heavy': True, 'risk': 'Knocking over (Xô ngã)'},
    'cow': {'type': 'animal', 'heavy': True, 'risk': 'Trampling (Giẫm đạp)'},
    
    # 3. Nhóm Đồ thể thao / Cứng
    'frisbee': {'type': 'sport', 'heavy': False, 'risk': 'Hitting (Va đập)'},
    'skis': {'type': 'sport', 'heavy': False, 'risk': 'Tripping/Sharp edge (Vấp ngã/Cạnh sắc)'},
    'snowboard': {'type': 'sport', 'heavy': True, 'risk': 'Tripping (Vấp ngã)'},
    'sports ball': {'type': 'sport', 'heavy': False, 'risk': 'Tripping hazard (Trượt ngã)'},
    'kite': {'type': 'sport', 'heavy': False, 'risk': 'Strangulation from string (Siết cổ do dây)'},
    'baseball bat': {'type': 'sport', 'heavy': True, 'risk': 'Blunt trauma (Chấn thương va đập)'},
    'skateboard': {'type': 'sport', 'heavy': True, 'risk': 'Rolling hazard/Fall (Trượt té)'},
    'tennis racket': {'type': 'sport', 'heavy': False, 'risk': 'Hitting (Va đập)'},
    
    # 4. Nhóm Nội thất / Leo trèo / Bị đè
    'chair': {'type': 'furniture', 'heavy': False, 'risk': 'Falling from climb (Té ngã từ trên ghế)'},
    'couch': {'type': 'furniture', 'heavy': True, 'risk': 'Falling (Té ngã)'},
    'potted plant': {'type': 'furniture', 'heavy': True, 'risk': 'Poisonous leaves/Dirt ingestion (Ăn đất/Lá độc)'},
    'bed': {'type': 'furniture', 'heavy': True, 'risk': 'Rolling off (Lăn rớt khỏi giường)'},
    'dining table': {'type': 'furniture', 'heavy': True, 'risk': 'Falling from height (Té ngã từ bàn)'},
    'toilet': {'type': 'bathroom', 'heavy': True, 'risk': 'Drowning/Hygiene (Ngạt nước/Nhiễm khuẩn)'},
    'sink': {'type': 'bathroom', 'heavy': True, 'risk': 'Slip/Scalding (Trượt chân/Bỏng vòi nước nóng)'},
    'bench': {'type': 'furniture', 'heavy': True, 'risk': 'Falling (Té ngã)'},
    
    # 5. Nhóm Điện tử / Nóng / Điện giật
    'microwave': {'type': 'appliance', 'heavy': True, 'risk': 'Burn/Electrical (Bỏng nhiệt/Điện)'},
    'oven': {'type': 'appliance', 'heavy': True, 'risk': 'Severe Burn (Bỏng cấp độ nặng)'},
    'toaster': {'type': 'appliance', 'heavy': False, 'risk': 'Burn/Electric shock (Bỏng/Điện giật)'},
    'hair drier': {'type': 'appliance', 'heavy': False, 'risk': 'Burn/Electrocution in water (Bỏng/Điện giật nước)'},
    
    # 6. Nhóm Sắc nhọn / Thủy tinh / Độc hại
    'bottle': {'type': 'liquid', 'heavy': False, 'risk': 'Poisoning/Chemicals (Ngộ độc hóa chất/Chất tẩy)'},
    'wine glass': {'type': 'glassware', 'heavy': False, 'risk': 'Shattering/Deep cuts (Rơi vỡ cắt đứt tay)'},
    'cup': {'type': 'glassware', 'heavy': False, 'risk': 'Scalding/Shattering (Bỏng nước sôi/Vỡ)'},
    'fork': {'type': 'sharp', 'heavy': False, 'risk': 'Puncture wound (Đâm chọc tổn thương)'},
    'knife': {'type': 'sharp', 'heavy': False, 'risk': 'Severe Laceration (Vết thương hở sâu/Cắt)'},
    'spoon': {'type': 'sharp', 'heavy': False, 'risk': 'Eye injury (Chọc vào mắt)'},
    'bowl': {'type': 'glassware', 'heavy': False, 'risk': 'Scalding (Đổ canh/cháo nóng)'},
    'vase': {'type': 'glassware', 'heavy': True, 'risk': 'Shattering/Crushing (Bình hoa đổ vỡ)'},
    'scissors': {'type': 'sharp', 'heavy': False, 'risk': 'Puncture/Cut (Đâm, cắt trúng mắt/tay)'},
    
    # 7. Nhóm Túi xách / Dây nhợ
    'backpack': {'type': 'bag', 'heavy': True, 'risk': 'Tripping/Strangulation (Vấp ngã/Dây vướng)'},
    'umbrella': {'type': 'sharp', 'heavy': False, 'risk': 'Eye puncture (Móc vào mắt)'},
    'handbag': {'type': 'bag', 'heavy': False, 'risk': 'Choking on small items inside (Hóc đồ trong túi)'},
    'tie': {'type': 'cloth', 'heavy': False, 'risk': 'Strangulation (Siết cổ)'},
    'suitcase': {'type': 'furniture', 'heavy': True, 'risk': 'Tripping/Crushing (Vấp ngã/Đè)'},
    
    # 8. Nhóm Khác
    'book': {'type': 'safe', 'heavy': False, 'risk': 'Paper cut (Đứt tay nhẹ)'},
    'clock': {'type': 'glassware', 'heavy': False, 'risk': 'Shattering/Battery (Rơi vỡ/Nuốt pin)'},
    'teddy bear': {'type': 'soft', 'heavy': False, 'risk': 'Suffocation for infants (Ngạt thở cho trẻ sơ sinh)'},
    'toothbrush': {'type': 'small', 'heavy': False, 'risk': 'Throat injury (Chọc vào họng)'}
}

# Các hằng số Cấu hình Heuristics
SAFE_DIST = 400    
WARNING_DIST = 200 
DANGER_DIST = 100  
HISTORY_FRAMES = 20 
VELOCITY_THRESHOLD = 15.0 # Pixel/frame. Nếu lớn hơn -> Trẻ đang chạy/lao tới

# ==========================================
# 3. KINEMATIC & RULE-BASED EXPERT ENGINE
# ==========================================
def evaluate_interaction(child_box, obj_box, obj_name, min_dist, dist_hist, child_vel):
    """
    Hệ chuyên gia đánh giá tương tác với Động học (Kinematics)
    Trả về: (Status, Action, Risk_Text)
    """
    cx1, cy1, cx2, cy2 = child_box
    ox1, oy1, ox2, oy2 = obj_box
    
    c_center_y = (cy1 + cy2) / 2
    o_center_y = (oy1 + oy2) / 2
    
    # 1. Logic Giao thoa (Intersection)
    x_left = max(cx1, ox1); y_top = max(cy1, oy1)
    x_right = min(cx2, ox2); y_bottom = min(cy2, oy2)
    is_intersecting = (x_right >= x_left) and (y_bottom >= y_top)
    
    rule = RISK_MATRIX.get(obj_name, None)
    if not rule: return "SAFE", "none", ""

    status, action, risk = "SAFE", "no interaction", ""

    # 2. Logic Thời gian (Reaching) & Động học (Velocity)
    is_reaching = False
    is_running = child_vel > VELOCITY_THRESHOLD
    
    if len(dist_hist) == HISTORY_FRAMES:
        first_half = sum(list(dist_hist)[:10]) / 10
        second_half = sum(list(dist_hist)[10:]) / 10
        if second_half < first_half - 5:
            is_reaching = True

    # 3. LUẬT LỚN (CORE RULES)
    if is_intersecting or min_dist <= DANGER_DIST:
        status = "DANGER"
        risk = rule['risk']
        
        o_type = rule['type']
        
        # Nhóm Nội thất / Toilet
        if o_type in ['furniture', 'bathroom']:
            if c_center_y < o_center_y - 20: # Trẻ cao hơn hẳn
                action = "climbing"
            elif rule['heavy'] and is_intersecting and (c_center_y > o_center_y + 30):
                action = "crushed / trapped under"
            else:
                action = "touching / pulling"
                
        # Nhóm Điện tử / Đồ gia dụng
        elif o_type in ['electronics', 'appliance']:
            action = "pulling / manipulating" if rule['heavy'] else "touching / holding"
            
        # Nhóm Phương tiện, Động vật
        elif o_type in ['vehicle', 'animal']:
            action = "close proximity / petting"
            if is_running: risk = f"COLLISION/ATTACK IMMINENT! ({risk})"
            
        # Nhóm Sắc nhọn / Ly tách
        elif o_type in ['sharp', 'glassware', 'liquid']:
            action = "holding / playing with"
            
        # Thể thao / Túi xách
        elif o_type in ['sport', 'bag']:
            action = "playing / tripping over"
            
        else:
            action = "interacting"

    # 4. Cảnh báo sớm (Early Warning)
    elif min_dist <= WARNING_DIST and is_reaching:
        status = "WARNING"
        action = "running towards" if is_running else "reaching for"
        risk = f"Approaching {obj_name.upper()}"
        if is_running: 
            status = "DANGER" # Nếu đang lao tốc độ cao tới vật nguy hiểm -> DANGER luôn
            risk = f"HIGH SPEED COLLISION RISK: {obj_name.upper()}"
            
    return status, action, risk


# ==========================================
# 4. TẢI MODEL & VIDEO
# ==========================================
@st.cache_resource
def load_model():
    model = YOLO("yolo26x.pt") 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, device

def download_youtube_video(url):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, 'demo_vip.mp4')
    if os.path.exists(output_path): os.remove(output_path)
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best',
        'outtmpl': output_path,
        'quiet': True, 'no_warnings': True, 'nocheckcertificate': True, 'geo_bypass': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# ==========================================
# 5. GIAO DIỆN ĐIỀU KHIỂN
# ==========================================
# YouTube URL input temporarily disabled
# col_in1, col_in2 = st.columns(2)
# with col_in1:
#     yt_url = st.text_input("🔗 Nhập Link YouTube Camera:", placeholder="https://www.youtube.com/watch?v=...")
yt_url = None
uploaded_file = st.file_uploader("📂 Upload Raw Video (MP4/AVI):", type=['mp4', 'avi', 'mov'])

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    process_btn = st.button("🚀 Proceeding Video")

if process_btn and uploaded_file:
    status_text = st.empty()
    video_path = None
    
    try:
        if uploaded_file is not None:
            status_text.info("Đang đọc video local...")
            temp_dir = tempfile.gettempdir()
            video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(video_path, "wb") as f: f.write(uploaded_file.read())
            
        status_text.success("Cơ sở dữ liệu 80 Lớp Đã Sẵn Sàng. Khởi động GPU Engine...")
        model, device_name = load_model()
        
        cap = cv2.VideoCapture(video_path)
        
        # --- DASHBOARD METRICS ---
        st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_fps = m_col1.empty()
        m_threat = m_col2.empty()
        m_objs = m_col3.empty()
        m_child = m_col4.empty()
        
        # Bố cục giao diện Real-time (2 Cột Video)
        st.markdown("<br>", unsafe_allow_html=True)
        vid_col1, vid_col2 = st.columns(2)
        vid_col1.markdown("<h4 style='text-align: center; color: #888;'>🎥 LUỒNG CAMERA GỐC</h4>", unsafe_allow_html=True)
        vid_col2.markdown("<h4 style='text-align: center; color: #ff003c;'>🛡️ LỚP PHÂN TÍCH SPATIO-TEMPORAL</h4>", unsafe_allow_html=True)
        orig_placeholder = vid_col1.empty()
        proc_placeholder = vid_col2.empty()
        
        st.markdown("<h4 style='color: #00e676; margin-top: 30px;'>📋 NHẬT KÝ KIỂM TOÁN TƯƠNG TÁC (AUDIT LOGS)</h4>", unsafe_allow_html=True)
        table_placeholder = st.empty()
        
        # Biến Bộ nhớ
        distance_history = {} 
        child_centroid_history = {} # Lưu tọa độ tâm trẻ em để tính Vận tốc
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
            prev_time = curr_time
            
            # Resize 640px cho mượt
            target_width = 640
            h, w = frame.shape[:2]
            target_height = int(h * (target_width / w))
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Hiện ảnh gốc
            orig_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_placeholder.image(orig_frame_rgb, channels="RGB", use_container_width=True)
            
            # --- AI INFERENCE ---
            proc_frame = frame.copy()
            # Confidence 0.05 quét siêu sâu
            results = model.track(proc_frame, persist=True, verbose=False, device=device_name, conf=0.05)[0]
            
            children = []
            objects = []
            
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                cls_ids = results.boxes.cls.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy()
                
                for box, cls, trk_id in zip(boxes, cls_ids, track_ids):
                    class_name = model.names[int(cls)]
                    x1, y1, x2, y2 = map(int, box)
                    
                    if class_name == 'person':
                        children.append({'id': trk_id, 'box': (x1, y1, x2, y2), 'center': ((x1+x2)//2, (y1+y2)//2)})
                    elif class_name in RISK_MATRIX:
                        objects.append({'id': trk_id, 'class': class_name, 'box': (x1, y1, x2, y2), 'center': ((x1+x2)//2, (y1+y2)//2)})
            
            table_rows = []
            global_threat_level = "SAFE"
            active_dangers = 0
            
            # TÍNH VẬN TỐC TRẺ EM
            for child in children:
                c_id = int(child['id'])
                c_cx, c_cy = child['center']
                
                if c_id not in child_centroid_history:
                    child_centroid_history[c_id] = deque(maxlen=5)
                child_centroid_history[c_id].append((c_cx, c_cy))
                
                child_vel = 0.0
                if len(child_centroid_history[c_id]) == 5:
                    old_c = child_centroid_history[c_id][0]
                    new_c = child_centroid_history[c_id][-1]
                    # Khoảng cách di chuyển chia cho số frame (5)
                    child_vel = math.hypot(new_c[0] - old_c[0], new_c[1] - old_c[1]) / 5.0
                child['velocity'] = child_vel

            # DUYỆT CÁC VẬT THỂ
            for obj in objects:
                ox1, oy1, ox2, oy2 = obj['box']
                ocx, ocy = obj['center']
                obj_cls = obj['class']
                obj_id = int(obj['id'])
                
                min_dist = float('inf')
                closest_child = None
                
                for child in children:
                    ccx, ccy = child['center']
                    dist = math.hypot(ccx - ocx, ccy - ocy)
                    if dist < min_dist:
                        min_dist = dist
                        closest_child = child
                
                if closest_child:
                    c_box = closest_child['box']
                    ccx, ccy = closest_child['center']
                    c_id = int(closest_child['id'])
                    c_vel = closest_child['velocity']
                    
                    if obj_id not in distance_history:
                        distance_history[obj_id] = deque(maxlen=HISTORY_FRAMES)
                    distance_history[obj_id].append(min_dist)
                    
                    # GỌI EXPERT ENGINE
                    status, action, risk = evaluate_interaction(c_box, obj['box'], obj_cls, min_dist, distance_history[obj_id], c_vel)
                    
                    # ==========================================
                    # VẼ LÊN VIDEO (GLOW EFFECT)
                    # ==========================================
                    if status in ["DANGER", "WARNING"]:
                        if status == "DANGER":
                            global_threat_level = "CRITICAL"
                            active_dangers += 1
                            box_color = (0, 0, 255) # Đỏ
                            
                            # Hiệu ứng Glow cho Box nguy hiểm
                            overlay = proc_frame.copy()
                            cv2.rectangle(overlay, (ox1, oy1), (ox2, oy2), box_color, 8)
                            cv2.addWeighted(overlay, 0.4, proc_frame, 0.6, 0, proc_frame)
                        else:
                            if global_threat_level == "SAFE": global_threat_level = "WARNING"
                            box_color = (0, 165, 255) # Cam
                        
                        # Vẽ Trẻ (Xanh neon)
                        cv2.rectangle(proc_frame, (c_box[0], c_box[1]), (c_box[2], c_box[3]), (0, 255, 0), 2)
                        
                        # Vẽ Vật & Action
                        cv2.rectangle(proc_frame, (ox1, oy1), (ox2, oy2), box_color, 2)
                        cv2.putText(proc_frame, f"{obj_cls.upper()}", (ox1, max(15, oy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        
                        cv2.line(proc_frame, (ccx, ccy), (ocx, ocy), box_color, 2)
                        mid_x, mid_y = (ccx + ocx) // 2, (ccy + ocy) // 2
                        
                        text = f"{action.upper()}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(proc_frame, (mid_x - 5, mid_y - th - 5), (mid_x + tw + 5, mid_y + 5), (0,0,0), -1)
                        cv2.putText(proc_frame, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    # Ghi Log
                    if status == "DANGER": css_class = "row-danger"
                    elif status == "WARNING": css_class = "row-warn"
                    else: css_class = "row-safe"
                    
                    row = f"<tr class='{css_class}'><td>Child {c_id} (v={c_vel:.1f})</td><td>{obj_cls.upper()}</td><td>{status}</td><td>{action}</td><td>{risk}</td></tr>"
                    if status != "SAFE": table_rows.insert(0, row) # Đẩy cảnh báo lên đầu
                    else: table_rows.append(row)

            # CẢNH BÁO TỔNG MÀN HÌNH
            if global_threat_level == "CRITICAL":
                cv2.rectangle(proc_frame, (0, 0), (target_width, 40), (0, 0, 255), -1)
                cv2.putText(proc_frame, "!!! SYSTEM ALERT: IMMEDIATE INTERVENTION REQUIRED !!!", (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            # Update Metrics Dashboard
            t_color = "#ff003c" if global_threat_level == "CRITICAL" else ("#ffaa00" if global_threat_level == "WARNING" else "#00e676")
            m_fps.markdown(f"<div class='metric-box'><div class='metric-title'>ENGINE SPEED</div><div class='metric-val'>{fps:.1f} FPS</div></div>", unsafe_allow_html=True)
            m_threat.markdown(f"<div class='metric-box'><div class='metric-title'>THREAT LEVEL</div><div class='metric-val' style='color:{t_color};'>{global_threat_level}</div></div>", unsafe_allow_html=True)
            m_objs.markdown(f"<div class='metric-box'><div class='metric-title'>MONITORED OBJECTS</div><div class='metric-val'>{len(objects)}</div></div>", unsafe_allow_html=True)
            m_child.markdown(f"<div class='metric-box'><div class='metric-title'>ACTIVE DANGERS</div><div class='metric-val' style='color:#ff003c;'>{active_dangers}</div></div>", unsafe_allow_html=True)

            # Render
            proc_frame_rgb = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            proc_placeholder.image(proc_frame_rgb, channels="RGB", use_container_width=True)
            
            if not table_rows:
                table_html = "<div style='color: #555; text-align:center;'>Không phát hiện vật thể rủi ro trong khung hình này.</div>"
            else:
                table_html = """
                <table class='log-table'>
                    <tr><th>Subject (Velocity)</th><th>Object Class</th><th>Status</th><th>Kinematic Action</th><th>Risk Profile</th></tr>
                    """ + "".join(table_rows[:15]) + "</table>" # Hiển thị 15 dòng log
            table_placeholder.markdown(table_html, unsafe_allow_html=True)
            
        cap.release()
        status_text.success("✅ Phân tích toàn diện hoàn tất!")
        
    except Exception as e:
        status_text.error(f"❌ Lỗi hệ thống: {str(e)}")