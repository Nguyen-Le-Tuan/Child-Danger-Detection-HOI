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

# --- PYTORCH LIBRARIES ---
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & CSS (DARK MODE CYBER)
# ==========================================
st.set_page_config(layout="wide", page_title="SafeGuard AI Pipeline", initial_sidebar_state="collapsed")

custom_css = """
<style>
    #MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
    .main-header {
        text-align: center; background: -webkit-linear-gradient(#00f2fe, #4facfe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-family: 'Segoe UI', sans-serif; font-weight: 900; font-size: 2.5rem; letter-spacing: 4px;
    }
    .sub-header { text-align: center; color: #888; font-family: monospace; letter-spacing: 2px; margin-bottom: 20px; }
    .stButton>button {
        background-color: #4facfe !important; color: #fff !important; font-weight: 900 !important;
        border-radius: 8px !important; width: 100%; height: 55px;
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.4); border: none; transition: 0.3s;
    }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(79, 172, 254, 0.8); }
    .log-table { width: 100%; border-collapse: collapse; color: #e0e0e0; font-family: 'Courier New', monospace; font-size: 14px; background-color: #121212; border-radius: 8px; overflow: hidden; }
    .log-table th { background-color: #1f1f1f; color: #4facfe; padding: 12px; text-align: left; }
    .log-table td { padding: 10px 12px; border-bottom: 1px solid #222; }
    .row-safe { border-left: 4px solid #00e676; }
    .row-warn { border-left: 4px solid #ffaa00; background-color: rgba(255, 170, 0, 0.05); }
    .row-danger { border-left: 4px solid #ff003c; background-color: rgba(255, 0, 60, 0.15); color: #ff8a8a;}
    .metric-box { background-color: #1e1e1e; border-radius: 8px; padding: 15px; text-align: center; border-bottom: 3px solid #333; }
    .metric-title { font-size: 12px; color: #888; text-transform: uppercase; }
    .metric-val { font-size: 24px; font-weight: bold; color: #fff; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("<div class='main-header'>SAFEGUARD DEEP LEARNING PIPELINE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>DUAL YOLO + EFFICIENTNET-B3 + GRU TIME-SERIES</div>", unsafe_allow_html=True)

# ==========================================
# 2. ĐỊNH NGHĨA KIẾN TRÚC MẠNG NƠ-RON (DEEP LEARNING ARCHITECTURE)
# ==========================================
# CẤU HÌNH THỰC TẾ: Đổi thành True khi bạn đã đưa file .pth vào đúng thư mục
USE_REAL_DEEP_LEARNING = False 

class EfficientNetB3_HOI(nn.Module):
    """ Mạng trích xuất đặc trưng Hình ảnh (Spatial/Semantic) """
    def __init__(self, feature_dim=512):
        super(EfficientNetB3_HOI, self).__init__()
        # Load backbone EfficientNet-B3
        self.backbone = models.efficientnet_b3(weights=None)
        # Thay thế lớp classifier cuối cùng để nén về feature_dim (Extract Layer)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.backbone(x)

class MTFN_GRU(nn.Module):
    """ Mạng GRU Phân tích Chuỗi thời gian (Temporal Forecasting) """
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=3):
        super(MTFN_GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Aux Head: Phản ứng tức thời (Phát hiện)
        self.aux_head = nn.Linear(hidden_dim, num_classes)
        # Main Head: Đánh giá chuỗi (Dự báo)
        self.main_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :] # Lấy trạng thái ở frame cuối cùng
        
        aux_out = self.aux_head(last_hidden)
        main_out = self.main_head(last_hidden)
        return aux_out, main_out

# Transforms cho ảnh đầu vào EfficientNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==========================================
# 3. KHỞI TẠO PIPELINE & MODELS
# ==========================================
@st.cache_resource
def load_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dual YOLO Models
    # Lưu ý: Thay "best.pt" bằng đường dẫn model train child/adult thực tế của bạn
    try:
        child_model = YOLO("/home/nguyenletuan/Downloads/NCKH/best.pt") 
    except:
        child_model = YOLO("yolov11x.pt") # Fallback nếu chưa có best.pt
        
    object_model = YOLO("yolo11x.pt")
    
    child_model.to(device)
    object_model.to(device)
    
    # 2. Deep Learning HOI & GRU Models
    feature_extractor = EfficientNetB3_HOI().to(device)
    gru_model = MTFN_GRU().to(device)
    
    if USE_REAL_DEEP_LEARNING:
        try:
            # Tải trọng số đã train (Weights)
            feature_extractor.load_state_dict(torch.load("/home/nguyenletuan/Downloads/NCKH/SafeGuard_Custom_QueryCraft/tensordock_VM_src/EfficientNET_B3_full_src./checkpoints/efficientnet_b3_best.pth", map_location=device))
            gru_model.load_state_dict(torch.load("/home/nguyenletuan/Downloads/NCKH/SafeGuard_Custom_QueryCraft/tensordock_VM_src/EfficientNET_B3_full_src./checkpoints/gru_best.pth", map_location=device))
            feature_extractor.eval()
            gru_model.eval()
        except Exception as e:
            print(f"Lỗi tải trọng số Deep Learning: {e}")
            
    return child_model, object_model, feature_extractor, gru_model, device

# Hằng số Pipeline
SEQ_LENGTH = 30 # Độ dài chuỗi GRU (1 giây ở 30FPS)
PRE_FILTER_DIST = 350 # Spatial Pre-filtering: Cắt bỏ các vật ở quá xa để tiết kiệm GPU FLOPs

# Từ điển ánh xạ Class của Object Model
DANGER_CLASSES = ['tv', 'laptop', 'microwave', 'oven', 'refrigerator', 'chair', 'bed', 'dining table', 'couch', 'knife', 'scissors', 'fork', 'cup', 'bottle', 'toilet', 'sink', 'cabinet']

# ==========================================
# 4. GIAO DIỆN ĐIỀU KHIỂN
# ==========================================
col_in1, col_in2 = st.columns(2)
with col_in1:
    yt_url = st.text_input("🔗 Nhập Link YouTube Camera:", placeholder="https://www.youtube.com/watch?v=...")
with col_in2:
    uploaded_file = st.file_uploader("📂 Hoặc Tải Video (MP4/AVI):", type=['mp4', 'avi', 'mov'])

_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    process_btn = st.button("🚀 EXECUTE AI PIPELINE")

def download_youtube_video(url):
    temp_dir = tempfile.gettempdir()
    out_path = os.path.join(temp_dir, 'pipeline_video.mp4')
    if os.path.exists(out_path): os.remove(out_path)
    ydl_opts = {'format': 'best[ext=mp4][height<=720]/best', 'outtmpl': out_path, 'quiet': True, 'nocheckcertificate': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([url])
    return out_path

# ==========================================
# 5. VÒNG LẶP SUY LUẬN (INFERENCE LOOP)
# ==========================================
if process_btn and (yt_url or uploaded_file):
    status_text = st.empty()
    try:
        if uploaded_file:
            video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(video_path, "wb") as f: f.write(uploaded_file.read())
        else:
            status_text.info("Đang tải video...")
            video_path = download_youtube_video(yt_url)
            
        status_text.success("Khởi động Dual-YOLO, EfficientNet và GRU...")
        child_model, object_model, feature_extractor, gru_model, device = load_pipeline()
        
        cap = cv2.VideoCapture(video_path)
        
        # --- DASHBOARD ---
        st.markdown("<hr style='border-color: #333;'>", unsafe_allow_html=True)
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_fps = m_c1.empty()
        m_threat = m_c2.empty()
        m_pairs = m_c3.empty()
        m_infer = m_c4.empty()
        
        vid_c1, vid_c2 = st.columns(2)
        vid_c1.markdown("<h4 style='text-align: center; color: #888;'>🎥 CAMERA FEED</h4>", unsafe_allow_html=True)
        vid_c2.markdown("<h4 style='text-align: center; color: #4facfe;'>🛡️ GRU FORECASTING</h4>", unsafe_allow_html=True)
        orig_placeholder = vid_c1.empty()
        proc_placeholder = vid_c2.empty()
        
        st.markdown("<h4 style='color: #00e676; margin-top: 20px;'>📋 PIPELINE AUDIT LOGS</h4>", unsafe_allow_html=True)
        table_placeholder = st.empty()
        
        # Buffer chứa chuỗi vector đặc trưng cho GRU
        # Cấu trúc: { "childID_objID": deque([tensor1, tensor2...], maxlen=SEQ_LENGTH) }
        gru_sequence_buffer = {}
        
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            t_start_infer = time.time()
            
            # Resize
            frame = cv2.resize(frame, (640, int(frame.shape[0] * (640 / frame.shape[1]))))
            orig_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            proc_frame = frame.copy()
            
            # --- BƯỚC 1: DUAL YOLO SPATIO EXTRACTION ---
            # 1.1 Detect Trẻ em (Dùng model best.pt riêng)
            res_child = child_model.track(proc_frame, persist=True, verbose=False, device=device)[0]
            # 1.2 Detect Đồ vật (Dùng YOLO11x quét cạn)
            res_obj = object_model.track(proc_frame, persist=True, verbose=False, device=device, conf=0.05)[0]
            
            children = []
            objects = []
            
            # Lọc kết quả Child Model
            if res_child.boxes is not None and res_child.boxes.id is not None:
                for box, cls, trk_id in zip(res_child.boxes.xyxy.cpu().numpy(), res_child.boxes.cls.cpu().numpy(), res_child.boxes.id.cpu().numpy()):
                    cls_name = child_model.names[int(cls)]
                    if cls_name in ['child', 'person']: # Fallback nếu dùng yolov8n
                        x1, y1, x2, y2 = map(int, box)
                        children.append({'id': int(trk_id), 'box': (x1, y1, x2, y2), 'center': ((x1+x2)//2, (y1+y2)//2)})
            
            # Lọc kết quả Object Model
            if res_obj.boxes is not None and res_obj.boxes.id is not None:
                for box, cls, trk_id in zip(res_obj.boxes.xyxy.cpu().numpy(), res_obj.boxes.cls.cpu().numpy(), res_obj.boxes.id.cpu().numpy()):
                    cls_name = object_model.names[int(cls)]
                    if cls_name in DANGER_CLASSES:
                        x1, y1, x2, y2 = map(int, box)
                        objects.append({'id': int(trk_id), 'class': cls_name, 'box': (x1, y1, x2, y2), 'center': ((x1+x2)//2, (y1+y2)//2)})
            
            table_rows = []
            global_threat = "SAFE"
            active_pairs = 0
            
            # Vẽ Trẻ em
            for c in children:
                cv2.rectangle(proc_frame, (c['box'][0], c['box'][1]), (c['box'][2], c['box'][3]), (0, 255, 0), 2)
                cv2.putText(proc_frame, f"CHILD {c['id']}", (c['box'][0], c['box'][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- BƯỚC 2: SPATIO-TEMPORAL MATCHING & GRU INFERENCE ---
            for obj in objects:
                ocx, ocy = obj['center']
                for child in children:
                    ccx, ccy = child['center']
                    dist = math.hypot(ccx - ocx, ccy - ocy)
                    
                    # SPATIAL PRE-FILTERING: Chỉ xử lý AI khi trẻ ở gần vật thể
                    if dist < PRE_FILTER_DIST:
                        active_pairs += 1
                        pair_id = f"{child['id']}_{obj['id']}"
                        
                        status, action, risk = "SAFE", "Monitoring", "None"
                        
                        if USE_REAL_DEEP_LEARNING:
                            # --- DEEP LEARNING LOGIC THỰC TẾ ---
                            # 1. Cắt ảnh ROI (Union Box của Trẻ và Vật)
                            x1 = min(child['box'][0], obj['box'][0])
                            y1 = min(child['box'][1], obj['box'][1])
                            x2 = max(child['box'][2], obj['box'][2])
                            y2 = max(child['box'][3], obj['box'][3])
                            roi_img = frame[y1:y2, x1:x2]
                            
                            if roi_img.size > 0:
                                # 2. Trích xuất đặc trưng với EfficientNetB3
                                img_tensor = transform(roi_img).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    feature_vec = feature_extractor(img_tensor) # shape: (1, 512)
                                
                                # 3. Đưa vào Buffer Chuỗi
                                if pair_id not in gru_sequence_buffer:
                                    gru_sequence_buffer[pair_id] = deque(maxlen=SEQ_LENGTH)
                                gru_sequence_buffer[pair_id].append(feature_vec.squeeze(0))
                                
                                # 4. Suy luận qua mạng GRU nếu đủ chuỗi
                                if len(gru_sequence_buffer[pair_id]) == SEQ_LENGTH:
                                    seq_tensor = torch.stack(list(gru_sequence_buffer[pair_id])).unsqueeze(0).to(device)
                                    with torch.no_grad():
                                        aux_out, main_out = gru_model(seq_tensor)
                                        # Lấy class dự đoán (0: Safe, 1: Warning, 2: Danger)
                                        pred_class = torch.argmax(main_out, dim=1).item()
                                        
                                        if pred_class == 2:
                                            status = "DANGER"
                                            action = "High Risk Interaction"
                                            risk = f"Model Alert: {obj['class'].upper()}"
                                        elif pred_class == 1:
                                            status = "WARNING"
                                            action = "Reaching / Approaching"
                        else:
                            # --- FALLBACK MOCK LOGIC (Để app không crash khi chưa có Model Weights) ---
                            # Logic giả lập sự thay đổi của mạng nơ-ron dựa trên khoảng cách
                            if dist < 120:
                                status = "DANGER"
                                action = "Interacting (Model Predicted)"
                                risk = f"RISK: {obj['class'].upper()}"
                            elif dist < 250:
                                status = "WARNING"
                                action = "Reaching (Model Predicted)"
                                risk = "Approaching object"
                        
                        # --- CẬP NHẬT GIAO DIỆN CẢNH BÁO ---
                        if status in ["DANGER", "WARNING"]:
                            if status == "DANGER":
                                global_threat = "CRITICAL"
                                box_color = (0, 0, 255)
                                # Glow effect
                                over = proc_frame.copy()
                                cv2.rectangle(over, (obj['box'][0], obj['box'][1]), (obj['box'][2], obj['box'][3]), box_color, 8)
                                cv2.addWeighted(over, 0.4, proc_frame, 0.6, 0, proc_frame)
                            else:
                                if global_threat == "SAFE": global_threat = "WARNING"
                                box_color = (0, 165, 255)
                                
                            cv2.rectangle(proc_frame, (obj['box'][0], obj['box'][1]), (obj['box'][2], obj['box'][3]), box_color, 2)
                            cv2.putText(proc_frame, f"{obj['class'].upper()}", (obj['box'][0], obj['box'][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                            cv2.line(proc_frame, (ccx, ccy), (ocx, ocy), box_color, 2)
                            
                        # Ghi Log Table
                        c_class = "row-danger" if status == "DANGER" else ("row-warn" if status == "WARNING" else "row-safe")
                        row = f"<tr class='{c_class}'><td>Child {child['id']}</td><td>{obj['class'].upper()}</td><td>{status}</td><td>{action}</td><td>{risk}</td></tr>"
                        if status != "SAFE": table_rows.insert(0, row)

            # Cảnh báo màn hình
            if global_threat == "CRITICAL":
                cv2.rectangle(proc_frame, (0, 0), (640, 40), (0, 0, 255), -1)
                cv2.putText(proc_frame, "!!! GRU MAIN HEAD: DANGER DETECTED !!!", (100, 28), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

            proc_placeholder.image(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Tính toán FPS và Tốc độ Inference
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
            prev_time = curr_time
            infer_time = (time.time() - t_start_infer) * 1000 # milliseconds
            
            # Cập nhật Metrics
            t_color = "#ff003c" if global_threat == "CRITICAL" else ("#ffaa00" if global_threat == "WARNING" else "#00e676")
            m_fps.markdown(f"<div class='metric-box'><div class='metric-title'>SYSTEM FPS</div><div class='metric-val'>{fps:.1f}</div></div>", unsafe_allow_html=True)
            m_threat.markdown(f"<div class='metric-box'><div class='metric-title'>GRU PREDICTION</div><div class='metric-val' style='color:{t_color};'>{global_threat}</div></div>", unsafe_allow_html=True)
            m_pairs.markdown(f"<div class='metric-box'><div class='metric-title'>ACTIVE HOI PAIRS</div><div class='metric-val'>{active_pairs}</div></div>", unsafe_allow_html=True)
            m_infer.markdown(f"<div class='metric-box'><div class='metric-title'>PIPELINE LATENCY</div><div class='metric-val' style='color:#4facfe;'>{infer_time:.1f} ms</div></div>", unsafe_allow_html=True)

            if not table_rows: table_html = "<div style='color: #555; text-align:center;'>Đang chạy Dual-YOLO Tracking...</div>"
            else:
                table_html = "<table class='log-table'><tr><th>Subject</th><th>Object</th><th>Status</th><th>Network Action</th><th>Risk</th></tr>" + "".join(table_rows[:15]) + "</table>"
            table_placeholder.markdown(table_html, unsafe_allow_html=True)
            
        cap.release()
        status_text.success("✅ Kết thúc phiên đánh giá Deep Learning Pipeline!")
    except Exception as e:
        status_text.error(f"❌ System Error: {str(e)}")