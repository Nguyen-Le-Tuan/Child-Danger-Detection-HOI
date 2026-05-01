## 🇻🇳 Tiếng Việt: Áp dụng học sâu trong cảnh báo một số tương tác nguy hiểm của trẻ em trong nhà qua camera an ninh theo thời gian thực
# 🛡️ Child Danger Detection via Human-Object Interaction (HOI) - NCKH

Dự án Nghiên cứu khoa học (NCKH) xây dựng Hệ thống nhận diện Hành vi tương tác giữa Người và Vật thể (Human-Object Interaction - HOI) nhằm mục đích **Cảnh báo sớm và ngăn chặn các rủi ro nguy hiểm cho trẻ em**. 

Hệ thống là sự kết hợp tiên tiến giữa kiến trúc phát hiện đối tượng kép (Dual-YOLO), trích xuất đặc trưng không gian (EfficientNet/ResNet/CLIP) và **mạng dự đoán chuỗi thời gian (GRU - Multi-Task Forecasting Network)** để phân tích quỹ đạo di chuyển và đánh giá mức độ rủi ro theo thời gian thực trước khi tai nạn thực sự xảy ra.

## 🚀 Tính năng nổi bật (Key Features)

- **Spatio-Temporal Pipeline (Phân tích Không gian - Thời gian):** Theo dõi liên tục sự thay đổi khoảng cách và hành vi giữa trẻ em và các vật thể nguy hiểm trong một chuỗi khung hình (time-series).
- **Dual-YOLO Object Detection:** Sử dụng YOLO11x và các mô hình YOLO được fine-tune để tách biệt chính xác `child` (trẻ em), `adult` (người lớn) và nhận diện các vật thể có khả năng gây sát thương (dao, kéo, ổ điện...).
- **Mạng dự đoán GRU (MTFN):** Tích hợp mạng Gated Recurrent Unit (GRU) đa tác vụ để dự báo các hành động tiến tới (reaching) hoặc chạm (touching) ở các khung hình tương lai.
- **Bộ công cụ Auto-Annotation & Data Cleaning (Streamlit):** Tích hợp sẵn các tool GUI chuyên nghiệp cho phép gán nhãn tự động (auto-forward bounding boxes), trích xuất vector đặc trưng CLIP và dọn dẹp nhiễu BBox trực quan.
- **Kiến trúc QueryCraft:** Thuật toán ghép cặp (Hungarian Bipartite Matching) để map chính xác mối quan hệ ngữ nghĩa giữa Người - Vật Thể.

---

## 📁 Cấu trúc thư mục cốt lõi (Repository Structure)

```text
├── GRU-second-version/                  # Mạng dự đoán chuỗi thời gian (MTFN) & Scripts huấn luyện GRU
│   ├── models/backbone.py               # Kiến trúc MTFN Triple Input kết hợp SemanticMixer
│   └── scripts/train.py                 # Huấn luyện mô hình GRU (hỗ trợ Early Stopping, Oversampling)
├── SafeGuard_Custom_QueryCraft/         # Code huấn luyện trích xuất đặc trưng HOI (ResNet/EfficientNet)
│   ├── train_hicodet.py                 # Script huấn luyện trên dataset HICO-DET
│   └── evaluate_hicodet.py              # Script đánh giá hiệu năng mô hình
├── completed_anno_training_data_tools.py # Ứng dụng Streamlit tự động gán nhãn HOI Workflow & nhúng CLIP
├── B_editing_processed_data_app.py      # Công cụ GUI kiểm duyệt Dataset, chỉnh sửa BBox và Auto-forward ID
├── clean_bbox.py / fix.py               # Các script làm sạch nhiễu BBox và tái cấu trúc dữ liệu JSON/CSV
├── main_deep_learning_app.py                     # Giao diện Dashboard hiển thị cảnh báo thời gian thực
├── requirements.txt                     # Danh sách thư viện Python cần thiết
└── .gitignore                           # Cấu hình bỏ qua Weights & Dataset nặng
```

---

## ⚙️ Cài đặt & Môi trường (Installation)

**1. Clone dự án & Thiết lập môi trường:**
```bash
git clone <https://github.com/your-username/Child-Danger-Detection-HOI.git>
cd Child-Danger-Detection-HOI

conda create -n nckh_env python=3.10 -y
conda activate nckh_env
```

**2. Cài đặt các thư viện phụ thuộc:**
```bash
pip install -r requirements.txt
```
*(Yêu cầu hệ thống: Linux/Ubuntu 22.04 LTS, NVIDIA GPU + CUDA 11.8/12.1 để chạy GRU/YOLO mượt mà nhất).*

---

## 💻 Hướng dẫn sử dụng (Quick Start)

### 1. Chuẩn bị Dữ liệu & Gán nhãn tự động
Khởi chạy công cụ Annotation Tool tự động kết hợp CLIP & YOLO:
```bash
streamlit run completed_anno_training_data_tools.py
```
Công cụ kiểm duyệt và chỉnh sửa BBox chuỗi thời gian:
```bash
streamlit run B_editing_processed_data_app.py
```

### 2. Huấn luyện mô hình (Training)
Huấn luyện mạng GRU (Dự đoán rủi ro chuỗi thời gian):
```bash
cd GRU-second-version
python scripts/train.py
```

### 3. Khởi chạy Hệ thống Cảnh báo (Dashboard)
```bash
streamlit run main_deep_learning_app.py
```

---

## 🇬🇧 English Version: Child Danger Detection via Human-Object Interaction

**An AI-powered Human-Object Interaction (HOI) system for detecting and proactively predicting dangerous interactions between children and surrounding objects.**

This Scientific Research Project (NCKH) introduces a proactive Spatio-Temporal Pipeline to monitor, analyze, and forecast hazardous situations involving children. Instead of merely detecting when an accident occurs, the system utilizes a **Multi-Task Forecasting Network (MTFN)** based on **Gated Recurrent Units (GRU)** to predict threatening trajectories (e.g., a child reaching for a knife or electrical socket) before they materialize.

### 🚀 Key Architectural Highlights
*   **Spatio-Temporal Deep Learning Pipeline:** Integrates spatial feature extraction (EfficientNet-B3 / ResNet / CLIP) with temporal sequence analysis (GRU) to understand complex behaviors over time.
*   **Dual-YOLO Object Detection:** Employs a decoupled YOLO architecture (YOLO11x) to accurately distinguish between adults and children while exhaustively detecting hazardous objects.
*   **Advanced HOI Engine (QueryCraft):** Utilizes Hungarian Bipartite Matching to map semantic relationships between humans and objects accurately.
*   **Proactive Temporal Forecasting:** Integrates MTFN to process a 30-frame sequence buffer, continuously analyzing child-object distances and kinematic features to compute dynamic threat levels.
*   **Comprehensive Custom Tools:** Features fully custom-built Streamlit GUI tools for automated CLIP embeddings, automated bounding-box forward propagation, and BBox noise cleaning.

### 📁 System Components
*   `GRU-second-version/`: Contains the core MTFN architecture and training scripts for time-series danger forecasting.
*   `SafeGuard_Custom_QueryCraft/`: Fine-tuning scripts for the Spatial HOI extraction engine using HICO-DET.
*   `completed_anno_training_data_tools.py`: A comprehensive Streamlit application for automated HOI annotation, clipping extraction, and BBox tracking.
*   `B_editing_processed_data_app.py`: A manual review dashboard to audit, edit, and sync BBox coordinates across frames.
*   `clean_bbox.py` & `fix.py`: Data preprocessing pipelines for sanitizing bounding box noise and restructuring nested JSON annotations.

### 📦 Weights & Datasets Note
In accordance with standard repository practices, heavy weights (`.pt`, `.pth`, `.npy`) and video datasets are ignored via `.gitignore` (Max limit 100MB/file). Pre-trained YOLO11 weights are downloaded automatically on the first inference run.

---

**Developed by:** Nguyễn Lê Tuấn & Đặng Trường Phát  
**Institution:** Le Hong Phong High School for the Gifted, Ho Chi Minh City (THPT Chuyên Lê Hồng Phong TPHCM).
