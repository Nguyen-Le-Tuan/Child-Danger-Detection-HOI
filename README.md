# Child Danger Detection via Human-Object Interaction (HOI)

An AI-based research project for detecting and forecasting dangerous child-object interactions from camera streams.

## Overview

This project combines object detection, HOI reasoning, and temporal forecasting to identify high-risk situations early (for example, a child reaching toward hazardous objects).  
The system integrates Dual-YOLO detection, spatial feature extraction (EfficientNet/ResNet/CLIP), and a GRU-based Multi-Task Forecasting Network (MTFN).

## Key Features

- **Spatio-temporal risk analysis:** Tracks behavior and distance changes across frame sequences.
- **Dual-YOLO pipeline:** Detects children, adults, and hazardous objects with role-aware modeling.
- **GRU forecasting (MTFN):** Predicts near-future interaction risks before incidents occur.
- **QueryCraft matching:** Uses Hungarian bipartite matching for precise human-object pairing.
- **Custom Streamlit tools:** Supports annotation, bounding-box cleanup, and dataset review workflows.

## Repository Structure

```text
├── GRU-second-version/                    # Time-series forecasting model and training scripts
│   ├── models/backbone.py
│   └── scripts/train.py
├── SafeGuard_Custom_QueryCraft/           # HOI feature extraction training/evaluation
│   ├── train_hicodet.py
│   └── evaluate_hicodet.py
├── completed_anno_training_data_tools.py  # Streamlit app for annotation and CLIP workflows
├── B_editing_processed_data_app.py        # Streamlit app for BBox review/editing
├── clean_bbox.py
├── fix.py                                 # Data cleaning and restructuring scripts
├── main_deep_learning_app.py              # Real-time warning dashboard
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/your-username/Child-Danger-Detection-HOI.git
cd Child-Danger-Detection-HOI

conda create -n nckh_env python=3.10 -y
conda activate nckh_env
pip install -r requirements.txt
```

Recommended environment: Ubuntu 22.04, NVIDIA GPU, CUDA 11.8/12.1.

## Quick Start

### 1) Annotation and data review

```bash
streamlit run completed_anno_training_data_tools.py
streamlit run B_editing_processed_data_app.py
```

### 2) Train forecasting model

```bash
cd GRU-second-version
python scripts/train.py
```

### 3) Run real-time dashboard

```bash
streamlit run main_deep_learning_app.py
```

## Weights and Datasets

Large model weights (`.pt`, `.pth`, `.npy`) and datasets are excluded via `.gitignore`.  
Pretrained YOLO weights are downloaded automatically on first inference.

## Credits

Developed by **Nguyễn Lê Tuấn** and **Đặng Trường Phát**  
Le Hong Phong High School for the Gifted, Ho Chi Minh City
