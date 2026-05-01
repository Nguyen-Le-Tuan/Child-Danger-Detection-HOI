import cv2
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng file YOLO Cache (.npy)")
    parser.add_argument("--img_path", type=str, required=True, help="Đường dẫn đến ảnh gốc (VD: images/train2015/HICO_train2015_00000001.jpg)")
    parser.add_argument("--npy_path", type=str, required=True, help="Đường dẫn đến file cache tương ứng (VD: yolo_cache/train2015/HICO_train2015_00000001.npy)")
    return parser.parse_args()

def cxcywh_to_xyxy(box, w, h):
    """Chuyển đổi tọa độ [cx, cy, w, h] chuẩn hóa về Pixel thực tế [x1, y1, x2, y2]"""
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2

def main():
    args = parse_args()
    
    print("="*60)
    print("🔍 CÔNG CỤ KIỂM TRA CHẤT LƯỢNG YOLO CACHE")
    print("="*60)

    if not os.path.exists(args.img_path):
        print(f"❌ Không tìm thấy ảnh tại: {args.img_path}")
        return
        
    if not os.path.exists(args.npy_path):
        print(f"❌ Không tìm thấy file Cache tại: {args.npy_path}")
        return

    # 1. Đọc ảnh bằng OpenCV
    img = cv2.imread(args.img_path)
    if img is None:
        print(f"❌ Lỗi khi đọc ảnh: {args.img_path}")
        return
    
    h_orig, w_orig = img.shape[:2]
    img_name = os.path.basename(args.img_path)

    # 2. Đọc file YOLO Cache (.npy)
    # Định dạng lưu là [N, 6]: [cx, cy, w, h, conf, class_id]
    boxes_with_info = np.load(args.npy_path)
    num_boxes = len(boxes_with_info)
    print(f"✅ Đã tải thành công file Cache. Tìm thấy {num_boxes} bounding boxes.")

    if num_boxes == 0:
        print("⚠️ File cache trống (YOLO không nhận diện được vật thể nào trong ảnh này).")
    
    # 3. Vẽ từng hộp lên ảnh
    for i, row in enumerate(boxes_with_info):
        # Kiểm tra xem row có đủ 6 phần tử không (tránh lỗi nếu lỡ dùng bản cũ lưu 4 cột)
        if len(row) >= 6:
            cx, cy, w, h, conf, cls_id = row[:6]
        else:
            cx, cy, w, h = row[:4]
            conf, cls_id = 1.0, -1 # Giá trị mặc định nếu file npy chỉ có 4 cột
            
        cls_id = int(cls_id)
        
        # Chuyển đổi tọa độ
        x1, y1, x2, y2 = cxcywh_to_xyxy([cx, cy, w, h], w_orig, h_orig)
        
        # Phân loại màu sắc: Người (Class 0) màu Xanh, Vật thể khác màu Đỏ
        if cls_id == 0:
            color = (255, 0, 0) # Xanh dương (BGR trong OpenCV)
            label = f"Person ({conf:.2f})"
        else:
            color = (0, 0, 255) # Đỏ
            label = f"Obj_{cls_id} ({conf:.2f})"
            
        # Vẽ hình chữ nhật
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Tạo viền đen cho chữ để dễ đọc trên nền sáng/tối
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_text = max(y1, text_height + 5)
        
        # Vẽ nền đen cho Text
        cv2.rectangle(img, (x1, y_text - text_height - 5), (x1 + text_width, y_text + baseline - 5), (0, 0, 0), cv2.FILLED)
        # Vẽ Text lên trên nền đen
        cv2.putText(img, label, (x1, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. Lưu ảnh kết quả
    output_filename = f"result_{img_name}"
    cv2.imwrite(output_filename, img)
    print(f"\n📸 Đã vẽ xong! Ảnh kết quả lưu tại: {output_filename}")

if __name__ == "__main__":
    main()