import cv2
import numpy as np
import os
import argparse
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Kiểm tra hàng loạt chất lượng file YOLO Cache (.npy)")
    parser.add_argument("--img_dir", type=str, required=True, help="Thư mục chứa ảnh gốc (VD: images/train2015/)")
    parser.add_argument("--npy_dir", type=str, required=True, help="Thư mục chứa file cache (VD: yolo_cache/train2015/)")
    parser.add_argument("--output_dir", type=str, default="sample_cache_results", help="Thư mục lưu các ảnh đã vẽ (Mặc định: sample_cache_results)")
    parser.add_argument("--num_images", type=int, default=1000, help="Số lượng ảnh muốn kiểm tra ngẫu nhiên (Mặc định: 1000)")
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
    print("🔍 CÔNG CỤ KIỂM TRA YOLO CACHE HÀNG LOẠT (RANDOM SAMPLE)")
    print("="*60)

    # 1. Kiểm tra đầu vào và tạo thư mục đầu ra
    if not os.path.exists(args.img_dir):
        print(f"❌ Không tìm thấy thư mục ảnh: {args.img_dir}")
        return
    if not os.path.exists(args.npy_dir):
        print(f"❌ Không tìm thấy thư mục cache: {args.npy_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Lấy danh sách toàn bộ ảnh và chọn ngẫu nhiên
    all_images = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))]
    total_imgs = len(all_images)
    print(f"📦 Tìm thấy tổng cộng {total_imgs} ảnh.")
    
    # Giới hạn số lượng ảnh bằng tổng số ảnh hiện có nếu num_images lớn hơn tổng
    sample_size = min(args.num_images, total_imgs)
    
    # Cố định seed để nếu bạn chạy lại lệnh này, nó vẫn ra đúng 1000 ảnh đó (dễ debug)
    random.seed(42) 
    sampled_images = random.sample(all_images, sample_size)
    
    print(f"🎲 Đã chọn ngẫu nhiên {sample_size} ảnh để kiểm tra.")
    print(f"📂 Ảnh kết quả sẽ được lưu tại: {args.output_dir}/")

    # 3. Vòng lặp vẽ và lưu ảnh
    empty_cache_count = 0
    missing_cache_count = 0
    
    for img_name in tqdm(sampled_images, desc="Đang vẽ Bounding Boxes"):
        img_path = os.path.join(args.img_dir, img_name)
        npy_name = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
        npy_path = os.path.join(args.npy_dir, npy_name)

        # Kiểm tra file cache có tồn tại không
        if not os.path.exists(npy_path):
            missing_cache_count += 1
            continue

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            continue
        h_orig, w_orig = img.shape[:2]

        # Đọc file numpy
        boxes_with_info = np.load(npy_path)
        
        if len(boxes_with_info) == 0:
            empty_cache_count += 1
            # Nếu không có vật thể, lưu ảnh gốc luôn với dòng cảnh báo trên góc
            cv2.putText(img, "YOLO: No Objects Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(args.output_dir, img_name), img)
            continue
        
        # Vẽ từng hộp
        for row in boxes_with_info:
            if len(row) >= 6:
                cx, cy, w, h, conf, cls_id = row[:6]
            else:
                cx, cy, w, h = row[:4]
                conf, cls_id = 1.0, -1
                
            cls_id = int(cls_id)
            x1, y1, x2, y2 = cxcywh_to_xyxy([cx, cy, w, h], w_orig, h_orig)
            
            # Phân màu: Người (Class 0) = Xanh dương, Vật = Đỏ
            if cls_id == 0:
                color = (255, 0, 0) # Xanh dương
                label = f"Person ({conf:.2f})"
            else:
                color = (0, 0, 255) # Đỏ
                label = f"Obj_{cls_id} ({conf:.2f})"
                
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ Text nền đen
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y_text = max(y1, text_height + 5)
            cv2.rectangle(img, (x1, y_text - text_height - 5), (x1 + text_width, y_text + baseline - 5), (0, 0, 0), cv2.FILLED)
            cv2.putText(img, label, (x1, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Lưu ảnh kết quả (Vẫn giữ nguyên tên ảnh để dễ đối chiếu)
        output_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(output_path, img)

    # 4. Báo cáo tổng kết
    print("\n" + "="*60)
    print("📊 BÁO CÁO TỔNG KẾT")
    print(f"✅ Đã xử lý thành công: {sample_size - missing_cache_count} ảnh.")
    print(f"⚠️ Ảnh không tìm thấy file Cache (Missing): {missing_cache_count} ảnh.")
    print(f"👻 Ảnh có Cache nhưng trống (YOLO mù): {empty_cache_count} ảnh.")
    print(f"👉 Vui lòng mở thư mục '{args.output_dir}' để kiểm tra bằng mắt thường.")
    print("="*60)

if __name__ == "__main__":
    main()