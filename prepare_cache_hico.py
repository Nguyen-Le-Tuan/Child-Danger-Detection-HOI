import os
import torch
import numpy as np
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Sinh YOLO Cache siêu tốc độ bằng Manual Batching")
    parser.add_argument("--img_dir", type=str, required=True, help="Đường dẫn đến thư mục chứa ảnh HICO-DET")
    parser.add_argument("--cache_dir", type=str, required=True, help="Đường dẫn thư mục lưu file .npy output")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="Phiên bản YOLO")
    parser.add_argument("--batch_size", type=int, default=32, help="Số lượng ảnh đút vào GPU mỗi lần")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Ngưỡng tự tin lọc rác")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 SAFEGUARD AI - HICO-DET YOLO CACHE GENERATOR (FIXED)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Thiết bị chạy: {device.upper()}")

    os.makedirs(args.cache_dir, exist_ok=True)
    if not os.path.exists(args.img_dir):
        raise FileNotFoundError(f"❌ Không tìm thấy thư mục ảnh tại: {args.img_dir}")

    print(f"⏳ Đang tải mô hình {args.model}...")
    model = YOLO(args.model)

    # Lấy danh sách ảnh
    image_names = [f for f in os.listdir(args.img_dir) if f.endswith(('.jpg', '.png'))]
    image_paths = [os.path.join(args.img_dir, f) for f in image_names]
    total_imgs = len(image_paths)
    
    print(f"📦 Tìm thấy {total_imgs} ảnh.")
    print(f"⚙️  Cấu hình: Batch Size = {args.batch_size} | Confidence = {args.conf_thres}")
    print("⚡ Bắt đầu băm nhỏ và quét GPU...")

    # =================================================================
    # CHIẾN THUẬT MỚI: TỰ CHIA BATCH THỦ CÔNG (MANUAL BATCHING)
    # Bỏ qua hàm .predict(stream=True) bị lỗi của YOLO
    # =================================================================
    
    # tqdm hiển thị theo số lượng Batch thay vì số lượng ảnh lẻ
    total_batches = (total_imgs + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(0, total_imgs, args.batch_size), total=total_batches, desc="Đang chạy Batches"):
        # Lấy ra 1 cục 32 ảnh
        batch_paths = image_paths[i : i + args.batch_size]
        batch_names = image_names[i : i + args.batch_size]
        
        # Đưa thẳng list 32 đường dẫn vào model (Gọi __call__ trực tiếp)
        results = model(batch_paths, verbose=False, device=device, conf=args.conf_thres)
        
        # Bóc tách kết quả của 32 ảnh này và lưu
        for j, result in enumerate(results):
            img_name = batch_names[j]
            
            boxes_xywhn = result.boxes.xywhn.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy().reshape(-1, 1)
            classes = result.boxes.cls.cpu().numpy().reshape(-1, 1)
            
            if len(boxes_xywhn) > 0:
                boxes_with_info = np.hstack((boxes_xywhn, confs, classes))
            else:
                boxes_with_info = np.zeros((0, 6))
                
            cache_filename = img_name.replace('.jpg', '.npy').replace('.png', '.npy')
            cache_path = os.path.join(args.cache_dir, cache_filename)
            np.save(cache_path, boxes_with_info)

    print(f"\n✅ Hoàn tất! Cache đã được lưu tại: {args.cache_dir}")

if __name__ == "__main__":
    main()