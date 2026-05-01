import pandas as pd
import os
import glob
import ast

def calculate_area(bbox_str):
    """
    Hàm tính diện tích của Bounding Box từ chuỗi string.
    Ví dụ: "[740, 252, 851, 388]" -> Diện tích = (851-740) * (388-252)
    """
    try:
        if bbox_str == '[]' or pd.isna(bbox_str):
            return 0
        bbox = ast.literal_eval(bbox_str)
        if len(bbox) == 4:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            return width * height
        return 0
    except Exception:
        return 0

def clean_csv_bboxes(input_file, output_file):
    """
    Đọc CSV, lọc Bbox tốt nhất cho mỗi (Frame_id, Human_ID) và lưu lại.
    """
    print(f"⏳ Đang xử lý: {os.path.basename(input_file)}...")
    df = pd.read_csv(input_file)
    
    # Gom nhóm theo Frame và ID Người
    grouped = df.groupby(['Frame_id', 'Human_ID'])
    best_bboxes = {}
    
    # BƯỚC 1: Tìm Bbox tốt nhất cho mỗi người trong từng frame
    for (frame_id, human_id), group in grouped:
        # Bỏ qua nếu không có ID
        if pd.isna(human_id) or str(human_id).lower() == 'none':
            continue
            
        unique_bboxes = group['Bbox_Human'].unique()
        
        # Lọc bỏ các box rỗng
        valid_bboxes = [b for b in unique_bboxes if str(b) != '[]' and not pd.isna(b)]
        
        if len(valid_bboxes) == 0:
            best_bboxes[(frame_id, human_id)] = '[]'
        elif len(valid_bboxes) == 1:
            # Chỉ có 1 box -> Chuẩn rồi, không bị nhiễu
            best_bboxes[(frame_id, human_id)] = valid_bboxes[0]
        else:
            # CÓ NHIỄU: Nhiều box khác nhau cho cùng 1 ID trong 1 Frame
            # -> Chọn box có diện tích lớn nhất (toàn diện nhất)
            best_bbox = max(valid_bboxes, key=calculate_area)
            best_bboxes[(frame_id, human_id)] = best_bbox
            
    # BƯỚC 2: Cập nhật lại Bbox chuẩn cho toàn bộ DataFrame
    def apply_best_bbox(row):
        frame_id = row['Frame_id']
        human_id = row['Human_ID']
        # Nếu có trong từ điển chuẩn hóa thì dùng box đã chuẩn hóa
        if (frame_id, human_id) in best_bboxes:
            return best_bboxes[(frame_id, human_id)]
        return row['Bbox_Human']

    df['Bbox_Human'] = df.apply(apply_best_bbox, axis=1)
    
    # Lưu ra file mới, giữ nguyên 100% cấu trúc cột
    df.to_csv(output_file, index=False)
    print(f"✅ Đã lưu xong: {os.path.basename(output_file)}\n")

def main():
    # ==========================================
    # CẤU HÌNH THƯ MỤC CỦA BẠN Ở ĐÂY
    # ==========================================
    input_folder = "/home/nguyenletuan/Downloads/NCKH/Processed_Data_Mar_20"   # Thư mục chứa các file CSV bị nhiễu
    output_folder = "/home/nguyenletuan/Downloads/NCKH/Processed_Data_with_startingFRAME/cleaned" # Thư mục sẽ lưu các file CSV đã làm sạch
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    # Tìm tất cả file CSV trong thư mục input
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"❌ Không tìm thấy file .csv nào trong thư mục: {input_folder}")
        return
        
    print(f"🔍 Tìm thấy {len(csv_files)} file CSV. Bắt đầu dọn dẹp nhiễu BBox...\n")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, file_name)
        clean_csv_bboxes(file_path, output_path)
        
    print("🎉 HOÀN THÀNH TOÀN BỘ!")

if __name__ == "__main__":
    main()