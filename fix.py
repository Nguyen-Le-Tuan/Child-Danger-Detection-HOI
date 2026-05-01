import os
import json
import re

def clean_extract(text):
    """
    Hàm làm sạch mạnh tay:
    1. Xóa hết ký tự đặc biệt (ngoặc, nháy, escape...), chỉ giữ lại chữ, số, gạch ngang, gạch dưới.
    2. Trả về chuỗi sạch.
    Ví dụ: "\"'no-interaction'}\"" -> "no-interaction"
    """
    if not isinstance(text, str):
        return str(text)
    # Thay thế mọi ký tự KHÔNG phải là a-z, A-Z, 0-9, -, _ bằng dấu cách
    return re.sub(r"[^a-zA-Z0-9\-_]", " ", text).strip()

def recursive_rebuild(data, stats):
    """
    Hàm này duyệt và XÂY DỰNG LẠI dictionary.
    Khác với các bản trước, bản này có khả năng ĐỔI KEY.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            # KIỂM TRA: Nếu KEY chứa "person_" và có vẻ bị lỗi (chứa ngoặc nhọn hoặc nháy)
            # Ví dụ k = "{\"{'person_1'\""
            if isinstance(k, str) and "person_" in k and ("{" in k or '"' in k or "'" in k):
                
                # 1. Trích xuất ID từ Key lỗi (Lấy 'person_1')
                clean_k_text = clean_extract(k)
                # Tìm chữ person_X trong đống hỗn độn của key
                key_match = re.search(r"(person_\d+)", clean_k_text)
                
                # 2. Trích xuất Hành động từ Value lỗi (Lấy 'no-interaction')
                # Value lúc này có thể là "\"'no-interaction'}\""
                clean_v_text = clean_extract(str(v))
                # Tìm từ khóa hành động (các chữ cái nối nhau bằng gạch ngang)
                val_match = re.search(r"([a-zA-Z0-9-]+)", clean_v_text)

                if key_match and val_match:
                    real_key = key_match.group(1)   # person_1
                    real_val = val_match.group(1)   # no-interaction
                    
                    # Lưu vào dict mới với KEY CHUẨN
                    new_dict[real_key] = real_val
                    stats['fixed_count'] += 1
                    # print(f"   [REBUILD] {k} : {v}  --->  {real_key} : {real_val}")
                else:
                    # Trường hợp xui xẻo không parse được thì giữ nguyên (để không mất dữ liệu)
                    new_dict[k] = recursive_rebuild(v, stats)
            
            else:
                # Nếu Key bình thường (ví dụ "interaction"), thì chỉ đệ quy vào Value
                # Lưu ý: Value bên trong có thể là một dict chứa key lỗi, nên phải đệ quy tiếp
                new_dict[k] = recursive_rebuild(v, stats)
                
        return new_dict
    
    elif isinstance(data, list):
        return [recursive_rebuild(item, stats) for item in data]
    
    return data

def process_folder(folder_path):
    print(f"--- Bắt đầu chế độ SỬA KEY & VALUE tại: {folder_path} ---")
    
    if not os.path.exists(folder_path):
        print(f"Lỗi: Không tìm thấy thư mục '{folder_path}'")
        return

    total_fixes = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            stats = {'fixed_count': 0}
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Xử lý tái cấu trúc data
                cleaned_data = recursive_rebuild(data, stats)
                
                if stats['fixed_count'] > 0:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
                    print(f"[OK] {filename}: Đã tái cấu trúc {stats['fixed_count']} cặp lỗi.")
                    total_fixes += stats['fixed_count']
                else:
                    print(f"[SKIP] {filename}: Chuẩn.")
                
            except Exception as e:
                print(f"[FAIL] Lỗi file {filename}: {e}")

    print(f"--- Hoàn tất! Tổng số lỗi đã sửa: {total_fixes} ---")

# --- CẤU HÌNH ---
INPUT_FOLDER = "/home/nguyenletuan/Downloads/NCKH/VIDEOS_new/danger1"

if __name__ == "__main__":
    process_folder(INPUT_FOLDER)