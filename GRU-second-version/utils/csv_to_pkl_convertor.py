import pandas as pd
import pickle
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

def str_to_arr(s):
    if pd.isna(s) or s == "" or s == "[]":
        return np.array([], dtype=np.float32)
    try:
        # Xử lý cả trường hợp phân tách bằng dấu phẩy hoặc khoảng trắng
        s_clean = s.strip('[]').replace(',', ' ')
        return np.fromstring(s_clean, sep=' ', dtype=np.float32)
    except:
        return np.array([], dtype=np.float32)
def pick_col(df, candidates):
                    for c in candidates:
                        if c in df.columns:
                            return c
                    return None

def process_directories_to_pickle(parent_dir, output_path):
    parent_path = Path(parent_dir)
    # Tìm tất cả các file .csv nằm trong các thư mục con danger{n}
    csv_files = list(parent_path.glob("danger*/merged_*.csv"))
    
    data_factory = {}
    
    print(f"Tìm thấy {len(csv_files)} video để xử lý...")
    if len(csv_files) == 0:
        print("No CSV files matched pattern 'danger*/merged_*.csv' under parent_dir.")
        print(f"Checked parent directory: {parent_path.resolve()}")
    else:
        print("Found CSV files (sample):")
        for i, p in enumerate(csv_files[:10]):
            try:
                hdr = pd.read_csv(p, nrows=3)
                print(f"  {i+1}. {p} - rows_preview={len(hdr)} columns={list(hdr.columns)}")
            except Exception as e:
                print(f"  {i+1}. {p} - Failed to read preview: {e}")

    for csv_file in tqdm(csv_files, desc="Processing Videos"):
        # Lấy ID video từ tên thư mục (ví dụ: 'danger1' -> '1')
        # Hoặc dùng nguyên tên thư mục 'danger1' làm ID
        video_id = csv_file.parent.name 
        
        # Đọc CSV
        df = pd.read_csv(csv_file)
        
        # Sắp xếp theo cặp và thời gian để đảm bảo tính tuần tự cho GRU
        df = df.sort_values(by=['pair_id', 'timestamp'])
        
        data_factory[video_id] = {}
        
        # Nhóm theo từng cặp Trẻ - Vật trong video đó
        group_col = 'pair_id' if 'pair_id' in df.columns else ( 'object_id' if 'object_id' in df.columns else None)
        groups = df.groupby(group_col) if group_col is not None else [(None, df)]
        for pid, pair_df in groups:
            # 1. Trích xuất Numeric (Distance, Velocity)
            # support both 'velocity' and 'relative_velocity'
            '''
            vel_col = 'velocity' if 'velocity' in pair_df.columns else ('relative_velocity' if 'relative_velocity' in pair_df.columns else None)
            num_cols = ['distance'] + ([vel_col] if vel_col is not None else [])
            numeric_seq = pair_df[[c for c in num_cols if c in pair_df.columns]].values.astype(np.float32)
            '''
            vel_col = pick_col(pair_df, ['velocity', 'relative_velocity'])
            dist_col = 'distance'

            # Tạo mảng 2 cột mặc định (Distance, Velocity)
            numeric_seq = np.zeros((len(pair_df), 2), dtype=np.float32)
            if dist_col in pair_df.columns:
                numeric_seq[:, 0] = pair_df[dist_col].values
            if vel_col in pair_df.columns:
                numeric_seq[:, 1] = pair_df[vel_col].values
            # 2. Trích xuất Semantic Embeddings (Chuyển từ string sang Matrix)
            try:
                # Try multiple possible column names for embeddings
                
                obj_col = pick_col(pair_df, ['obj_embedding', 'object_embedding', 'object_emb', 'object_vector'])
                int_col = pick_col(pair_df, ['int_embedding', 'interaction_clip_embedding', 'interaction_embedding', 'embedding'])
                img_col = pick_col(pair_df, ['img_embedding', 'image_embedding', 'img_emb'])

                # Helper to build embedding matrix with fallback zeros when column missing
                DEFAULT_EMB_DIM = 512
                def build_emb_matrix(series, colname, n_rows, default_dim=DEFAULT_EMB_DIM):
                    if colname is None:
                        return np.zeros((n_rows, default_dim), dtype=np.float32)
                    arrs = [str_to_arr(x) for x in series]
                    # determine embedding dim from most common non-zero length
                    lengths = [len(a) for a in arrs if len(a) > 0]
                    if len(lengths) > 0:
                        # choose the most common length
                        emb_dim = max(set(lengths), key=lengths.count)
                    else:
                        emb_dim = default_dim
                    mat = np.zeros((n_rows, emb_dim), dtype=np.float32)
                    for i, a in enumerate(arrs):
                        if len(a) == emb_dim:
                            mat[i] = a
                        elif len(a) > 0:
                            # truncate or pad
                            mat[i, :min(len(a), emb_dim)] = a[:emb_dim]
                        else:
                            # leave zeros
                            pass
                    return mat

                n_rows = len(pair_df)
                obj_embs = build_emb_matrix(pair_df[obj_col] if obj_col is not None else None, obj_col, n_rows)
                int_embs = build_emb_matrix(pair_df[int_col] if int_col is not None else None, int_col, n_rows)
                img_embs = build_emb_matrix(pair_df[img_col] if img_col is not None else None, img_col, n_rows)
                
                # 3. Lấy nhãn (Dựa trên khung hình cuối hoặc giá trị trung bình)
                # prefer common label column names
                label_col = pick_col(pair_df, ['label', 'danger_label', 'status_label'])
                if label_col is None:
                    raise ValueError('Missing label column')
                # Dựa trên biểu đồ 06b, nhãn cuối chuỗi là quan trọng nhất
                label_series = pair_df[label_col].dropna()

                if len(label_series) == 0:
                    continue  # DROP WHOLE PAIR

                label = int(label_series.iloc[-1])

                # Build per-frame dict list for compatibility with RobustVideoDataset
                frames_list = []
                for i in range(n_rows):
                    # label per frame if available else use overall label
                    raw_label = pair_df[label_col].iloc[i]

                    if pd.isna(raw_label):
                        continue   # DROP FRAME

                    frame_label = int(raw_label)

                    frames_list.append({
                        'numeric': numeric_seq[i].tolist() if numeric_seq.size else np.zeros(0).tolist(),
                        'obj_emb': obj_embs[i].astype(np.float32),
                        'int_emb': int_embs[i].astype(np.float32),
                        'img_emb': img_embs[i].astype(np.float32),
                        'label': frame_label
                    })

                data_factory[video_id][pid] = frames_list
            except Exception as e:
                print(f"Lỗi tại Video {video_id}, Cặp {pid}: {e}")
                continue

    # Lưu toàn bộ vào file .pkl
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(data_factory, f)
    
    print(f"\nThành công! Đã lưu {len(data_factory)} videos vào {output_path}")

output_file = "dataset/final_dataset.pkl"
if __name__ == "__main__":
    # Thay đổi đường dẫn thư mục cha và tên file đầu ra theo nhu cầu
    time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Use workspace-relative fine_data directory by default
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    parent_directory = os.path.join(repo_root, 'fine_data')
    output_file = f"dataset/train_valid_{time}.pkl"  # Tên file pickle đầu ra
    
    process_directories_to_pickle(parent_directory, output_file)
# Cách sử dụng:
# process_directories_to_pickle('duong/dan/den/thu/muc/cha', 'dataset_final.pkl')

import pickle
with open(output_file, 'rb') as f:
    data = pickle.load(f)

# Kiểm tra tỉ lệ nhãn
total_positive = 0
total_samples = 0
for vid in data:
    for pid in data[vid]:
        labels = [f['label'] for f in data[vid][pid]]
        if any(l == 1 for l in labels):
            total_positive += 1
        total_samples += 1

print(f"Tổng số cặp: {total_samples}")
print(f"Số cặp có chứa nhãn nguy hiểm (1): {total_positive}")
print(f"Tỉ lệ: {total_positive/total_samples:.2%}")