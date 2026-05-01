import streamlit as st
import pandas as pd
import cv2
import os
import ast

st.set_page_config(layout="wide", page_title="Review Tool: Dataset for GRU")

st.title("🛡️ Công cụ Kiểm duyệt Dữ liệu GRU (Auto-Forward & Color Sync)")
st.markdown("""
- **Phân loại màu BBox tự động:** `person` mặc định (Xanh dương) | `child` (Xanh lá) | `adult` (Đỏ). Hộp BBox trên ảnh sẽ **lập tức đổi màu** khi bạn chọn Vai trò ở bảng bên phải.
- **Sửa BBox & ID:** Sửa trực tiếp tọa độ Bbox hoặc ID Người ngay trên bảng.
- **Tự động lan truyền:** Những thay đổi của bạn sẽ tự động ghi đè lên các Frame phía sau (nếu bật công tắc).
""")

# --- 1. SETUP THƯ MỤC ---
with st.sidebar:
    st.header("📁 Cấu hình Dữ liệu")
    # Thay đổi đường dẫn mặc định này cho khớp với máy của bạn B
    csv_path = st.text_input("Đường dẫn file CSV:", value="/home/user/Processed_Data_Mar_20/danger1/danger1.csv")
    img_dir = st.text_input("Đường dẫn thư mục chứa Frames:", value="/home/user/Processed_Data_Mar_20/danger1/frames")
    
    if st.button("Tải Dữ Liệu"):
        if os.path.exists(csv_path) and os.path.exists(img_dir):
            df = pd.read_csv(csv_path)
            if 'Human_Label' not in df.columns:
                df['Human_Label'] = 'person'
            st.session_state['df'] = df
            st.session_state['csv_path'] = csv_path
            st.session_state['img_dir'] = img_dir
            st.success("Tải dữ liệu thành công!")
        else:
            st.error("Không tìm thấy file CSV hoặc thư mục ảnh!")

# --- 2. MAIN WORKSPACE ---
if 'df' in st.session_state:
    df = st.session_state['df']
    frame_list = sorted(df['Frame_id'].unique())
    
    if len(frame_list) == 0:
        st.warning("File CSV hiện không có dữ liệu.")
        st.stop()
    
    # Điều hướng
    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    current_idx = st.session_state.get('curr_idx', 0)
    
    with col_nav1:
        if st.button("⬅️ Trở về") and current_idx > 0:
            st.session_state['curr_idx'] -= 1
            st.rerun()
    with col_nav2:
        selected_frame = st.selectbox("Chọn Frame ID", frame_list, index=current_idx)
        st.session_state['curr_idx'] = frame_list.index(selected_frame)
    with col_nav3:
        if st.button("Tiếp theo ➡️") and current_idx < len(frame_list) - 1:
            st.session_state['curr_idx'] += 1
            st.rerun()

    current_frame_id = frame_list[st.session_state['curr_idx']]
    frame_df = df[df['Frame_id'] == current_frame_id].copy()
    frame_df['_original_index'] = frame_df.index 
    
    # Chia cột hiển thị
    col_img, col_data = st.columns([1.2, 1])
    
    # ==============================================================
    # BƯỚC 1: HIỂN THỊ BẢNG DATA EDITOR (LẤY DỮ LIỆU REAL-TIME)
    # ==============================================================
    with col_data:
        auto_propagate = st.checkbox("🔄 Tự động áp dụng thay đổi cho TẤT CẢ các Frame phía sau", value=True)
        
        display_df = frame_df[['_original_index', 'Human_ID', 'Human_Label', 'Object_ID', 'Interaction', 'Label', 'Bbox_Human', 'Bbox_Object']]
        
        # Biến edited_df sẽ lưu trữ mọi thao tác nhấp chuột/gõ phím ngay lập tức
        edited_df = st.data_editor(
            display_df,
            column_config={
                "_original_index": None, # Ẩn đi
                # ĐÃ SỬA Ở ĐÂY: disabled=False để cho phép sửa ID Người
                "Human_ID": st.column_config.TextColumn("ID Người", disabled=False),
                "Human_Label": st.column_config.SelectboxColumn("Vai trò", options=["person", "child", "adult"], required=True),
                "Object_ID": st.column_config.TextColumn("ID Vật Thể", disabled=True),
                "Interaction": st.column_config.TextColumn("Interaction"),
                "Label": st.column_config.CheckboxColumn("Nguy hiểm (Label=1)", default=False),
                "Bbox_Human": st.column_config.TextColumn("BBox Người"),
                "Bbox_Object": st.column_config.TextColumn("BBox Vật")
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            key=f"editor_{current_frame_id}"
        )
        
        # Xử lý Logic Lưu file CSV
        if st.button("💾 LƯU LẠI (SAVE)", type="primary", use_container_width=True):
            surviving_indices = edited_df['_original_index'].dropna().tolist()
            original_indices = frame_df['_original_index'].tolist()
            deleted_indices = [idx for idx in original_indices if idx not in surviving_indices]
            
            if deleted_indices:
                df = df.drop(index=deleted_indices)
            
            for _, row in edited_df.iterrows():
                orig_idx = row['_original_index']
                if pd.isna(orig_idx): continue 
                
                # ĐÃ SỬA Ở ĐÂY: Lấy ID cũ để làm mốc tìm kiếm cho auto_propagate
                old_h_id = frame_df.loc[orig_idx, 'Human_ID']
                o_id = row['Object_ID']
                
                # Các giá trị MỚI từ bảng
                new_h_id = row['Human_ID'] # Lấy ID mới nếu user có sửa
                new_label_role = row['Human_Label']
                new_interaction = row['Interaction']
                new_danger_label = 1 if row['Label'] else 0
                new_h_box, new_o_box = row['Bbox_Human'], row['Bbox_Object']
                
                # Lưu vào df cho frame hiện tại
                df.at[orig_idx, 'Human_ID'] = new_h_id
                df.at[orig_idx, 'Human_Label'] = new_label_role
                df.at[orig_idx, 'Interaction'] = new_interaction
                df.at[orig_idx, 'Label'] = new_danger_label
                df.at[orig_idx, 'Bbox_Human'] = new_h_box
                df.at[orig_idx, 'Bbox_Object'] = new_o_box
                
                if auto_propagate:
                    # Tìm kiếm frame tương lai dựa trên ID CŨ
                    future_mask = (df['Frame_id'] > current_frame_id) & (df['Human_ID'] == old_h_id) & (df['Object_ID'] == o_id)
                    if future_mask.any():
                        df.loc[future_mask, 'Human_ID'] = new_h_id # Cập nhật sang ID mới
                        df.loc[future_mask, 'Human_Label'] = new_label_role
                        df.loc[future_mask, 'Interaction'] = new_interaction
                        df.loc[future_mask, 'Label'] = new_danger_label
                        df.loc[future_mask, 'Bbox_Human'] = new_h_box
                        df.loc[future_mask, 'Bbox_Object'] = new_o_box
                
            df = df.reset_index(drop=True)
            st.session_state['df'] = df
            df.to_csv(st.session_state['csv_path'], index=False)
            
            st.toast("✅ Đã lưu thay đổi!", icon="💾")
            st.rerun()

    # ==============================================================
    # BƯỚC 2: VẼ ẢNH DỰA THEO BẢNG ĐANG ĐƯỢC CHỈNH SỬA (REAL-TIME)
    # ==============================================================
    with col_img:
        frame_filename = f"frame_{int(current_frame_id):06d}.jpg"
        img_path = os.path.join(st.session_state['img_dir'], frame_filename)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            
            # Lặp qua dữ liệu để vẽ BBox và Text
            for _, row in edited_df.iterrows():
                if pd.isna(row['_original_index']) or row['Bbox_Human'] == "[]":
                    continue
                    
                try:
                    h_box = ast.literal_eval(row['Bbox_Human'])
                    
                    # QUYẾT ĐỊNH MÀU SẮC THEO CLASS CHỌN Ở BẢNG (RGB)
                    human_label = str(row.get('Human_Label', 'person')).strip().lower()
                    if human_label == 'child':
                        h_color = (0, 255, 0)      # Xanh lá
                    elif human_label == 'adult':
                        h_color = (255, 0, 0)      # Đỏ
                    else:
                        h_color = (0, 100, 255)    # Xanh dương (Mặc định Person)
                        
                    # Vẽ BBox Người
                    cv2.rectangle(img, (h_box[0], h_box[1]), (h_box[2], h_box[3]), h_color, 2)
                    cv2.putText(img, f"{human_label.upper()} {row['Human_ID']}", (h_box[0], max(15, h_box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, h_color, 2)
                    
                    # Vẽ BBox Vật (nếu có)
                    if pd.notna(row['Bbox_Object']) and str(row['Bbox_Object']).lower() not in ["[]", "none"]:
                        o_box = ast.literal_eval(row['Bbox_Object'])
                        o_color = (255, 140, 0) # Cam cho vật thể
                        cv2.rectangle(img, (o_box[0], o_box[1]), (o_box[2], o_box[3]), o_color, 2)
                        cv2.putText(img, f"Obj: {row['Object_ID']}", (o_box[0], max(15, o_box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, o_color, 2)
                        
                except Exception as e:
                    pass
                    
            st.image(img, use_container_width=True, caption=f"Frame {current_frame_id} (Ảnh cập nhật thời gian thực theo bảng)")
        else:
            st.error(f"Không tìm thấy file ảnh: {img_path}")