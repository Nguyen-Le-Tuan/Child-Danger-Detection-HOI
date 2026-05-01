import os
import json
import numpy as np
import cv2
import re
import shutil

# --- CONFIGURATION ---
VIDEOS_DIR = ""

def get_safe_filename(s):
    """Sanitize object ID for use in filenames."""
    return "".join(x for x in s if x.isalnum() or x in "-_").strip()

def migrate_video_folder(video_id):
    video_path = os.path.join(VIDEOS_DIR, video_id)
    if not os.path.isdir(video_path):
        return

    print(f"📂 Processing video: {video_id}...")
    
    # Find all old JSON files (anno_f*.json) in the root of the video folder
    # We exclude files that are already inside subdirectories
    files = [f for f in os.listdir(video_path) 
             if f.endswith('.json') and f.startswith('anno_f') 
             and os.path.isfile(os.path.join(video_path, f))]
    
    if not files:
        print(f"   No flat JSON files found in {video_id}. Skipping.")
        return

    count_migrated = 0
    
    for json_file in files:
        old_json_path = os.path.join(video_path, json_file)
        
        try:
            with open(old_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"   ❌ [ERROR] Could not read {json_file}: {e}")
            continue

        # 1. Determine Frame ID
        frame_id = data.get('frame_id')
        if frame_id is None:
            # Fallback: extract from filename (anno_f123.json)
            match = re.search(r'anno_f(\d+).json', json_file)
            if match:
                frame_id = int(match.group(1))
            else:
                print(f"   ⚠️ [SKIP] Could not determine frame_id for {json_file}")
                continue

        # 2. Measure Image Dimensions
        # Assuming standard naming convention from extract_keyframes
        img_filename = f"frame_{frame_id:06d}.jpg"
        img_path = os.path.join(video_path, img_filename)
        
        h, w = 0, 0
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
        else:
            print(f"   ⚠️ [WARN] Image {img_filename} not found. Width/Height set to 0.")

        # Update Metadata
        data['image_width'] = w
        data['image_height'] = h

        # 3. Create New Directory Structure
        # Structure: VIDEOS_new/video_id/anno_fX/
        new_anno_dir = os.path.join(video_path, f"anno_f{frame_id}")
        embeddings_dir = os.path.join(new_anno_dir, "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)

        # 4. Extract Embeddings & Update JSON
        objects = data.get('objects', [])
        for obj in objects:
            embed = obj.get('object_embedding')
            
            # Only process if it's a list (raw embedding vector)
            if isinstance(embed, list) and len(embed) > 0:
                obj_id = obj.get('id', 'unknown')
                safe_obj_id = get_safe_filename(obj_id)
                
                # Generate .npy filename
                npy_filename = f"{video_id}_f{frame_id}_{safe_obj_id}.npy"
                npy_path = os.path.join(embeddings_dir, npy_filename)
                
                # Save numpy file
                np.save(npy_path, np.array(embed))
                
                # Update JSON to point to relative path
                obj['object_embedding'] = f"embeddings/{npy_filename}"

        # 5. Save New JSON
        new_json_path = os.path.join(new_anno_dir, f"anno_f{frame_id}.json")
        with open(new_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        count_migrated += 1
        # Optional: Rename old file to .bak so we know it's processed
        # os.rename(old_json_path, old_json_path + ".bak")

    print(f"   ✅ Migrated {count_migrated} files in {video_id}.")

if __name__ == "__main__":
    print(f"🚀 Starting migration in: {VIDEOS_DIR}")
    for folder in os.listdir(VIDEOS_DIR):
        migrate_video_folder(folder)
    print("🎉 Migration Complete!")