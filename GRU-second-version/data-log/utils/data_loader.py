"""
Data loading utilities for analysis pipeline.
"""

import os
import json
import glob
import pandas as pd
import numpy as np


def load_all_danger_data(fine_data_dir='./fine_data', pattern='danger*/merged_danger*.csv'):
    paths = sorted(glob.glob(os.path.join(fine_data_dir, pattern)))
    dfs = []

    for csv_path in paths:
        df = pd.read_csv(csv_path, low_memory=False)

        video_id = os.path.basename(os.path.dirname(csv_path))
        df['video_id'] = video_id

        if 'label' in df.columns:
            # 1. Convert to numeric safely
            df['label_numeric'] = pd.to_numeric(df['label'], errors='coerce')

            # 2. Report invalid labels (VERY IMPORTANT for research)
            invalid_cnt = df['label_numeric'].isna().sum()
            if invalid_cnt > 0:
                print(
                    f'[Warning] {video_id}: {invalid_cnt} rows have invalid label'
                )

            # 3. Decide policy: drop rows without ground-truth label
            df = df[df['label_numeric'].notna()]

            # 4. Cast safely
            df['danger_label'] = df['label_numeric'].astype(int)

        else:
            # Fallback: derive from interaction text
            danger_interactions = {'crawling', 'climbing', 'falling', 'dangerous', 'drowning'}
            if 'interaction' in df.columns:
                df['danger_label'] = df['interaction'].apply(
                    lambda x: 1 if isinstance(x, str) and x.lower() in danger_interactions else 0
                )
            else:
                df['danger_label'] = 0

        dfs.append(df)

    if not dfs:
        print(f'[!] No CSV files found in {fine_data_dir}/{pattern}')
        return None

    return pd.concat(dfs, ignore_index=True)



def load_clip_embeddings(fine_data_dir='./fine_data', pattern='danger*/merged_danger*.json'):
    """Load CLIP embeddings from JSON files."""
    paths = sorted(glob.glob(os.path.join(fine_data_dir, pattern)))
    embeddings_data = []
    
    for json_path in paths:
        video_id = os.path.basename(os.path.dirname(json_path))
        
        with open(json_path) as f:
            data = json.load(f)
            
        for frame in data.get('frames', []):
            frame_id = frame.get('frame_id')
            timestamp = frame.get('timestamp')
            
            for obj in frame.get('objects', []):
                obj_id = obj.get('id')
                interaction = obj.get('interaction', {})
                
                for person_id, inter_data in interaction.items():
                    # Extract interaction label and embedding
                    if isinstance(inter_data, list) and len(inter_data) == 2:
                        inter_label = inter_data[0]
                        embedding = inter_data[1]
                        
                        danger_interactions = {'crawling', 'climbing', 'falling', 'dangerous'}
                        danger_label = 1 if inter_label.lower() in danger_interactions else 0
                        
                        embeddings_data.append({
                            'video_id': video_id,
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'object_id': obj_id,
                            'person_id': person_id,
                            'interaction': inter_label,
                            'danger_label': danger_label,
                            'embedding': np.array(embedding)
                        })
    
    if not embeddings_data:
        print('[!] No embedding data found')
        return None
    
    return pd.DataFrame(embeddings_data)
