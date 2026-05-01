import pandas as pd
import json


def frames_to_df(frames, include_children=False):
    """Convert merged frames into DataFrame with one row per (pair_id, frame_id).
    
    Structure:
    - Each unique (person_id, object_id) pair gets a unique pair_id
    - Each row represents one frame for one specific pair
    - Columns: pair_id, person_id, object_id, frame_id, timestamp, 
              distance, relative_velocity, label, bbox, interaction, embedding
    
    Args:
        frames: iterable of frame dicts
        include_children: unused (kept for compatibility)
    
    Returns:
        pd.DataFrame sorted by (pair_id, frame_id)
    """
    rows = []
    pair_id_map = {}  # (person_id, object_id) -> pair_id
    next_pair_id = 1
    
    # First pass: collect all unique pairs and assign pair_ids
    for f in frames:
        for obj in f.get('objects', []):
            oid = obj.get('id')
            distances = obj.get('distances', {})
            
            # Handle distances as dict {person_id: distance}
            if isinstance(distances, dict):
                for pid in distances.keys():
                    key = (pid, oid)
                    if key not in pair_id_map:
                        pair_id_map[key] = next_pair_id
                        next_pair_id += 1
    
    # Second pass: create rows
    for f in frames:
        fid = f.get('frame_id')
        ts = f.get('timestamp')
        
        for obj in f.get('objects', []):
            oid = obj.get('id')
            
            bbox = obj.get('bbox')
            if not bbox:
                continue
            
            otype = obj.get('object_type')  # Get object_type
            label = obj.get('label')  # Can be scalar or dict
            distances = obj.get('distances', {})
            relative_velocities = obj.get('relative_velocities', {})
            interaction = obj.get('interaction', {})
            embedding = obj.get('object_embedding')  # Get the actual embedding from object
            
            # Handle distances as dict {person_id: distance}
            if isinstance(distances, dict):
                for pid, dist in distances.items():
                    key = (pid, oid)
                    pair_id = pair_id_map.get(key, 0)
                    
                    # Extract person-specific values
                    person_label = label.get(pid) if isinstance(label, dict) else label
                    person_rel_vel = relative_velocities.get(pid) if isinstance(relative_velocities, dict) else relative_velocities
                    person_interaction = interaction.get(pid) if isinstance(interaction, dict) else interaction
                    
                    # Handle interaction with CLIP embedding: [label, embedding] or just label
                    interaction_text = None
                    interaction_clip_emb = None
                    if isinstance(person_interaction, list) and len(person_interaction) == 2:
                        interaction_text = person_interaction[0]
                        interaction_clip_emb = person_interaction[1]
                    elif isinstance(person_interaction, str):
                        interaction_text = person_interaction
                    
                    rows.append({
                        'pair_id': pair_id,
                        'person_id': pid,
                        'object_id': oid,
                        'object_type': otype,
                        'object_type_embedding': json.dumps(obj.get('object_type_embedding'), ensure_ascii=False) if obj.get('object_type_embedding') is not None else None,
                        'frame_id': fid,
                        'timestamp': ts,
                        'distance': dist if dist is not None else None,
                        'relative_velocity': person_rel_vel if isinstance(person_rel_vel, (int, float, type(None))) else person_rel_vel,
                        'label': person_label if person_label is not None else None,
                        'bbox': json.dumps(bbox, ensure_ascii=False),
                        'interaction': interaction_text,
                        'interaction_clip_embedding': json.dumps(interaction_clip_emb, ensure_ascii=False) if interaction_clip_emb is not None else None,
                        'embedding': json.dumps(embedding, ensure_ascii=False) if embedding is not None else None
                    })
            else:
                # Scalar distances (backward compatibility)
                for pid in [f.get('children', [{}])[0].get('id') for _ in [None]]:  # fallback
                    key = (pid, oid)
                    pair_id = pair_id_map.get(key, 0)
                    
                    rows.append({
                        'pair_id': pair_id,
                        'person_id': pid,
                        'object_id': oid,
                        'object_type': otype,
                        'object_type_embedding': json.dumps(obj.get('object_type_embedding'), ensure_ascii=False) if obj.get('object_type_embedding') is not None else None,
                        'frame_id': fid,
                        'timestamp': ts,
                        'distance': distances,
                        'relative_velocity': relative_velocities,
                        'label': label,
                        'bbox': json.dumps(bbox, ensure_ascii=False),
                        'interaction': interaction if interaction is not None else None,
                        'embedding': json.dumps(embedding, ensure_ascii=False) if embedding is not None else None
                    })
    
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    
    # Sort by pair_id, then frame_id for grouped analysis
    df = df.sort_values(
        by=['pair_id', 'frame_id'],
        na_position='last'
    ).reset_index(drop=True)
    
    return df