'''
python data-preparation/merge_pipeline.py \
    --input_dir ./raw-data/danger10 \
    --video_id danger10 \
    --out_dir fine_data/danger10
'''
# ========================= merge_pipeline.py =========================
import os
import argparse
from collections import OrderedDict

from io_utils import read_keyframe_jsons, sort_frames_by_time, write_json, write_csv
from tracking import detect_duplicates, build_tracks
from filling import fill_bbox_track
from distance import compute_distances, compute_relative_velocities
from clip_encoding import compute_interaction_clip_embeddings, encode_text
from export_csv import frames_to_df
from plotting import (
    plot_all_distances_combined,
    plot_label_timeline,
    plot_interaction_timeline,
    plot_trajectories_2d,
    plot_relative_velocities
)
from report import write_report


# ---------------- helpers ----------------
def nearest_frame(target, frames):
    return min(frames, key=lambda x: abs(x - target))


# ---------------- pipeline ----------------
def run_pipeline(input_dir, video_id, out_dir, pattern='anno_f*.json'):
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    print('[1] Read keyframe json files')
    frames = read_keyframe_jsons(input_dir, pattern)
    frames = sort_frames_by_time(frames)
    fps = frames[0][1].get('fps', 30)

    print('[2] Detect duplicate frame_id')
    duplicates = detect_duplicates(frames)

    print('[3] Build sparse tracks')
    children_tracks, object_tracks_raw, timestamps = build_tracks(frames)
    frame_ids = sorted(timestamps.keys())
    frame_range = (frame_ids[0], frame_ids[-1])

    print('[4] Fill missing frames (children)')
    children_filled = {
        cid: fill_bbox_track(track, frame_range)
        for cid, track in children_tracks.items()
    }

    print('[5] Fill missing frames (objects)')
    object_bbox_tracks = {
        oid: {fid: v['bbox'] for fid, v in fmap.items()}
        for oid, fmap in object_tracks_raw.items()
    }
    object_meta = object_tracks_raw

    objects_filled = {
        oid: fill_bbox_track(track, frame_range)
        for oid, track in object_bbox_tracks.items()
    }

    print('[6] Merge per-frame data')
    merged_frames = OrderedDict()
    for fid in range(frame_range[0], frame_range[1] + 1):
        frame = {
            'frame_id': fid,
            'timestamp': timestamps.get(fid, fid / fps),
            'children': [],
            'objects': []
        }

        for cid, track in children_filled.items():
            frame['children'].append({'id': cid, 'bbox': track[fid]})

        for oid, track in objects_filled.items():
            bbox = track.get(fid)
            if bbox is None:
                continue

            meta_frames = sorted(object_meta.get(oid, {}).keys())
            if meta_frames:
                nf = nearest_frame(fid, meta_frames)
                meta = object_meta[oid][nf]
                label = meta.get('label')
                interaction = meta.get('interaction', {})
                embedding = meta.get('object_embedding')
                otype = meta.get('object_type')
            else:
                label = interaction = embedding = otype = None

            frame['objects'].append({
                'id': oid,
                'object_type': otype,
                'object_type_embedding': encode_text(otype) if otype else None,
                'bbox': bbox,
                'label': label,
                'interaction': interaction,
                'object_embedding': embedding,
                'distances': {}
            })

        merged_frames[fid] = frame

    print('[7] Compute distances')
    for fid, frame in merged_frames.items():
        child_map = {c['id']: c['bbox'] for c in frame['children']}
        for obj in frame['objects']:
            obj['distances'] = compute_distances(child_map, {obj['id']: obj['bbox']})[obj['id']]

    print('[7b] Compute relative velocities')
    velocities = compute_relative_velocities(merged_frames)
    for fid, frame in merged_frames.items():
        for obj in frame['objects']:
            obj['relative_velocities'] = velocities.get(fid, {}).get(obj['id'], {})

    print('[7c] Compute CLIP interaction embeddings')
    compute_interaction_clip_embeddings(merged_frames)

    print('[8] Export JSON / CSV')
    json_path = os.path.join(out_dir, f'merged_{video_id}.json')
    write_json({'video_id': video_id, 'frames': list(merged_frames.values())}, json_path)

    csv_path = os.path.join(out_dir, f'merged_{video_id}.csv')
    write_csv(frames_to_df(merged_frames.values()), csv_path)

    print('[9] Plot')
    plot_all_distances_combined(merged_frames, os.path.join(plot_dir, f'distance_{video_id}.png'))
    plot_label_timeline(merged_frames, os.path.join(plot_dir, f'label_{video_id}.png'))
    plot_interaction_timeline(merged_frames, os.path.join(plot_dir, f'interaction_{video_id}.png'))
    plot_trajectories_2d(merged_frames, os.path.join(plot_dir, f'traj_{video_id}.png'))
    plot_relative_velocities(merged_frames, os.path.join(plot_dir, f'vel_{video_id}.png'))

    print('[10] Write report')
    write_report({
        'duplicates': duplicates,
        'num_frames': len(merged_frames),
        'num_children': len(children_filled),
        'num_objects': len(objects_filled)
    }, os.path.join(out_dir, 'report.json'))

    print('[DONE]')


# ---------------- CLI ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--video_id', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--pattern', default='anno_f*.json')
    args = parser.parse_args()

    run_pipeline(args.input_dir, args.video_id, args.out_dir, args.pattern)
