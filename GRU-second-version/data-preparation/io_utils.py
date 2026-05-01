import os, json, glob

def read_keyframe_jsons(input_dir, pattern='anno_f*.json'):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    frames = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            frames.append((f, json.load(fh)))
    return frames

def sort_frames_by_time(frames):
    return sorted(frames, key=lambda x: (x[1].get('frame_id', 0), x[1].get('timestamp', 0.0)))


def write_json(obj, path):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def write_csv(df, path):
    df.to_csv(path, index=False)