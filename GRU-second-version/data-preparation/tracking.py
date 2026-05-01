from collections import defaultdict

def detect_duplicates(frames):
    seen, dups = set(), []
    for fn, d in frames:
        fid = d.get('frame_id')
        if fid in seen: dups.append((fid, fn))
        seen.add(fid)
    return dups


def build_tracks(frames):
    children, objects = defaultdict(dict), defaultdict(dict)
    timestamps = {}
    for _, d in frames:
        fid = d['frame_id']
        timestamps[fid] = d.get('timestamp')
        for c in d.get('children', []):
            cid = c.get('id')
            if cid is not None:  # Skip None ids
                children[cid][fid] = c['bbox']
        for o in d.get('objects', []):
            oid = o.get('id')
            if oid is not None:  # Skip None ids (placeholder objects)
                objects[oid][fid] = o
    return children, objects, timestamps