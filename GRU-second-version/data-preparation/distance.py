import numpy as np

# ================= geometry =================

def bbox_center(b):
    if b is None:
        return None
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)


def center_distance(b1, b2):
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    if c1 is None or c2 is None:
        return np.nan
    return float(np.linalg.norm(c1 - c2))


def safe_unit_vector(v, eps=1e-6):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


# ================= distances =================

def compute_distances(children, objects):
    """
    children: dict {child_id: bbox}
    objects: dict {object_id: bbox}
    return: dict {object_id: {child_id: distance}}
    """
    out = {}

    for cid, cb in children.items():
        if cb is None:
            continue

        for oid, ob in objects.items():
            if ob is None:
                continue

            dist = center_distance(cb, ob)
            out.setdefault(oid, {})[cid] = dist

    return out


# ================= relative velocity (NEW, CORRECT) =================

def compute_relative_velocities(merged_frames):
    """
    Physically-correct relative velocity:
    v_rel = (v_child - v_object) · unit(child → object)

    Returns:
        dict {frame_id: {object_id: {child_id: v_rel}}}
    """

    # sort frames
    if isinstance(merged_frames, dict):
        frames = sorted(merged_frames.values(), key=lambda f: f["frame_id"])
    else:
        frames = merged_frames

    centers = {}

    for f in frames:
        fid = f["frame_id"]
        centers[fid] = {"child": {}, "obj": {}}

        for c in f.get("children", []):
            cc = bbox_center(c.get("bbox"))
            if cc is not None:
                centers[fid]["child"][c["id"]] = cc

        for o in f.get("objects", []):
            oc = bbox_center(o.get("bbox"))
            if oc is not None:
                centers[fid]["obj"][o["id"]] = oc

    frame_vel = {}

    for i in range(1, len(frames)):
        f0 = frames[i - 1]
        f1 = frames[i]

        fid = f1["frame_id"]
        dt = f1["timestamp"] - f0["timestamp"]
        if dt <= 0:
            continue

        for oid, o1 in centers[fid]["obj"].items():
            if oid not in centers[f0["frame_id"]]["obj"]:
                continue

            o0 = centers[f0["frame_id"]]["obj"][oid]
            v_obj = (o1 - o0) / dt

            for cid, c1 in centers[fid]["child"].items():
                if cid not in centers[f0["frame_id"]]["child"]:
                    continue

                c0 = centers[f0["frame_id"]]["child"][cid]
                v_child = (c1 - c0) / dt

                direction = o1 - c1
                u = safe_unit_vector(direction)
                if u is None:
                    continue

                v_rel = float(np.dot(v_child - v_obj, u))

                frame_vel.setdefault(fid, {}) \
                         .setdefault(oid, {})[cid] = v_rel

    return frame_vel
