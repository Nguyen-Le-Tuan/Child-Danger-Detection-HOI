# ========================= plotting_fixed.py =========================
# Clean, robust plotting utilities
# - Normalizes labels to hashable values
# - Ensures strict time ordering
# - Avoids zig-zag artifacts
# - Handles missing / dict labels safely

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
import numpy as np
import os

# ---------------- helpers ----------------

def _center_from_bbox(bbox):
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return (bbox['x1'] + bbox['x2']) / 2, (bbox['y1'] + bbox['y2']) / 2
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return None


def _normalize_label(label, pid=None):
    """
    Always return a hashable scalar (str / int) or None.
    Robust to dict / list / nested structures.
    """
    if label is None:
        return None

    # dict: pick by pid
    if isinstance(label, dict):
        if pid is not None and pid in label:
            return _normalize_label(label[pid])
        return None

    # list / tuple: join to string
    if isinstance(label, (list, tuple)):
        if not label:
            return None
        return '+'.join(str(x) for x in label)

    # numpy / others
    try:
        hash(label)
        return label
    except TypeError:
        return str(label)

# ---------------- trajectories ----------------

def plot_trajectories_2d(merged_frames, out, spacing=10):
    if isinstance(merged_frames, dict):
        merged_frames = list(merged_frames.values())

    merged_frames = sorted(merged_frames, key=lambda f: f.get('timestamp', 0))

    traj = defaultdict(lambda: {'x': [], 'y': [], 'type': None})

    for f in merged_frames:
        for c in f.get('children', []):
            ctr = _center_from_bbox(c.get('bbox'))
            if ctr:
                k = f"child_{c['id']}"
                traj[k]['x'].append(ctr[0])
                traj[k]['y'].append(ctr[1])
                traj[k]['type'] = 'child'
        for o in f.get('objects', []):
            ctr = _center_from_bbox(o.get('bbox'))
            if ctr:
                k = f"obj_{o['id']}"
                traj[k]['x'].append(ctr[0])
                traj[k]['y'].append(ctr[1])
                traj[k]['type'] = 'object'

    if not traj:
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(traj)))

    for i, (k, v) in enumerate(traj.items()):
        x = np.array(v['x']); y = np.array(v['y'])
        if len(x) < 2:
            continue
        if v['type'] == 'child':
            ax.plot(x, y, color='black', lw=3, label=k)
        else:
            ax.plot(x, y, color=colors[i], lw=1.5, alpha=0.7, label=k)

    ax.invert_yaxis()
    ax.set_title('2D Trajectories')
    ax.set_xlabel('X (px)'); ax.set_ylabel('Y (px)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()


# ---------------- distance ----------------

def plot_all_distances_combined(merged_frames, out, max_plots=25):
    if isinstance(merged_frames, dict):
        merged_frames = list(merged_frames.values())

    merged_frames = sorted(merged_frames, key=lambda f: f.get('timestamp', 0))

    pair = defaultdict(lambda: {'t': [], 'd': [], 'lab': []})

    for f in merged_frames:
        t = f.get('timestamp', 0)
        for o in f.get('objects', []):
            oid = o['id']
            for pid, dist in (o.get('distances') or {}).items():
                if dist is None or np.isnan(dist):
                    continue
                pair[(pid, oid)]['t'].append(t)
                pair[(pid, oid)]['d'].append(dist)
                pair[(pid, oid)]['lab'].append(_normalize_label(o.get('label'), pid))

    items = list(pair.items())[:max_plots]
    if not items:
        return

    cols = 4
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(16, 3 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for i, ((pid, oid), v) in enumerate(items):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        t = np.array(v['t']); d = np.array(v['d'])
        idx = np.argsort(t)
        t = t[idx]; d = d[idx]
        lab = [v['lab'][j] for j in idx if v['lab'][j] is not None]
        title_lab = Counter(lab).most_common(1)[0][0] if lab else 'unknown'
        ax.plot(t, d, lw=1.5)
        ax.set_title(f'C{pid}-O{oid}: {title_lab}', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()


# ---------------- velocity ----------------

def plot_relative_velocities(merged_frames, out, spacing=10):
    if isinstance(merged_frames, dict):
        merged_frames = list(merged_frames.values())

    merged_frames = sorted(merged_frames, key=lambda f: f.get('timestamp', 0))

    pair = defaultdict(lambda: {'t': [], 'v': [], 'lab': []})

    for f in merged_frames:
        t = f.get('timestamp', 0)
        for o in f.get('objects', []):
            oid = o['id']
            for pid, vel in (o.get('relative_velocities') or {}).items():
                if vel is None or np.isnan(vel):
                    continue
                pair[(pid, oid)]['t'].append(t)
                pair[(pid, oid)]['v'].append(vel)
                pair[(pid, oid)]['lab'].append(_normalize_label(o.get('label'), pid))

    if not pair:
        return

    cols = 3
    rows = (len(pair) + cols - 1) // cols
    fig = plt.figure(figsize=(15, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for i, ((pid, oid), v) in enumerate(pair.items()):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        t = np.array(v['t']); vel = np.array(v['v'])
        idx = np.argsort(t)
        t = t[idx]; vel = vel[idx]
        lab = [v['lab'][j] for j in idx if v['lab'][j] is not None]
        title_lab = Counter(lab).most_common(1)[0][0] if lab else 'unknown'
        ax.plot(t, vel, lw=1.2, color='gray')
        ax.axhline(0, ls='--', lw=0.8)
        ax.set_title(f'P{pid}-O{oid}: {title_lab}', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=100)
    plt.close()
def plot_label_timeline(merged_frames, out, max_rows=25):
    """
    Plot label timeline for (child, object) pairs.
    Each row = one (child, object) pair
    X-axis = time
    Color = label category
    """
    if isinstance(merged_frames, dict):
        merged_frames = list(merged_frames.values())

    merged_frames = sorted(merged_frames, key=lambda f: f.get('timestamp', 0))

    # (pid, oid) -> list of (t, label)
    pair = defaultdict(list)

    for f in merged_frames:
        t = f.get('timestamp', 0)
        for o in f.get('objects', []):
            oid = o['id']
            label = o.get('label')
            for pid in (o.get('distances') or {}).keys():
                lab = _normalize_label(label, pid)
                if lab is not None:
                    pair[(pid, oid)].append((t, lab))

    items = list(pair.items())[:max_rows]
    if not items:
        return

    # Encode labels to integers
    all_labels = sorted({lab for _, v in items for _, lab in v})
    label_to_int = {l: i for i, l in enumerate(all_labels)}

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(items) + 2))

    for y, ((pid, oid), seq) in enumerate(items):
        seq = sorted(seq, key=lambda x: x[0])
        t = [x[0] for x in seq]
        lab = [label_to_int[x[1]] for x in seq]

        ax.scatter(t, [y] * len(t), c=lab, cmap='tab20', s=25)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f'C{pid}-O{oid}' for (pid, oid), _ in items], fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_title('Label Timeline')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()

def plot_interaction_timeline(merged_frames, out, max_rows=25):
    """
    Plot interaction timeline for (child, object) pairs.
    Supports:
      - interaction as string
      - interaction as dict {child_id: interaction}
    """
    if isinstance(merged_frames, dict):
        merged_frames = list(merged_frames.values())

    merged_frames = sorted(merged_frames, key=lambda f: f.get('timestamp', 0))

    pair = defaultdict(list)

    for f in merged_frames:
        t = f.get('timestamp', 0)
        for o in f.get('objects', []):
            oid = o['id']
            interaction = o.get('interaction')

            for pid in (o.get('distances') or {}).keys():
                it = _normalize_label(interaction, pid)
                if it is not None:
                    pair[(pid, oid)].append((t, it))

    items = list(pair.items())[:max_rows]
    if not items:
        return

    # Encode interaction to integers
    all_inter = sorted({it for _, v in items for _, it in v})
    inter_to_int = {l: i for i, l in enumerate(all_inter)}

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(items) + 2))

    for y, ((pid, oid), seq) in enumerate(items):
        seq = sorted(seq, key=lambda x: x[0])
        t = [x[0] for x in seq]
        it = [inter_to_int[x[1]] for x in seq]

        ax.scatter(t, [y] * len(t), c=it, cmap='tab20b', s=25)

    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f'C{pid}-O{oid}' for (pid, oid), _ in items], fontsize=9)
    ax.set_xlabel('Time (s)')
    ax.set_title('Interaction Timeline')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
