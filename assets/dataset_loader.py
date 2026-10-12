import numpy as np
import torch
from torch_geometric.data import Data
import json
from pathlib import Path
import pickle
import random
from scipy.ndimage import gaussian_filter1d
from assets.dataset_utils import *

# ----------------------------
# Skeleton edges
# ----------------------------

# 21-landmark MediaPipe hand edges
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Mouth edges: simple chain + closing loop
MOUTH_EDGES = [(i, i + 1) for i in range(39)] + [(39, 0)]


# Pose edges for 7-point upper body layout
# Assumes something like:
# 0=head/nose, 1=left shoulder, 2=right shoulder,
# 3=left elbow, 4=right elbow, 5=left wrist, 6=right wrist
POSE_EDGES = [
    (0, 1), (0, 2),
    (1, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
]


def load_episode_split_paths(manifest_path, split, feature_root):
    manifest_path = Path(manifest_path)
    feature_root = Path(feature_root)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    episode_ids = set(manifest[split])
    paths = []

    for p in feature_root.rglob("*.npz"):
        episode_id = p.name.split(".mp4_")[0]
        if episode_id in episode_ids:
            paths.append(p)

    return paths, episode_ids



from pathlib import Path
import pickle

def load_split_paths(splits_path, split, feature_root="features"):
    """
    Loads file paths for a given split from splits.pkl
    Optionally maps to feature paths if feature_root is provided.
    """

    splits_path = Path(splits_path)

    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    paths = splits.get(split, [])

    print(f"[load_split_paths] split={split}")
    print(f"[load_split_paths] total samples={len(paths)}")

    # -------------------------
    # OPTIONAL: map to features
    # -------------------------
    if feature_root is not None:
        feature_root = Path(feature_root)

        mapped_paths = []
        missing = 0

        for p in paths:
            # preserve class folder structure
            rel = Path(p).parts[-2:]  # class/video.mp4_0001.mp4
            fp = feature_root / Path(p).with_suffix(".npz").name
            fp = feature_root / Path(p).parts[-2] / Path(p).with_suffix(".npz").name

            if fp.exists():
                mapped_paths.append(fp)
            else:
                missing += 1

        print(f"[load_split_paths] feature paths found={len(mapped_paths)} missing={missing}")
        return mapped_paths

    return paths

 
def load_split_paths__(manifest_path, split, feature_root="features"):
    manifest_path = Path(manifest_path)
    feature_root = Path(feature_root)

   # print(f"[load_split_paths] split={split}")
   # print(f"[load_split_paths] manifest_path={manifest_path.resolve()}")
    #print(f"[load_split_paths] feature_root={feature_root.resolve()}")
    #print(f"[load_split_paths] manifest exists={manifest_path.exists()}")
    #print(f"[load_split_paths] feature_root exists={feature_root.exists()}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    assignments = manifest["assignments"].get(split, {})
    print(f"[load_split_paths] words in split={len(assignments)}")

    words = sorted(manifest["summary"].keys())
    label2id = {w: i for i, w in enumerate(words)}

    paths = []
    missing = 0

    for word, files in assignments.items():
        for fn in files:
            p = feature_root / word / fn
            if p.exists():
                paths.append(p)
            else:
                missing += 1

    print(f"[load_split_paths] found paths={len(paths)} missing={missing}")
    return paths, label2id


def normalize_pose_and_hands(hand, pose):
    """
    hand: (T, 2, 21, 3) before flattening
    pose: (T, 7, 4)

    We normalize xyz using:
      center = midpoint of shoulders
      scale  = shoulder distance
    """
    hand = hand.copy()
    pose = pose.copy()

    # pose indices:
    # 0=head/nose, 1=left shoulder, 2=right shoulder,
    # 3=left elbow, 4=right elbow, 5=left wrist, 6=right wrist
    lsho = pose[:, 1, :3]
    rsho = pose[:, 2, :3]

    center = ((lsho + rsho) / 2.0)[:, None, :]          # (T, 1, 3)
    scale = np.linalg.norm(rsho - lsho, axis=-1, keepdims=True)  # (T, 1)
    scale = np.clip(scale, 1e-6, None)[:, None, :]      # (T, 1, 1)

    # normalize pose xyz
    pose[:, :, :3] = (pose[:, :, :3] - center) / scale

    # normalize hand xyz
    hand[:, :, :, :3] = (hand[:, :, :, :3] - center[:, None, :, :]) / scale[:, None, :, :]

    return hand, pose


def normalize_mouth(mouth):
    # mouth: (T, 40, 3)
    mouth = mouth.copy()

    # example corners, adjust indices to your landmark layout
    left_corner = mouth[:, 0, :3]
    right_corner = mouth[:, 6, :3]

    center = mouth[:, :, :3].mean(axis=1, keepdims=True)   # (T, 1, 3)
    width = np.linalg.norm(right_corner - left_corner, axis=-1, keepdims=True)  # (T, 1)
    width = np.clip(width, 1e-6, None)[:, None, :]  # (T, 1, 1)

    mouth[:, :, :3] = (mouth[:, :, :3] - center) / width
    return mouth

def zscore_feature(arr, mean, std, eps=1e-8):
    arr = arr.astype(np.float32)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    return (arr - mean) / (std + eps)

def compute_mean_std(feature_list):
    """
    feature_list: list of numpy arrays of shape (D,)
    returns mean, std of shape (D,)
    """
    X = np.stack(feature_list, axis=0).astype(np.float32)   # (N, D)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.clip(std, 1e-6, None)
    return mean, std


# ----------------------------
# Compute graph features
# ----------------------------

def compute_graph_feats(
    hand,
    mouth,
    pose,
    use_hand_velocity=True,
    use_mouth_velocity=True,
    use_smoothing=True,
    smoothing_sigma=1.0
):
    """
    Node features:

      Base:
        4 -> [x,y,z,confidence]

      Node type:
        3 -> [hand,mouth,pose]

      Hand:
        3 -> velocity xyz
        1 -> velocity magnitude
        3 -> acceleration xyz

      Mouth:
        3 -> velocity xyz
        1 -> velocity magnitude
        3 -> acceleration xyz

      Mouth stats:
        2 -> [open,width]


    Total dimension = 23
    """

    T = hand.shape[0]
    # =====================================================
    # Base features
    # =====================================================

    feat = np.concatenate([hand, mouth, pose],axis=1).astype(np.float32)
    # =====================================================
    # Node types
    # =====================================================
    hand_type = np.tile(np.array([1, 0, 0], dtype=np.float32), (T, 42, 1))
    mouth_type = np.tile(np.array([0, 1, 0], dtype=np.float32), (T, 40, 1))
    pose_type = np.tile(np.array([0, 0, 1], dtype=np.float32), (T, 7, 1))
    node_type = np.concatenate([hand_type, mouth_type, pose_type], axis=1)   # (T, 89, 3)


    # =====================================================
    # Containers
    # =====================================================
    hand_vel_full = np.zeros((T,89,3), dtype=np.float32)
    hand_speed_full = np.zeros((T,89,1), dtype=np.float32)
    hand_acc_full = np.zeros((T,89,3), dtype=np.float32)


    mouth_vel_full = np.zeros((T,89,3), dtype=np.float32)
    mouth_speed_full = np.zeros((T,89,1), dtype=np.float32)
    mouth_acc_full = np.zeros((T,89,3), dtype=np.float32)

    # =====================================================
    # Helper function
    # =====================================================

    def compute_motion(xyz):
        if use_smoothing:
            #print("using msmoothing")
            xyz = gaussian_filter1d(xyz, sigma=smoothing_sigma, axis=0)

        velocity = np.zeros_like(xyz, dtype=np.float32)
        # -----------------------------
        # Central difference
        # -----------------------------
        if xyz.shape[0] > 2:
            velocity[1:-1] = (xyz[2:] - xyz[:-2]) * 0.5
            # boundary frames
            velocity[0] = xyz[1] - xyz[0]
            velocity[-1] = xyz[-1] - xyz[-2]
        else:
            velocity[1:] = xyz[1:] - xyz[:-1]

        # -----------------------------
        # Speed magnitude
        # -----------------------------
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)

        # -----------------------------
        # Acceleration
        # -----------------------------
        acceleration = np.zeros_like(velocity)
        acceleration[1:-1] = (velocity[2:] - velocity[:-2]) * 0.5
        acceleration[0] = (velocity[1] - velocity[0])
        acceleration[-1] = (velocity[-1] - velocity[-2])


        return (
            velocity.astype(np.float32),
            speed.astype(np.float32),
            acceleration.astype(np.float32)
        )

    # =====================================================
    # Hand motion
    # =====================================================

    if use_hand_velocity:
        #print("using hand velocity")
        hand_xyz = hand[..., :3]
        (hand_vel, hand_speed, hand_acc) = compute_motion(hand_xyz)
        hand_vel_full[:,0:42] = hand_vel
        hand_speed_full[:,0:42] = hand_speed
        hand_acc_full[:,0:42] = hand_acc

    # =====================================================
    # Mouth motion
    # =====================================================

    if use_mouth_velocity:
        #print("using mouth velocity")
        mouth_xyz = mouth[..., :3]
        (mouth_vel, mouth_speed, mouth_acc) = compute_motion(mouth_xyz)
        mouth_vel_full[:,42:82] = mouth_vel
        mouth_speed_full[:,42:82] = mouth_speed
        mouth_acc_full[:,42:82] = mouth_acc

    # =====================================================
    # Mouth shape features
    # =====================================================

    left_corner_idx = 0
    right_corner_idx = 6
    upper_lip_idx = 13
    lower_lip_idx = 19

    mouth_width = np.linalg.norm(
        mouth[:,right_corner_idx,:3] - mouth[:,left_corner_idx,:3],
        axis=-1,
        keepdims=True
    )

    mouth_open = np.linalg.norm(
        mouth[:,upper_lip_idx,:3] - mouth[:,lower_lip_idx,:3],
        axis=-1,
        keepdims=True
    )

    mouth_stats = np.zeros((T,89,2), dtype=np.float32)

    mouth_stats[:,42:82,0] = mouth_open
    mouth_stats[:,42:82,1] = mouth_width


    # =====================================================
    # Final feature vector
    # =====================================================

    feat = np.concatenate(
        [
            feat, node_type,
            hand_vel_full, hand_speed_full, hand_acc_full,
            mouth_vel_full, mouth_speed_full, mouth_acc_full,
            mouth_stats
        ],
        axis=-1
    )


    return feat.astype(np.float32)







# ----------------------------
# Loading
# ----------------------------

def load_valid_frames(path):
    """
    Loads one .npz clip and returns a dict:
        hand:  (T, 42, 4)
        mouth: (T, 40, 4)
        pose:  (T, 7, 4)

    Output feature format for all node groups is:
        [x, y, z, conf]
    """
    d = np.load(path, allow_pickle=False)

    hand = d["hand"].astype(np.float32)   
    mouth = d["mouth"].astype(np.float32)  
    pose = d["pose"].astype(np.float32) 


    mouth_compact = d["mouth_compact"].astype(np.float32) if "mouth_compact" in d.files else None

    mouth_stats = d["mouth_stats"].astype(np.float32) if "mouth_stats" in d.files else None
  

    hand, pose = normalize_pose_and_hands(hand, pose)
    mouth = normalize_mouth(mouth)

    # Flatten 2 hands -> 42 nodes
    hand = hand.reshape(hand.shape[0], 42, 3)  # (T, 42, 3)

    # Add dummy confidence channel to hand/mouth so all nodes become 4D
    hand_conf = np.zeros((hand.shape[0], hand.shape[1], 1), dtype=np.float32)
    hand = np.concatenate([hand, hand_conf], axis=-1)  # (T, 42, 4)

    mouth_conf = np.zeros((mouth.shape[0], mouth.shape[1], 1), dtype=np.float32)
    mouth = np.concatenate([mouth, mouth_conf], axis=-1)  # (T, 40, 4)

    # pose already has 4 dims
    return {
        "hand": hand,    # (T, 42, 4)
        "mouth": mouth,  # (T, 40, 4)
        "pose": pose,
        "mouth_compact": mouth_compact,
        "mouth_stats": mouth_stats,
    }


# ----------------------------
# Edge building
# ----------------------------

def make_spatial_edges_per_frame():
    """
    Node order per frame:
      left hand:  0..20
      right hand: 21..41
      mouth:      42..81
      pose:       82..88
    """
    edges = []

    # Left hand
    for a, b in HAND_EDGES:
        edges.append((a, b))
        edges.append((b, a))

    # Right hand
    for a, b in HAND_EDGES:
        edges.append((a + 21, b + 21))
        edges.append((b + 21, a + 21))

    # Mouth
    for a, b in MOUTH_EDGES:
        edges.append((a + 42, b + 42))
        edges.append((b + 42, a + 42))

    # Pose
    for a, b in POSE_EDGES:
        edges.append((a + 82, b + 82))
        edges.append((b + 82, a + 82))

    # Minimal cross-region edges
    # left hand wrist (0) <-> pose left wrist (5 -> global 87)
    edges.append((0, 82 + 5))
    edges.append((82 + 5, 0))

    # right hand wrist (21) <-> pose right wrist (6 -> global 88)
    edges.append((21, 82 + 6))
    edges.append((82 + 6, 21))

    # mouth center approx (42 + 20) <-> pose head/nose (82 + 0)
    edges.append((42 + 20, 82 + 0))
    edges.append((82 + 0, 42 + 20))


    def add_bi(a, b):
        edges.append((a, b))
        edges.append((b, a))

    # pose offsets
    HEAD = 82 + 0
    LSHO = 82 + 1
    RSHO = 82 + 2
    LWRIST_POSE = 82 + 5
    RWRIST_POSE = 82 + 6

    # hand anchors
    LH_WRIST = 0
    RH_WRIST = 21
    LH_INDEX_TIP = 8
    RH_INDEX_TIP = 21 + 8
    LH_MID_TIP = 12
    RH_MID_TIP = 21 + 12

    # mouth anchor
    MOUTH_CENTER = 42 + 20

    # existing
    add_bi(LH_WRIST, LWRIST_POSE)
    add_bi(RH_WRIST, RWRIST_POSE)
    add_bi(MOUTH_CENTER, HEAD)

    # stronger cross-region
    add_bi(LH_WRIST, MOUTH_CENTER)
    add_bi(RH_WRIST, MOUTH_CENTER)
    add_bi(LH_INDEX_TIP, MOUTH_CENTER)
    add_bi(RH_INDEX_TIP, MOUTH_CENTER)
    add_bi(LH_MID_TIP, MOUTH_CENTER)
    add_bi(RH_MID_TIP, MOUTH_CENTER)

    add_bi(LH_WRIST, HEAD)
    add_bi(RH_WRIST, HEAD)
    add_bi(MOUTH_CENTER, LSHO)
    add_bi(MOUTH_CENTER, RSHO)
    add_bi(LH_WRIST, RH_WRIST)

    return edges




def build_spatiotemporal_edge_index(T: int, V: int):
    spatial = make_spatial_edges_per_frame()
    edge_list = []

    # more computaitionally cheaper but not sure
    """
    LEFT_HAND_CENTRE = 0
    RIGHT_HAND_CENTRE = 21
    MOUTH_CENTRE = 62
    LEFT_SHOULDER = 82
    RIGHT_SHOULDER = 83

    extra_edges = [
        (LEFT_HAND_CENTRE, MOUTH_CENTRE),
        (MOUTH_CENTRE, LEFT_HAND_CENTRE),

        (RIGHT_HAND_CENTRE, MOUTH_CENTRE),
        (MOUTH_CENTRE, RIGHT_HAND_CENTRE),

        (LEFT_HAND_CENTRE, RIGHT_HAND_CENTRE),
        (RIGHT_HAND_CENTRE, LEFT_HAND_CENTRE),

        (LEFT_HAND_CENTRE, LEFT_SHOULDER),
        (RIGHT_HAND_CENTRE, RIGHT_SHOULDER),
    ]
    """
    
    def add(a, b):
        edge_list.append((a, b))

    # spatial
    for t in range(T):
        base = t * V
        for a, b in spatial:
            add(base + a, base + b)

    # temporal same-node
    for t in range(T - 1):
        base0 = t * V
        base1 = (t + 1) * V
        for j in range(V):
            add(base0 + j, base1 + j)
            add(base1 + j, base0 + j)

        # temporal cross-region anchor edges
        LH_WRIST = 0
        RH_WRIST = 21
        MOUTH_CENTER = 42 + 20
        HEAD = 82 + 0

        anchors = [
            (LH_WRIST, MOUTH_CENTER),
            (RH_WRIST, MOUTH_CENTER),
            (LH_WRIST, HEAD),
            (RH_WRIST, HEAD),
            (MOUTH_CENTER, HEAD),
        ]

        for a, b in anchors:
            add(base0 + a, base1 + b)
            add(base1 + b, base0 + a)
            add(base0 + b, base1 + a)
            add(base1 + a, base0 + b) 

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return edge_index

_EDGE_CACHE = {}


def build_spatiotemporal_edge_index_cached(T: int, V: int):
    key = (T, V)
    if key not in _EDGE_CACHE:
        _EDGE_CACHE[key] = build_spatiotemporal_edge_index(T, V)
    return _EDGE_CACHE[key]


def edge_index_to_adj(edge_index, V):
    adj = torch.zeros(V, V)

    adj[edge_index[0], edge_index[1]] = 1

    return adj


# ----------------------------
# DEEP FEATURES
# ----------------------------

def compute_videomae_stats(
    train_paths,
    hand_videomae_root=None,
    upper_mouth_videomae_root=None,
    use_upper_body_feature=False,
    use_mouth_feature=False,
    use_hand_feature=False,
    save_path=None,
):
    hand_videomae_root = Path(hand_videomae_root) if hand_videomae_root is not None else None
    upper_mouth_videomae_root = Path(upper_mouth_videomae_root) if upper_mouth_videomae_root is not None else None

    hands_feats = []
    upper_feats = []
    mouth_feats = []

    for landmark_path in train_paths:
        landmark_path = Path(landmark_path)
        word = landmark_path.parent.name
        filename = landmark_path.name

        if upper_mouth_videomae_root is not None:
            upper_mouth_path = upper_mouth_videomae_root / word / filename
            if upper_mouth_path.exists():
                d_um = np.load(upper_mouth_path, allow_pickle=False)

                if use_upper_body_feature and "upper_body_feature" in d_um.files:
                    upper_feats.append(d_um["upper_body_feature"].astype(np.float32))

                if use_mouth_feature and "mouth_feature" in d_um.files:
                    mouth_feats.append(d_um["mouth_feature"].astype(np.float32))

        if hand_videomae_root is not None:
            hand_path = hand_videomae_root / word / filename
            if hand_path.exists():
                d_hand = np.load(hand_path, allow_pickle=False)

                if use_hand_feature and "hands_feature" in d_hand.files:
                    hands_feats.append(d_hand["hands_feature"].astype(np.float32))

    stats = {}

    if len(hands_feats) > 0:
        mean, std = compute_mean_std(hands_feats)
        stats["hands_mean"] = mean
        stats["hands_std"] = std
        print(f"[stats] hands: {len(hands_feats)} samples, dim={mean.shape[0]}")

    if len(upper_feats) > 0:
        mean, std = compute_mean_std(upper_feats)
        stats["upper_body_mean"] = mean
        stats["upper_body_std"] = std
        print(f"[stats] upper_body: {len(upper_feats)} samples, dim={mean.shape[0]}")

    if len(mouth_feats) > 0:
        mean, std = compute_mean_std(mouth_feats)
        stats["mouth_mean"] = mean
        stats["mouth_std"] = std
        print(f"[stats] mouth: {len(mouth_feats)} samples, dim={mean.shape[0]}")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, **stats)
        print(f"[stats] saved to {save_path}")

    return stats







# ----------------------------
# Dataset
# ----------------------------

class SignClipGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        label2id,
        load_fn=load_valid_frames,
        use_hand_velocity=True,
        use_mouth_velocity=True,
        use_mouth_compact=False,
        preload=True,
        hand_videomae_root=None,
        upper_mouth_videomae_root=None,
        deepFeat_stats_path=None,
        deepFeat_norm="none",
        use_upper_body_feature=False,
        use_mouth_feature=False,
        use_hand_feature=False,
        augment=True,
        videomae_norm="none",
        
    ):
        self.paths = list(paths)
        self.label2id = label2id
        self.load_fn = load_fn
        self.use_hand_velocity = use_hand_velocity
        self.use_mouth_velocity = use_mouth_velocity
        self.use_mouth_compact = use_mouth_compact
        self.preload = preload

        self.hand_videomae_root = Path(hand_videomae_root) if hand_videomae_root is not None else None
        self.upper_mouth_videomae_root = Path(upper_mouth_videomae_root) if upper_mouth_videomae_root is not None else None

        self.use_upper_body_feature = use_upper_body_feature
        self.use_mouth_feature = use_mouth_feature
        self.use_hand_feature = use_hand_feature
        self.deepFeat_norm = deepFeat_norm
        self.videomae_norm = videomae_norm
        self.videomae_stats = {}

        # Track missing VideoMAE features without spamming the terminal
        self.missing_upper_mouth_count = 0
        self.missing_hand_count = 0
        self.missing_upper_body_feature_count = 0
        self.missing_mouth_feature_count = 0
        self.missing_hands_feature_count = 0

        self.data_list = []

        self.augment = augment

       
        if deepFeat_stats_path is not None:
            stats_file = np.load(deepFeat_stats_path, allow_pickle=False)
            self.videomae_stats = {k: stats_file[k].astype(np.float32) for k in stats_file.files}
            print(f"[Dataset] Loaded deepFeat stats from {deepFeat_stats_path}")
            print(f"[Dataset] Available stats keys: {list(self.videomae_stats.keys())}")
            
            

        if self.preload:
            print(f"[Dataset] Preloading {len(self.paths)} clips into RAM...")

            for p in self.paths:
                clip = self.load_fn(p)

              

                hand = clip["hand"]    # (T, 42, 4)
                mouth = clip["mouth"]  # (T, 40, 4)
                pose = clip["pose"]    # (T, 7, 4)
                mouth_stats = clip.get("mouth_stats", None)

                if self.augment: 
                    print("augmenting training data")
                    hand, mouth, pose = temporal_augment(hand, mouth, pose)

                         
                feat = compute_graph_feats(
                    hand,
                    mouth,
                    pose,
                    use_hand_velocity=self.use_hand_velocity,
                    use_mouth_velocity=self.use_mouth_velocity,
                )



                y = torch.tensor(self.label2id[p.parent.name], dtype=torch.long)
                data = self._build_data_from_feat(feat, y)

                vm_feats = self._load_videomae_for_path(p)

                data.upper_body_feature = vm_feats.get(
                    "upper_body_feature",
                    torch.zeros(768, dtype=torch.float32)
                ).unsqueeze(0)

                data.mouth_feature = vm_feats.get(
                    "mouth_feature",
                    torch.zeros(768, dtype=torch.float32)
                ).unsqueeze(0)

                data.hands_feature = vm_feats.get(
                    "hands_feature",
                    torch.zeros(768, dtype=torch.float32)
                ).unsqueeze(0)
               

               

                self.data_list.append(data)

            print("[Dataset] Preloading complete.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.preload:
            return self.data_list[idx]

        p = self.paths[idx]

        clip = self.load_fn(p)
        

        hand = clip["hand"]
        mouth = clip["mouth"]
        pose = clip["pose"]
        mouth_stats = clip.get("mouth_stats", None)

        if self.augment: 
            hand, mouth, pose = temporal_augment(hand, mouth, pose)
      

        feat = compute_graph_feats(
            hand,
            mouth,
            pose,
            use_hand_velocity=self.use_hand_velocity,
            use_mouth_velocity=self.use_mouth_velocity,
        )

        y = torch.tensor(self.label2id[p.parent.name], dtype=torch.long)
        data = self._build_data_from_feat(feat, y)

        vm_feats = self._load_videomae_for_path(p)

        data.upper_body_feature = vm_feats.get(
            "upper_body_feature",
            torch.zeros(768, dtype=torch.float32)
        ).unsqueeze(0)

        data.mouth_feature = vm_feats.get(
            "mouth_feature",
            torch.zeros(768, dtype=torch.float32)
        ).unsqueeze(0)

        data.hands_feature = vm_feats.get(
            "hands_feature",
            torch.zeros(768, dtype=torch.float32)
        ).unsqueeze(0)
       


        return data


    def _normalize_videomae_feature(self, feat, feature_type):
        """
        feat: numpy array (D,)
        feature_type: one of 'hands', 'upper_body', 'mouth'
        """
        feat = feat.astype(np.float32)

        if self.videomae_norm == "none":
            return feat

        elif self.videomae_norm == "zscore":
            mean_key = f"{feature_type}_mean"
            std_key = f"{feature_type}_std"

            if mean_key not in self.videomae_stats or std_key not in self.videomae_stats:
                raise ValueError(
                    f"Missing z-score stats for {feature_type}. "
                    f"Expected keys {mean_key} and {std_key} in stats file."
                )

            mean = self.videomae_stats[mean_key]
            std = self.videomae_stats[std_key]
            return (feat - mean) / (std + 1e-8)

        else:
            raise ValueError(f"Unknown videomae_norm: {self.videomae_norm}")

    

    def _build_data_from_feat(self, feat, y):
        T, V, F = feat.shape
        x = torch.from_numpy(feat.reshape(T * V, F)).float()
        edge_index = build_spatiotemporal_edge_index_cached(T, V)
        return Data(x=x, edge_index=edge_index, y=y)

    def _load_videomae_for_path(self, landmark_path):
        landmark_path = Path(landmark_path)
        word = landmark_path.parent.name
        filename = landmark_path.name

        out = {}

        # ------------------------------------------------------------
        # Upper-body / mouth VideoMAE
        # ------------------------------------------------------------
        if self.upper_mouth_videomae_root is not None:

            upper_mouth_path = (
                self.upper_mouth_videomae_root / word / filename
            )

            if upper_mouth_path.exists():

                d_um = np.load(
                    upper_mouth_path,
                    allow_pickle=False
                )

                # -------------------------
                # Upper-body feature
                # -------------------------
                if self.use_upper_body_feature:

                    if "upper_body_feature" in d_um.files:

                        feat = d_um["upper_body_feature"].astype(
                            np.float32
                        )

                        feat = self._normalize_videomae_feature(
                            feat,
                            "upper_body"
                        )

                        out["upper_body_feature"] = torch.from_numpy(feat)

                    else:
                        self.missing_upper_body_feature_count += 1

                # -------------------------
                # Mouth feature
                # -------------------------
                if self.use_mouth_feature:

                    if "mouth_feature" in d_um.files:

                        feat = d_um["mouth_feature"].astype(
                            np.float32
                        )

                        feat = self._normalize_videomae_feature(
                            feat,
                            "mouth"
                        )

                        out["mouth_feature"] = torch.from_numpy(feat)

                    else:
                        self.missing_mouth_feature_count += 1

            else:

                if self.use_upper_body_feature:
                    self.missing_upper_mouth_count += 1

                if self.use_mouth_feature:
                    self.missing_upper_mouth_count += 1

        # ------------------------------------------------------------
        # Hand VideoMAE
        # ------------------------------------------------------------
        if self.hand_videomae_root is not None:

            hand_path = (
                self.hand_videomae_root / word / filename
            )

            if hand_path.exists():

                d_hand = np.load(
                    hand_path,
                    allow_pickle=False
                )

                if self.use_hand_feature:

                    if "hands_feature" in d_hand.files:

                        feat = d_hand["hands_feature"].astype(
                            np.float32
                        )

                        feat = self._normalize_videomae_feature(
                            feat,
                            "hands"
                        )

                        out["hands_feature"] = torch.from_numpy(feat)

                    else:
                        self.missing_hands_feature_count += 1

            else:

                if self.use_hand_feature:
                    self.missing_hand_count += 1

        # ------------------------------------------------------------
        # Print ONE feature-shape check only
        # ------------------------------------------------------------
        if not hasattr(self, "_printed_vm_shape"):

            print("---- VideoMAE feature check ----")

            if self.upper_mouth_videomae_root is not None:
                print(
                    "upper_mouth_videomae_root:",
                    self.upper_mouth_videomae_root
                )

            if self.hand_videomae_root is not None:
                print(
                    "hand_videomae_root:",
                    self.hand_videomae_root
                )

            print(
                "videomae_norm:",
                self.videomae_norm
            )

            for k, v in out.items():

                print(
                    f"{k}: {tuple(v.shape)} | "
                    f"mean={float(v.mean()):.4f} "
                    f"std={float(v.std()):.4f}"
                )

            self._printed_vm_shape = True

        return out