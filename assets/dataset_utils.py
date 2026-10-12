import numpy as np
import random
from scipy.ndimage import gaussian_filter1d


# ----------------------------
# Augment training data
# ----------------------------
def temporal_crop(hand, mouth, pose, crop_ratio=(0.85, 1.0)):
    T = hand.shape[0]

    ratio = random.uniform(*crop_ratio)
    new_len = max(4, int(T * ratio))

    start = random.randint(0, T - new_len)

    hand = hand[start:start+new_len]
    mouth = mouth[start:start+new_len]
    pose = pose[start:start+new_len]

    idx = np.linspace(0, new_len-1, T).astype(int)

    return hand[idx], mouth[idx], pose[idx]


def speed_perturb(hand, mouth, pose, speed=(0.8, 1.2)):
    T = hand.shape[0]

    factor = random.uniform(*speed)

    new_len = max(4, int(T / factor))

    idx = np.linspace(0, T-1, new_len).astype(int)

    hand = hand[idx]
    mouth = mouth[idx]
    pose = pose[idx]

    idx = np.linspace(0, new_len-1, T).astype(int)

    return hand[idx], mouth[idx], pose[idx]


def landmark_noise(hand, mouth, pose, sigma=0.003):
    hand[..., :3] += np.random.normal(0, sigma,hand[..., :3].shape)
    mouth[..., :3] += np.random.normal(0, sigma,mouth[..., :3].shape)
    pose[..., :3] += np.random.normal(0, sigma,pose[..., :3].shape)

    return hand, mouth, pose


def frame_drop(hand, mouth, pose, p=0.5, max_drop=0.2):
    """
    Randomly removes a small percentage of frames.

    hand : (T,42,4)
    mouth: (T,40,4)
    pose : (T,7,4)

    Returns tensors with the SAME sequence length.
    """

    if random.random() > p:
        return hand, mouth, pose

    T = hand.shape[0]
    # number of frames to remove
    n_drop = max(1, int(T * random.uniform(0.05, max_drop)))
    keep = np.ones(T, dtype=bool)
    drop_idx = np.random.choice(T, n_drop, replace=False)
    keep[drop_idx] = False

    hand = hand[keep]
    mouth = mouth[keep]
    pose = pose[keep]

    return hand, mouth, pose

def resample_sequence(seq, target_len):
    old_len = seq.shape[0]

    old_idx = np.arange(old_len)
    new_idx = np.linspace(0, old_len - 1, target_len)

    out = np.empty((target_len, *seq.shape[1:]), dtype=seq.dtype)

    flat = seq.reshape(old_len, -1)

    out_flat = np.stack([
        np.interp(new_idx, old_idx, flat[:, i])
        for i in range(flat.shape[1])
    ], axis=1)

    return out_flat.reshape(target_len, *seq.shape[1:])



# spatial augmentation 

def random_translation(hand, mouth, pose, max_shift=0.03):
    tx = random.uniform(-max_shift, max_shift)
    ty = random.uniform(-max_shift, max_shift)

    for x in (hand, mouth, pose):
        x[..., 0] += tx
        x[..., 1] += ty

    return hand, mouth, pose





def random_scaling(hand, mouth, pose, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)

    all_pts = np.concatenate([
        hand[..., :2].reshape(-1,2),
        mouth[..., :2].reshape(-1,2),
        pose[..., :2].reshape(-1,2)
    ])

    centre = all_pts.mean(axis=0)

    for x in (hand, mouth, pose):
        x[..., :2] = (x[..., :2] - centre) * scale + centre

    return hand, mouth, pose


def random_rotation(hand, mouth, pose, max_angle=10):
    angle = np.deg2rad(random.uniform(-max_angle, max_angle))

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    all_pts = np.concatenate([
        hand[..., :2].reshape(-1,2),
        mouth[..., :2].reshape(-1,2),
        pose[..., :2].reshape(-1,2)
    ])

    centre = all_pts.mean(axis=0)

    for x in (hand, mouth, pose):
        pts = x[..., :2] - centre
        pts = pts @ R.T
        x[..., :2] = pts + centre

    return hand, mouth, pose

def landmark_mask(hand, mouth, pose,
                  mask_prob=0.10):

    for x in (hand, mouth, pose):

        mask = np.random.rand(*x[...,0].shape) < mask_prob

        x[..., :3][mask] = 0.0

        # confidence
        x[..., 3][mask] = 0.0

    return hand, mouth, pose


def random_hand_dropout(hand, p=0.15):

    if random.random() > p:
        return hand

    if random.random() < 0.5:
        hand[:, :21] = 0
    else:
        hand[:, 21:] = 0

    return hand


def temporal_augment(hand, mouth, pose):

    T = hand.shape[0]

    if random.random() < 0.30:
        hand, mouth, pose = temporal_crop(hand, mouth, pose)

    if random.random() < 0.30:
        hand, mouth, pose = speed_perturb(hand, mouth, pose)

    if random.random() < 0.50:
        hand, mouth, pose = frame_drop(hand, mouth, pose)

    hand = resample_sequence(hand, T)
    mouth = resample_sequence(mouth, T)
    pose = resample_sequence(pose, T)

    # Spatial augmentations
    if random.random() < 0.5:
        hand, mouth, pose = random_translation(hand, mouth, pose)

    if random.random() < 0.5:
        hand, mouth, pose = random_scaling(hand, mouth, pose)

    if random.random() < 0.5:
        hand, mouth, pose = random_rotation(hand, mouth, pose)

    if random.random() < 0.3:
        hand, mouth, pose = landmark_mask(hand, mouth, pose)

    if random.random() < 0.15:
        hand = random_hand_dropout(hand)

    if random.random() < 0.5:
        hand, mouth, pose = landmark_noise(hand, mouth, pose)

    return hand, mouth, pose





