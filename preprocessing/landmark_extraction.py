"""
MediaPipe Feature Extraction Script for BSL Video Clips

This script processes word-level video clips and extracts landmark-based features
using MediaPipe. It is used during preprocessing to convert video clips into
structured feature files for model training and evaluation.

Features extracted:
- Hand landmarks for left and right hands
- Mouth landmarks and compact mouth features
- Upper-body pose landmarks

Processing steps:
- Reads video clips from the input folder
- Extracts frame-level landmarks using MediaPipe
- Normalises hand, mouth, and pose coordinates
- Resamples each sequence to a fixed length of 32 frames
- Saves the extracted features as compressed .npz files

Output format:
- features/<word>/<video_name>.npz

Purpose:
This script prepares the landmark-based input features used by the graph neural
network for British Sign Language recognition.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np 
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mediapipe.tasks import python

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# hand mediapipe
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# body pose mediapipe
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# mouth media pipe 
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# normalise features 
def normalize_hand_points(pts_21x3, eps=1e-6):
    """
    Normalize one hand using wrist as origin and hand size as scale.
    Input: (21, 3)
    Output: (21, 3)
    """
    if np.allclose(pts_21x3, 0):
        return pts_21x3.copy()

    pts = pts_21x3.copy()

    wrist = pts[0]  # landmark 0
    pts = pts - wrist

    # use average distance from wrist to MCP joints as scale
    mcp_ids = [5, 9, 13, 17]
    scale = np.mean([np.linalg.norm(pts[i]) for i in mcp_ids])
    scale = max(scale, eps)

    pts = pts / scale
    return pts


def normalize_mouth_points(pts_nx3, eps=1e-6):
    """
    Normalize mouth landmarks using mouth center and mouth width.
    Input: (N, 3)
    Output: (N, 3)
    """
    if np.allclose(pts_nx3, 0):
        return pts_nx3.copy()

    pts = pts_nx3.copy()

    center = pts.mean(axis=0)
    pts = pts - center

    # width from leftmost to rightmost x
    width = pts[:, 0].max() - pts[:, 0].min()
    width = max(width, eps)

    pts = pts / width
    return pts


def normalize_pose_points(pts_7x4, eps=1e-6):
    """
    Normalize upper-body pose using shoulder midpoint as origin
    and shoulder distance as scale.
    Input: (7, 4) -> x, y, z, visibility
    Output: (7, 4)
    """
    if np.allclose(pts_7x4[:, :3], 0):
        return pts_7x4.copy()

    pts = pts_7x4.copy()

    # indices in your selected upper_body_idx = [0, 11, 12, 13, 14, 15, 16]
    # shoulders are local positions 1 and 2
    left_shoulder = pts[1, :3]
    right_shoulder = pts[2, :3]

    center = (left_shoulder + right_shoulder) / 2.0
    pts[:, :3] = pts[:, :3] - center

    shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
    shoulder_dist = max(shoulder_dist, eps)

    pts[:, :3] = pts[:, :3] / shoulder_dist
    return pts


def resample_sequence(seq, target_len=32):
    """
    Resample a sequence to fixed length.
    seq: numpy array (T, ...)
    """
    T = len(seq)
    if T == 0:
        raise ValueError("Empty sequence cannot be resampled.")

    if T == target_len:
        return seq

    idxs = np.linspace(0, T - 1, target_len).astype(np.int32)
    return seq[idxs]



# HAND FEATURES 
def extract_hand_features(frame, landmarker, timestamp_ms):

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)


    frame_features = {
        "Left": np.zeros((21, 3), dtype=np.float32),
        "Right": np.zeros((21, 3), dtype=np.float32)
    }

    if hand_landmarker_result.hand_landmarks and hand_landmarker_result.handedness:
        for hand_landmarks, handedness in zip(
            hand_landmarker_result.hand_landmarks,
            hand_landmarker_result.handedness
        ):
            label = handedness[0].category_name  # "Left" or "Right"
            features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
            features = normalize_hand_points(features)
            frame_features[label] = features

    return np.stack([frame_features["Left"], frame_features["Right"]], axis=0)
    

    


# mouth FEATURES 
MOUTH_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405,
    321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39,
    37, 0, 267, 269, 270, 409, 415, 310,
    311, 312, 13, 82, 81, 42, 183, 78
]
def pt(face_landmarks, idx):
    lm = face_landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def dist(a, b):
    return np.linalg.norm(a - b)

def extract_mouth_features(frame, landmarker, timestamp_ms):
       
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if face_landmarker_result.face_landmarks:
        face_landmarks = face_landmarker_result.face_landmarks[0] 

      
        mouth_features = np.array([[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z] for i in MOUTH_IDXS], dtype=np.float32)
        mouth_features = normalize_mouth_points(mouth_features)

        left_corner = pt(face_landmarks, 61)
        right_corner = pt(face_landmarks, 291)
        upper_lip = pt(face_landmarks, 13)
        lower_lip = pt(face_landmarks, 14)

        mouth_width = dist(left_corner, right_corner)
        mouth_open = dist(upper_lip, lower_lip)
        mouth_ratio = mouth_open / (mouth_width + 1e-6)

        compact = np.array([mouth_width, mouth_open, mouth_ratio], dtype=np.float32)

    
    else:
        mouth_features = np.zeros((len(MOUTH_IDXS), 3), dtype=np.float32)
        compact = np.zeros(3, dtype=np.float32)

    return mouth_features, compact



# body pose FEATURES

def extract_body_pose_features(frame, landmarker, timestamp_ms):
    upper_body_idx = [0, 11, 12, 13, 14, 15, 16]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    
    pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)


    if pose_landmarker_result.pose_landmarks:
        lm = pose_landmarker_result.pose_landmarks[0]
        features = np.array([[lm[i].x, lm[i].y, lm[i].z, lm[i].visibility] for i in upper_body_idx], dtype=np.float32)
        features = normalize_pose_points(features)
    else:
        features = np.zeros((len(upper_body_idx), 4), dtype=np.float32)

    
    return features



def mediapipe_process_for_clips(video_path_str, output_root_str):
    video_path = Path(video_path_str)
    output_root = Path(output_root_str)
    label = video_path.parent.name

    BaseOptions = mp.tasks.BaseOptions

    delegate = BaseOptions.Delegate.CPU
    #print("Running MediaPipe with delegate:", delegate)

    label_out_dir = output_root / label
    label_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = label_out_dir / f"{video_path.stem}.npz"

    if out_path.exists():
        return "skipped"


    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task",delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2
    )

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="face_landmarker.task",delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1
    )

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task",delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.VIDEO
    )

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
         FaceLandmarker.create_from_options(face_options) as face_landmarker, \
         PoseLandmarker.create_from_options(pose_options) as pose_landmarker:

    
        #print(f"Processing: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        step_ms = max(1, int(round(1000.0 / fps))) if fps and fps > 1e-3 else 33

        hand_seq = []
        mouth_pose_seq = []
        mouth_compact_seq = []
        pose_seq = []
        timestamp_ms = 0

        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            hand_feats = extract_hand_features(frame, hand_landmarker, timestamp_ms)          
            mouth_pose, mouth_compact = extract_mouth_features(frame, face_landmarker, timestamp_ms) 
            body_pose = extract_body_pose_features(frame, pose_landmarker, timestamp_ms) 

            hand_seq.append(hand_feats)
            mouth_pose_seq.append(mouth_pose)
            mouth_compact_seq.append(mouth_compact)
            pose_seq.append(body_pose)

            timestamp_ms += step_ms    

        cap.release()

    if len(hand_seq) == 0:
        hand_feats = np.zeros((32, 2, 21, 3), dtype=np.float32)
        mouth_pose = np.zeros((32, len(MOUTH_IDXS), 3), dtype=np.float32)
        mouth_compact = np.zeros((32, 4), dtype=np.float32)
        body_pose = np.zeros((32, 7, 4), dtype=np.float32)
    else:
        hand_feats = np.stack(hand_seq, axis=0)
        mouth_pose = np.stack(mouth_pose_seq, axis=0)
        mouth_compact = np.stack(mouth_compact_seq, axis=0)
        body_pose = np.stack(pose_seq, axis=0)

        motion = np.zeros((mouth_compact.shape[0], 1), dtype=np.float32)
        if mouth_compact.shape[0] > 1:
            motion[1:, 0] = np.abs(np.diff(mouth_compact[:, 2]))
        mouth_compact = np.concatenate([mouth_compact, motion], axis=1)

        mean = mouth_compact.mean(axis=0, keepdims=True)
        std = mouth_compact.std(axis=0, keepdims=True) + 1e-6
        mouth_compact = (mouth_compact - mean) / std

        hand_feats = resample_sequence(hand_feats, target_len=32)
        mouth_pose = resample_sequence(mouth_pose, target_len=32)
        mouth_compact = resample_sequence(mouth_compact, target_len=32)
        body_pose = resample_sequence(body_pose, target_len=32)

    np.savez_compressed(
        out_path,
        hand=hand_feats,
        mouth=mouth_pose,
        mouth_compact=mouth_compact,
        pose=body_pose
    )


    return "saved"

def silence_worker_output():
    devnull = open(os.devnull, "w")
    os.dup2(devnull.fileno(), 1)  # stdout
    os.dup2(devnull.fileno(), 2)  # stderr


def main():
    input_dir = Path("video")         
    output_dir = Path("features")
    output_dir.mkdir(exist_ok=True)

    video_paths = sorted(input_dir.rglob("*.mp4"))
    print(f"Found {len(video_paths)} videos")
    print("Current folder:", Path.cwd())
    print("Input exists:", input_dir.exists())
    cores = os.cpu_count() or 8
    max_workers = min(4, cores)  

    failed = 0 
    saved = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=silence_worker_output) as executor:
        futures = [
            executor.submit(mediapipe_process_for_clips, str(video_path), str(output_dir))
            for video_path in video_paths
        ]

        with tqdm(total=len(futures), desc="Extracting features", dynamic_ncols=True) as pbar:
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    if result == "saved":
                        saved += 1
                    elif result == "skipped":
                        skipped += 1
                except Exception:
                    failed += 1

                pbar.set_postfix(saved=saved, skipped=skipped, failed=failed)
                pbar.update(1)



if __name__ == "__main__":
    main()














