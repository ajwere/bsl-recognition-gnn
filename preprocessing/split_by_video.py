import os
import random
import pickle
from collections import defaultdict

root = "../../dataset_98/video"

with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)

classes = vocab_data["classes"]

print("vocab length:", len(classes))

# -------------------------
# STEP 1: GROUP BY LONG VIDEO ID
# -------------------------
video_groups = defaultdict(list)

for cls in classes:
    class_path = os.path.join(root, cls)

    if not os.path.isdir(class_path):
        continue

    for f in os.listdir(class_path):
        if not f.endswith(".mp4"):
            continue

        full_path = os.path.join(class_path, f)

        # -------------------------
        # EXTRACT LONG VIDEO ID
        # -------------------------
        # "5130...mp4_000104_000105.mp4"
        video_id = f.split(".mp4_")[0] + ".mp4"

        video_groups[video_id].append(full_path)

# -------------------------
# STEP 2: SPLIT BY VIDEO ID
# -------------------------
video_ids = list(video_groups.keys())
random.seed(42)
random.shuffle(video_ids)

n = len(video_ids)

train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_ids = set(video_ids[:train_end])
val_ids = set(video_ids[train_end:val_end])
test_ids = set(video_ids[val_end:])

# -------------------------
# STEP 3: ASSIGN CLIPS
# -------------------------
splits = {"train": [], "val": [], "test": []}

for vid, clips in video_groups.items():
    if vid in train_ids:
        splits["train"].extend(clips)
    elif vid in val_ids:
        splits["val"].extend(clips)
    else:
        splits["test"].extend(clips)

# -------------------------
# SAVE
# -------------------------
with open("splits_by_video.pkl", "wb") as f:
    pickle.dump(splits, f)

print("Total videos:", len(video_ids))
print("Train videos:", len(train_ids))
print("Val videos:", len(val_ids))
print("Test videos:", len(test_ids))


print("\nSample grouped videos:")
for i, (vid, clips) in enumerate(video_groups.items()):
    print(vid, "->", len(clips))
    if i == 5:
        break


        