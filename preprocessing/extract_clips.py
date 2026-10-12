"""
Clip Extraction Script for BSL-1K Dataset

This script processes the BSL-1K mouthing annotations and extracts short video clips
corresponding to individual sign instances.

Overview:
- Reads mouthing annotation data (.pkl file)
- Filters annotations based on confidence threshold
- Extracts short clips around each detected sign using ffmpeg
- Saves clips into a structured directory: clips/<split>/<word>/*.mp4

Key Features:
- Uses a fixed time window around detected sign peaks
- Filters out words with too few instances
- Supports optional reprocessing of clips

Purpose:
This script is used during preprocessing to convert continuous sign language videos
into word-level clips that can be used for training and evaluation.
Usage:
    python extract_clips.py --output_dir data/bsl1k_annotated
"""

import argparse
import os
import pickle as pkl
import subprocess
from collections import defaultdict

MIN_INSTANCES = 100
def extract_clip(source_file, output_filename, start_time, end_time):
    """Extract a clip from a video.

    arguments:
    ---------
    source_file: str
        Path to untrimmed video
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Defensive argument checking.
    assert isinstance(output_filename, str), "output_filename must be string"
    status = False
    # Construct command to trim the videos (ffmpeg required).
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-t",
        str(end_time - start_time),
        "-i",
        "'%s'" % source_file,
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-threads",
        "1",
        "-loglevel",
        "panic",
        '"%s"' % output_filename,
    ]
    command = " ".join(command)
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status


def take_interval_from_peak(time_in_sec, window_sec=1.2):
    half = window_sec / 2
    start_time = max(0.0, time_in_sec - half)
    end_time = time_in_sec + half
    return start_time, end_time


def construct_video_filename(
    output_dir, set_name, word, name, start_time, end_time, trim_format
):
    output_filename = "%s_%s_%s.mp4" % (
        name,
        trim_format % start_time,
        trim_format % end_time,
    )
    return os.path.join(output_dir, set_name, word, output_filename)


def construct_original_filename(episode_name):
    # episode_name already ends with .mp4
    return os.path.join("bsl-1k", "video", episode_name)

def mkdir_p(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def build_mouthing_format_from_v2(pkl_path, prob_thres=0.9, keep_type="M*", test_episode=None):
    """
    Converts mouthing_spottings_v2.pkl (dict-of-lists) into:
    data[split][word] = {"names":[], "times":[], "probs":[]}

    Since you don’t have official splits in the pkl, we put everything in "test"
    for now (you can later split by episode into train/val/test).
    """
    d = pkl.load(open(pkl_path, "rb"))

    out = {"test": {}}  # keep it simple for now

    for ep, w, t, p, ty in zip(
        d["episode_name"], d["annot_word"], d["annot_time"], d["annot_prob"], d["annot_type"]
    ):
        if test_episode and ep != test_episode:
            continue
        if keep_type and ty != keep_type:
            continue
        if p < prob_thres:
            continue

        if w not in out["test"]:
            out["test"][w] = {"names": [], "times": [], "probs": []}

        out["test"][w]["names"].append(ep)   # episode filename, e.g. 5085....mp4
        out["test"][w]["times"].append(float(t))
        out["test"][w]["probs"].append(float(p))

    return out

def main(output_dir, prob_thres, trim_format, anno_pkl_path, refresh, window_sec):
    # Where extracted clips will go
    output_dir_videos = os.path.join(output_dir, "clips")
    mkdir_p(output_dir_videos)

    # Build data in the format the original script expects
    data = build_mouthing_format_from_v2(
        anno_pkl_path,
        prob_thres=prob_thres,
        keep_type="M*",
        test_episode=None,   # set to None to do all videos
    )
    # Filter words with fewer than MIN_INSTANCES
    filtered_words = {}

    for word, worddict in data["test"].items():
        num_instances = len(worddict["probs"])
        if num_instances >= MIN_INSTANCES:
            filtered_words[word] = worddict

    print(f"Words before filtering: {len(data['test'])}")
    print(f"Words after filtering (>= {MIN_INSTANCES}): {len(filtered_words)}")

    data["test"] = filtered_words
    # Quick count
    count = 0
    for worddict in data["test"].values():
        count += len(worddict["probs"])
    print(f"Count after filtering: {count}")

    cnts = {}
    output_filenames = defaultdict(list)
    failed = []
    for s in data.keys():
        cnts[s] = []
        for word in data[s].keys():
            cnts[s].append(0)
            mkdir_p(os.path.join(output_dir_videos, s, word))
            N = len(data[s][word]["names"])
            for i in range(N):
                if data[s][word]["probs"][i] > prob_thres:
                    cnts[s][-1] += 1
                    start_time, end_time = take_interval_from_peak(
                        data[s][word]["times"][i],
                        window_sec=window_sec
                    )
                    output_filename = construct_video_filename(
                        output_dir_videos,
                        s,
                        word,
                        data[s][word]["names"][i],
                        start_time,
                        end_time,
                        trim_format,
                    )
                    output_filenames[output_filename].append((start_time, end_time))
                    if not os.path.exists(output_filename) or refresh:
                        source_file = construct_original_filename(data[s][word]["names"][i])
                        status = extract_clip(
                            source_file, output_filename, start_time, end_time
                        )
                        print(output_filename, source_file)
                        print(status)
                        if not status:
                            failed.append(source_file)
    fnames = sorted(output_filenames, key=lambda x: len(x[1]))
    for fname in fnames[:10]:
        print(f"{fname}: {output_filenames[fname]}")


if __name__ == "__main__":
    description = "Script to extract clips from the BSL-1K annotated split."
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--output_dir",
        type=str,
        default="data/bsl1k_annotated",
        help="Output directory where videos will be saved.",
    )
    p.add_argument(
        "--prob_thres",
        type=float,
        default=0.5,
        help="Threshold for the mouthing probability.",
    )

    p.add_argument(
        "--refresh", action="store_true", help="whether to refresh the videos"
    )
    p.add_argument(
        "--anno_pkl_path",
        default="data/bsl1k_annotated/info/annotations_in_mouthing_format_2020.03.04.pkl",
        help="Location of the pkl file containing the human verified annotations.",
    )
    p.add_argument(
        "-f",
        "--trim-format",
        type=str,
        default="%06d",
        help=(
            "This will be the format for the "
            "filename of trimmed videos: "
            "videoid_%%0xd(start_time)_%%0xd(end_time).mp4"
        ),
    )
    p.add_argument(
        "--window_sec",
        type=float,
        default=1.2,
        help="Clip length in seconds (e.g. 1.2).",
    )
    main(**vars(p.parse_args()))