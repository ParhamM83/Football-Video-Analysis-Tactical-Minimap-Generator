import os
from typing import List, Optional
import numpy as np
import cv2


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Read all frames from a video file into memory.

    :param video_path: Path to the video file.
    :return: List of BGR frames as numpy ndarrays.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


def save_video(frames: List[np.ndarray], video_path: str, fps: float = 24.0) -> None:
    """
    Save a list of video frames to an output video file using XVID codec.

    :param frames: List of frames to write.
    :param video_path: Destination path for the video file.
    :param fps: Video frame rate (default: 24.0).
    """
    if not frames:
        print("[Warning] No frames to save.")
        return

    # Ensure parent directory exists
    parent_dir = os.path.dirname(video_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    try:
        for frame in frames:
            out.write(frame)
    finally:
        out.release()
