import argparse
from config import Config
import utils

from tracking import Tracker, PitchTracker
from assigner import Assigner
from renderer import Renderer
from homography import HomographyTransformer


def parse_args(argv, config: Config):
    """Parse command line arguments for video analysis."""
    parser = argparse.ArgumentParser(
        description="Run football match video analysis and minimap generation pipeline"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=config.input_video_path,
        help=f"Path to input video (default: {config.input_video_path})"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=config.output_video_path,
        help=f"Path to save output video (default: {config.output_video_path})"
    )

    parser.add_argument(
        "--player-model",
        type=str,
        default=config.Analyzer.player_model_path,
        help=f"Path to player detection model (default: {config.Analyzer.player_model_path})"
    )

    parser.add_argument(
        "--field-model",
        type=str,
        default=config.Analyzer.field_model_path,
        help=f"Path to field keypoint detection model (default: {config.Analyzer.field_model_path})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        help=f"Inference device (cuda, mps, or cpu; default: {config.device})"
    )

    return parser.parse_args(argv)


def run_analyzer(args, config: Config) -> None:
    """
    Full analysis pipeline:
      1. Read video frames.
      2. Scan frames to compute homography keyframes and optical flow tracking.
      3. Track players, goalkeepers, ball, and others across all frames.
      4. Assign every player / goalkeeper to a team via jersey HSV clustering.
      5. Annotate frames and write output video.
    """
    parsed_args = parse_args(args, config)

    input_path = parsed_args.input
    output_path = parsed_args.output
    player_model_path = parsed_args.player_model
    field_model_path = parsed_args.field_model
    device = parsed_args.device

    print(f"Loading video from: {input_path}")
    video_frames = utils.read_video(input_path)
    if not video_frames:
        print(f"Error: No frames could be read from {input_path}")
        return

    print(f"Loaded {len(video_frames)} frames from {input_path}")

    print("Initializing player and field detection models...")
    tracker = Tracker(
        player_model_path,
        device,
    )

    field_tracker = PitchTracker(
        field_model_path,
        device
    )

    print("Scanning video for pitch keypoints & computing homography keyframes...")
    homography = HomographyTransformer(field_tracker)
    homography.precompute_keyframes(video_frames)

    print("Tracking players and ball across frames...")
    tracks = tracker.track_detections(video_frames)

    print("Clustering jersey colors to assign teams...")
    assigner = Assigner()

    bootstrap_players = {}
    for frame_idx in range(min(30, len(video_frames))):
        for track_id, player in tracks["players"][frame_idx].items():
            if track_id not in bootstrap_players:
                bootstrap_players[track_id] = player

    assigner.assign_team(video_frames[0], bootstrap_players)

    for frame_num, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_num]
        for track_id, track in player_track.items():
            pid = assigner.get_player_team(
                frame, track["bounding_box"], track_id,
            )
            if pid is None:
                continue
            team = assigner.get_team(pid)
            track["global_id"]  = pid
            track["team"]       = team
            track["team_color"] = assigner.get_team_color(team)

    print("Rendering tactical minimap and player overlays...")
    renderer = Renderer(homography)
    output_frames = renderer.render_items(video_frames, tracks)

    print(f"Saving output video to: {output_path}")
    utils.save_video(output_frames, output_path)
    print(f"Analysis complete! Saved: {output_path}")
