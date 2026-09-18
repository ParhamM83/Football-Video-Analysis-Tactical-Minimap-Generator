#!/usr/bin/env python
"""
Main entry point for the Football Analytics & Tactical Minimap Pipeline.
Usage:
    python main.py analyze --input input_video.mp4 --output output_video.avi
    python main.py train --data custom_data.yaml --epochs 50
"""

import argparse
import sys
import cli
from config import Config



def main():
    """Main entry point"""
    config = Config()

    parser = argparse.ArgumentParser(
        description="Machine Learning Pipeline CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    subparsers.add_parser(
        "train",
        add_help=False,
        help="Train a model"
    )

    subparsers.add_parser(
        "analyze",
        add_help=False,
        help="Analyze a video"
    )

    args, remaining = parser.parse_known_args()

    match args.command:
        case "train":
            cli.train_model(remaining, config)
        case "analyze":
            cli.run_analyzer(remaining, config)
        case _:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)

if __name__ == "__main__":
    main()
