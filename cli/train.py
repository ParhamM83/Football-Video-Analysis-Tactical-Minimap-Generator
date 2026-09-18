#!/usr/bin/env python
"""
Training command for machine learning models.

Usage:
    python main.py train --data custom_data.csv --epochs 20
    python -m cli.train --data custom_data.csv --epochs 20
"""

import argparse
import logging
from ultralytics import YOLO
from config import Config

logger = logging.getLogger(__name__)

def parse_args(argv, config: Config):
    """Parse command line arguments with config defaults"""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        default=config.Train.train_data_path,
        help=f"Path to training data (default: {config.Train.train_data_path})"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/best.pt",
        help="Path to save the trained model (default: models/best.pt)"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.Train.batch_size,
        help=f"Batch size for training (default: {config.Train.batch_size})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        help=f"Device to use (default: {config.device})"
    )

    parser.add_argument(
        "--yolo-model",
        type=str,
        default=config.Train.yolo_base_model,
        help=f"YOLO base model (default: {config.Train.yolo_base_model})"
    )

    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=config.Train.conf,
        help=f"Confidence threshold (default: {config.Train.conf})"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=not config.Train.use_cache,
        help="Disable caching"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args(argv)


def train_model(args, config: Config):
    """Train the model with given arguments and config"""
    parsed_args = parse_args(args, config)

    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Training configuration:")
    logger.info("  Data: %s", parsed_args.data)
    logger.info("  Model save path: %s", parsed_args.output)
    logger.info("  Epochs: %s", parsed_args.epochs)
    logger.info("  Device: %s", parsed_args.device)
    logger.info("  Batch size: %s", parsed_args.batch_size)
    logger.info("  Confidence: %s", parsed_args.confidence)
    logger.info("  YOLO model: %s", parsed_args.yolo_model)
    logger.info("  Cache: %s", 'Disabled' if parsed_args.no_cache else 'Enabled')

    model = YOLO(parsed_args.yolo_model)
    model.train(
        data=parsed_args.data,
        epochs=parsed_args.epochs,
        batch=parsed_args.batch_size,
        device=parsed_args.device,
        conf=parsed_args.confidence,
        imgsz=config.Train.imgsz,
        cache='disk' if not parsed_args.no_cache else False,
        verbose=parsed_args.verbose,
        workers=6,
        nbs=32,
        half=False,
        amp=False,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        save_period=5,
        save=True,
        exist_ok=True,
        resume=parsed_args.resume,
    )

    model.save(parsed_args.output)
    logger.info("Model saved to: %s", parsed_args.output)
    logger.info("Training completed successfully!")
