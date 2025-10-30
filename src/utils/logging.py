"""
Logging configuration for MurmurNet system.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import config


def setup_logging() -> None:
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()
    
    # Add console logger with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=config.system.log_level.value,
        colorize=True,
    )
    
    # Ensure log directory exists
    log_dir = config.system.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file logger
    logger.add(
        log_dir / "murmurnet_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.system.log_level.value,
        rotation="00:00",  # Rotate at midnight
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
    )
    
    # Add error-only file logger
    logger.add(
        log_dir / "murmurnet_errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="00:00",
        retention="90 days",
        compression="zip",
    )
    
    logger.info("Logging configured successfully")
