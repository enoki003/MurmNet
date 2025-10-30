"""
Main entry point for MurmurNet system.
"""

import asyncio
from pathlib import Path

import uvicorn
from loguru import logger

from src.api.server import app
from src.config import config
from src.utils.logging import setup_logging


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    
    logger.info("=" * 60)
    logger.info(f"Starting {config.system.system_name}")
    logger.info("=" * 60)
    
    # Ensure all directories exist
    config.ensure_directories()
    
    # Start API server
    uvicorn.run(
        app,
        host=config.api.api_host,
        port=config.api.api_port,
        workers=config.api.api_workers,
        log_level=config.system.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
