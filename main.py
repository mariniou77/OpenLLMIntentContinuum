#!/usr/bin/env python3
"""
IntentContinuum - Main Entry Point

This is the main entry point for the IntentContinuum system.
It initializes all components and starts the Intent Watch Loop.

Usage:
    python main.py                  # Run with default config
    python main.py --config my.yaml # Run with custom config
    python main.py --iterations 10  # Run for 10 iterations only
"""

import argparse
import logging
import signal
import sys
from pathlib import Path

import yaml

from intent_watch_loop import IntentWatchLoop


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def print_banner():
    """Print application banner."""
    banner = """
║                                                                  ║
║   LLM-Powered Intent-Based Resource Management                   ║
║   for the Compute Continuum                                      ║
║                                                                  ║
"""
    print(banner)


def print_config_summary(config: dict):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"  Intent Thresholds:")
    print(f"    Upper: {config['intent']['upper_threshold']}s")
    print(f"    Lower: {config['intent']['lower_threshold']}s")
    print(f"    EMA Alpha: {config['intent']['ema_alpha']}")
    print(f"  Check Interval: {config['intent']['check_interval']}s")
    print(f"  Wait After Action: {config['intent']['wait_after_action']}s")
    print(f"  Application Endpoint: {config['application']['entry_point']}")
    print(f"  LLM Model: {config['llm']['model']}")
    print(f"  Enabled Actions:")
    for action, enabled in config['actions'].items():
        status = "✅" if enabled else "❌"
        print(f"    {status} {action}")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="IntentContinuum - LLM-Powered Intent-Based Resource Management"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=None,
        help="Number of iterations to run (default: run forever)"
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress banner and config summary"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML configuration: {e}")
        sys.exit(1)
    
    # Print config summary
    if not args.quiet:
        print_config_summary(config)
    
    # Initialize the Intent Watch Loop
    watch_loop = IntentWatchLoop(config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping...")
        watch_loop.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check component health before starting
    logger.info("Checking component health...")
    health = watch_loop.data_collector.get_health_status()
    
    all_healthy = True
    for component, status in health.items():
        icon = "✅" if status else "❌"
        logger.info(f"  {icon} {component}: {'healthy' if status else 'not responding'}")
        if not status:
            all_healthy = False
    
    if not watch_loop.decision_maker.is_healthy():
        logger.warning("  ❌ Ollama (LLM): not responding")
        all_healthy = False
    else:
        logger.info("  ✅ Ollama (LLM): healthy")
    
    if not all_healthy:
        logger.warning("Some components are not healthy. Continuing anyway...")
    
    # Start the watch loop
    logger.info("Starting Intent Watch Loop...")
    
    try:
        watch_loop.run(max_iterations=args.iterations)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    
    logger.info("IntentContinuum stopped.")


if __name__ == "__main__":
    main()