# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: utils/logging.py
# Description: Advanced logging setup for the project.
# Created: 2025-04-15
# Updated: 2025-04-15

import logging
import os
import sys # Needed for StreamHandler target
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- Logging Configuration ---
# Read configuration from environment variables with sensible defaults
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE_ENABLED = os.environ.get('LOG_FILE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOG_CONSOLE_ENABLED = os.environ.get('LOG_CONSOLE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
LOG_FILE_NAME = os.environ.get('LOG_FILE_NAME', 'dropout_disco.log') # Updated project name
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10 * 1024 * 1024))  # 10 MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s | %(name)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s') # Added filename/lineno
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGER_NAME = "DropoutDisco" # Define a specific logger name

# --- Logger Instance ---
# Get OUR specific logger instance. Other modules will import this.
logger = logging.getLogger(LOGGER_NAME)

# Flag to ensure setup runs only once
_logging_initialized = False

# --- Logging Setup Function ---
def setup_logging():
    """
    Configures the project-specific logger ('DropoutDisco') with handlers.
    Reads configuration from environment variables.
    Intended to be called automatically once on first import.
    """
    global _logging_initialized
    if _logging_initialized:
        # print("Skipping redundant logging setup.") # Optional debug print
        return

    print("⚙️  Configuring DropoutDisco logging...") # Use print before logger is fully set up

    # Determine the logging level
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logger.setLevel(level) # Set level on OUR logger
    print(f"  Logger '{LOGGER_NAME}' level set to: {LOG_LEVEL}")

    # Remove existing handlers from OUR logger to prevent duplicates
    if logger.hasHandlers():
        print("  Clearing existing logging handlers...")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console Handler ---
    if LOG_CONSOLE_ENABLED:
        console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
        console_handler.setLevel(level) # Set level on handler
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        print("  ✅ Console logging handler added.")
    else:
        print("  ❌ Console logging disabled by environment variable.")

    # --- Rotating File Handler ---
    if LOG_FILE_ENABLED:
        try:
            os.makedirs(LOGS_DIR, exist_ok=True)
            log_file_path = os.path.join(LOGS_DIR, LOG_FILE_NAME)

            # Check writability
            with open(log_file_path, 'a', encoding='utf-8') as f:
                 f.write(f"# Log session started at: {datetime.now().isoformat()}\n")

            file_handler = RotatingFileHandler(
                log_file_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding='utf-8'
            )
            # File handler often logs more detail (e.g., DEBUG if root logger allows)
            # Let's set it to the same level as root for simplicity here,
            # or set to DEBUG if LOG_LEVEL allows finer control desired.
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"  ✅ File logging handler added: {log_file_path}")

        except PermissionError:
             print(f"  ❌ ERROR: Permission denied for log file: {log_file_path}. File logging disabled.")
        except Exception as e:
            print(f"  ❌ ERROR: Failed to set up file logger: {e}. File logging disabled.")
    else:
        print("  ❌ File logging disabled by environment variable.")

    if logger.hasHandlers():
        logger.info("🎉 Logging system initialized successfully!")
        for handler in logger.handlers:
             handler.flush()
    else:
        print(f"⚠️ Warning: No logging handlers configured for logger '{LOGGER_NAME}'.")

    _logging_initialized = True # Mark as initialized

# --- Automatically configure logging on first import ---
# This ensures that simply importing 'logger' from utils gets a configured logger.
# Be mindful if complex import orders or circular dependencies exist in larger projects.
if not _logging_initialized:
    setup_logging()

# --- Example Usage (for testing logging.py directly) ---
if __name__ == "__main__":
    # Setup has already run automatically on import if this is the main script

    logger.debug("🐛 This is a debug message.")
    logger.info("✅ This is an info message.")
    logger.warning("⚠️ This is a warning message.")
    logger.error("❌ This is an error message.")
    logger.critical("🚨 This is a critical message!")

    user = "DropoutDiscoDev"
    action = "tested logger"
    logger.info(f"👤 User '{user}' successfully {action} 👍")