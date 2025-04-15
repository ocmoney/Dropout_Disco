# Hacker News Upvote Prediction
# Copyright (c) 2025 Dropout Disco Team (Yurii, James, Ollie, Emil)
# File: src/utils.py
# Description: Shared utility functions, including logging setup.
# Created: 2025-04-15
# Updated: 2025-04-15

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

# --- Logging Configuration ---

# Read configuration from environment variables with sensible defaults
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
LOG_FILE_ENABLED = os.environ.get('LOG_FILE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOG_CONSOLE_ENABLED = os.environ.get('LOG_CONSOLE_ENABLED', 'true').lower() in ('true', 'yes', '1')
LOGS_DIR = os.environ.get('LOGS_DIR', 'logs') # Matches project structure
LOG_FILE_NAME = os.environ.get('LOG_FILE_NAME', 'hn_predictor.log') # Project-specific name
LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10 * 1024 * 1024))  # Default: 10 MB
LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5)) # Keep 5 backup logs
LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s') # Added '|' for readability
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# --- Logging Setup Function ---

def setup_logging():
    """
    Configures the root logger with console and rotating file handlers.
    Reads configuration from environment variables.
    Should be called once at application startup.
    """
    print("⚙️  Configuring logging...") # Use print here as logger might not be ready

    # Get the root logger
    root_logger = logging.getLogger()

    # Determine the logging level
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    root_logger.setLevel(level)
    print(f"  Root logger level set to: {LOG_LEVEL}")

    # Remove existing handlers to avoid duplicates if called again
    # Important for environments like FastAPI with --reload
    if root_logger.hasHandlers():
        print("  Clearing existing logging handlers...")
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close() # Close handler to release file locks if any

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- Console Handler ---
    if LOG_CONSOLE_ENABLED:
        console_handler = logging.StreamHandler()
        # Console handler level can be same as root or more verbose if needed
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        print("  ✅ Console logging handler added.")
    else:
        print("  ❌ Console logging disabled by environment variable.")

    # --- Rotating File Handler ---
    if LOG_FILE_ENABLED:
        try:
            # Ensure log directory exists
            os.makedirs(LOGS_DIR, exist_ok=True)
            log_file_path = os.path.join(LOGS_DIR, LOG_FILE_NAME)

            # Check writability (optional but good practice)
            with open(log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"# Log session started at: {datetime.now().isoformat()}\n")

            # Create and add file handler
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            # File handler often logs more detail (e.g., DEBUG)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"  ✅ File logging handler added: {log_file_path}")

        except PermissionError:
             print(f"  ❌ ERROR: Permission denied writing to log file: {log_file_path}. File logging disabled.")
        except Exception as e:
            print(f"  ❌ ERROR: Failed to set up file logger at {LOGS_DIR}/{LOG_FILE_NAME}: {e}. File logging disabled.")
    else:
        print("  ❌ File logging disabled by environment variable.")

    # Final initialization log message using the configured logger
    if root_logger.hasHandlers():
        logging.info("🎉 Logging system initialized successfully!")
        # Explicitly flush handlers after setup (good practice)
        for handler in root_logger.handlers:
            handler.flush()
    else:
        print("⚠️ Warning: No logging handlers configured.")


# --- Example Usage (Optional - for testing utils.py directly) ---
if __name__ == "__main__":
    setup_logging() # Call setup first

    # Now log with emojis!
    logging.debug("🐛 This is a debug message with a bug.")
    logging.info("✅ This is an info message indicating success.")
    logging.warning("⚠️ This is a warning message about a potential issue.")
    logging.error("❌ This is an error message reporting a failure.")
    logging.critical("🚨 This is a critical message! System meltdown imminent!")

    # You can use them anywhere in the string
    user = "DropoutDiscoUser"
    action = "logged in"
    logging.info(f"👤 User '{user}' successfully {action} 👍")

    item_count = 51211
    logging.info(f"📊 Processed {item_count:,} items.")