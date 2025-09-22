import logging
import os

# Purpose: Centralized logger for the entire project
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

try:
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
except Exception:
    # Fallback: log to console if file writing fails
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("doc_classifier")
