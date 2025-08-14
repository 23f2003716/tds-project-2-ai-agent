import logging
import os
from datetime import datetime, timezone, timedelta
from google import genai
from pathlib import Path
from random import randint

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

ist = timezone(timedelta(hours=5, minutes=30))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / f'api_{datetime.now(ist).strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Configure timezone for all log timestamps
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if hasattr(handler.formatter, 'converter'):
        handler.formatter.converter = lambda *args: datetime.now(
            ist).timetuple()

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv(f"GEMINI_API_KEY_{randint(1, 3)}")
client = genai.Client(api_key=GEMINI_API_KEY)
