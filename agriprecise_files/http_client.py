import requests
import json
import logging

from logging.handlers import RotatingFileHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- LOGGING CONFIGURATION ---
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "client.log"

LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
    "[%(name)s] | "
    "[%(funcName)s] | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler],
    force=True)

logger = logging.getLogger(__name__)

def client():

    url = "http://localhost:8000/anomaly_detection"
    #File name to analyse. The rover ID is optional and can be removed
    payload = {
        "file": "example_data.rtcm3",
        "rover": "ROVER_01"
    }
        
    try:
        logger.info(f"Sending POST request to {url} for file: {payload['file']}")
        # timeout de 10 segons
        resposta = requests.post(url, json=payload, timeout=120)
        logger.info(f"HTTP Status: {resposta.status_code}")

        resposta.raise_for_status()

        try:
            dades_retorn = resposta.json()
            preds = dades_retorn.get('Predictions', [])
            logger.info(f"Successfully received response containing {len(preds)} predictions.")
            logger.debug(f"Full JSON Response:\n{json.dumps(dades_retorn, indent=2)}")
            print("Resposta JSON:")
            print(json.dumps(dades_retorn, indent=2))
        except ValueError:
            logger.error("Invalid JSON received from server.")
            logger.error(f"Raw response: {resposta.text}")
            print("Error: JSON no valid")
            print(f"rebut: {resposta.text}")
            
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e}")
    except requests.exceptions.ConnectionError:
        logger.error("Connection error: Is the server running on localhost:8000?")
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 120 seconds.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        
if __name__ == "__main__":
    client()