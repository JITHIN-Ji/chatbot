# app/services/ocr_reader.py
import platform
from pathlib import Path
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

# ---- 1. locate tesseract once ----
if platform.system() == "Windows":
    DEFAULT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(DEFAULT_PATH).exists():
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_PATH
    else:
        logger.warning("Tesseract executable not found – update path in OCRReader!")

class OCRReader:
    """Very small wrapper around pytesseract."""
    def extract_text(self, image_path: str) -> str:
        try:
            with Image.open(image_path) as im:
                return pytesseract.image_to_string(im)
        except Exception as exc:
            logger.error(f"OCR failed for {image_path}: {exc}", exc_info=True)
            return ""
