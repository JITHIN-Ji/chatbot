# app/services/document_processor.py
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    • extract_text_from_pdf -> List[dict] each with
        { "index": 0‑based index,
          "label": printed label (e.g. '6', 'ii'),
          "text" : page text }
    • chunk_text returns (chunks, meta_per_chunk)
        meta_per_chunk item = {
            "page"       : int page if label is numeric else -1,
            "label"      : label string,
            "page_index" : 0‑based index,
            "chunk_index": running chunk counter
        }
    """

    # ---------- 1. Extract ----------
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        pages: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
            for idx, page in enumerate(doc):
                text  = page.get_text() or ""
                label = page.get_label() or str(idx + 1)   # printed page; fallback
                pages.append({"index": idx, "label": label, "text": text})
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}", exc_info=True)
        return pages

    # ---------- 2. Chunk ----------
    def chunk_text(
        self,
        pages: List[Dict],
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> Tuple[List[str], List[Dict]]:
        chunks: List[str] = []
        meta  : List[Dict] = []

        for page in pages:
            page_idx   = page["index"]
            page_label = page["label"]
            text       = page["text"]

            start = 0
            while start < len(text):
                end   = start + chunk_size
                chunk = text[start:end]

                chunks.append(chunk)
                meta.append({
                    "page"       : int(page_label) if page_label.isdigit() else -1,
                    "label"      : page_label,
                    "page_index" : page_idx,
                    "chunk_index": len(chunks) - 1
                })

                start += chunk_size - chunk_overlap

        return chunks, meta


# ---- quick test ----
if __name__ == "__main__":
    proc = DocumentProcessor()
    pages = proc.extract_text_from_pdf("example.pdf")
    ch, md = proc.chunk_text(pages)
    print(md[0])   # {'page': 1, 'label': '1', 'page_index': 0, 'chunk_index': 0}
