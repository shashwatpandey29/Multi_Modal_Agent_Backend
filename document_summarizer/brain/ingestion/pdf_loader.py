import pdfplumber
from document_summarizer.brain.logger import get_logger
from document_summarizer.brain.exceptions import PDFLoadError

logger = get_logger("PDFLoader")

def load_pdf_text(path: str) -> str:
    try:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
        if not text.strip():
            raise PDFLoadError("Empty PDF content")
        return text
    except Exception as e:
        logger.error(f"PDF loading failed: {e}")
        raise PDFLoadError(str(e))
