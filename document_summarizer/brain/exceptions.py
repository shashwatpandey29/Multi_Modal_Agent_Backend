class BrainError(Exception):
    pass

class PDFLoadError(BrainError):
    pass

class LLMError(BrainError):
    pass

class EmbeddingError(BrainError):
    pass
