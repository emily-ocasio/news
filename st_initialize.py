"""
Initialize the SentenceTransformer model
"""
from sentence_transformers import SentenceTransformer

def st_model():
    """
    Load and return the SentenceTransformer model.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class SentenceTransformerModel:
    """
    Wrapper class for SentenceTransformer model
    """
    def __init__(self):
        self._model = None

    @property
    def model(self):
        """
        Lazy load and return the SentenceTransformer model
        """
        if self._model is None:
            self._model = st_model()
        return self._model
