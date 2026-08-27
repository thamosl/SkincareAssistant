import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    """
    Configuration for managing file paths and other settings in the application.
    """

    PDF_SOURCE_DIRECTORY: str = os.path.join(BASE_DIR, "data")
    CHROMA_PERSIST_DIRECTORY: str = os.path.join(BASE_DIR, "docs", "chroma")
    COLLECTION_NAME: str = "skincare"

    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    GROQ_API_KEY: str = "gsk_*" #Your Groq API KEY

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents directory: '{self.PDF_SOURCE_DIRECTORY}'.")

config = Config()
