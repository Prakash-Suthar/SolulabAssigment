from pydantic import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    MODELPATH: str = os.getenv("MODEL_PATH")
    VECTORIZERPATH: str = os.getenv("VECTORIZER_PATH")
    DataPath: str = os.getenv("DATA_PATH")

settings = Settings()
