import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")

# Vector DB Settings
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", str(BASE_DIR / "vectorstore"))

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "llama3-8b-8192")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

# Path configurations
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
