
# config.py

import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Default LLM model
DEFAULT_MODEL = "gemini-1.5-flash"
MODEL_TEMPERATURE = 0.1

# Optional: logging level, max tokens, etc.