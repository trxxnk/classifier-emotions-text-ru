import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    model_name = os.getenv("MODEL_NAME", "")
    model_path = os.getenv("MODEL_PATH", "")
    hf_api_token = os.getenv("HF_API_TOKEN", "")

config = ModelConfig()
