import torch
from transformers import pipeline
from huggingface_hub import InferenceClient
from config import config

class EmotionClassifier:
    def __init__(self, mode:['api', 'local'] = 'api'):
        self.mode = mode
        
        if self.mode == "local":
            self.classifier = pipeline(
                "text-classification",
                model=config.model_path,
                device=0 if torch.cuda.is_available() else -1
            )
            print(f"✅ Локальная модель загружена: {config.model_path}")
            
        elif self.mode == "api":
            token = config.hf_api_token
            print(f"{token=}")
            if token:
                self.client = InferenceClient(
                    model=config.model_name,
                    token=token,
                    timeout=30.0
                )
                print(f"✅ Настроен клиент для HF API: {config.model_name}")
            else:
                print(f"❌ HF токен не задан!")
    
    def predict(self, text):
        if self.mode == "local":
            result = self.classifier(text)[0]
            return {
                "label": result["label"],
                "score": round(result["score"], 4)
            }
        else:
            try:
                result = self.client.text_classification(text)[0]
                return {
                    "label": result.label,
                    "score": round(result.score, 4)
                }
            except Exception as e:
                return {"error": f"API error: {str(e)}"}

