from fastapi import FastAPI, HTTPException
# from functools import lru_cache
import logging
from pydantic import BaseModel

from classifier import EmotionClassifier


app = FastAPI(title="Emotion Classifier API")
classifier = EmotionClassifier()
logger = logging.getLogger(__name__)


@app.post("/classify")
async def classify_text(text: str):
    text = text.strip()
    if not text or len(text) > 1000:
        raise HTTPException(400, "Текст должен быть от 1 до 1000 символов")
    
    from_cache = False   
    
    result = classifier.predict(text)

    if "error" in result:
        logger.error(f"Ошибка классификации: {result['error']}")
        raise HTTPException(503, result['error'])


    answer = {
        "text": text,
        "emotion": result["label"],
        "confidence": result["score"],
        "from_cache": from_cache
    }

    return answer


# Альтернативный GET endpoint (для тестов)
@app.get("/classify")
async def classify_via_get(text: str):
    """Для быстрых тестов через браузер: /classify?text=Привет"""
    return await classify_text(text=text)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_mode": classifier.mode
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)