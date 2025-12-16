from fastapi import FastAPI, HTTPException
from functools import lru_cache
import logging

from classifier import EmotionClassifier


# Базовая настройка логгирования в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] app.py: %(message)s",
)

app = FastAPI(title="Emotion Classifier API")
classifier = EmotionClassifier()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def _cached_predict(text: str):
    """Cache classifier results to avoid recomputation."""
    return classifier.predict(text)


@app.post("/classify")
async def classify_text(text: str):
    proc_text = text.strip().lower()
    if proc_text == "" or len(text) > 1000:
        raise HTTPException(400, "Текст должен быть от 1 до 1000 символов")

    logger.info("Получен запрос на классификацию: %r", text)
    
    # Detect cache hit using lru_cache stats diff
    hits_before = _cached_predict.cache_info().hits
    result = _cached_predict(proc_text)
    hits_after = _cached_predict.cache_info().hits
    from_cache = hits_after > hits_before

    if "error" in result:
        logger.error(f"Ошибка классификации: {result['error']}")
        raise HTTPException(503, result['error'])


    answer = {
        "text": text,
        "emotion": result["label"],
        "confidence": result["score"],
        "from_cache": from_cache
    }

    logger.info(
        "Результат классификации: emotion=%s, confidence=%.4f, from_cache=%s",
        answer["emotion"],
        answer["confidence"],
        answer["from_cache"],
    )

    return answer


# Альтернативный GET endpoint (для тестов)
@app.get("/classify")
async def classify_via_get(text: str):
    """Для быстрых тестов через браузер: /classify?text=Привет"""
    return await classify_text(text)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_mode": classifier.mode
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False
    )