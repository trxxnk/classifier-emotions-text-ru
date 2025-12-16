import argparse
import json
import sys

import requests


def classify(endpoint: str, text: str) -> dict:
    response = requests.post(endpoint, params={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="CLI клиент для Emotion Classifier API")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/classify",
        help="URL POST эндпоинта /classify (по умолчанию http://localhost:8000/classify)",
    )
    args = parser.parse_args()

    print("Введите текст для классификации (или пустую строку для выхода):")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not text:
            print("Завершение работы.")
            break

        try:
            result = classify(args.endpoint, text)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except requests.Timeout:
            print("Таймаут: сервер не ответил вовремя.")
        except requests.ConnectionError:
            print("Ошибка сети: не удалось подключиться к серверу.")
        except requests.HTTPError as http_err:
            print(f"HTTP ошибка: {http_err} / ответ: {http_err.response.text}")
        except ValueError:
            print("Ошибка формата ответа: не удалось разобрать JSON.")
        except Exception as exc:
            print(f"Ошибка запроса: {exc}")


if __name__ == "__main__":
    sys.exit(main())

