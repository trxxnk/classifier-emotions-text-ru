FROM python:3.12-alpine

ENV PYTHONUNBUFFERED=1

WORKDIR /python-app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
