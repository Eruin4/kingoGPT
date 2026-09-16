FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY pyproject.toml README.md ./
COPY kingogpt ./kingogpt
RUN pip install --no-cache-dir --no-deps .

RUN mkdir -p /app/state

EXPOSE 8000

CMD ["kingogpt-openai-server"]
