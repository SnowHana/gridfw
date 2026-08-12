FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app" , "--host", "0.0.0.0", "--port", "8000"]
