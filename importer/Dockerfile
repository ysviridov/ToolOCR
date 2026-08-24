FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY toolocr_importer ./toolocr_importer
ENTRYPOINT ["python", "-m", "toolocr_importer.main"]
