# Lightweight image for the Streamlit error explorer (reads predictions.parquet
# + classes.json). Training runs on the GPU cluster, not here.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir numpy pandas pyarrow pillow matplotlib streamlit

COPY src ./src
COPY web ./web

EXPOSE 8501
ENV IMGCLS_OUT=data/run
# docker run -p 8501:8501 -v "$PWD/data:/app/data" image-classification
CMD ["streamlit", "run", "web/explorer.py", "--server.address=0.0.0.0", "--server.port=8501"]
