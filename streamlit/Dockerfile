FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    streamlit==1.38.0 \
    requests==2.32.3

COPY app.py .

RUN useradd -m -u 1000 app && chown -R app:app /app
USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
