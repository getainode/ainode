# BenchChat — fleet-aware chat + benchmark harness.
# Stdlib only, so there is nothing to install. Set BENCHCHAT_PASSWORD before
# putting this on a hostname; without it the server runs open and says so.
FROM python:3.12-slim
WORKDIR /app
COPY bench-serve.py bench-chat.html ./
ENV BENCHCHAT_HOST=0.0.0.0 BENCHCHAT_PORT=8899 PYTHONUNBUFFERED=1
EXPOSE 8899
CMD ["python", "bench-serve.py"]
