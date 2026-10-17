# Package the app + production model into one container.
# Build:  docker build -t spam-lab .
# Run:    docker run -p 8501:8501 spam-lab
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py app.py ./
COPY models/production.pkl models/production.pkl

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
