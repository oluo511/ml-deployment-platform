FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Streamlit default port
EXPOSE 8501

# Start Streamlit web server
# streamlit run app.py = run the Streamlit application
# --server.port=8501 = use port 8501 (Streamlit default)
# --server.address=0.0.0.0 = accept connections from anywhere (not just localhost)  
# --server.headless=true = run without browser popup (good for containers)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]