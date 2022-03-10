FROM python:3.7-slim

COPY requirements.txt .

COPY anomaly.joblib .

COPY main.py . 
# install dependencies
RUN pip install -r requirements.txt



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

