import re
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from prometheus_client import make_asgi_app, Counter, Histogram
import time

app = FastAPI()
app_prom = make_asgi_app()
app.mount("/metrics", app_prom)
clf = load('anomaly.joblib') 

predictions_counter = Counter('predictions', 'Number of predictions')
model_info_counter = Counter(
    'model_info', 'Number of executions of model information')
predictions_output_hist = Histogram('predictions_output', 'Predictions output')
predictions_scores_hist = Histogram('predictions_scores', 'Predictions scores')
predictions_latency_hist = Histogram(
    'predictions_latency', 'Latency of predictions')

class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: bool


class PredictionResponse(BaseModel):
    score: float

@app.post("/prediction")
def predict(request: PredictionRequest):
    start_time = time.time()
    predictions_counter.inc()

    prediction = clf.predict([request.feature_vector])
    response = {"is_inlier": int(prediction[0])}
    predictions_output_hist.observe(int(0))

    if request.score:

        scores = clf.score_samples([request.feature_vector])
        response['anomaly_score'] = scores[0]
        predictions_scores_hist.observe(scores[0])

    delta = time.time() - start_time    
    response['time'] = delta
    predictions_latency_hist.observe(delta)

    return response

@app.get("/model_information")
def model_information():
    model_info_counter.inc()
    return clf.get_params()



if __name__ == "__main__":
    print("running...")