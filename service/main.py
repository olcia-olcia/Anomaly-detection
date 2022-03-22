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

pred_counter = Counter('predictions', 'Number of predictions')
predictions_output_hist = Histogram('predictions_output', 'Predictions output')
predictions_scores_hist = Histogram('predictions_scores', 'Predictions scores')
predictions_latency_hist = Histogram('predictions_latency', 'Latency of predictions')

class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: bool

@app.post("/prediction")
def predict(request: PredictionRequest):
    start_time = time.time()
    pred_counter.inc()
    prediction = clf.predict([request.feature_vector])
    response = {"is_inlier": int(prediction[0])}

    if request.score:

        scores = clf.score_samples([request.feature_vector])
        response['anomaly_score'] = scores[0]

    response['time'] = time.time() - start_time

    predictions_output_hist.observe(response['is_inliner'])
    predictions_scores_hist.observe(response['anomaly_score'])
    predictions_latency_hist.observe(response['time'])

    return response

@app.get("/model_information")
def model_information():
    return clf.get_params()



if __name__ == "__main__":
    print("running...")