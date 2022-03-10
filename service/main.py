from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


app = FastAPI()
clf = load('anomaly.joblib') 

class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: bool

@app.post("/prediction")
def predict(request: PredictionRequest):

    prediction = clf.predict([request.feature_vector])
    response = {"is_inlier": int(prediction[0])}

    if request.score:

        scores = clf.score_samples([request.feature_vector])
        response['anomaly_score'] = scores[0]

    return response

@app.get("/model_information")
def model_information():
    return clf.get_params()


if __name__ == "__main__":
    print("running...")