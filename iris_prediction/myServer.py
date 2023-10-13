
from fastapi import FastAPI, Request
from joblib import  load
import numpy as np
from fastapi.templating import Jinja2Templates
from sklearn.datasets import load_iris  

templates = Jinja2Templates(directory="templates")


filename = "myFirstSavedModel.joblib"
clfUploaded = load(filename)
dataSet = load_iris()
labelsNames =list(dataSet.target_names)

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/predict/")
async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float):

    testData = np.array([L1, W1, L2, W2]).reshape(-1, 4)
    probalities = clfUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probality = probalities[predicted]
    predicted = labelsNames[predicted]

    
    return  templates.TemplateResponse("prediction.html", {"request": request, 
                                                           "probalities": probalities,
                                                           "predicted": predicted,
                                                           "probality": probality})
