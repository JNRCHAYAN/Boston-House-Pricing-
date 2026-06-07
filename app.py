from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np
import pandas as pd



app = FastAPI()

# Load Model
model = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.get("/about.html")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={}
    )

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])
    return {"prediction": float(output[0])}


