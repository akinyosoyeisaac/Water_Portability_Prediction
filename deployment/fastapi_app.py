import pandas as pd
import yaml
import pickle as pk
import json
from fastapi import FastAPI, Form
import uvicorn


app = FastAPI()
with open("params.yaml") as file:
    config = yaml.safe_load(file)


def loading_model(config):
    with open(config["paths"]["model"], "rb") as file:
        model = pk.load(file)
    return model

model = loading_model(config=config)

columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']


# Building the path to our home page
@app.get("/")
async def home():
    """
    This page runs a prediction that output the whether a water is suitable for consumption
    
    To run this app click on the try it out then enter your values and click on execute.
    
    Your result will be generated in the response body
    """
    return "Check the new page to run a prediction"


# Building the path to the prediction page
@app.post("/predict")
async def prediction(pH_value: int = Form(gt=0, lt=15), Hardness: int = Form(), Solid: int = Form(), Chloramine: int = Form(), Sulfate: int = Form(), Conductivity: int = Form(), Organic_carbon: int = Form(), Trihalomethanes: int = Form(), Turbidity: int = Form()):

    """
    Run prediction using form
    """

    # features to be used to create dataframe object
    values = [pH_value, Hardness, Solid, Chloramine, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]
    

    # Dataframe object of the raw data
    test_data = pd.DataFrame(data = [values], columns=columns)
    
    
    # Generating prediction from our model
    predicted_value = model.predict(test_data)[0]
    
    # Generating our prediction in a human readable format
    if predicted_value == 0:
        prediction = "Water not potable"
    else:
        prediction = "Water potable"
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run("deployment.fastapi_app:app", host="127.0.0.1", port=5000)
