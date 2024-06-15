# Create a FastAPI app
# Root endpoint returns the app description

from fastapi import FastAPI
from irisclassifier import predict_species
from pydantic import BaseModel

app = FastAPI()

# function to return description


def get_app_description():
    return (
        "Welcome to the Iris species prediction API"
        "This API allows you to predict the species of an IRIS"
        "Use the '/predict/' endpoint with a POST request with JSON data containing sepal_length, sepal_width,petal_length, petal_width "
    )


class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
async def root():
    return {"message": get_app_description()}


@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
    species = predict_species(
        iris_data.sepal_length,
        iris_data.sepal_width,
        iris_data.petal_length,
        iris_data.petal_width,
    )
    return {"species": species}
