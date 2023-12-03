import tensorflow as tf
from tensorflow import keras
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import cv2
import torch
print("Loading model....")
model = torch.load("misc/")
print(model.summary())

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")


@app.get("/home")
async def upload():
    return RedirectResponse("/static/index.html")


@app.post("/home")
async def upload(file: UploadFile = File(...)):
    # Read the image using OpenCV
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the uploaded photo
    processed_image = preprocess(image)

    # Make predictions using the ML model
    predictions = await make_predictions(processed_image)

    return {"predictions": predictions}


# Preprocess the image (placeholder function)
def preprocess(image):
    # Perform necessary pre-processing steps on the image
    # e.g., resizing, normalizing, etc.
    return image


# Make predictions using the ML model
async def make_predictions(photo):
    # Use the ML model to make predictions on the photo
    predictions = model.predict(np.expand_dims(photo, axis=0))
    # Adjust the above line based on the input shape expected by the model

    return predictions.tolist()


# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
