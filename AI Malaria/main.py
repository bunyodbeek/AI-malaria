from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

model = load_model("my_model.h5")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class mapping (0: parasitized, 1: uninfected) based on TFDS 'malaria'
class_map = {0: "Parasitized", 1: "Uninfected"}


@app.get("/")
async def root():
    return {"message": "Malaria cell classification API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        prediction = model.predict(image)[0][0]
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = float(prediction) if predicted_class == 1 else 1 - float(prediction)

        return JSONResponse(content={
            "prediction": class_map[predicted_class],
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
