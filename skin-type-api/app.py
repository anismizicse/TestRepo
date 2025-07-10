import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model import load_model, predict

app = FastAPI(title="Skin Type Classifier API")

# Load model and labels once at startup
model, class_names = load_model()

@app.get("/")
async def root():
    return {"message": "Skin Type Classifier API is running!"}

@app.post("/predict")
async def predict_skin_type(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(status_code=400, content={"error": "Invalid file type. Upload JPEG or PNG."})
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = predict(model, image, class_names)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "classes": class_names}
