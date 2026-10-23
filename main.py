from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Tazemba AI Engine")

# Load model weights
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "Tazemba AI Engine is Live 24/7"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    res = model(img)[0]
    
    detections = [{
        "class": model.names[int(b.cls)],
        "conf": round(float(b.conf), 2),
        "box": [round(float(x), 2) for x in b.xyxy[0]]
    } for b in res.boxes]
    
    return {"filename": file.filename, "detections": detections}
