from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

MODEL_NAME = "kkkkkjjjjjj/credit-card-complaint-classifier"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

label_map = {
    0: "billing",
    1: "customer_service",
    2: "features",
    3: "fraud",
    4: "trouble_using_card"
}

class ComplaintInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Credit Card Complaint Classifier API is live!"}

@app.post("/predict")
def predict(data: ComplaintInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    predicted_label = label_map.get(predicted_class_id, "unknown")

    return {"prediction": predicted_label}
