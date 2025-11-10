# main.py (Vilofury Proxy API - lightweight version)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, requests

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables (set these in Render dashboard)
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")  # Your own key for protection
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face read token
HF_MODEL = "vishnu00l/vilofury-finetuned"  # Your model name on HF

@app.get("/")
def home():
    return {"message": "Vilofury API (HuggingFace proxy) is live 🚀"}

@app.get("/ask")
def ask(q: str, key: str = None):
    """Forward requests to Hugging Face Inference API."""
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": q},
            timeout=60
        )

        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            reply = data[0]["generated_text"].replace(q, "").strip()
        else:
            reply = str(data)

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
