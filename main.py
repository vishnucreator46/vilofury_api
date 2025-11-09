from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import random
import json
from difflib import SequenceMatcher
from wikipedia_api import get_wikipedia_summary  # your custom module

# Load environment variables
load_dotenv()

# --- ENVIRONMENT VARIABLES ---
VILOFURY_API_KEY = os.getenv("VILOFURY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="ViloFury API", version="2.0")

# --- CORS (allow all origins for frontend access) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
print("⚙️ Loading Vilofury fine-tuned model from Hugging Face...")

try:
    model_name = "vishnu00l/vilofury-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer = None, None


# --- API KEY VALIDATION ---
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Skip the home endpoint
    if request.url.path in ["/", "/docs", "/openapi.json"]:
        return await call_next(request)

    api_key = request.query_params.get("key") or request.headers.get("X-API-Key")

    if not api_key:
        return JSONResponse(status_code=401, content={"error": "Missing API key"})
    if api_key != VILOFURY_API_KEY:
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})

    return await call_next(request)


# --- ROOT ENDPOINT ---
@app.get("/")
def home():
    return {"message": "🚀 Welcome to VILOFURY API — Your Intelligent Assistant"}


# --- ASK ENDPOINT ---
@app.get("/ask")
async def ask(q: str, key: str = None):
    if not q:
        raise HTTPException(status_code=400, detail="Missing query parameter 'q'")

    if not model or not tokenizer:
        return {"error": "Model not loaded on the server."}

    try:
        # Prepare input text
        input_text = f"User: {q}\nViloFury:"
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate model output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.8,
                top_p=0.95,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ViloFury:")[-1].strip()

        # If output empty or nonsensical, fallback to Wikipedia
        if not response or len(response) < 2:
            wiki_summary = get_wikipedia_summary(q)
            if wiki_summary:
                return {"reply": wiki_summary}
            return {"reply": "I'm not sure about that, could you rephrase?"}

        return {"reply": response}

    except Exception as e:
        print(f"⚠️ Error during /ask: {e}")
        return {"error": "Something went wrong. Please try again later."}


# --- EXTRA: SIMPLE TEST ENDPOINT ---
@app.get("/test")
def test():
    return {"status": "ok", "model_loaded": model is not None}
