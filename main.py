from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher
import torch
import random
import os
import json
from wikipedia_api import get_wikipedia_summary  # Your custom module

# --- Load environment variables ---
load_dotenv()
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")

# --- Initialize FastAPI ---
app = FastAPI(title="VILOFURY API", version="1.0")

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# --- Load intents ---
print("🧠 Loading intents...")
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents = json.load(f)
    print("✅ Intents loaded successfully.")
except Exception as e:
    print(f"❌ Error loading intents.json: {e}")
    intents = {"intents": []}

# --- Load model from Hugging Face ---
print("⚙️ Loading Vilofury fine-tuned model from Hugging Face...")
try:
    HF_REPO = "vishnucreator46/vilofury-finetuned"  # 👈 your Hugging Face repo
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO)
    model = AutoModelForCausalLM.from_pretrained(HF_REPO)
    model = model.to("cpu")
    print("✅ Model loaded successfully from Hugging Face!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- Middleware for API Key Authentication ---
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/", "/docs", "/openapi.json"]:
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if VILOFURY_KEY and api_key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return await call_next(request)

# --- Root Route ---
@app.get("/")
async def home():
    return {"message": "🚀 Welcome to VILOFURY API — Your Intelligent Assistant"}

# --- Ask Endpoint ---
@app.get("/ask")
async def ask_vilofury(q: str):
    user_input = q.strip()
    if not user_input:
        return {"reply": "I didn’t catch that. Could you say something?"}

    # Step 1: Check intents.json
    best_match = None
    best_score = 0.0
    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            score = SequenceMatcher(None, user_input.lower(), pattern.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = intent

    if best_match and best_score > 0.8:
        return {"reply": random.choice(best_match["responses"])}

    # Step 2: Try Wikipedia summary
    summary = get_wikipedia_summary(user_input)
    if summary:
        return {"reply": summary}

    # Step 3: Use Vilofury fine-tuned model
    if model:
        prompt = f"User: {user_input}\nViloFury:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        full_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = full_reply[len(prompt):].strip()
        return {"reply": reply or "I'm still learning. Could you rephrase that?"}

    return {"reply": "⚠️ Model not loaded. Please check your Hugging Face path."}

