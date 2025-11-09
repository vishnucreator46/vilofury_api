from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher
import torch
import random
import os
import json
from wikipedia_api import get_wikipedia_summary  # your custom module

# Load environment variables
load_dotenv()
VILOFURY_API_KEY = os.getenv("VILOFURY_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="ViloFury API", version="1.0")

# ================================
# Intent Data (local fallback)
# ================================
def load_intents():
    try:
        with open("intents.json", "r", encoding="utf-8") as file:
            data = json.load(file)
            print("🧠 Intents loaded successfully.")
            return data
    except Exception as e:
        print("⚠️ Error loading intents:", e)
        return {"intents": []}

intents = load_intents()

# ================================
# Load fine-tuned Hugging Face model
# ================================
print("⚙️ Loading Vilofury fine-tuned model from Hugging Face...")

try:
    model_name = "vishnucreator46/vilofury-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)
    print("✅ Model loaded successfully from Hugging Face!")
except Exception as e:
    print("❌ Error loading model:", e)
    tokenizer, model = None, None

# ================================
# Helper: match user input to intent
# ================================
def get_intent_response(text):
    best_match = None
    highest_ratio = 0.0

    for intent in intents.get("intents", []):
        for pattern in intent.get("patterns", []):
            ratio = SequenceMatcher(None, text.lower(), pattern.lower()).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = intent

    if best_match and highest_ratio > 0.7:
        return random.choice(best_match.get("responses", []))
    return None

# ================================
# Root endpoint
# ================================
@app.get("/")
def root():
    return {"message": "Welcome to the ViloFury API 🚀"}

# ================================
# /ask endpoint (Main)
# ================================
@app.get("/ask")
async def ask(q: str, key: str, request: Request):
    # Verify API key
    if key != VILOFURY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key.")

    user_input = q.strip()

    # 1️⃣ Check local intent-based response
    intent_response = get_intent_response(user_input)
    if intent_response:
        return {"reply": intent_response}

    # 2️⃣ Try Wikipedia summary if no intent matches
    wiki_summary = get_wikipedia_summary(user_input)
    if wiki_summary:
        return {"reply": wiki_summary}

    # 3️⃣ Use AI model if available
    if model and tokenizer:
        try:
            inputs = tokenizer.encode(user_input, return_tensors="pt")
            outputs = model.generate(inputs, max_length=80, num_return_sequences=1)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"reply": reply}
        except Exception as e:
            print("⚠️ Model inference error:", e)

    # 4️⃣ Default fallback
    return {"reply": "They call me ViloFury in these digital streets!"}
