from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os
import json
import torch
import random
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer
from wikipedia_api import get_wikipedia_summary # Using your custom wikipedia script

load_dotenv()
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")

app = FastAPI(title="VILOFURY API", version="1.0")

# --- Define the path to the model assets folder ---
script_dir = os.path.dirname(os.path.realpath(__file__))
ASSETS_DIR = os.path.join(script_dir, "model_assets")

# --- Define absolute paths for all required files ---
LABEL_ENCODER_PATH = os.path.join(ASSETS_DIR, "vilofury_label_encoder.pkl")
MODEL_WEIGHTS_PATH = os.path.join(ASSETS_DIR, "vilofury_model_weights.pth")
INTENTS_PATH = os.path.join(script_dir, "intents.json")

# --- Load intents.json for pattern matching ---
print("🧠 Loading intents for pattern matching...")
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as file:
        intents = json.load(file)
except FileNotFoundError:
    print("❌ ERROR: intents.json not found.")
    exit()
print("✅ Intents loaded.")

# --- Load Vilofury conversational model ---
print("🧠 Loading Vilofury conversational model...")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
convo_model_name = os.path.join(script_dir, "vilofury_finetuned")
convo_tokenizer = AutoTokenizer.from_pretrained(convo_model_name)
convo_model = AutoModelForCausalLM.from_pretrained(convo_model_name).to("cpu")
print("✅ Vilofury model loaded successfully!")


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Allow open access for docs and home routes
    if request.url.path in ["/", "/docs", "/openapi.json"]:
        return await call_next(request)

    api_key = request.headers.get("x-api-key")
    if api_key != VILOFURY_KEY and request.client.host != "127.0.0.1":
        raise HTTPException(status_code=401, detail="Invalid API key")
    response = await call_next(request)
    return response

@app.get("/")
async def home():
    return {"message": "Welcome to VILOFURY API!"}

@app.get("/ask")
async def ask_vilofury(q: str):
    user_input = q.strip()
    if not user_input:
        return {"reply": "I didn’t catch that. Could you say something?"}

    # Step 1: Try matching with intents.json
    best_match = None
    best_score = 0.0
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
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

    # Step 3: Use fine-tuned conversational model as fallback
    prompt = f"User: {user_input}\nViloFury:"
    inputs = convo_tokenizer(prompt, return_tensors="pt")
    outputs = convo_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=convo_tokenizer.eos_token_id
    )
    full_reply = convo_tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = full_reply[len(prompt):].strip()

    return {"reply": reply or "I'm still learning, could you rephrase that?"}


