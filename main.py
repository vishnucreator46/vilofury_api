from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with your website URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Load model ---
try:
    print("⚙️ Loading Vilofury fine-tuned model from Hugging Face...")
    model_name = "vishnu00l/vilofury-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer = None, None


@app.get("/")
def home():
    return {"message": "Vilofury API is live 🚀"}


@app.get("/ask")
def ask(q: str, key: str = None):
    # Verify API key
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    # Tokenize & generate response
    inputs = tokenizer(q, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Cleanup
    reply = reply.replace(q, "").strip()
    if not reply:
        reply = "I'm not sure how to answer that right now."

    return {"reply": reply}
