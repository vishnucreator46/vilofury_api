# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os, gc

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model name (INT8 version) ---
model_name = "vishnu00l/vilofury-finetuned-int8"

# --- Load model and tokenizer once at startup ---
print("Loading model and tokenizer... This may take a while.")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN,
    load_in_8bit=True,
    device_map="auto",
)
print("Model loaded successfully!")

@app.get("/")
def home():
    return {"message": "Vilofury API is live 🚀"}

@app.get("/ask")
def ask(q: str, key: str = None):
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        # Tokenize and move inputs to model device
        inputs = tokenizer(q, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=50)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.replace(q, "").strip()

        if not reply:
            reply = "I'm not sure how to answer that right now."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

    finally:
        # Optional memory cleanup
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"reply": reply}
