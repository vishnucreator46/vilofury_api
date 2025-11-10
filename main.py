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

# --- Model repo name ---
model_name = "vishnu00l/vilofury-finetuned"

# --- Load model once on startup (CPU mode) ---
print("🚀 Loading Vilofury model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.to("cpu")

    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error during model loading: {e}")
    model = None
    tokenizer = None

@app.get("/")
def home():
    return {"message": "Vilofury API is live 🚀"}

@app.get("/ask")
def ask(q: str, key: str = None):
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialized. Check server logs.")

    try:
        inputs = tokenizer(q, return_tensors="pt").to("cpu")
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.replace(q, "").strip()
        if not reply:
            reply = "I'm not sure how to answer that right now."

        return {"reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    finally:
        gc.collect()
