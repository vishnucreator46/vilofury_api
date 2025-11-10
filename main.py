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


@app.get("/")
def home():
    return {"message": "Vilofury API is live 🚀"}


@app.get("/ask")
def ask(q: str, key: str = None):
    # --- API key verification ---
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        # --- Load model and tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # Use CPU-safe precision
            low_cpu_mem_usage=True
        ).to("cpu")

        # --- Generate reply ---
        inputs = tokenizer(q, return_tensors="pt").to("cpu")
        outputs = model.generate(**inputs, max_new_tokens=100)
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean the reply
        reply = reply.replace(q, "").strip()
        if not reply:
            reply = "I'm not sure how to answer that right now."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    finally:
        # --- Free memory safely ---
        for var in ["model", "tokenizer", "inputs", "outputs"]:
            if var in locals():
                del locals()[var]
        gc.collect()

    return {"reply": reply}
