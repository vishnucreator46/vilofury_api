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
# NOTE: Ensure these variables are set in your Render environment settings.
VILOFURY_KEY = os.getenv("VILOFURY_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model repo name ---
model_name = "vishnu00l/vilofury-finetuned"

# --- Load model once on startup (CPU only, no quantization) ---
print("🚀 Loading Vilofury model...")
try:
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    
    # Load Model - The fix is adding device_map="cpu" to bypass 8-bit loading logic.
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.to("cpu")

    
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error during model loading: {e}")
    # In a real application, you might want to raise this error to prevent startup
    # raise e

@app.get("/")
def home():
    """Returns a simple status message."""
    return {"message": "Vilofury API is live 🚀"}


@app.get("/ask")
def ask(q: str, key: str = None):
    """
    Accepts a query 'q' and an API key 'key' to generate a response.
    """
    # --- API key check ---
    if key != VILOFURY_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    try:
        if 'model' not in locals() and 'model' not in globals():
            # Basic check if model failed to load at startup
            raise Exception("Model not initialized. Check server logs.")

        # --- Generate reply ---
        # NOTE: Using a simple .to("cpu") for inputs is generally fine
        inputs = tokenizer(q, return_tensors="pt").to("cpu")
        
        # Increase max_new_tokens for potentially better responses if desired
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            do_sample=True, # For more creative answers
            temperature=0.7 # A good balance of randomness and coherence
        )
        
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Basic cleaning of the reply
        reply = reply.replace(q, "").strip()
        if not reply:
            reply = "I'm not sure how to answer that right now."

        return {"reply": reply}

    except Exception as e:
        # Catch and report any errors during generation
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    finally:
        # Clean up memory after generation
        gc.collect()

