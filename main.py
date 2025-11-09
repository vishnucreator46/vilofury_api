from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from difflib import SequenceMatcher
import torch
import os
import random
import json

from wikipedia_api import get_wikipedia_summary  # Make sure this file exists

load_dotenv()  # Load environment variables

app = FastAPI()

# ✅ Your API key (must match Render environment variable)
VILOFURY_API_KEY = os.getenv("VILOFURY_API_KEY")

# ✅ Load model from Hugging Face
print("⚙️ Loading Vilofury fine-tuned model from Hugging Face...")
try:
    model_name = "vishnu00l/vilofury-finetuned"  # ✅ corrected model repo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model, tokenizer = None, None


# Helper: Find similarity between strings
def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


@app.get("/")
async def root():
    return {"message": "🚀 Welcome to VILOFURY API — Your Intelligent Assistant"}


@app.get("/ask")
async def ask(request: Request, q: str = "", key: str = None):
    """
    The main endpoint — takes query `q` and optional `key` (for security)
    Example: /ask?q=what%20is%20your%20name&key=YOUR_API_KEY
    """

    # ✅ Key check
    if not key or key != VILOFURY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    # Check if model is loaded
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded on the server.")

    # Handle empty queries
    if not q.strip():
        return JSONResponse({"reply": "Please enter a valid question."})

    # Check for Wikipedia-related queries
    if any(word in q.lower() for word in ["who", "what", "when", "where", "tell me about"]):
        wiki_result = get_wikipedia_summary(q)
        if wiki_result:
            return JSONResponse({"reply": wiki_result})

    # ✅ Generate response using the Vilofury model
    try:
        inputs = tokenizer.encode(q, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=150,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup the response text
        response = response.replace(q, "").strip()
        if not response:
            response = random.choice([
                "I'm still learning to answer that.",
                "Let me think about it...",
                "Can you please rephrase your question?"
            ])

        return JSONResponse({"reply": response})
    except Exception as e:
        print("Error generating reply:", e)
        raise HTTPException(status_code=500, detail="Error generating response.")


# ✅ Custom error handler for debugging
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


# ✅ Run locally (Render runs uvicorn automatically)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

