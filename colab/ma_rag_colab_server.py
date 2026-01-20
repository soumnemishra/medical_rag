# =============================================================================
# MA-RAG COLAB SERVER
# =============================================================================
# This file contains all the code for your Google Colab notebook.
# 
# SETUP INSTRUCTIONS:
# 1. Go to https://colab.research.google.com
# 2. Create a new notebook
# 3. Copy each CELL section below into separate Colab cells
# 4. Run cells in order (Cell 1 → Cell 6)
# 5. Copy the ngrok URL to your local config.py
#
# FREE TIER: Works with T4 GPU (default on Colab)
# =============================================================================

# =============================================================================
# CELL 1: Check GPU and Install Dependencies
# =============================================================================
# Copy everything between the triple quotes into your first Colab cell

CELL_1 = '''
# Check GPU
!nvidia-smi

# Install packages
!pip install -q transformers accelerate bitsandbytes
!pip install -q sentence-transformers
!pip install -q flask pyngrok
!pip install -q huggingface_hub

# Login to HuggingFace (needed for Llama models)
from huggingface_hub import login
login()  # Enter your HF token when prompted
'''

# =============================================================================
# CELL 2: Configure ngrok
# =============================================================================

CELL_2 = '''
from pyngrok import ngrok

# Get free token from: https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # <-- REPLACE THIS!

ngrok.set_auth_token(NGROK_TOKEN)
print("✅ ngrok configured!")
'''

# =============================================================================
# CELL 3: Load Models (Takes ~2-3 minutes)
# =============================================================================

CELL_3 = '''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import gc

# Clear memory
gc.collect()
torch.cuda.empty_cache()

print("🔄 Loading LLM...")

# Model choice - Llama 3.2 3B works on free T4
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
print(f"✅ LLM loaded: {MODEL_ID}")

# Load embedding model
print("🔄 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
print("✅ Embedder loaded")

# Load reranker
print("🔄 Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")
print("✅ Reranker loaded")

print(f"\\n🎉 All models ready! GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
'''

# =============================================================================
# CELL 4: Define Generation Function
# =============================================================================

CELL_4 = '''
def generate_chat_response(messages, max_tokens=512, temperature=0.1):
    """Generate response from chat messages (OpenAI-compatible format)."""
    
    # Build prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n{content}<|eot_id|>")
        elif role == "user":
            prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\\n{content}<|eot_id|>")
        elif role == "assistant":
            prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\\n{content}<|eot_id|>")
    
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\\n")
    prompt = "".join(prompt_parts)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Quick test
test_response = generate_chat_response([{"role": "user", "content": "Say hello in 5 words."}])
print(f"Test response: {test_response}")
'''

# =============================================================================
# CELL 5: Create Flask API Server
# =============================================================================

CELL_5 = '''
from flask import Flask, request, jsonify
import time
import uuid

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu": torch.cuda.is_available()})

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 512)
        temperature = data.get("temperature", 0.1)
        
        # Generate response
        start = time.time()
        response_text = generate_chat_response(messages, max_tokens, temperature)
        elapsed = time.time() - start
        
        # Format as OpenAI response
        return jsonify({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(response_text.split()),
                "total_tokens": 0
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/embeddings", methods=["POST"])
def embeddings():
    """Generate embeddings for texts."""
    try:
        data = request.json
        texts = data.get("input", [])
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        vectors = embedder.encode(texts, convert_to_numpy=True).tolist()
        
        return jsonify({
            "object": "list",
            "data": [{"object": "embedding", "index": i, "embedding": vec} 
                     for i, vec in enumerate(vectors)],
            "model": "all-MiniLM-L6-v2"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/v1/rerank", methods=["POST"])
def rerank():
    """Rerank documents by relevance to query."""
    try:
        data = request.json
        query = data.get("query", "")
        documents = data.get("documents", [])
        top_k = data.get("top_k", 10)
        
        # Create pairs and score
        pairs = [[query, doc] for doc in documents]
        scores = reranker.predict(pairs).tolist()
        
        # Sort by score
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        return jsonify({
            "results": [{"index": idx, "score": score} for idx, score in ranked[:top_k]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("✅ Flask API defined!")
'''

# =============================================================================
# CELL 6: Start Server with ngrok (RUN THIS LAST!)
# =============================================================================

CELL_6 = '''
from pyngrok import ngrok
import threading

# Start ngrok tunnel
public_url = ngrok.connect(5000)
print("=" * 60)
print("🚀 MA-RAG COLAB SERVER IS RUNNING!")
print("=" * 60)
print(f"\\n📡 PUBLIC URL: {public_url}")
print(f"\\n👆 Copy this URL to your local config.py:")
print(f'   COLAB_API_URL = "{public_url}"')
print("=" * 60)
print("\\nEndpoints available:")
print(f"  POST {public_url}/v1/chat/completions  - LLM inference")
print(f"  POST {public_url}/v1/embeddings        - Text embeddings")
print(f"  POST {public_url}/v1/rerank            - Document reranking")
print(f"  GET  {public_url}/health               - Health check")
print("=" * 60)

# Run Flask (blocking)
app.run(port=5000)
'''

# =============================================================================
# QUICK REFERENCE: All cells in order
# =============================================================================
if __name__ == "__main__":
    print("MA-RAG Colab Server Setup")
    print("=" * 50)
    print("Copy each CELL_X variable's content into separate Colab cells.")
    print("Run them in order: CELL_1 → CELL_2 → CELL_3 → CELL_4 → CELL_5 → CELL_6")
    print("\nAfter CELL_6, you'll get a public URL like:")
    print("  https://xxxx-xx-xx-xxx-xx.ngrok-free.app")
    print("\nAdd this to your local config.py to use Colab's GPU!")
