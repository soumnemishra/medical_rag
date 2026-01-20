# MA-RAG Colab Integration

## Quick Setup

### Step 1: Start Colab Server

1. Open [Google Colab](https://colab.research.google.com)
2. Create new notebook with **GPU runtime** (Runtime → Change runtime type → T4 GPU)
3. Copy each cell from `ma_rag_colab_server.py` into Colab
4. Run cells 1-6 in order
5. Copy the ngrok URL from the output

### Step 2: Configure Local Pipeline

Edit your `.env` file:
```env
USE_COLAB=true
COLAB_API_URL=https://xxxx-xx-xx-xxx-xx.ngrok-free.app
```

Or set in code:
```python
from src.config import settings
settings.USE_COLAB = True
settings.COLAB_API_URL = "https://your-ngrok-url.ngrok-free.app"
```

### Step 3: Run Pipeline

```bash
python run_pipeline.py
```

Your local pipeline will now use Colab's T4 GPU for:
- LLM inference (Planner, Extractor, QA, Summarizer)
- Embeddings (dense retrieval)
- Reranking

## Prerequisites

1. **ngrok account** (free): https://ngrok.com
2. **HuggingFace account** (for Llama models): https://huggingface.co
3. Request access to `meta-llama/Llama-3.2-3B-Instruct`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection timeout | Check if Colab cell 6 is running |
| ngrok URL changed | Copy new URL to config |
| Model not loading | Check HuggingFace login in Cell 1 |
| GPU OOM | Restart runtime, try smaller model |
