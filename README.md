# 🏥 Medical RAG Chatbot

A Streamlit-based chat application for medical doctors to query PubMed literature using real-time search, powered by **Pydantic AI** and **Google Gemini**.

## Features

- 🔬 **Real-time PubMed Search** - Queries NCBI E-utilities API for latest medical literature
- 🤖 **Gemini-powered Responses** - Uses Google's Gemini model for intelligent answers
- 📚 **Citation Tracking** - Automatically cites PMID sources
- 💬 **Conversational Interface** - Natural chat-based interaction
- 🎨 **Premium Dark UI** - Modern, professional medical theme

## Project Structure

```
RAG_CHAT_BOT_MRAGE/
├── app.py                    # Streamlit main application
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── pubmed_client.py      # PubMed E-utilities client
│   └── agent.py              # Pydantic AI medical agent
├── tests/
│   ├── __init__.py
│   └── test_pubmed_client.py # Unit tests
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Set Up Google Cloud

```bash
# Authenticate with GCP
gcloud auth application-default login

# Or set GOOGLE_API_KEY environment variable for Gemini API
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `PUBMED_API_KEY` | NCBI API key (optional) | - |
| `MAX_SEARCH_RESULTS` | Max PubMed results | `10` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Running Tests

```bash
pytest tests/ -v --cov=src
```

## Architecture

```
User Query → Streamlit UI → Pydantic AI Agent → PubMed Search
                                    ↓
                              Gemini LLM
                                    ↓
                         Evidence-based Response
```

## Disclaimer

This tool provides information for educational purposes only. Always verify information and use clinical judgment. Not a substitute for professional medical advice.

## License

MIT
