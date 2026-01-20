# Vertex AI Integration Plan

## Goal
Enable "Heavy" agents (Planner, Extractor, QA) to use Google Vertex AI (Gemini 2.0 Flash) via `ChatVertexAI`.

## User Review Required
> [!IMPORTANT]
> You must authenticate with Google Cloud on your machine.
> Run this command in your terminal: `gcloud auth application-default login`
> And verify your Project ID is set correctly in `src/config.py`.

## Proposed Changes

### 1. Dependencies (`requirements.txt`)
-   Add `langchain-google-vertexai>=0.0.1`.

### 2. Configuration (`src/config.py`)
-   Ensure `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are used.
-   Add `USE_VERTEX` flag (or reuse `USE_HYBRID` implies Vertex if configured/available).

### 3. Registry (`src/core/registry.py`)
-   Update `get_heavy_llm()` to import and return `ChatVertexAI`.
-   Parameters: `project`, `location`, `model_name`.

### 4. Verification
-   Creating `test_vertex_api.py` to verify connectivity.
-   Run `test_vertex_api.py`.

## Verification Plan
1.  **Manual Step**: User runs `gcloud auth application-default login`.
2.  **Automated Test**: `python test_vertex_api.py` sends a "Hello" prompt to Vertex AI.
3.  **Pipeline Test**: Run `run_pipeline.py` (Hybrid) to confirm end-to-end integration.
