# Hybrid Model Implementation Plan

## Goal
Implement the "Heavy/Light" model strategy from the MA-RAG paper using the user's available resources:
-   **Heavy Model**: Gemini (via API) -> Planner, Extractor, QA.
-   **Light Model**: Llama 3.2 (Local Ollama) -> Step Definer.

## User Review Required
> [!IMPORTANT]
> Ensure `GOOGLE_API_KEY` is set in your environment variables or `.env` file.

## Proposed Changes

### 1. Update `src/core/registry.py`
-   Modify `ModelRegistry` to support tiered access.
-   `get_heavy_llm()`: Returns `ChatGoogleGenerativeAI` (Gemini Pro).
-   `get_light_llm()`: Returns `ChatOllama` (Llama 3.2).
-   Update `get_llm()` to warn or deprecate, or map to `get_light_llm` by default.

### 2. Update `src/config.py`
-   Ensure `USE_HYBRID` flag (optional) or just rely on explicit calls.
-   Confirm `GOOGLE_API_KEY` loading.

### 3. Refactor Agents
-   **Planner (`src/agents/planner.py`)**: Call `ModelRegistry.get_heavy_llm()`.
-   **Extractor (`src/agents/extractor.py`)**: Call `ModelRegistry.get_heavy_llm()`.
-   **RagAgent (`src/agents/rag.py`)**: Call `ModelRegistry.get_heavy_llm()` (QA is heavy).
-   **StepDefiner (`src/agents/step_definer.py`)**: Call `ModelRegistry.get_light_llm()`.
-   **Executor (`src/agents/executor.py`)**: Aggregation -> `get_heavy_llm()`.

### 4. Verification
-   Run `run_pipeline.py`.
-   Observed behavior: Planning and QA should use Gemini (higher quality), Step Definition uses Ollama (fast).
