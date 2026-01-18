# MA-RAG Implementation Plan

## Goal
Implement the Model-Augmented RAG (MA-RAG) architecture to handle complex medical queries through iterative planning and execution.

## Architecture
Reference: `MA-RAG-main` (iterative planner-executor graph).
Target: `src/` (Clean architecture).

### Components
1.  **Planner Agent** (`src/agents/planner.py`):
    -   Deconstructs queries into a list of atomic steps.
    -   Uses `GraphState` history to refine plans.
    -   Powered by LLM (Ollama/Gemini) + Pydantic Parser.

2.  **Executor Graph** (`src/agents/executor.py`):
    -   Nested LangGraph workflow.
    -   Loop: `StepDefiner` <-> `ExecutionNode`.
    -   **StepDefiner** (`src/agents/step_definer.py`): Decides next task or stops.
    -   **ExecutionNode**: Routes to Aggregate logic or RAG.

3.  **RAG Agent** (`src/agents/rag.py`):
    -   Atomic QA using retrieval.
    -   Uses `RetrieverTool` (PubMedClient adapter).

4.  **Orchestrator** (`src/orchestrator/graph.py`):
    -   Main Graph: `Start` -> `Planner` -> `Executor` -> `End`.

## Technical Decisions
-   **LangGraph**: For stateful, cyclic workflows.
-   **PydanticOutputParser**: For robust structured output generation (JSON) compatible with Ollama/Llama 3.
-   **Lazy Imports**: In `Registry` to support flexible dependencies.
-   **State Management**: TypedDict based `GraphState` mirroring the reference implementation.

## Status
-   [x] Analysis of MA-RAG reference.
-   [x] Ported State definitions.
-   [x] Ported Prompt templates.
-   [x] Implemented Planner Agent.
-   [x] Implemented Step Definer.
-   [x] Implemented RAG Agent.
-   [x] Implemented Executor Graph.
-   [x] Implemented Orchestrator Graph.
-   [x] Created `run_pipeline.py` CLI.
-   [ ] Dependency Installation (In Progress).
-   [ ] Verification Run.
