# Extraction Agent Implementation Plan

## Goal
Implement the **Extractor Agent** as described in the MA-RAG paper to filter retrieved documents into concise "notes". This solves the "Lost-in-the-Middle" phenomenon and improves QA accuracy.

## User Review Required
> [!IMPORTANT]
> This changes the execution pipeline from `Retrieve -> QA` to `Retrieve -> Extract -> QA`. This will increase latency (one extra LLM call per step) but significantly improve accuracy.

## Proposed Changes

### 1. New Agent: `src/agents/extractor.py`
-   **Class**: `ExtractorAgent`.
-   **Input**: `step_question` (query), `retrieved_docs` (raw content).
-   **Output**: `extracted_notes` (concise relevant facts).
-   **Model**: Use `llama3.2:3b` with `json_mode`.
-   **Prompt**: "Extract only sentences relevant to the query. If no relevance, return empty."

### 2. Update Executor: `src/agents/executor.py`
-   **Current Flow**: `task_define` -> `single_task_execute` (calls Retriever -> RagAgent).
-   **New Flow**: 
    1.  `task_define` (step i)
    2.  `retrieve_node` (fetch docs)
    3.  `extractor_node` (docs -> notes)
    4.  `qa_node` (notes -> answer)
-   *Refactoring*: Split `single_task_execute` into granular nodes or chain them within the node. Chaining within `single_task_execute` is simpler for now to maintain graph structure.

### 3. State Update: `src/state/state.py`
-   Ensure `PlanExecState` tracks `step_notes` (already defined in TypedDict, just need to populate it).

### 4. Verification
-   Run `run_pipeline.py` to verify flow.
-   Run `run_evaluation.py` (Pilot) to check if "Unknown" count decreases.

## Verification Plan
1.  **Unit Test**: Test `ExtractorAgent` with a dummy long document and specific query.
2.  **Pipeline Test**: Verify `step_notes` are populated in the logs.
3.  **Benchmarking**: Re-run PubMedQA Pilot.
