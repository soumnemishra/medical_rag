# MA-RAG Evaluation Plan

## Goal
Evaluate the correctness and performance of the MA-RAG pipeline using the **PubMedQA** benchmark.

## Methodology

### 1. Dataset
-   **Source**: PubMedQA (`pqa_labeled` split) via HuggingFace `datasets` (or local file if available).
-   **Size**: Pilot run on **20 RANDOM SAMPLES** (due to local inference constraints). Full run can be triggered later.
-   **Fields**: `question`, `context` (abstracts), `final_decision` (yes/no/maybe), `long_answer`.

### 2. Execution Pipeline
-   **Input**: Feed `question` to `src/orchestrator/graph.py`.
-   **Configuration**: Use `llama3.2:3b` (Ollama) with `json_mode`.
-   **Output Capture**:
    -   `final_answer` (Text)
    -   `trace` (Plan steps)

### 3. Metrics
-   **Accuracy (Exact/Loose)**: Map generated answer to {yes, no, maybe} using a lightweight "Judge" LLM call. match against ground truth `final_decision`.
-   **Latency**: Time per query.
-   **Step Count**: Average number of steps taken.

## Implementation Steps

### Phase 1: Setup
1.  [ ] Install `datasets`, `scikit-learn` (for metrics).
2.  [ ] Create `scripts/evaluate_pipeline.py`.

### Phase 2: Evaluation Script Logic
1.  Load Dataset.
2.  Initialize Graph (Ollama).
3.  Loop through samples:
    -   Run Pipeline.
    -   Save Result (JSONL).
    -   Compare Result (Judge).
4.  Print Report.

### Phase 3: Reports
-   Generate `evaluation_report.md` with:
    -   Overall Accuracy.
    -   Failure Analysis (which steps failed).
    -   Example Traces.

## Risks & Mitigations
-   **Slowness**: Limit to 20 samples.
-   **Parsing Failures**: The pipeline is now robust, but if it fails, treat as "Incorrect".
-   **Context Limit**: PubMedQA provides contexts (abstracts).
    -   *Crucial Decision*: **MA-RAG is a retrieval system**. Should we give it the context provided by PubMedQA (reading comprehension) OR force it to search PubMed (open-domain)?
    -   **Decision**: **Open-Domain**. The user asked to "evaluate this pipeline". The pipeline HAS a Retriever (`PubMedClient`). Integrating provided contexts bypasses the retriever.
    -   *However*: Comparing Open-Domain retrieval generation against Closed-Domain Ground Truth (based on specific abstracts) is tricky. The retriever might find *different* but *correct* papers.
    -   *Refined Decision*: Evaluate in **Open-Domain** mode (Real-world scenario). We will judge correctness of the answer factually, not just strict label matching if the evidence differs. BUT for "benchmark" comparability, matching the Label is standard. We will use the Label as the "Gold Standard" and assume PubMed retrieval *should* find the relevant papers.

## User Approval Required
-   **Dataset**: Confirm using PubMedQA `pqa_labeled`?
-   **Mode**: Confirm Open-Domain (using our Retriever) vs Closed-Domain (feeding abstracts)? I propose **Open-Domain** to test the full Agentic Pipeline.
