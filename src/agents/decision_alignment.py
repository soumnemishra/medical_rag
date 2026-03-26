# what the important of the this file is that it checks 
'''
--> evidence section says that the 5 evidence collect state that the treatment works
-->but the conclusion says that system doesnot works 
-->your conlcusion contradicts your evidence 
--> that reviewner is the decision alignment node 
'''


import logging
import re
from typing import Dict, Any

from src.state.state import GraphState

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Answer extraction patterns                                         #
# ------------------------------------------------------------------ #
# These regex patterns extract the final yes/no/maybe decision from
# the summariser's answer text.
#
# Why regex over LLM: this is a deterministic structural check, not
# semantic reasoning. Regex is fast, free, and testable.
#
# Ordered from most specific to least specific — re.findall returns
# ALL matches; we take the LAST one (the final decision in the text).

ANSWER_PATTERNS = [
    r"\*\*final\s+answer:\s*(yes|no|maybe)\*\*",
    r"final\s+answer:\s*\*?\*?(yes|no|maybe)\*?\*?",
    r"(?:final\s+)?answer\s*(?:is|:)\s*[\"']?\b(yes|no|maybe)\b[\"']?",
    r"(?:^|\n)\s*\**\s*(yes|no|maybe)\s*\**\s*(?:$|\n)",
]

# Polarity → yes/no/maybe mapping used by forced resolution
POLARITY_TO_ANSWER = {
    "support":      "yes",
    "refute":       "no",
    "mixed":        "maybe",
    "insufficient": None,   # no forced answer for insufficient evidence
}


def _get_current_answer(text: str) -> str:
    """
    Extract the final yes/no/maybe decision from an answer string.

    Takes the LAST match so that if the summariser writes multiple
    candidate answers and then a final decision, we get the final one.

    Returns "unknown" if no standard answer tag is found.
    """
    if not text:
        return "unknown"

    text_lower = text.lower()
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].lower()

    return "unknown"


# ------------------------------------------------------------------ #
#  Node                                                               #
# ------------------------------------------------------------------ #

async def decision_alignment_node(state: GraphState) -> Dict[str, Any]:
    """
    Enforces consistency between Evidence Polarity and the Final Answer.

    The core rule: never make an answer MORE confident than the evidence
    supports. Only DOWNGRADE (to "maybe"), never upgrade.

    Example:
        Polarity = "refute" (confidence 0.85) + Answer = "yes"
        → override to "maybe"  (evidence contradicts the answer)

        Polarity = "support" (confidence 0.80) + Answer = "yes"
        → no change needed

    This node MUST run AFTER the summariser writes final_answer to state.
    If final_answer is empty, it exits early with a visible warning so
    graph.py ordering problems are immediately detectable in logs.

    Eval mode forced resolution:
        When evaluation_mode=True AND retry_count >= 1 AND confidence >= 0.6,
        forces a binary answer that matches the polarity. This ensures
        PubMedQA benchmark queries always get a scoreable yes/no/maybe
        rather than a hedged non-answer. Every override is recorded in
        safety_flags for full audit traceability.

    Returns:
        {} if no alignment change needed (most queries)
        {"final_answer": ..., "safety_flags": ...} when override applied
    """
    final_answer  = state.get("final_answer", "")
    polarity_data = state.get("evidence_polarity", {})
    safety_flags  = state.get("safety_flags", [])

    # Guard: must run after summariser
    if not final_answer:
        logger.warning(
            "decision_alignment_node: final_answer is empty. "
            "This node must run AFTER the summariser. "
            "Check node ordering in graph.py."
        )
        return {}

    polarity   = polarity_data.get("polarity",   "insufficient") \
                 if isinstance(polarity_data, dict) else "insufficient"
    confidence = float(polarity_data.get("confidence", 0.0)) \
                 if isinstance(polarity_data, dict) else 0.0

    current_answer = _get_current_answer(final_answer)

    logger.info(
        f"Decision alignment | "
        f"answer='{current_answer}' polarity='{polarity}' "
        f"confidence={confidence:.2f}"
    )

    # ── Eval mode forced resolution ────────────────────────────────
    # Converts hedged answers to binary for benchmark scoring.
    # ONLY fires when: evaluation_mode=True AND retries exhausted AND
    # evidence confidence is meaningful.
    is_eval_mode = state.get("evaluation_mode", False)
    retry_count  = state.get("retry_count", 0)

    if (
        is_eval_mode
        and retry_count >= 1
        and polarity != "insufficient"
        and confidence >= 0.6
    ):
        forced_answer = POLARITY_TO_ANSWER.get(polarity)

        if forced_answer and forced_answer != current_answer:
            reason = (
                f"Forced resolution (eval mode): "
                f"polarity={polarity} confidence={confidence:.2f} "
                f"→ overriding '{current_answer}' to '{forced_answer}'"
            )
            logger.warning(reason)

            correction = (
                f"\n\n**Decision Alignment Override (Eval Policy)**\n"
                f"{reason}\n"
                f"**Final Answer: {forced_answer}**"
            )

            return {
                "final_answer": final_answer + correction,
                # Every override is recorded — safety critic and evaluator
                # can see this happened and weight results accordingly
                "safety_flags": safety_flags + ["forced_resolution_applied"],
            }

    # ── Standard alignment rules ───────────────────────────────────
    # Never upgrade confidence. Only downgrade to "maybe" when the
    # evidence direction contradicts the stated answer.

    new_decision = None
    reason       = None

    # Rule 1: Strong supporting evidence + answer says "no" → downgrade
    if polarity == "support" and confidence >= 0.75:
        if current_answer == "no":
            new_decision = "maybe"
            reason = (
                f"Evidence strongly supports the claim "
                f"(confidence={confidence:.2f}) but answer was 'no'. "
                f"Downgraded to 'maybe'."
            )

    # Rule 2: Strong refuting evidence + answer says "yes" → downgrade
    elif polarity == "refute" and confidence >= 0.75:
        if current_answer == "yes":
            new_decision = "maybe"
            reason = (
                f"Evidence refutes the claim "
                f"(confidence={confidence:.2f}) but answer was 'yes'. "
                f"Downgraded to 'maybe'."
            )

    # Rule 3: Mixed or insufficient evidence + binary answer → downgrade
    elif polarity in ("mixed", "insufficient"):
        if current_answer in ("yes", "no"):
            new_decision = "maybe"
            reason = (
                f"Evidence is {polarity} (confidence={confidence:.2f}). "
                f"Binary certainty not justified. "
                f"Downgraded to 'maybe'."
            )

    if new_decision:
        logger.warning(f"Alignment override: {reason}")

        correction = (
            f"\n\n**Decision Alignment Override**\n"
            f"{reason}\n"
            f"**Final Answer: {new_decision}**"
        )

        return {
            "final_answer": final_answer + correction,
            "safety_flags": safety_flags + ["alignment_override_applied"],
        }

    # No alignment change needed
    logger.info("Decision alignment: no override needed.")
    return {}