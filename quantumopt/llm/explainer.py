"""
quantumopt.llm.explainer — Claude API integration for optimization explanations.

Provides plain-English explanations of circuit optimization decisions
and parses user intent for the compilation pipeline.

Uses Claude claude-opus-4-6 with structured prompts designed to produce
output suitable for citation in academic papers.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file if present
load_dotenv()


def _get_client():
    """Get an Anthropic client, or None if API key is not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_key_here":
        logger.info("ANTHROPIC_API_KEY not set — Claude explanations disabled")
        return None

    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("anthropic package not installed")
        return None
    except Exception as e:
        logger.warning(f"Failed to create Anthropic client: {e}")
        return None


def explain_optimization(
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any],
    decisions_made: List[str],
) -> Optional[str]:
    """Generate a plain-English explanation of optimization decisions using Claude.

    Calls Claude API (model: claude-opus-4-6, max_tokens: 400) to produce
    a 3-5 sentence explanation suitable for citation in academic papers.

    Args:
        before_stats: Dict with original circuit stats (depth, gate_count, etc.).
        after_stats: Dict with optimized circuit stats.
        decisions_made: List of optimization actions applied (e.g.,
            ["cancel_redundant_gates", "merge_rotations"]).

    Returns:
        String explanation, or None if ANTHROPIC_API_KEY is not set.
    """
    client = _get_client()
    if client is None:
        return None

    # Build the prompt
    prompt = f"""You are a quantum computing expert writing for a quantum research paper.
Explain the following circuit optimization results in 3-5 sentences. Be precise and
technical enough for citation in an academic paper, but use clear language.

Original Circuit Statistics:
- Depth: {before_stats.get('depth', 'N/A')}
- Total gate count: {before_stats.get('gate_count', 'N/A')}
- Two-qubit gate count: {before_stats.get('two_qubit_gate_count', before_stats.get('cx_count', 'N/A'))}
- Number of qubits: {before_stats.get('num_qubits', 'N/A')}

Optimized Circuit Statistics:
- Depth: {after_stats.get('depth', 'N/A')}
- Total gate count: {after_stats.get('gate_count', 'N/A')}
- Two-qubit gate count: {after_stats.get('two_qubit_gate_count', after_stats.get('cx_count', 'N/A'))}
- Estimated fidelity: {after_stats.get('estimated_fidelity', 'N/A')}

Optimization techniques applied: {', '.join(decisions_made) if decisions_made else 'standard transpilation'}

Write a concise explanation of what was optimized and why these changes improve
circuit execution on IBM Quantum hardware. Focus on the practical impact on
fidelity, decoherence, and execution time."""

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Claude API call failed: {e}")
        return None


def explain_optimization_fallback(
    before_stats: Dict[str, Any],
    after_stats: Dict[str, Any],
    decisions_made: List[str],
) -> str:
    """Generate a template-based explanation when Claude API is unavailable.

    Returns:
        A structured explanation string.
    """
    depth_before = before_stats.get("depth", 0)
    depth_after = after_stats.get("depth", 0)
    gates_before = before_stats.get("gate_count", 0)
    gates_after = after_stats.get("gate_count", 0)
    fidelity = after_stats.get("estimated_fidelity", "N/A")

    depth_reduction = ((depth_before - depth_after) / max(depth_before, 1)) * 100
    gate_reduction = ((gates_before - gates_after) / max(gates_before, 1)) * 100

    techniques = ", ".join(decisions_made) if decisions_made else "standard Qiskit transpilation"

    return (
        f"The circuit was optimized using {techniques}, reducing circuit depth "
        f"from {depth_before} to {depth_after} ({depth_reduction:.1f}% reduction) "
        f"and total gate count from {gates_before} to {gates_after} "
        f"({gate_reduction:.1f}% reduction). "
        f"The estimated circuit fidelity is {fidelity}. "
        f"Reducing circuit depth minimizes decoherence effects, while lowering "
        f"two-qubit gate count directly improves execution fidelity on "
        f"superconducting quantum hardware."
    )


def parse_user_intent(
    user_message: str,
    circuit_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse user intent from a natural language message using Claude.

    Extracts target hardware, optimization priority, and special constraints
    from the user's message.

    Args:
        user_message: Natural language description of what the user wants.
        circuit_info: Dict with info about the circuit being compiled.

    Returns:
        Dict with keys: target_hardware, optimization_priority, special_constraints.
        Returns defaults if Claude API is unavailable.
    """
    defaults = {
        "target_hardware": "ibm_brisbane",
        "optimization_priority": "fidelity",
        "special_constraints": [],
    }

    client = _get_client()
    if client is None:
        return defaults

    prompt = f"""You are a quantum compiler assistant. Parse the following user request
and extract structured parameters.

User message: "{user_message}"

Circuit info: {circuit_info}

Extract and return ONLY a JSON object with these keys:
- "target_hardware": string (IBM backend name, default "ibm_brisbane")
- "optimization_priority": string (one of: "fidelity", "depth", "speed")
- "special_constraints": list of strings (any special requirements mentioned)

Return ONLY the JSON object, no other text."""

    try:
        import json
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Try to parse JSON from response
        result = json.loads(text)
        return {
            "target_hardware": result.get("target_hardware", defaults["target_hardware"]),
            "optimization_priority": result.get("optimization_priority", defaults["optimization_priority"]),
            "special_constraints": result.get("special_constraints", defaults["special_constraints"]),
        }
    except Exception as e:
        logger.warning(f"Intent parsing failed: {e}")
        return defaults
