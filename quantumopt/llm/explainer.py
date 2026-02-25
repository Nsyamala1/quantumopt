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
    before_stats: dict,
    after_stats: dict, 
    decisions: list
) -> str:
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return None  # triggers fallback template
    
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    depth_change = before_stats['depth'] - after_stats['depth']
    gate_change = before_stats['gate_count'] - after_stats['gate_count']
    fidelity = after_stats.get('estimated_fidelity', 0)
    
    prompt = f"""You are an expert quantum compiler engineer 
writing for a quantum physics research audience.

A quantum circuit was compiled with these results:

BEFORE compilation:
- Circuit depth: {before_stats['depth']}
- Total gates: {before_stats['gate_count']}

AFTER optimization for IBM Brisbane hardware:
- Circuit depth: {after_stats['depth']} 
  ({abs(depth_change)} {'fewer' if depth_change > 0 else 'more'} layers)
- Total gates: {after_stats['gate_count']}
  ({abs(gate_change)} {'fewer' if gate_change > 0 else 'more'} gates)
- Estimated fidelity: {fidelity:.4f}

Optimization techniques applied: {', '.join(decisions)}

Write exactly 3 sentences explaining:
1. What was optimized and by how much
2. Which technique had the most impact and why
3. How this improves execution on real IBM hardware

Requirements:
- Technical language appropriate for a research paper
- Mention specific gate counts and depth numbers
- Do not start with "The circuit was optimized"
- Write as if this will be cited in a paper
- Be specific not generic
"""
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        logger.warning(f"Claude API call failed: {e}")
        return None  # triggers fallback template


def explain_optimization_fallback(
    before_stats: dict,
    after_stats: dict,
    decisions: list
) -> str:
    depth_reduction = before_stats['depth'] - after_stats['depth']
    gate_reduction = before_stats['gate_count'] - after_stats['gate_count']
    fidelity = after_stats.get('estimated_fidelity', 0)
    
    return (
        f"Circuit compilation reduced depth by {depth_reduction} "
        f"layers and eliminated {gate_reduction} gates through "
        f"{', '.join(decisions)}. "
        f"The optimized circuit targets IBM Brisbane native gate set "
        f"with estimated fidelity of {fidelity:.4f}. "
        f"Reduced gate count directly lowers accumulated error rates "
        f"on NISQ hardware."
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
            max_tokens=300,
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
