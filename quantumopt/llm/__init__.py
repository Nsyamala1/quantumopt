"""
quantumopt.llm — LLM integration module.

Uses Claude API (Anthropic) to generate plain-English explanations
of optimization decisions and parse user intent.
"""

from quantumopt.llm.explainer import explain_optimization, parse_user_intent

__all__ = ["explain_optimization", "parse_user_intent"]
