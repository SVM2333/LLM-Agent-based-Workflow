"""Agent 2: Formula Converter"""
from typing import Dict, Any, Iterator
from agents.base_agent import BaseAgent


class FormulaConverterAgent(BaseAgent):
    """Formula Converter - Convert structured sequences to mathematical formulas"""

    def __init__(self):
        super().__init__("agent2_formula_converter")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute formula conversion

        Args:
            input_data: Structured sequence (JSON)

        Returns:
            Dictionary containing mathematical formulas
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary type")

        # Convert structured data to string input
        user_input = self._format_input(input_data)

        # Call LLM
        response = self.call_llm(user_input, temperature=0.5)

        # Validate output
        if not self.validate_output(response):
            raise ValueError("Output validation failed")

        return {
            "success": True,
            "output": response,
            "input_summary": self._get_summary(input_data)
        }

    def execute_stream(self, input_data: Any, examples: str = "") -> Iterator[str]:
        """
        Streaming formula conversion

        Args:
            input_data: Structured sequence (JSON)

        Yields:
            Each chunk of the LLM response
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary type")

        user_input = self._format_input(input_data)

        # Streaming LLM call
        for chunk in self.call_llm_stream(user_input, temperature=0.5, examples=examples):
            yield chunk

        # Use pure content part for validation (excluding thinking)
        content_only = getattr(self, '_last_stream_content', '')

        # Validate output
        if self.validate_output(content_only):
            yield "\n\n✅ Conversion complete"
        else:
            yield "\n\n⚠️ Output validation failed"

    def _format_input(self, structured_data: Dict[str, Any]) -> str:
        """Format input as prompt"""
        import json
        return f"Please convert the following structured problem description into mathematical formulas:\n\n```json\n{json.dumps(structured_data, ensure_ascii=False, indent=2)}\n```"

    def _get_summary(self, data: Dict[str, Any]) -> str:
        """Get input summary"""
        problem_type = data.get("problem_type", "Unknown")
        objectives = ", ".join(data.get("objectives", []))
        return f"{problem_type} - Objectives: {objectives}"

    def validate_output(self, output: Any) -> bool:
        """Validate mathematical formula output"""
        if not isinstance(output, str):
            return False

        # Check for key mathematical symbols
        math_indicators = ["min", "max", "Σ", "∑", "≤", "≥", "=", "P_", "C_"]
        return any(indicator in output for indicator in math_indicators)
