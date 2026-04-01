"""Agent 1: Requirement Parser"""
import json
import re
from typing import Dict, Any, Iterator
from agents.base_agent import BaseAgent


class RequirementParserAgent(BaseAgent):
    """Requirement Parser - Convert natural language to structured sequences"""

    def __init__(self):
        super().__init__("agent1_requirement_parser")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute requirement parsing

        Args:
            input_data: User's natural language input

        Returns:
            Dictionary containing structured sequence
        """
        if isinstance(input_data, dict):
            user_input = input_data.get("user_input", "")
        else:
            user_input = str(input_data)

        if not user_input.strip():
            raise ValueError("Input cannot be empty")

        # Call LLM
        response = self.call_llm(user_input, temperature=0.3)

        # Parse JSON output
        structured_data = self._parse_json_response(response)

        # Validate output
        if not self.validate_output(structured_data):
            raise ValueError("Output validation failed")

        return {
            "success": True,
            "output": structured_data,
            "raw_response": response
        }

    def execute_stream(self, input_data: Any, examples: str = "") -> Iterator[str]:
        """
        Streaming requirement parsing

        Args:
            input_data: User's natural language input

        Yields:
            Each chunk of the LLM response
        """
        if isinstance(input_data, dict):
            user_input = input_data.get("user_input", "")
        else:
            user_input = str(input_data)

        if not user_input.strip():
            raise ValueError("Input cannot be empty")

        # Streaming LLM call
        for chunk in self.call_llm_stream(user_input, temperature=0.3, examples=examples):
            yield chunk

        # Use pure content part for parsing (excluding thinking)
        content_only = getattr(self, '_last_stream_content', '')

        # Parse and store result for external callers
        self._parsed_result = None
        try:
            structured_data = self._parse_json_response(content_only)
            if self.validate_output(structured_data):
                self._parsed_result = structured_data
                yield "\n\n✅ Parsing complete"
        except:
            yield "\n\n⚠️ JSON parsing failed"

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from response"""
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse JSON from response: {response[:200]}")

    def validate_output(self, output: Any) -> bool:
        """Validate structured output"""
        if not isinstance(output, dict):
            return False

        required_keys = ["problem_type", "objectives", "constraints", "components"]
        return all(key in output for key in required_keys)
