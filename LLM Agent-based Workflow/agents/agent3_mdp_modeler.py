"""Agent 3: Reinforcement Learning Designer"""
import json
import re
from typing import Dict, Any, Iterator
from agents.base_agent import BaseAgent


class MDPModelerAgent(BaseAgent):
    """RL Designer - Convert mathematical formulas to complete RL scheme (MDP + Algorithm + Network Architecture)"""

    def __init__(self):
        super().__init__("agent3_mdp_modeler")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute RL scheme design

        Args:
            input_data: Mathematical formulas (string)

        Returns:
            Dictionary containing complete RL scheme (MDP definition, algorithm selection, network architecture)
        """
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string type (mathematical formulas)")

        # Build prompt
        user_input = f"Please convert the following mathematical formulas into a complete reinforcement learning scheme (including MDP definition, algorithm selection, and network architecture design):\n\n{input_data}"

        # Call LLM
        response = self.call_llm(user_input, temperature=0.4)

        # Parse JSON output
        rl_design = self._parse_json_response(response)

        # Validate output
        if not self.validate_output(rl_design):
            raise ValueError("Output validation failed")

        return {
            "success": True,
            "output": rl_design,
            "raw_response": response
        }

    def execute_stream(self, input_data: Any, examples: str = "") -> Iterator[str]:
        """
        Streaming RL scheme design

        Args:
            input_data: Mathematical formulas (string)

        Yields:
            Each chunk of the LLM response
        """
        if not isinstance(input_data, str):
            raise ValueError("Input must be a string type (mathematical formulas)")

        user_input = f"Please convert the following mathematical formulas into a complete reinforcement learning scheme (including MDP definition, algorithm selection, and network architecture design):\n\n{input_data}"

        # Streaming LLM call
        for chunk in self.call_llm_stream(user_input, temperature=0.4, examples=examples):
            yield chunk

        # Use pure content part for parsing (excluding thinking)
        content_only = getattr(self, '_last_stream_content', '')

        # Parse and store result
        self._parsed_result = None
        try:
            rl_design = self._parse_json_response(content_only)
            if self.validate_output(rl_design):
                self._parsed_result = rl_design
                yield "\n\n✅ RL scheme design complete"
            else:
                yield "\n\n⚠️ Output validation failed"
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
        """Validate RL scheme output"""
        if not isinstance(output, dict):
            return False

        # Check required top-level keys
        required_keys = ["mdp_definition", "rl_algorithm", "network_architecture"]
        if not all(key in output for key in required_keys):
            return False

        # Check MDP definition required keys
        mdp_keys = ["state_space", "action_space", "reward_function", "transition_dynamics"]
        if not all(key in output.get("mdp_definition", {}) for key in mdp_keys):
            return False

        # Check algorithm config
        if "name" not in output.get("rl_algorithm", {}):
            return False

        # Check network architecture
        network_arch = output.get("network_architecture", {})
        if "actor_network" not in network_arch or "critic_network" not in network_arch:
            return False

        return True
