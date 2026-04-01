"""Agent 4 Enhanced: Full Code Generator - Generate complete environment + training + evaluation + visualization code"""
import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Iterator
from agents.base_agent import BaseAgent


class FullCodeGeneratorAgent(BaseAgent):
    """Full Code Generator - Generate complete, directly trainable NMG+RL code"""

    def __init__(self):
        # Don't call parent __init__ because we use our own prompt file
        self.agent_name = "agent4_code_generator_full"
        self._load_full_prompt()

        # Initialize LLM client (reuse parent logic)
        from openai import OpenAI
        from config.settings import Config
        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )
        self.execution_count = 0
        self.total_time = 0.0

    def _load_full_prompt(self):
        """Load enhanced prompt from prompts_full.yaml"""
        prompts_file = Path(__file__).parent.parent / "config" / "prompts_full.yaml"
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)
        config = prompts.get(self.agent_name, {})
        self.name = config.get("name", "Full Code Generator")
        self.description = config.get("description", "")
        self.system_prompt = config.get("system_prompt", "")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute full code generation"""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary type (MDP model)")

        user_input = self._format_input(input_data)
        response = self.call_llm(user_input, temperature=0.3)
        python_code = self._extract_python_code(response)

        if not self.validate_output(python_code):
            raise ValueError("Output validation failed")

        return {
            "success": True,
            "output": python_code,
            "raw_response": response
        }

    def execute_stream(self, input_data: Any, examples: str = "") -> Iterator[str]:
        """Streaming full code generation"""
        if not isinstance(input_data, dict):
            raise ValueError("Input must be a dictionary type (MDP model)")

        user_input = self._format_input(input_data)

        self._extracted_code = None
        for chunk in self.call_llm_stream(user_input, temperature=0.3, examples=examples):
            yield chunk

        # Use pure content part for validation (excluding thinking)
        content_only = getattr(self, '_last_stream_content', '')

        try:
            python_code = self._extract_python_code(content_only)
            self._extracted_code = python_code
            if self.validate_output(python_code):
                yield "\n\n✅ Full code generation complete (includes environment + training + evaluation + visualization)"
            else:
                yield "\n\n⚠️ Code validation failed, some modules may be missing"
        except Exception:
            yield "\n\n⚠️ Code extraction failed"

    def _format_input(self, mdp_model: Dict[str, Any]) -> str:
        """Format input - emphasize need for complete code"""
        return (
            "Based on the following complete RL scheme design (including MDP definition, algorithm selection, network architecture), "
            "generate a **complete, directly runnable and trainable** Python file.\n\n"
            "The code must include:\n"
            "1. Gymnasium environment class (strictly implement according to MDP definition)\n"
            "2. Custom Actor-Critic network (implement according to network_architecture)\n"
            "3. Training script using Stable-Baselines3 (configure with rl_algorithm hyperparameters)\n"
            "4. Evaluation and visualization functions (plot dispatch result charts)\n"
            "5. Complete main entry point\n\n"
            f"RL scheme design:\n```json\n{json.dumps(mdp_model, ensure_ascii=False, indent=2)}\n```"
        )

    def _extract_python_code(self, response: str) -> str:
        """Extract Python code from response (supports concatenating multiple code blocks)"""
        # 1. Find all ```python ... ``` code blocks
        code_blocks = re.findall(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            # Concatenate all code blocks
            return "\n\n".join(block.strip() for block in code_blocks)

        # 2. Find all ``` ... ``` code blocks (no language identifier)
        code_blocks = re.findall(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if code_blocks:
            python_blocks = [b.strip() for b in code_blocks
                            if "import" in b or "class" in b or "def" in b]
            if python_blocks:
                return "\n\n".join(python_blocks)

        # 3. If no code blocks, try finding code starting with import statement
        if "import" in response:
            lines = response.split("\n")
            start_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("import") or line.strip().startswith("from"):
                    start_idx = i
                    break
            if start_idx is not None:
                return "\n".join(lines[start_idx:]).strip()

        raise ValueError(f"Failed to extract Python code from response: {response[:200]}")

    def validate_output(self, output: Any) -> bool:
        """Validate full code output - check if all required parts are present"""
        if not isinstance(output, str):
            return False

        # Basic checks
        required_keywords = ["import", "class", "def"]
        if not all(kw in output for kw in required_keywords):
            return False

        # Check Gymnasium environment
        has_gym = "gym" in output.lower() or "gymnasium" in output.lower()

        # Check training-related
        has_training = "stable_baselines3" in output or "train" in output.lower()

        # Check visualization
        has_plot = "matplotlib" in output or "plt" in output

        return has_gym and has_training and has_plot
