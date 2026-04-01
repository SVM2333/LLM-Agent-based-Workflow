"""Agent Base Class"""
import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from config.settings import Config


class BaseAgent:
    """Agent Base Class"""

    def __init__(self, agent_name: str):
        """
        Initialize Agent

        Args:
            agent_name: Agent config name (e.g., agent1_requirement_parser)
        """
        self.agent_name = agent_name
        self.config = Config.get_agent_prompt(agent_name)
        self.name = self.config.get("name", "Unknown Agent")
        self.description = self.config.get("description", "")
        self.system_prompt = self.config.get("system_prompt", "")

        # Initialize LLM client
        self.client = OpenAI(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL
        )

        # Execution statistics
        self.execution_count = 0
        self.total_time = 0.0

    def call_llm(self, user_input: str, temperature: float = 0.7) -> str:
        """
        Call LLM (non-streaming)

        Args:
            user_input: User input
            temperature: Temperature parameter

        Returns:
            LLM response
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=temperature,
                timeout=Config.TIMEOUT,
                extra_body={"reasoning": {"enabled": True}}  # Enable GLM-5 reasoning mode
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")

    def call_llm_stream(self, user_input: str, temperature: float = 0.7, examples: str = ""):
        """
        Streaming LLM call

        Args:
            user_input: User input
            temperature: Temperature parameter
            examples: User-provided reference example text

        Yields:
            Each chunk of the LLM response (including thinking and content)

        Note:
            After the call ends, self._last_stream_content holds the pure content part
            (excluding thinking), which can be used for subsequent JSON parsing and other
            scenarios requiring pure output.
        """
        self._last_stream_content = ""
        try:
            # If examples are provided, append them to the user message
            full_input = user_input
            if examples and examples.strip():
                full_input = f"Here are some reference examples:\n\n{examples.strip()}\n\n---\n\n{user_input}"

            stream = self.client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_input}
                ],
                temperature=temperature,
                timeout=Config.TIMEOUT,
                stream=True,  # Enable streaming output
                extra_body={"reasoning": {"enabled": True}}  # Enable GLM-5 reasoning mode
            )

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Output thinking content (if available)
                    if hasattr(delta, 'reasoning') and delta.reasoning:
                        yield delta.reasoning

                    # Output normal content (if available)
                    if delta.content:
                        self._last_stream_content += delta.content
                        yield delta.content

        except Exception as e:
            raise Exception(f"LLM streaming call failed: {str(e)}")

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute Agent task (subclasses must implement)

        Args:
            input_data: Input data

        Returns:
            Dictionary containing output and metadata
        """
        raise NotImplementedError("Subclasses must implement the execute method")

    def run(self, input_data: Any, max_retries: int = None) -> Dict[str, Any]:
        """
        Run Agent (with retry mechanism)

        Args:
            input_data: Input data
            max_retries: Maximum number of retries

        Returns:
            Execution result
        """
        if max_retries is None:
            max_retries = Config.MAX_RETRIES

        start_time = time.time()
        last_error = None

        for attempt in range(max_retries):
            try:
                result = self.execute(input_data)
                execution_time = time.time() - start_time

                # Update statistics
                self.execution_count += 1
                self.total_time += execution_time

                # Add metadata
                result["metadata"] = {
                    "agent_name": self.name,
                    "execution_time": round(execution_time, 2),
                    "attempt": attempt + 1,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                return result

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"{self.name} execution failed (tried {max_retries} times): {str(last_error)}")

    def validate_output(self, output: Any) -> bool:
        """
        Validate output (subclasses can override)

        Args:
            output: Output data

        Returns:
            Whether valid
        """
        return output is not None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
