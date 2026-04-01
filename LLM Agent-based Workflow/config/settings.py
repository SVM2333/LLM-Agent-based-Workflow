"""Configuration Management Module"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Global Configuration Class"""

    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "z-ai/glm-5")

    # Application Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    TIMEOUT = int(os.getenv("TIMEOUT", "60"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Path Configuration
    BASE_DIR = Path(__file__).parent.parent
    CONFIG_DIR = BASE_DIR / "config"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    PROMPTS_FILE = CONFIG_DIR / "prompts.yaml"

    @classmethod
    def load_prompts(cls) -> Dict[str, Any]:
        """Load prompt configuration"""
        with open(cls.PROMPTS_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @classmethod
    def get_agent_prompt(cls, agent_name: str) -> Dict[str, str]:
        """Get prompt configuration for specified Agent"""
        prompts = cls.load_prompts()
        return prompts.get(agent_name, {})
