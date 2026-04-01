"""State schema for workflow"""
from typing import TypedDict, List, Dict, Optional, Any


class AgentState(TypedDict):
    """Workflow state definition"""

    # User input
    user_input: str

    # Agent outputs
    structured_sequence: Optional[Dict[str, Any]]
    math_formula: Optional[str]
    mdp_model: Optional[Dict[str, Any]]
    generated_code: Optional[str]

    # Execution status
    current_step: int
    total_steps: int
    step_status: List[str]  # ["completed", "running", "pending", "error"]

    # Logs and errors
    execution_log: List[Dict[str, Any]]
    errors: List[str]

    # Metadata
    timestamp: str
    session_id: str
