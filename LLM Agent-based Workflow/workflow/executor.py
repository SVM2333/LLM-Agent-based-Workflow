"""Workflow Executor"""
from typing import Dict, Any, Iterator
from workflow.graph_builder import WorkflowBuilder
from workflow.state_schema import AgentState


class WorkflowExecutor:
    """Workflow executor"""

    def __init__(self):
        """Initialize executor"""
        self.builder = WorkflowBuilder()
        self.workflow = self.builder.create_workflow()

    def execute(self, user_input: str) -> AgentState:
        """
        Execute full workflow.

        Args:
            user_input: User input

        Returns:
            Final state
        """
        initial_state = self.builder.create_initial_state(user_input)
        final_state = self.workflow.invoke(initial_state)
        return final_state

    def stream_execute(self, user_input: str) -> Iterator[AgentState]:
        """
        Stream execute workflow (for real-time UI updates).

        Args:
            user_input: User input

        Yields:
            State after each step
        """
        initial_state = self.builder.create_initial_state(user_input)

        for output in self.workflow.stream(initial_state):
            # LangGraph's stream returns {node_name: state} dict
            for node_name, state in output.items():
                yield state
