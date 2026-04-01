"""LangGraph Workflow Builder"""
import time
import uuid
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from workflow.state_schema import AgentState
from agents import (
    RequirementParserAgent,
    FormulaConverterAgent,
    MDPModelerAgent,
    CodeGeneratorAgent
)


class WorkflowBuilder:
    """Workflow builder"""

    def __init__(self):
        """Initialize workflow"""
        self.agents = {
            "agent1": RequirementParserAgent(),
            "agent2": FormulaConverterAgent(),
            "agent3": MDPModelerAgent(),
            "agent4": CodeGeneratorAgent()
        }

    def create_workflow(self) -> StateGraph:
        """Create workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent1", self._agent1_node)
        workflow.add_node("agent2", self._agent2_node)
        workflow.add_node("agent3", self._agent3_node)
        workflow.add_node("agent4", self._agent4_node)

        # Define edges
        workflow.set_entry_point("agent1")
        workflow.add_edge("agent1", "agent2")
        workflow.add_edge("agent2", "agent3")
        workflow.add_edge("agent3", "agent4")
        workflow.add_edge("agent4", END)

        return workflow.compile()

    def _agent1_node(self, state: AgentState) -> AgentState:
        """Agent 1 node: requirement parsing"""
        try:
            state["current_step"] = 1
            state["step_status"][0] = "running"

            result = self.agents["agent1"].run(state["user_input"])

            state["structured_sequence"] = result["output"]
            state["step_status"][0] = "completed"

            log_entry = {
                "step": 1,
                "agent": "Requirement Parser",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][0] = "error"
            state["errors"].append(f"Agent 1 Error: {str(e)}")
            log_entry = {
                "step": 1,
                "agent": "Requirement Parser",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _agent2_node(self, state: AgentState) -> AgentState:
        """Agent 2 node: formula conversion"""
        try:
            state["current_step"] = 2
            state["step_status"][1] = "running"

            result = self.agents["agent2"].run(state["structured_sequence"])

            state["math_formula"] = result["output"]
            state["step_status"][1] = "completed"

            log_entry = {
                "step": 2,
                "agent": "Formula Converter",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][1] = "error"
            state["errors"].append(f"Agent 2 Error: {str(e)}")
            log_entry = {
                "step": 2,
                "agent": "Formula Converter",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _agent3_node(self, state: AgentState) -> AgentState:
        """Agent 3 node: RL design"""
        try:
            state["current_step"] = 3
            state["step_status"][2] = "running"

            result = self.agents["agent3"].run(state["math_formula"])

            state["mdp_model"] = result["output"]
            state["step_status"][2] = "completed"

            log_entry = {
                "step": 3,
                "agent": "RL Designer",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][2] = "error"
            state["errors"].append(f"Agent 3 Error: {str(e)}")
            log_entry = {
                "step": 3,
                "agent": "RL Designer",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def _agent4_node(self, state: AgentState) -> AgentState:
        """Agent 4 node: code generation"""
        try:
            state["current_step"] = 4
            state["step_status"][3] = "running"

            result = self.agents["agent4"].run(state["mdp_model"])

            state["generated_code"] = result["output"]
            state["step_status"][3] = "completed"

            log_entry = {
                "step": 4,
                "agent": "Code Generator",
                "status": "completed",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": result["metadata"]["execution_time"]
            }
            state["execution_log"].append(log_entry)

        except Exception as e:
            state["step_status"][3] = "error"
            state["errors"].append(f"Agent 4 Error: {str(e)}")
            log_entry = {
                "step": 4,
                "agent": "Code Generator",
                "status": "error",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            state["execution_log"].append(log_entry)
            raise

        return state

    def create_initial_state(self, user_input: str) -> AgentState:
        """Create initial state"""
        return {
            "user_input": user_input,
            "structured_sequence": None,
            "math_formula": None,
            "mdp_model": None,
            "generated_code": None,
            "current_step": 0,
            "total_steps": 4,
            "step_status": ["pending", "pending", "pending", "pending"],
            "execution_log": [],
            "errors": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": str(uuid.uuid4())
        }
