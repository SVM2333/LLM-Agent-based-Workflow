"""Enhanced Streaming UI - Full NMG+RL Dispatch Code Generation"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from agents import (
    RequirementParserAgent,
    FormulaConverterAgent,
    MDPModelerAgent,
)
from agents.agent4_code_generator_full import FullCodeGeneratorAgent

st.set_page_config(
    page_title="LLM-Agent+NMG (Full)",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ LLM-Agent+NMG Collaborative Dispatch Generation (Full)")
st.markdown("LangGraph + GLM-5 + 4-agent | Full Version - Environment + Training + Evaluation + Visualization")

# --- Sidebar ---
with st.sidebar:
    st.header("📋 System Guide")
    st.markdown("""
    ### Agent Pipeline
    1. **Requirement Parser** - Parse natural language requirements
    2. **Formula Converter** - Convert to mathematical formulas
    3. **RL Designer** - Design complete RL solution
    4. **Full Code Generator** - Generate complete Python code
    """)
    st.markdown("---")
    st.info("**Full Version Note:** Agent 4 generates complete code including Gymnasium environment, "
            "PyTorch networks, SB3 training scripts, evaluation, and visualization.")

# --- Tabs ---
tab_run, tab_examples = st.tabs(["🚀 Run", "📝 Example Config"])

# ========== Example Config Tab ==========
with tab_examples:
    st.markdown("Provide reference examples for each Agent. They will be appended as supplemental context. Leave empty for no examples.")

    ex1, ex2, ex3, ex4 = st.tabs([
        "Agent 1: Requirement Parser",
        "Agent 2: Formula Converter",
        "Agent 3: RL Designer",
        "Agent 4: Full Code Generator (Full)"
    ])

    with ex1:
        st.session_state.setdefault("ex_agent1", "")
        st.session_state["ex_agent1"] = st.text_area(
            "Agent 1 Reference Example", value=st.session_state["ex_agent1"],
            height=200, key="input_ex1",
            placeholder="e.g., When inputting 'optimize a microgrid day-ahead dispatch', the expected JSON structure output...")

    with ex2:
        st.session_state.setdefault("ex_agent2", "")
        st.session_state["ex_agent2"] = st.text_area(
            "Agent 2 Reference Example", value=st.session_state["ex_agent2"],
            height=200, key="input_ex2",
            placeholder="e.g., Reference for objective function mathematical formula format...")

    with ex3:
        st.session_state.setdefault("ex_agent3", "")
        st.session_state["ex_agent3"] = st.text_area(
            "Agent 3 Reference Example", value=st.session_state["ex_agent3"],
            height=200, key="input_ex3",
            placeholder="e.g., MDP state space definition reference, PPO hyperparameters reference...")

    with ex4:
        st.session_state.setdefault("ex_agent4", "")
        st.session_state["ex_agent4"] = st.text_area(
            "Agent 4 Reference Example (Full)", value=st.session_state["ex_agent4"],
            height=200, key="input_ex4",
            placeholder="e.g., Complete training code structure reference including environment + SB3 training + evaluation + visualization...")

# ========== Run Tab ==========
with tab_run:
    user_input = st.text_area(
        "📝 Input Requirement Description",
        value="Optimize a microgrid dispatch system with xx generators, goal is to minimize total cost while satisfying load demand and unit constraints.",
        height=100
    )

    if st.button("🚀 Start Processing", type="primary"):

        ex1_text = st.session_state.get("ex_agent1", "")
        ex2_text = st.session_state.get("ex_agent2", "")
        ex3_text = st.session_state.get("ex_agent3", "")
        ex4_text = st.session_state.get("ex_agent4", "")

        # --- Agent 1 ---
        st.markdown("---")
        st.markdown("### 🤖 Agent 1: Requirement Parser")
        st.caption("Converting natural language to structured sequence...")
        try:
            agent1 = RequirementParserAgent()
            agent1_text = st.write_stream(agent1.execute_stream(user_input, examples=ex1_text))
            structured_data = agent1._parsed_result
            if structured_data is None:
                st.error("❌ Failed to parse Agent 1 output (JSON parse failed or validation failed)")
                st.stop()
        except Exception as e:
            st.error(f"❌ Agent 1 Error: {str(e)}")
            st.stop()

        # --- Agent 2 ---
        st.markdown("---")
        st.markdown("### 🤖 Agent 2: Formula Converter")
        st.caption("Converting structured sequence to mathematical formulas...")
        try:
            agent2 = FormulaConverterAgent()
            agent2_text = st.write_stream(agent2.execute_stream(structured_data, examples=ex2_text))
        except Exception as e:
            st.error(f"❌ Agent 2 Error: {str(e)}")
            st.stop()

        # --- Agent 3 ---
        st.markdown("---")
        st.markdown("### 🤖 Agent 3: RL Designer")
        st.caption("Designing complete RL solution (MDP + Algorithm + Network Architecture)...")
        try:
            agent3 = MDPModelerAgent()
            agent3_input = agent2._last_stream_content or agent2_text
            agent3_text = st.write_stream(agent3.execute_stream(agent3_input, examples=ex3_text))
            rl_design = agent3._parsed_result
            if rl_design is None:
                st.error("❌ Failed to parse Agent 3 output (JSON parse failed or validation failed)")
                st.stop()
        except Exception as e:
            st.error(f"❌ Agent 3 Error: {str(e)}")
            st.stop()

        # --- Agent 4 (Full Version) ---
        st.markdown("---")
        st.markdown("### 🤖 Agent 4: Full Code Generator (Full)")
        st.caption("Generating complete NMG+RL code (Environment + Network + Training + Evaluation + Visualization)...")
        try:
            agent4 = FullCodeGeneratorAgent()
            agent4_text = st.write_stream(agent4.execute_stream(rl_design, examples=ex4_text))
        except Exception as e:
            st.error(f"❌ Agent 4 Error: {str(e)}")
            st.stop()

        # --- Done ---
        st.markdown("---")
        st.success("✅ All agents completed!")
        download_code = getattr(agent4, '_extracted_code', None) or agent4._last_stream_content or agent4_text
        st.download_button(
            label="📥 Download Full Training Code",
            data=download_code,
            file_name="nmg_rl_dispatch_full.py",
            mime="text/x-python"
        )
