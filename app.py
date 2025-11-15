# app.py
# Web UI for the HealthFlow Agentic System.

import asyncio
import re
import toml
from pathlib import Path
import streamlit as st

# Ensure the project root is in the path to import healthflow modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from healthflow.system import HealthFlowSystem
from healthflow.core.config import get_config, setup_logging, HealthFlowConfig, LLMProviderConfig, SystemConfig, EvaluationConfig, LoggingConfig

# --- Page Configuration ---
st.set_page_config(
    page_title="HealthFlow Agent UI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Functions ---

def get_llm_options_from_config(config_data):
    """Parses the config file to get available LLM names."""
    if "llm" in config_data:
        return list(config_data["llm"].keys())
    return []

@st.cache_resource
def load_config_data():
    """Loads the config.toml file. Caches the result for performance."""
    config_path = Path("config.toml")
    if config_path.exists():
        return toml.load(config_path)
    # Fallback for deployed environment where config.toml might not exist
    st.warning("`config.toml` not found. Relying on Streamlit secrets for configuration.")
    return {}

def initialize_system_from_secrets(active_llm_name: str) -> HealthFlowSystem:
    """
    Initializes HealthFlowSystem using Streamlit's secrets management.
    This is the preferred way for deployed apps.
    """
    try:
        llm_config_data = st.secrets.llm[active_llm_name]
        llm_config = LLMProviderConfig(**llm_config_data)

        # Use default system/evaluation/logging configs or pull from secrets if defined
        system_config = SystemConfig(**st.secrets.get("system", {}))
        evaluation_config = EvaluationConfig(**st.secrets.get("evaluation", {}))
        logging_config = LoggingConfig(**st.secrets.get("logging", {"log_level": "INFO", "log_file": "healthflow_streamlit.log"}))

        config = HealthFlowConfig(
            active_llm_name=active_llm_name,
            llm=llm_config,
            system=system_config,
            evaluation=evaluation_config,
            logging=logging_config
        )
        setup_logging(config)
        return HealthFlowSystem(config=config, experience_path=Path(config.system.workspace_dir) / "experience.jsonl")
    except KeyError:
        st.error(f"Configuration for LLM '{active_llm_name}' not found in Streamlit secrets. Please check your secrets.toml.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize HealthFlow system from secrets: {e}")
        st.stop()


def initialize_system_from_file(active_llm_name: str) -> HealthFlowSystem:
    """

    Initializes HealthFlowSystem from the local config.toml file.
    Used for local development.
    """
    try:
        config_path = Path("config.toml")
        experience_path = Path("workspace/experience.jsonl")
        config = get_config(config_path, active_llm_name)
        setup_logging(config)
        return HealthFlowSystem(config=config, experience_path=experience_path)
    except Exception as e:
        st.error(f"Failed to initialize HealthFlow system from config.toml: {e}")
        st.stop()

def run_healthflow_task_async(system: HealthFlowSystem, task: str):
    """
    A wrapper to run the asynchronous HealthFlow task from Streamlit's
    synchronous execution environment.
    """
    # Streamlit runs in a sync context. We need to run our async code
    # in a new event loop.
    return asyncio.run(system.run_task(task))


# --- Main Application UI ---

st.title("🌊 HealthFlow: Autonomous AI for Healthcare Research")
st.markdown("Welcome to HealthFlow. Enter a complex healthcare research task below, and the AI agent will autonomously generate and execute a plan to find the answer.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

    # Determine if running on Streamlit Cloud (where secrets are available)
    IS_DEPLOYED = hasattr(st.secrets, 'llm')

    if IS_DEPLOYED:
        # In deployed environment, get LLM options from secrets
        llm_options = list(st.secrets.llm.keys())
        st.info("Running in Cloud mode. Configuration is loaded from Streamlit secrets.")
    else:
        # In local environment, get LLM options from config.toml
        config_data_local = load_config_data()
        llm_options = get_llm_options_from_config(config_data_local)
        st.info("Running in Local mode. Configuration is loaded from `config.toml`.")

    if not llm_options:
        st.error("No LLM configurations found. Please add them to your `config.toml` or Streamlit secrets.")
        st.stop()

    active_llm = st.selectbox(
        "Select Reasoning LLM",
        options=llm_options,
        index=0,
        help="Choose the Large Language Model that will be used for planning, evaluation, and reflection."
    )

    st.markdown("---")
    st.markdown(
        "Built for the COMP9501 Final Group Project."
    )


# --- Main Content Area ---

# Use session state to store the task and result between reruns
if 'task_result' not in st.session_state:
    st.session_state.task_result = None

# Input form
with st.form("task_form"):
    task_input = st.text_area(
        "Enter your research task:",
        height=150,
        placeholder="e.g., Analyze the provided 'patients.csv' to identify the top 3 risk factors for readmission. Anonymize any patient identifiers in the output."
    )
    submit_button = st.form_submit_button(label="Run HealthFlow Agent")

if submit_button and task_input:
    # Clear previous results and run the new task
    st.session_state.task_result = None

    with st.spinner("HealthFlow is orchestrating... This may take several minutes."):
        try:
            # Initialize the system based on the environment
            if IS_DEPLOYED:
                system = initialize_system_from_secrets(active_llm)
            else:
                system = initialize_system_from_file(active_llm)

            # Run the task
            result = run_healthflow_task_async(system, task_input)
            st.session_state.task_result = result
            st.rerun() # Rerun the script to display the results below
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.exception(e) # Display full traceback for debugging

# Display results if they exist in the session state
if st.session_state.task_result:
    result = st.session_state.task_result
    st.markdown("---")
    st.header("Results")

    if result.get("success"):
        st.success("**Task Completed Successfully!**")
    else:
        st.error("**Task Failed.**")

    # Display the final answer
    st.subheader("Final Answer")
    st.markdown(result.get("answer", "No answer was generated."))

    # Display details in an expander
    with st.expander("Show Execution Details"):
        st.subheader("Summary")
        st.write(result.get("final_summary", "No summary available."))

        st.metric(label="Execution Time", value=f"{result.get('execution_time', 0):.2f} seconds")

        workspace_path_str = result.get('workspace_path', '')
        st.subheader("Workspace Artifacts")
        st.code(workspace_path_str, language="bash")

        # Try to list and display files from the workspace
        if workspace_path_str:
            workspace_path = Path(workspace_path_str)
            if workspace_path.exists() and workspace_path.is_dir():
                st.write("Files generated in the workspace:")
                files_to_display = {
                    "Execution Log": "execution.log",
                    "Full History (JSON)": "full_history.json",
                    "Final Plan": "task_list_v1.md" # Simple assumption, might need to find latest
                }

                for display_name, file_name in files_to_display.items():
                    file_path = workspace_path / file_name
                    # Handle versioned plans (e.g., task_list_v2.md)
                    if "task_list" in file_name:
                        plan_files = sorted(workspace_path.glob("task_list_v*.md"), reverse=True)
                        if plan_files:
                            file_path = plan_files[0]

                    if file_path.exists():
                        try:
                            content = file_path.read_text(encoding="utf-8")
                            st.text(f"--- {display_name} ({file_path.name}) ---")
                            st.code(content, language="markdown" if ".md" in file_path.name else "json" if ".json" in file_path.name else "log")
                        except Exception as e:
                            st.warning(f"Could not read or display {file_path.name}: {e}")