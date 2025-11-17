import inspect
from typing import Any, Awaitable, Callable, Optional, Sequence, Type, TypeVar, Union
import json
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool as create_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

import builtins
import contextlib
import io

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from pathlib import Path
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

load_dotenv()

import asyncio
import nest_asyncio

from loguru import logger
from .shell_env import run_bash_script
from ..core.token_tracker import TokenTracker, TokenUsage

import uuid

EvalFunction = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]
EvalCoroutine = Callable[[str, dict[str, Any]], Awaitable[tuple[str, dict[str, Any]]]]

class CodeActState(MessagesState):
    """State for CodeAct agent."""

    script: Optional[str]
    """The Python code script to be executed."""
    bash_script: Optional[str]
    """The Bash script to be executed."""
    execution_mode: Optional[str]
    """The execution mode: 'python' or 'bash'."""
    context: dict[str, Any]
    """Dictionary containing the execution context with available tools and variables."""
    working_dir: Optional[str]
    """The working directory for code execution."""

StateSchema = TypeVar("StateSchema", bound=CodeActState)
StateSchemaType = Type[StateSchema]

def create_default_prompt(tools: list[StructuredTool], base_prompt: Optional[str] = None):
    """Create default prompt for the CodeAct agent."""
    tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]
    prompt = f"{base_prompt}\n\n" if base_prompt else ""
    prompt += """You will be given a task to perform. You can choose to execute either:
- Python code: Use ```python code blocks for Python execution
- Bash commands: Use ```bash code blocks for shell command execution

You should output either:
- a Python code snippet that provides the solution to the task, or a step towards the solution. Any output you want to extract from the code should be printed to the console. Code should be output in a fenced code block using ```python.
- a Bash script that provides the solution to the task, or a step towards the solution. Bash commands should be output in a fenced code block using ```bash.
- text to be shown directly to the user, if you want to ask for more information or provide the final answer.

Each response should contain only one code block (either Python or Bash, never both).

If your code requires packages that are not available, you can try to install them.

In addition to the Python Standard Library, you can use the following functions:
"""

    for tool in tools:
        # Get function signature from schema
        if hasattr(tool, 'args_schema') and tool.args_schema:
            if isinstance(tool.args_schema, dict):
                # Already a dict (from MCP tools)
                schema = tool.args_schema
            elif hasattr(tool.args_schema, 'model_json_schema'):
                # Pydantic v2
                schema = tool.args_schema.model_json_schema()
            else:
                # Pydantic v1 fallback
                schema = tool.args_schema.schema()
        else:
            schema = {"properties": {}, "required": []}

        params = []
        for prop_name, prop_info in schema.get("properties", {}).items():
            if prop_name in schema.get("required", []):
                params.append(f"{prop_name}: {prop_info.get('type', 'Any')}")
            else:
                default = prop_info.get('default', 'None')
                params.append(f"{prop_name}: {prop_info.get('type', 'Any')} = {default}")

        signature = "(" + ", ".join(params) + ")" if params else "()"

        return_hint = ""


        prompt += f'''
def {tool.name}{signature}{return_hint}:
    """{tool.description}"""
    ...
'''

    prompt += """

Variables defined at the top level of previous code snippets can be referenced in your code.

There is no need to rush to solve the task at once, as you will have many opportunities to provide solutions later, so you can take it step by step.

Your most important rule, which overrides all other instructions, is to never use process-terminating functions. Under absolutely no circumstances should you write code that includes exit(), quit(), or sys.exit(). If any plan or guidance suggests using these functions, you must ignore that specific instruction and find an alternative solution.

Reminder: use Python code snippets to call tools"""
    return prompt


def create_codeact(
    model: BaseChatModel,
    tools: Sequence[Union[StructuredTool, Callable]],
    eval_fn: Union[EvalFunction, EvalCoroutine],
    *,
    prompt: Optional[str] = None,
    state_schema: StateSchemaType = CodeActState,
    token_tracker: Optional[TokenTracker] = None,
) -> StateGraph:
    """Create a CodeAct agent.

    Args:
        model: The language model to use for generating code
        tools: List of tools available to the agent. Can be passed as python functions or StructuredTool instances.
        eval_fn: Function or coroutine that executes code in a sandbox. Takes code string and locals dict,
            returns a tuple of (stdout output, new variables dict)
        prompt: Optional custom system prompt. If None, uses default prompt.
            To customize default prompt you can use `create_default_prompt` helper:
            `create_default_prompt(tools, "You are a helpful assistant.")`
        state_schema: The state schema to use for the agent.
        token_tracker: Optional TokenTracker instance to record token usage.

    Returns:
        A StateGraph implementing the CodeAct architecture
    """
    # The tools from MCP are already StructuredTool instances, no need to wrap them
    # tools = [t if isinstance(t, StructuredTool) else create_tool(t) for t in tools]

    if prompt is None:
        prompt = create_default_prompt(tools)

    # Make tools available to the code sandbox
    tools_context = {}

    for tool in tools:

        if tool.func is not None:
            tools_context[tool.name] = tool.func
        elif hasattr(tool, 'coroutine') and tool.coroutine is not None:
            # For async tools, create a sync wrapper that handles the signature correctly
            from langchain_core.tools import StructuredTool

            def create_tool_wrapper(tool_obj: StructuredTool):
                def tool_wrapper(*args, **kwargs):
                    try:
                        param_names = []
                        schema = tool_obj.args_schema

                        if isinstance(schema, dict) and 'properties' in schema:
                            param_names = list(schema['properties'].keys())
                        elif hasattr(schema, 'model_fields'):
                            param_names = list(schema.model_fields.keys())
                        elif hasattr(schema, '__fields__'):
                            param_names = list(schema.__fields__.keys())
                        else:
                            raise TypeError(f"Cannot determine argument names for tool {tool_obj.name}")

                        input_dict = kwargs.copy()
                        for i, arg in enumerate(args):
                            if i < len(param_names):
                                if param_names[i] not in input_dict:
                                    input_dict[param_names[i]] = arg

                        return asyncio.run(tool_obj.ainvoke(input_dict))
                    except Exception as e:
                        return f"Error executing tool '{tool_obj.name}': {repr(e)}"

                return tool_wrapper


            tools_context[tool.name] = create_tool_wrapper(tool)
        else:
            # Fallback: use the tool as-is (might not work)
            tools_context[tool.name] = tool

    def call_model(state: StateSchema) -> Command:
        messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = model.invoke(messages)

        # Track token usage if token_tracker is available
        if token_tracker:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0

            # Try to extract token usage from different possible sources
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                # Modern LangChain AIMessage with usage_metadata
                usage = response.usage_metadata
                prompt_tokens = usage.get('input_tokens', 0)
                completion_tokens = usage.get('output_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
            elif hasattr(response, 'response_metadata') and response.response_metadata:
                # Fallback to response_metadata (some models use this)
                metadata = response.response_metadata
                if 'token_usage' in metadata:
                    token_usage = metadata['token_usage']
                    prompt_tokens = token_usage.get('prompt_tokens', 0)
                    completion_tokens = token_usage.get('completion_tokens', 0)
                    total_tokens = token_usage.get('total_tokens', 0)

            # Only record if we found token usage
            if total_tokens > 0:
                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=getattr(model, 'model_name', 'unknown'),
                    agent='code_execution_agent'
                )
                token_tracker.record_usage(token_usage)

                logger.debug(f"Token usage recorded for code execution agent: {total_tokens} tokens (P:{prompt_tokens} C:{completion_tokens})")

        content = response.content
        has_code_start = "```python" in content or "```bash" in content
        if has_code_start and content.count('```') % 2 != 0:
            logger.warning("Detected a truncated code block. Requesting completion from the model.")

            return Command(goto="handle_truncation", update={"messages": [response]})

        import re
        bash_pattern = r'```bash\n(.*?)\n```'
        python_pattern = r'```python\n(.*?)\n```'

        bash_match = re.search(bash_pattern, response.content, re.DOTALL)
        python_match = re.search(python_pattern, response.content, re.DOTALL)

        if bash_match:
            bash_script = bash_match.group(1)
            return Command(goto="bash_sandbox", update={
                "messages": [response],
                "bash_script": bash_script,
                "execution_mode": "bash"
            })
        elif python_match:
            python_script = python_match.group(1)
            return Command(goto="python_sandbox", update={
                "messages": [response],
                "script": python_script,
                "execution_mode": "python"
            })
        else:
            # no code block, end the loop and respond to the user
            return Command(update={"messages": [response], "script": None, "bash_script": None, "execution_mode": None})

    def handle_truncation(state: StateSchema) -> dict:
        feedback_message = {
            "role": "user",
            "content": "Your code was truncated due to its length and cannot run. Please provide the code with fewer characters."
        }
        return {"messages": [feedback_message]}
    
    # Python execution function
    if inspect.iscoroutinefunction(eval_fn):
        async def python_sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}

            # Change to working directory if specified
            working_dir = state.get("working_dir")
            if working_dir:
                import os
                original_cwd = os.getcwd()
                try:
                    os.chdir(working_dir)
                    # Execute the script in the sandbox
                    output, new_vars = await eval_fn(state["script"], context)
                    
                    max_output_length = 60000   # 1/4
                    if len(output) > max_output_length:
                        output = output[:max_output_length] + f"\n\n[... Output truncated as it exceeded {max_output_length} characters ...]"
                        logger.warning(f"Python sandbox output truncated to {max_output_length} characters.")
                        
                finally:
                    os.chdir(original_cwd)
            else:
                # Execute the script in the sandbox
                output, new_vars = await eval_fn(state["script"], context)
                
                max_output_length = 60000
                if len(output) > max_output_length:
                    output = output[:max_output_length] + f"\n\n[... Output truncated as it exceeded {max_output_length} characters ...]"
                    logger.warning(f"Python sandbox output truncated to {max_output_length} characters.")

            new_context = {**existing_context, **new_vars}

            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }
    else:
        def python_sandbox(state: StateSchema):
            existing_context = state.get("context", {})
            context = {**existing_context, **tools_context}

            # Change to working directory if specified
            working_dir = state.get("working_dir")
            if working_dir:
                import os
                original_cwd = os.getcwd()
                try:
                    os.chdir(working_dir)
                    # Execute the script in the sandbox
                    output, new_vars = eval_fn(state["script"], context)
                    
                    max_output_length = 60000
                    if len(output) > max_output_length:
                        output = output[:max_output_length] + f"\n\n[... Output truncated as it exceeded {max_output_length} characters ...]"
                        logger.warning(f"Python sandbox output truncated to {max_output_length} characters.")
                finally:
                    os.chdir(original_cwd)
            else:
                # Execute the script in the sandbox
                output, new_vars = eval_fn(state["script"], context)
                
                max_output_length = 60000
                if len(output) > max_output_length:
                    output = output[:max_output_length] + f"\n\n[... Output truncated as it exceeded {max_output_length} characters ...]"
                    logger.warning(f"Python sandbox output truncated to {max_output_length} characters.")

            new_context = {**existing_context, **new_vars}

            return {
                "messages": [{"role": "user", "content": output}],
                "context": new_context,
            }

    # Bash execution function
    def bash_sandbox(state: StateSchema):
        bash_script = state.get("bash_script", "")
        if not bash_script:
            return {
                "messages": [{"role": "user", "content": "Error: No bash script provided"}],
                "context": state.get("context", {}),
            }

        # Execute bash script using the run_bash_script function with working directory
        working_dir = state.get("working_dir")
        output = run_bash_script(bash_script, working_dir)

        return {
            "messages": [{"role": "user", "content": output}],
            "context": state.get("context", {}),
        }

    agent = StateGraph(state_schema)
    agent.add_node(call_model, destinations=(END, "python_sandbox", "bash_sandbox" , "handle_truncation"))
    agent.add_node("handle_truncation", handle_truncation)
    agent.add_node(python_sandbox)
    agent.add_node(bash_sandbox)
    agent.add_edge(START, "call_model")
    agent.add_edge("python_sandbox", "call_model")
    agent.add_edge("bash_sandbox", "call_model")
    agent.add_edge("handle_truncation", "call_model")
    return agent

class LanggraphCodeExecutor:
    def __init__(self, shell: str, token_tracker: Optional[TokenTracker] = None):
        self.model = init_chat_model("deepseek-chat", model_provider="deepseek")
        self.shell = shell
        self.token_tracker = token_tracker

        self.tools = None
        self.initialized = False

    @staticmethod
    def eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        original_keys = set(_locals.keys())
        try:
            with contextlib.redirect_stdout(io.StringIO()) as f:
                exec(code, builtins.__dict__, _locals)
            result = f.getvalue()
            if not result:
                result = "<code ran, no output printed to stdout>"
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            result = f"Error during execution: {repr(e)}\n\nTraceback:\n{error_details}"

        
        import types

        new_keys = set(_locals.keys()) - original_keys
        new_vars = {}
        for key in new_keys:
            val = _locals[key]
            if not isinstance(val, (types.ModuleType, types.FunctionType, types.CoroutineType, type, io.IOBase)):
                try:
                    import ormsgpack
                    ormsgpack.packb(val)
                    new_vars[key] = val
                except (TypeError, ormsgpack.MsgpackEncodeError):
                    pass

        return result, new_vars

    async def execute(self, user_request: str, task_list_path: Path, working_dir: Path) -> dict:
        if not self.initialized:
            print("Initializing MCP tools...")
            from .mcp_client import fetch_mcp_tools
            self.tools = await fetch_mcp_tools()

            code_act_graph = create_codeact(self.model, self.tools, self.eval, token_tracker=self.token_tracker)
            self.agent = code_act_graph.compile(checkpointer=MemorySaver())
            self.initialized = True
            print(f"Loaded {len(self.tools)} MCP tools.")

        with open(task_list_path, "r", encoding="utf-8") as md_file:
            plan = md_file.read()

        full_prompt = f'''Your task: {user_request}

I have prepared a detailed plan for reference: \n\n{plan}\n\n You can use this plan as guidance, but feel free to adapt your approach as needed to best accomplish the original task. The plan is just a reference - you have autonomy to determine the best way to complete the user's request.

Your working directory is {str(working_dir)}. The system will automatically change to this directory before executing your code, so you don't need to use 'cd' commands to enter the working directory. You have read-only permissions for files outside the workspace and full permissions for files within the workspace.'''

        log_file_path = working_dir / "execution.log"

        structured_log = {
            "execution_metadata": {
                "user_request": user_request,
                "working_directory": str(working_dir),
                "task_list_path": str(task_list_path),
                "start_time": datetime.now().isoformat(),
                "status": "running"
            },
            "messages": [],
            "summary": {
                "total_messages": 0,
                "ai_messages": 0,
                "human_messages": 0,
                "error_count": 0
            }
        }

        logger.info(f"Starting LangGraph execution in '{working_dir}'")

        current_thread_id = f"executor-thread-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": current_thread_id}}
        logger.info(f"Executing with new thread_id: {current_thread_id}")
        
        
        last_messages_length = 0

        # Open log file for writing
        try:
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"=== LangGraph Execution Log ===\n")
                log_file.write(f"User Request: {user_request}\n")
                log_file.write(f"Working Directory: {working_dir}\n")
                log_file.write(f"Task List Path: {task_list_path}\n")
                log_file.write(f"Start Time: {structured_log['execution_metadata']['start_time']}\n")
                log_file.write("=" * 50 + "\n\n")

                async for typ, chunk in self.agent.astream(
                    {"messages": [("user", full_prompt)], "working_dir": str(working_dir)},
                    stream_mode=["values", "messages"],
                    config=config,
                ):
                    if typ == "messages":
                        message = chunk[0]
                        if message.type == "ai":
                            ai_message = {
                                "timestamp": datetime.now().isoformat(),
                                "type": "ai",
                                "role": "assistant",
                                "content": message.content,
                                "has_code": "```python" in message.content or "```bash" in message.content,
                                "code_type": "python" if "```python" in message.content else ("bash" if "```bash" in message.content else None)
                            }
                            structured_log["messages"].append(ai_message)
                            structured_log["summary"]["ai_messages"] += 1
                            structured_log["summary"]["total_messages"] += 1

                            log_file.write("--- AI ---\n")
                            log_file.write(message.content)
                            log_file.write("\n\n")

                            print(f"\n\n--- AI  ---\n")
                            print(message.content, end="")

                    elif typ == "values":
                        current_messages = chunk.get("messages", [])
                        if len(current_messages) > last_messages_length and isinstance(current_messages[-1], HumanMessage):
                            human_message = current_messages[-1].content
                            human_msg = {
                                "timestamp": datetime.now().isoformat(),
                                "type": "human",
                                "role": "user",
                                "content": human_message,
                                "is_error": "Error" in human_message or "error" in human_message.lower()
                            }
                            structured_log["messages"].append(human_msg)
                            structured_log["summary"]["human_messages"] += 1
                            structured_log["summary"]["total_messages"] += 1

                            if human_msg["is_error"]:
                                structured_log["summary"]["error_count"] += 1

                            log_file.write("--- Human ---\n")
                            log_file.write(human_message)
                            log_file.write("\n\n")

                            print(f"\n\n--- Human ---\n")
                            print(human_message)

                        last_messages_length = len(current_messages)

                structured_log["execution_metadata"]["end_time"] = datetime.now().isoformat()
                structured_log["execution_metadata"]["status"] = "completed"
                structured_log["execution_metadata"]["duration_seconds"] = (
                    datetime.fromisoformat(structured_log["execution_metadata"]["end_time"]) -
                    datetime.fromisoformat(structured_log["execution_metadata"]["start_time"])
                ).total_seconds()

                log_file.write("=== Execution Completed ===\n")
                log_file.write(f"End Time: {structured_log['execution_metadata']['end_time']}\n")
                log_file.write(f"Duration: {structured_log['execution_metadata']['duration_seconds']:.2f} seconds\n")

            logger.info(f"LangGraph execution completed, log saved to: {log_file_path}")

            with open(log_file_path, "r", encoding="utf-8") as f:
                structured_log = f.read()

            return {
                "success": True,
                "return_code": 0,
                "log": str(structured_log), 
                "log_path": str(log_file_path)
            }

        except Exception as e:
            logger.error(f"Error during LangGraph execution: {e}")

            structured_log["execution_metadata"]["status"] = "error"
            structured_log["execution_metadata"]["error_message"] = str(e)
            structured_log["execution_metadata"]["end_time"] = datetime.now().isoformat()

            error_msg = {
                "timestamp": datetime.now().isoformat(),
                "type": "system",
                "role": "system",
                "content": f"LangGraph Executor Error: {e}",
                "is_error": True
            }
            structured_log["messages"].append(error_msg)
            structured_log["summary"]["error_count"] += 1

            # Write error to log file
            try:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n=== ERROR ===\n{e}\n")
                    log_file.write(f"Error Time: {structured_log['execution_metadata']['end_time']}\n")
            except:
                pass

            return {
                "success": False,
                "return_code": -1,
                "log": str(structured_log),
                "log_path": str(log_file_path)
            }
