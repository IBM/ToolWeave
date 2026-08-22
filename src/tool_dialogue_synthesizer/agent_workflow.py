import logging
import os
import yaml
from typing import LiteralString

from langgraph.graph import StateGraph, START, END

from src.tool_dialogue_synthesizer.agents.assistant_agent import AssistantAgent
from src.tool_dialogue_synthesizer.agents.memory_agent import MemoryAgent
from src.tool_dialogue_synthesizer.agents.tool_agent import ToolAgent
from src.tool_dialogue_synthesizer.agents.user_agent import UserAgent

from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM

from src.tool_dialogue_synthesizer.schema import DialogueState


logger = logging.getLogger('dialogue_generator')


# This function runs AFTER the AssistantAgent node.
def should_continue_based_on_plan(state: DialogueState) -> str | LiteralString:
    """
    Checks if the plan is complete based on current_step_idx.
    If not complete, determines the next node based on Assistant's action (tool_call or not).

    Args:
        state: DialogueState containing current_step_idx, plan, and tool_call info.

    Returns:
        "tool" if Assistant decided to call a tool and plan is not complete,
        "user" if Assistant did not call a tool and plan is not complete,
        END if the plan is complete or if there was an error in the previous step.
    """
    if state.get("abort_due_to_error", False):
        logger.debug("Aborting due to error in previous step. Ending.\n")
        return END

    num_plan_steps = len(state.get('plan', []))
    # Get index updated by AssistantAgent in its *previous* execution
    current_step_idx = state.get('current_step_idx', 0)

    logger.debug(f"Entering should_continue_based_on_plan | Current Step: {current_step_idx + 1} | Plan length: {num_plan_steps}")

    # Check if the plan is complete
    if current_step_idx >= num_plan_steps:
        logger.debug("Plan complete. Ending.\n")
        return END
    else:
        # Plan not complete, check if Assistant decided to call a tool
        logger.debug("Plan not complete. Checking assistant action.")
        if state.get("tool_call"):
            logger.debug("Routing to Tool.\n")
            return "tool" # Route to Tool node
        else:
            # No tool call, means Assistant asked clarification or gave response/chitchat
            # Needs input from user next.
            logger.debug("Routing to User.\n")
            return "user" # Route to User node
# ---------------------------------------------------------


def should_continue_after_tool(state: DialogueState) -> str | LiteralString:
    """
    Checks if workflow should continue or abort after tool execution.
    If an error occurred during tool execution, ends the workflow.
    Otherwise, continues to the memory agent.

    Args:
        state: DialogueState containing tool execution results and error flags.

    Returns:
        "memory" if tool executed successfully and workflow should continue,
        END if there was an error during tool execution and workflow should abort.
    """
    if state.get("abort_due_to_error", False):
        logger.debug("Aborting due to error in tool execution. Ending.\n")
        return END
    else:
        logger.debug("Tool executed successfully. Continuing to memory update.\n")
        return "memory"


def build_graph(
        llm: VLLMClient | WatsonxLLM, tools_schema: list[dict],
        partition_cumsum: list[int], prompts_config_path: str,
        generation_strategy: str = "chat", max_retries: int = 2
    ) -> StateGraph:
    """
    Build and compile a LangGraph workflow for multi-agent dialogue synthesis.

    Initializes User, Assistant, Tool, and Memory agents with their respective prompts
    and defines the control flow: User -> Memory -> Assistant -> (Tool/User) with 
    conditional routing based on plan completion and assistant actions.

    Args:
        llm: Language model client (WatsonxLLM or VLLMClient)
        tools_schema: List of tool definitions
        partition_cumsum: Cumulative partition indices for dialogue structure
        prompts_config_path: Path to YAML config containing prompt file paths
        generation_strategy: Strategy for text generation (default: "chat")
        max_retries: Maximum retry attempts for tool/assistant agents (default: 2)

    Returns:
        Compiled LangGraph workflow

    Raises:
        FileNotFoundError: If any prompt file paths are invalid
    """
    workflow = StateGraph(DialogueState)

    # Load the paths of the prompt files from the YAML config
    with open(prompts_config_path, 'r') as file:
        prompts_config = yaml.safe_load(file)

    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    prompt_keys = [
        "USER_UTTERER_PROMPT_PATH",
        "USER_CLARIFIER_PROMPT_PATH",
        "USER_CHITCHATTER_PROMPT_PATH",

        "ASSISTANT_CLARIFIER_PROMPT_PATH",
        "ASSISTANT_CHITCHATTER_PROMPT_PATH",
        "ASSISTANT_TOOL_CALLER_PROMPT_PATH",
        "ASSISTANT_CLARIFICATION_REFINER_PROMPT_PATH",
        "ASSISTANT_TOOL_RESPONSE_SUMMARIZER_PROMPT_PATH",

        "TOOL_PROMPT_PATH",
        "MEMORY_PROMPT_PATH",
    ]

    prompt_paths = {key: os.path.join(project_root, prompts_config.get(key)) for key in prompt_keys}

    user_utterer_prompt_path = prompt_paths["USER_UTTERER_PROMPT_PATH"]
    user_clarifier_prompt_path = prompt_paths["USER_CLARIFIER_PROMPT_PATH"]
    user_chitchatter_prompt_path = prompt_paths["USER_CHITCHATTER_PROMPT_PATH"]

    assistant_clarifier_prompt_path = prompt_paths["ASSISTANT_CLARIFIER_PROMPT_PATH"]
    assistant_chitchatter_prompt_path = prompt_paths["ASSISTANT_CHITCHATTER_PROMPT_PATH"]
    assistant_tool_caller_prompt_path = prompt_paths["ASSISTANT_TOOL_CALLER_PROMPT_PATH"]

    assistant_clarification_refiner_prompt_path = prompt_paths["ASSISTANT_CLARIFICATION_REFINER_PROMPT_PATH"]
    assistant_tool_response_summarizer_prompt_path = prompt_paths["ASSISTANT_TOOL_RESPONSE_SUMMARIZER_PROMPT_PATH"]

    tool_prompt_path = prompt_paths["TOOL_PROMPT_PATH"]
    memory_prompt_path = prompt_paths["MEMORY_PROMPT_PATH"]

    if not all(os.path.exists(path) for path in prompt_paths.values()):
        raise FileNotFoundError("One or more prompt files do not exist. Please check the paths.")

    workflow.add_node("tool", ToolAgent(
            llm, tools_schema, tool_prompt_path,
            generation_strategy=generation_strategy,
            max_retries=max_retries,
        )
    )
    workflow.add_node("memory", MemoryAgent(
            llm, tools_schema, memory_prompt_path,
            generation_strategy=generation_strategy,
        )
    )
    workflow.add_node("user", UserAgent(
            llm, tools_schema, partition_cumsum,
            user_utterer_prompt_path, user_clarifier_prompt_path, user_chitchatter_prompt_path,
            generation_strategy=generation_strategy,
        )
    )
    workflow.add_node("assistant", AssistantAgent(
            llm, tools_schema, partition_cumsum,
            assistant_tool_caller_prompt_path, assistant_clarifier_prompt_path,
            assistant_tool_response_summarizer_prompt_path, assistant_chitchatter_prompt_path,
            assistant_clarification_refiner_prompt_path,
            generation_strategy=generation_strategy, max_retries=max_retries,
        )
    )

    workflow.set_entry_point("user")

    # --- Define Edges for the Separate MemoryAgent Flow ---
    # User runs -> Update Memory based on user message
    workflow.add_edge("user", "memory")

    # Memory runs -> Assistant decides action based on updated memory
    workflow.add_edge("memory", "assistant")

    # Assistant runs -> Check if plan complete OR route to Tool/User
    workflow.add_conditional_edges(
        "assistant",
        should_continue_based_on_plan, # Use the function checking completed_idx
        {
            "tool": "tool", # If function returns "tool", go to tool node
            "user": "user", # If function returns "user", go to user node
            END: END        # If function returns END, terminate graph
        }
    )

    # Tool runs -> Check for errors, then update Memory or end
    workflow.add_conditional_edges(
        "tool",
        should_continue_after_tool,
        {
            "memory": "memory", # If no errors, continue to memory
            END: END            # If errors, terminate graph
        }
    )
    # --------------------------------------------------

    # Compile the graph
    custom_graph = workflow.compile()

    return custom_graph
