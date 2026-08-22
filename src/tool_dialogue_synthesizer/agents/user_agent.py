import json
import logging
import os
import re

from langchain_core.runnables import Runnable

from src.tool_dialogue_synthesizer.llm.vllm_llm import LLMResponse as VLLMResponse, VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import LLMResponse as WatsonxLLMResponse, WatsonxLLM
from src.tool_dialogue_synthesizer.schema import DialogueState
from src.tool_dialogue_synthesizer.utils.prompts import load_prompt
from src.tool_dialogue_synthesizer.utils.tool_call_schema import filter_tool_schemas


logger = logging.getLogger('dialogue_generator')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def replace_placeholders(
    template: str,
    conversations: list[dict],
    tool_definitions: list | dict | None,
    params_to_be_provided: list[str] | None,
    curr_plan_desc: str | None,
)-> str:
    """
    Replace placeholder variables in a template string with actual user agent context values.

    This function processes a template string and substitutes placeholder markers with
    JSON-formatted data from the user agent context, including conversation history,
    tool definitions, and parameters to be provided.

    Args:
        template: The template string containing placeholder markers
        conversations: List of conversation turn dictionaries with 'role' and 'content' keys
        tool_definitions: Optional list or dictionary of tool definitions to include in the prompt
        params_to_be_provided: Optional list of parameter names that the user needs to provide
        curr_plan_desc: Optional description of the current plan step for user utterances

    Returns:
        The template string with all placeholders replaced by their corresponding values
        in JSON format
    """
    if isinstance(tool_definitions, (list, dict)):
        tool_definitions = json.dumps(tool_definitions, indent=2)

    if isinstance(conversations, list):
        history = json.dumps(conversations, indent=2)

    mod_template = template.replace("{{history}}", history)

    if tool_definitions is not None:
        mod_template = mod_template.replace("{{tool_definitions}}", tool_definitions)

    if params_to_be_provided is not None and len(params_to_be_provided) > 0:
        mod_template = mod_template.replace("{{params_to_be_provided}}", ", ".join(params_to_be_provided))
    else:
        mod_template = mod_template.replace("{{params_to_be_provided}}", "None")

    if curr_plan_desc is not None:
        mod_template = mod_template.replace("{{curr_plan_desc}}", curr_plan_desc)

    return mod_template


def parse_user_response(llm_output_content: str, sample_idx: int) -> str:
    """
    Extract the user message from LLM output with flexible tag parsing.

    Attempts to extract content from within <user>...</user> tags using multiple
    fallback strategies to handle cases where the LLM may not properly format
    the output with complete tags.

    Args:
        llm_output_content: The raw output content from the LLM
        sample_idx: The index of the current sample for logging purposes

    Returns:
        The extracted user message content, stripped of whitespace
    """
    # Pattern to match content within <user>...</user> (non-greedy)
    # Also handles cases where <user> might be missing but </user> is present at the end.
    # It prioritizes the full <user>...</user> match first.
    # Then tries to match if the string simply ends with </user>, capturing content before it.

    # Try standard extraction first
    match_full = re.search(r"<user>(.*?)</user>", llm_output_content, re.DOTALL)
    if match_full:
        return match_full.group(1).strip()

    # Fallback: Try to find content if only the closing tag is present at the very end
    # This assumes the relevant content is everything before the final </user>
    # Note: This might be too greedy if there's unrelated preceding text.
    match_end_tag = re.search(r"(.*)</user>\s*$", llm_output_content, re.DOTALL)
    if match_end_tag:
        # Check if <user> is somewhere before the final </user>
        potential_content = match_end_tag.group(1)
        start_tag_pos = potential_content.rfind('<user>')
        if start_tag_pos != -1:
            # If <user> exists, take content after the *last* <user>
            return potential_content[start_tag_pos + len('<user>'):].strip()
        else:
            # If no <user> tag found before the final </user>, assume all content before it is the message
            # This might need adjustment depending on typical LLM failure modes
            return potential_content.strip()

    # If neither pattern matches, return the original stripped content as a last resort
    logger.warning(
        f"[UserAgent] Sample {sample_idx} -- Could not find standard <user>...</user> tags. Returning raw content."
    )
    return llm_output_content.strip()


class UserAgent(Runnable):
    """
    Agent responsible for simulating user behavior in dialogues.

    This agent handles multiple user dialogue tasks including initial utterances,
    and responses to clarifications. It orchestrates the user's side of conversations
    by generating appropriate messages based on the current step in a predefined plan.

    The agent supports two generation strategies:
    - "generate": Uses prompt templates for text generation
    - "chat": Uses chat-based prompts with system/user message separation

    Key Responsibilities:
    - USER_UTTERANCE: Generates initial user requests with parameter information
    - USER_RESPONSE_TO_CLARIFICATION: Provides answers to assistant clarification questions

    The agent filters tool schemas to show only relevant parameters based on the plan,
    ensuring the user message context includes appropriate tool information. It uses
    partition indices to select relevant tool subsets and maintains conversation coherence
    throughout multi-step dialogue interactions.

    Attributes:
        llm: Language model client for generating messages (VLLMClient or WatsonxLLM)
        tools_schema: List of tool definitions with parameter schemas
        partition_cumsum: Cumulative sum indices for partitioning tools by context
        generation_strategy: Strategy for prompt generation ("generate" or "chat")

    The agent loads different prompt templates based on the generation strategy and uses
    these to construct appropriate prompts for each user dialogue task.
    """
    def __init__(
        self, llm: VLLMClient | WatsonxLLM,
        tools_schema: list[dict], partition_cumsum: list[int],
        utterer_prompt_path: str, clarifier_prompt_path: str,
        generation_strategy: str = "chat",
    ) -> None:
        """
        Set up the user agent with LLM client, tool schemas, and prompt templates.

        Initializes the agent by storing the LLM client, tool definitions, and partition indices,
        then loading the appropriate prompt templates based on the generation strategy. For "chat"
        mode, loads separate system and user prompts for both utterance and clarification scenarios.
        For "generate" mode, loads single-file templates.

        Args:
            llm: Language model client for generating user messages (VLLMClient or WatsonxLLM)
            tools_schema: List of tool schema dictionaries defining available tools and their parameters
            partition_cumsum: Cumulative sum indices used to partition tools into context-specific subsets
            utterer_prompt_path: Path to prompt template(s) for initial user utterances - directory
                for "chat" mode or single file for "generate" mode
            clarifier_prompt_path: Path to prompt template(s) for clarification responses - directory
                for "chat" mode or single file for "generate" mode
            generation_strategy: LLM invocation mode - "generate" for completion-based or "chat"
                for message-based generation (default: "chat")

        Note:
            For "chat" strategy, each path should be a directory containing "system.txt" and
            "user.txt" files. For "generate" strategy, paths should point to single template files.
        """
        self.llm = llm
        self.tools_schema = tools_schema
        self.partition_cumsum = partition_cumsum
        self.generation_strategy = generation_strategy

        if generation_strategy != "chat":
            self.utterer_prompt_template = load_prompt(utterer_prompt_path)
            self.clarifier_prompt_template = load_prompt(clarifier_prompt_path)

        if generation_strategy != "generate":
            if generation_strategy == "chat":
                self.utterer_system_prompt, self.utterer_user_prompt = [
                    load_prompt(os.path.join(utterer_prompt_path, f"{role}.txt"))
                    for role in ("system", "user")
                ]
                self.clarifier_system_prompt, self.clarifier_user_prompt = [
                    load_prompt(os.path.join(clarifier_prompt_path, f"{role}.txt"))
                    for role in ("system", "user")
                ]


    def _invoke_llm(self, prompt_content: str | list[dict], **kwargs) -> VLLMResponse | WatsonxLLMResponse:
        """
        Invoke the LLM with the provided prompt content and optional parameters.

        Args:
            prompt_content: The prompt to send to the LLM (can be a string or list of message dicts)
            **kwargs: Additional keyword arguments to pass to the LLM invoke method
                (e.g., use_chat_mode, tools, tool_choice_option)

        Returns:
            LLM response object containing the generated content, either
            VLLMResponse or WatsonxLLMResponse depending on the LLM client type
        """
        return self.llm.invoke(prompt_content, **kwargs)


    def invoke(self, state: DialogueState, config=None) -> DialogueState:
        """
        Process the dialogue state and generate appropriate user messages based on the plan.

        This method is the main entry point for the UserAgent. It processes the current
        plan step to determine the type of user turn (USER_UTTERANCE or USER_RESPONSE_TO_CLARIFICATION),
        constructs appropriate prompts with filtered tool schemas and conversation context,
        invokes the LLM to generate user messages, and updates the dialogue state.

        The method handles two types of user turns:
        - USER_UTTERANCE: Generates initial user requests where parameters are provided
        - USER_RESPONSE_TO_CLARIFICATION: Generates responses to assistant clarification requests

        Args:
            state: The current dialogue state containing plan, conversations, current_step_idx,
                sample_idx, and partition_idx
            config: Optional configuration parameter (unused, for compatibility with Runnable interface)

        Returns:
            Updated DialogueState with the user message added to conversations, user_message field
            set, and current_step_idx incremented. Returns unchanged state if step type is
            unrecognized or plan length is exceeded
        """
        current_step_idx = state.get('current_step_idx', 0)
        plan = state.get("plan", [])
        conversations = state.get("conversations", [])
        sample_idx = state.get("sample_idx", 0)

        if current_step_idx >= len(plan):
            logger.warning(
                f"[UserAgent] Sample {sample_idx} -- "
                f"Step {current_step_idx + 1} exceeds plan length {len(plan)}"
            )
            return state

        step_type = plan[current_step_idx].get("type", "")

        if step_type == "USER_UTTERANCE":
            curr_plan_desc = plan[current_step_idx].get("utterance", "")
            provided_params = list(plan[current_step_idx].get("provided_params", {}).keys())

            if provided_params == []:
                relevant_tools = []

            else:
                partition_idx = state.get("partition_idx", 0)
                if partition_idx >= len(self.partition_cumsum) - 1:
                    logger.warning(
                        f"[UserAgent] Sample {sample_idx} -- "
                        f"Partition index {partition_idx} exceeds partition size {len(self.partition_cumsum) - 1}"
                    )
                    return state

                relevant_tools = self.tools_schema[
                    self.partition_cumsum[partition_idx]:self.partition_cumsum[partition_idx + 1]
                ]

            if relevant_tools:
                filtered_tools = filter_tool_schemas(
                    relevant_tools, provided_params, filter_mode="keep_inputs",
                )
            else:
                filtered_tools = []

            prompt = replace_placeholders(
                self.utterer_prompt_template if self.generation_strategy != "chat" else self.utterer_user_prompt,
                conversations,
                filtered_tools if self.generation_strategy != "chat" else None,
                provided_params,
                curr_plan_desc
            )

            if self.generation_strategy == "chat":
                prompt = [
                    {"role": "system", "content": self.utterer_system_prompt},
                    {"role": "user", "content": prompt}
                ]

                response = self._invoke_llm(
                    prompt, use_chat_mode=True,
                    tools=relevant_tools, tool_choice_option="none",
                )
                llm_output_content = response.content.strip()
                user_message = llm_output_content

            else:
                response = self._invoke_llm(prompt, use_chat_mode=False)
                llm_output_content = response.content.strip()
                user_message = parse_user_response(llm_output_content, sample_idx)


        elif step_type == "USER_RESPONSE_TO_CLARIFICATION":
            params_to_be_provided = list(plan[current_step_idx].get("provides_params", {}).keys())

            if not params_to_be_provided:
                logger.warning(
                    f"[UserAgent] Sample {sample_idx} -- "
                    f"No parameters to be provided in step {current_step_idx + 1} ({step_type})"
                )
                return state

            partition_idx = state.get("partition_idx", 0)
            if partition_idx >= len(self.partition_cumsum) - 1:
                logger.warning(
                    f"[UserAgent] Sample {sample_idx} -- "
                    f"Partition index {partition_idx} exceeds partition size {len(self.partition_cumsum) - 1}"
                )
                return state

            relevant_tools = self.tools_schema[
                self.partition_cumsum[partition_idx]:self.partition_cumsum[partition_idx + 1]
            ]

            if relevant_tools:
                filtered_tools = filter_tool_schemas(
                    relevant_tools, params_to_be_provided, filter_mode="keep_inputs",
                )
            else:
                filtered_tools = []

            prompt = replace_placeholders(
                self.clarifier_prompt_template if self.generation_strategy != "chat" else self.clarifier_user_prompt,
                conversations,
                filtered_tools if self.generation_strategy != "chat" else None,
                params_to_be_provided,
                curr_plan_desc=None
            )

            if self.generation_strategy == "chat":
                prompt = [
                    {"role": "system", "content": self.clarifier_system_prompt},
                    {"role": "user", "content": prompt}
                ]

                response = self._invoke_llm(
                    prompt, use_chat_mode=True,
                    tools=relevant_tools, tool_choice_option="none",
                )
                llm_output_content = response.content.strip()
                user_message = llm_output_content

            else:
                response = self._invoke_llm(prompt, use_chat_mode=False)
                llm_output_content = response.content.strip()
                user_message = parse_user_response(llm_output_content, sample_idx)


        else:
            logger.warning(
                f"[UserAgent] Sample {sample_idx} -- "
                f"Unrecognized step type '{step_type}' at step {current_step_idx + 1}."
            )
            return state

        if step_type == "USER_UTTERANCE":
            LOG_PREFIX = "Utterance"
        elif step_type == "USER_RESPONSE_TO_CLARIFICATION":
            LOG_PREFIX = "Clarification"

        logger.debug(f"[UserAgent {LOG_PREFIX}] {user_message}")

        logger.info(
            f"[UserAgent] Sample {sample_idx} -- "
            f"Step {current_step_idx+1}/{len(plan)} - {step_type} - DONE"
        )

        conversations.append({
            "role": "user",
            "content": user_message
        })

        return {
            **state,
            "user_message": user_message,
            "conversations": conversations,
            "current_step_idx": current_step_idx + 1
        }
