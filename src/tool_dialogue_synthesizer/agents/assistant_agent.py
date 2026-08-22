import json
import logging
import os
import re
from pprint import pformat
from typing import Any

from langchain_core.runnables import Runnable

from src.tool_dialogue_synthesizer.llm.vllm_llm import LLMResponse as VLLMResponse, VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import LLMResponse as WatsonxLLMResponse, WatsonxLLM
from tool_dialogue_synthesizer.schema import DialogueState
from src.tool_dialogue_synthesizer.utils.prompts import load_prompt
from src.tool_dialogue_synthesizer.utils.tool_call_schema import (
    coerce_types_from_schema,
    filter_tool_schemas,
    validate_value_against_schema,
)


logger = logging.getLogger('dialogue_generator')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def replace_placeholders(
    template: str,
    conversations: list[dict],
    memory_cache: dict,
    tool_definitions: list | dict | None,
    params_to_be_provided: list[str] | list[dict[str, str]] | dict | None,
    input_schema: dict | None = None,
    error_msg: str | None = None,
)-> str:
    """
    Replace placeholder tokens in a prompt template with actual values.

    This utility function populates a template string by replacing predefined
    placeholder tokens with their corresponding values. It handles various data
    types and formats them appropriately (JSON serialization for complex structures,
    comma-separated strings for lists, etc.).

    Args:
        template: The template string containing placeholder tokens
        conversations: List of conversation message dictionaries with 'role' and 'content'
        memory_cache: Dictionary of cached parameter values organized by tool name
        tool_definitions: Optional tool schema(s) as a list or single dictionary
        params_to_be_provided: Optional parameters to be provided. Can be:
            - List of strings: Joined with ", "
            - List of dicts: JSON-formatted
            - Dict: JSON-formatted
            - Other: Converted to string
        input_schema: Optional input parameter schema dictionary
        error_msg: Optional error message from previous attempt

    Returns:
        The template string with all placeholders replaced by their values.
    """
    mod_template = template.replace("{{history}}", json.dumps(conversations, indent=2))

    if memory_cache is not None:
        mod_template = mod_template.replace("{{memory_cache}}", json.dumps(memory_cache, indent=2))

    if tool_definitions is not None:
        mod_template = mod_template.replace("{{tool_definitions}}", json.dumps(tool_definitions, indent=2))

    if input_schema is not None:
        mod_template = mod_template.replace("{{input_schema}}", json.dumps(input_schema, indent=2))

    if params_to_be_provided is not None:
        if isinstance(params_to_be_provided, list):
            if params_to_be_provided and isinstance(params_to_be_provided[0], str):
                mod_template = mod_template.replace("{{params_to_be_provided}}", ", ".join(params_to_be_provided))
            elif params_to_be_provided and isinstance(params_to_be_provided[0], dict):
                mod_template = mod_template.replace("{{params_to_be_provided}}", json.dumps(params_to_be_provided, indent=2))
            else:
                mod_template = mod_template.replace("{{params_to_be_provided}}", str(params_to_be_provided))

        elif isinstance(params_to_be_provided, dict):
            mod_template = mod_template.replace("{{params_to_be_provided}}", json.dumps(params_to_be_provided, indent=2))

    if error_msg:
        mod_template = mod_template.replace("{{error_msg}}", error_msg)
    else:
        mod_template = mod_template.replace("{{error_msg}}", "")

    return mod_template


def parse_assistant_response(llm_output_content: str, sample_idx: int) -> str:
    """
    Extract and clean assistant response content from LLM output.

    This function handles LLM outputs that may be wrapped in <assistant> tags,
    removing the tags and extracting the actual content. It also handles malformed
    tags by attempting to complete or clean them.

    Args:
        llm_output_content: Raw output string from the LLM
        sample_idx: Sample index for logging purposes

    Returns:
        Cleaned assistant message content with tags removed
    """
    assistant_message_raw = llm_output_content.strip()
    if assistant_message_raw.startswith("<assistant>") and not assistant_message_raw.endswith("</assistant>"):
        if "</assistant>" not in assistant_message_raw[len("<assistant>"):]:
            assistant_message_raw += "</assistant>"

    assistant_message_processed = re.sub(r"(<assistant>)+", "<assistant>", assistant_message_raw)
    assistant_message_processed = re.sub(r"(</assistant>)+", "</assistant>", assistant_message_processed)

    assistant_message_cleaned = assistant_message_processed
    extraction_successful = False

    match_full = re.search(r"<assistant>(.*?)</assistant>", assistant_message_processed, re.DOTALL)

    if match_full:
        assistant_message_cleaned = match_full.group(1).strip()
        extraction_successful = True

    elif assistant_message_processed.endswith("</assistant>"):
        match_end_tag = re.search(r"(.*)</assistant>\s*$", assistant_message_processed, re.DOTALL)
        if match_end_tag:
            potential_content = match_end_tag.group(1)
            start_tag_pos = potential_content.rfind('<assistant>')
            if start_tag_pos != -1:
                assistant_message_cleaned = potential_content[start_tag_pos + len('<assistant>'):].strip()
                extraction_successful = True
            else:
                assistant_message_cleaned = potential_content.strip()
                extraction_successful = True

    if not extraction_successful:
        if assistant_message_processed.startswith("<assistant>") or assistant_message_processed.endswith(
                "</assistant>"):
            logger.warning(
                f"[AssistantAgent] Sample {sample_idx} -- Could not reliably extract content from assistant "
                f"tags in: '{assistant_message_processed}'. Using processed content."
            )
        assistant_message_cleaned = assistant_message_processed

    return assistant_message_cleaned


class AssistantAgent(Runnable):
    """
    Agent responsible for simulating assistant behavior in the dialogues.

    This agent handles multiple dialogue tasks including tool calling, clarification requests,
    and summarizing tool responses. It orchestrates the assistant's side of conversations
    by generating appropriate responses based on the current step in a predefined plan.

    The agent supports two generation strategies:
    - "generate": Uses prompt templates for text generation
    - "chat": Uses chat-based prompts with system/user message separation

    Key Responsibilities:
    - CALL_TOOL: Extracts and validates parameters for tool calls from conversation context
    - ASSISTANT_CLARIFICATION: Generates questions to clarify missing parameter values
    - ASSISTANT_RESPONSE_TOOL: Summarizes tool outputs into natural language responses

    The agent implements retry logic with validation to ensure tool calls meet schema requirements.
    It leverages memory cache to prioritize previously known parameter values and maintains
    conversation state throughout multi-step dialogue interactions.

    Attributes:
        llm: Language model client for generating responses (VLLMClient or WatsonxLLM)
        tools_schema: List of tool definitions with parameter schemas
        partition_cumsum: Cumulative sum indices for partitioning tools
        generation_strategy: Strategy for prompt generation ("generate" or "chat")
        max_retries: Maximum number of retry attempts for tool parameter extraction

    The agent loads different prompt templates based on the generation strategy and uses
    these to construct appropriate prompts for each dialogue task.
    """
    def __init__(
        self, llm: VLLMClient | WatsonxLLM, tools_schema: list[dict],
        partition_cumsum: list[int], tool_caller_prompt_path: str, clarifier_prompt_path: str,
        tool_response_summarizer_prompt_path: str, clarification_refiner_prompt_path: str,
        generation_strategy: str = "chat", max_retries: int = 2,
    ) -> None:
        """
        Initialize the assistant agent with LLM client and prompt templates.

        Loads prompt templates from file paths according to the specified generation strategy.
        For "generate" strategy, loads single-file templates. For "chat" strategy, loads
        separate system and user prompt files from subdirectories.

        Args:
            llm: Language model client instance (VLLMClient or WatsonxLLM)
            tools_schema: Complete list of tool schema definitions with parameter specifications
            partition_cumsum: Cumulative indices for dividing tools into logical groups
            tool_caller_prompt_path: Path to tool calling prompt template(s)
            clarifier_prompt_path: Path to clarification prompt template(s)
            tool_response_summarizer_prompt_path: Path to tool response summary prompt template(s)
            clarification_refiner_prompt_path: Path to clarification refinement prompt template(s)
            generation_strategy: Generation mode - "generate" for completion or "chat" for messages (default: "chat")
            max_retries: Maximum retry attempts for tool parameter validation (default: 2)

        Note:
            For "chat" strategy, each prompt path should be a directory containing
            "system.txt" and "user.txt" files. For "generate" strategy, each path
            should point directly to a single template file.
        """
        self.llm = llm
        self.tools_schema = tools_schema
        self.partition_cumsum = partition_cumsum
        self.generation_strategy = generation_strategy
        self.max_retries = max_retries

        if generation_strategy == "generate":
            self.clarifier_prompt_template = load_prompt(clarifier_prompt_path)
            self.tool_caller_prompt_template = load_prompt(tool_caller_prompt_path)
            self.clarification_refiner_prompt_template = load_prompt(clarification_refiner_prompt_path)
            self.tool_response_summarizer_prompt_template = load_prompt(tool_response_summarizer_prompt_path)

        else:
            self.clarifier_system_prompt, self.clarifier_user_prompt = [
                load_prompt(os.path.join(clarifier_prompt_path, f"{role}.txt"))
                for role in ("system", "user")
            ]
            self.tool_caller_system_prompt, self.tool_caller_user_prompt = [
                load_prompt(os.path.join(tool_caller_prompt_path, f"{role}.txt"))
                for role in ("system", "user")
            ]
            self.clarification_refiner_system_prompt, self.clarification_refiner_user_prompt = [
                load_prompt(os.path.join(clarification_refiner_prompt_path, f"{role}.txt"))
                for role in ("system", "user")
            ]
            self.tool_response_summarizer_system_prompt, self.tool_response_summarizer_user_prompt = [
                load_prompt(os.path.join(tool_response_summarizer_prompt_path, f"{role}.txt"))
                for role in ("system", "user")
            ]


    def _invoke_llm(self, prompt_content: str | list[dict], **kwargs) -> VLLMResponse | WatsonxLLMResponse:
        """
        Invoke the language model with the given prompt content.

        Args:
            prompt_content: The prompt to send to the LLM. Can be a string or list of
                message dictionaries (for chat mode).
            **kwargs: Additional keyword arguments to pass to the LLM, such as:
                - use_chat_mode: Whether to use chat-based generation
                - tools: List of tool schemas for tool-calling models
                - tool_choice: Specific tool to force the model to use
                - tool_choice_option: Option for tool selection ("none", "auto", etc.)

        Returns:
            VLLMResponse or WatsonxLLMResponse: The LLM response object containing
                the generated content.
        """
        return self.llm.invoke(prompt_content, **kwargs)


    def _validate_tool_parameters(
        self,
        tool_name: str,
        llm_provided_params: dict[str, Any],
        memory_cache: dict[str, Any],
        sample_idx: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Validate and merge tool parameters from LLM and memory cache against the tool schema.

        This method validates tool call parameters by checking them against the tool's schema
        definition. It implements a priority system where memory cache values are preferred
        over LLM-provided values for required parameters. The validation process includes:

        1. Validates all required parameters, prioritizing memory cache values
        2. Validates optional parameters provided by the LLM
        3. Applies default values for optional parameters when available
        4. Performs schema validation (type, format, enum, etc.) for all parameter values

        Validation Priority for Required Parameters:
        - Priority 1: Valid value from memory_cache (if present and schema-valid)
        - Priority 2: Valid value from LLM-provided parameters (if not found/valid from memory)

        Args:
            tool_name: Name of the tool being validated
            llm_provided_params: Parameters extracted from the LLM's tool call response
            memory_cache: Dictionary containing cached parameter values from previous interactions,
                organized by tool name
            sample_idx: Index of the current sample for logging purposes

        Returns:
            A tuple containing:
            - dict[str, Any] | None: Validated parameters dictionary if successful, None if validation fails
            - str | None: Detailed error message string if validation fails, None if successful
                The error message includes diagnostic information about which parameters failed
                validation and why (missing, invalid type, failed schema validation, etc.).
        """
        tool_definition = next((t for t in self.tools_schema if t.get("function", {}).get("name") == tool_name), None)

        if not tool_definition:
            logger.error(
                f"[AssistantAgent Validator] Sample {sample_idx} -- "
                f"Tool definition not found for tool: '{tool_name}'"
            )
            return None, [f"Tool '{tool_name}' is not a recognized tool."]

        # Get the parameter properties and required list from the schema
        param_properties = tool_definition.get("function", {}).get("parameters", {}).get("properties", {})
        required_param_names_raw = tool_definition.get("function", {}).get("parameters", {}).get("required", [])

        required_param_names = [str(p) for p in required_param_names_raw]

        validated_params = {}
        missing_or_invalid_required_params_log = []

        tool_specific_cached_params = memory_cache.get(tool_name, {})
        logger.debug(
            f"[AssistantAgent Validator] For tool '{tool_name}', "
            f"tool-specific cached params: {tool_specific_cached_params}"
        )

        for req_param_name in required_param_names:
            param_value = None
            is_value_valid_and_set = False
            param_schema_for_validation = param_properties.get(req_param_name, {})
            log_entry_for_param = [f"Checking required param '{req_param_name}'"]

            # Priority 1: Value from memory_cache (if present and schema-valid)
            if req_param_name in tool_specific_cached_params:
                val_from_memory = tool_specific_cached_params[req_param_name]
                log_entry_for_param.append(f"Memory has: '{val_from_memory}'")

                if val_from_memory is not None and str(val_from_memory).strip() != "":
                    is_value_valid, error_msg = validate_value_against_schema(val_from_memory, param_schema_for_validation)
                    if is_value_valid:
                        param_value = val_from_memory
                        is_value_valid_and_set = True

                    else:
                        log_entry_for_param.append(error_msg or "Value from memory FAILED schema validation.")

                else:
                    log_entry_for_param.append("Value from memory cache is None or empty.")

            # Priority 2: Value from LLM's proposed parameters (if not found/valid from memory)
            if not is_value_valid_and_set and req_param_name in llm_provided_params:
                val_from_llm = llm_provided_params[req_param_name]
                log_entry_for_param.append(f"LLM provided: '{val_from_llm}'")

                if val_from_llm is not None and str(val_from_llm).strip() != "":
                    is_value_valid, error_msg = validate_value_against_schema(val_from_llm, param_schema_for_validation)
                    if is_value_valid:
                        param_value = val_from_llm
                        is_value_valid_and_set = True

                    else:
                        log_entry_for_param.append(error_msg or "Value provided by LLM FAILED schema validation.")

                else:
                    log_entry_for_param.append("Value provided by LLM is None or empty.")

            if is_value_valid_and_set:
                validated_params[req_param_name] = param_value

            else:
                # Add param name to the list of missing, and the detailed log entry
                missing_or_invalid_required_params_log.append(
                    f"  - Param '{req_param_name}': Required but not found or invalid. "
                    f"Log:\n    - " + "\n    - ".join(log_entry_for_param)
                )

        # If there are any missing or invalid required parameters, validation fails.
        if missing_or_invalid_required_params_log:
            joined = "\n".join(missing_or_invalid_required_params_log)
            logger.error(
                f"[AssistantAgent Validator] Sample {sample_idx} -- Detailed failure log for '{tool_name}':\n{joined}"
            )
            return None, joined

        for param_name, param_value_from_llm in llm_provided_params.items():
            if param_name not in required_param_names:
                if param_name in param_properties:
                    param_schema_for_validation = param_properties.get(param_name, {})

                    if param_value_from_llm is not None and str(param_value_from_llm).strip() != "":
                        is_value_valid, error_msg = validate_value_against_schema(param_value_from_llm, param_schema_for_validation)
                        if is_value_valid:
                            validated_params[param_name] = param_value_from_llm

                        else:
                            logger.debug(
                                f"[AssistantAgent Validator] Optional param '{param_name}' from LLM "
                                f"('{param_value_from_llm}') failed schema validation with error message: "
                                f"'{error_msg}'. Not included."
                            )

        for param_name, param_schema in param_properties.items():
            if param_name not in validated_params and "default" in param_schema:
                default_value = param_schema["default"]

                if default_value is None or str(default_value).strip() == "":
                    logger.debug(
                        f"[AssistantAgent Validator] Default value for optional "
                        f"param '{param_name}' is None or empty. Not included."
                    )
                    continue

                is_value_valid, error_msg = validate_value_against_schema(default_value, param_schema)
                if is_value_valid:
                    validated_params[param_name] = default_value

                else:
                    logger.debug(
                        f"[AssistantAgent Validator] Default value for optional param "
                        f"'{param_name}' ('{default_value}') failed schema validation with error message: "
                        f"'{error_msg}'. Not included."
                    )

        return validated_params, None


    def invoke(self, state: DialogueState, config=None) -> DialogueState:
        """
        Execute the assistant agent's logic for the current dialogue step.

        This method processes different types of dialogue steps based on the current plan,
        handling tool calls, clarifications, and tool response summarization. It implements
        retry logic for error recovery and maintains conversation state throughout.

        The method handles three main step types:
        1. CALL_TOOL: Extracts and validates tool parameters, then creates a tool call
        2. ASSISTANT_CLARIFICATION: Generates questions to clarify missing parameter values
        3. ASSISTANT_RESPONSE_TOOL: Summarizes tool outputs into natural language

        Args:
            state: The current dialogue state containing:
                - conversations: List of conversation messages
                - plan: List of dialogue steps to execute
                - current_step_idx: Index of the current step in the plan
                - memory_cache: Cached parameter values from previous interactions
                - sample_idx: Index of the current sample for logging
                - retry_count: Number of retry attempts (for CALL_TOOL steps)
                - error_msg: Error message from previous attempt (for CALL_TOOL steps)
                - partition_idx: Index for partitioning tool schemas
            config: Optional configuration (unused, for compatibility with Runnable interface)

        Returns:
            DialogueState: Updated dialogue state with:
            - Modified conversations list with new assistant messages/tool calls
            - Updated current_step_idx (incremented for completed steps)
            - tool_call: Tool call object (used in the langgraph workflow
                to determine whether to pass control to the tool agent)
            - retry_count: Reset to 0 after successful tool call
            - error_msg: Reset to None after successful tool call
            - partition_idx: Incremented after ASSISTANT_RESPONSE_TOOL steps
            - abort_due_to_error: True if max retries exceeded

        Behavior by Step Type:
        - CALL_TOOL:
            - Validates parameters against tool schema with retry logic
            - Prioritizes memory cache values over LLM-provided values
            - Returns error state if max retries exceeded
            - Appends tool call to conversations on success

        - ASSISTANT_CLARIFICATION:
            - Refines parameter list based on conversation context
            - Skips clarification if refinement determines it's unnecessary
            - Generates natural language clarification questions
            - Advances to next step

        - ASSISTANT_RESPONSE_TOOL:
            - Filters tool schemas to relevant outputs
            - Generates summary of tool execution results
            - Advances both step and partition indices
        """
        conversations = state.get("conversations", [])
        plan = state.get("plan", [])
        current_step_idx = state.get('current_step_idx', 0)
        memory_cache = state.get('memory_cache', {})
        sample_idx = state.get('sample_idx', 0)

        if current_step_idx >= len(plan):
            logger.warning(
                f"[AssistantAgent] Sample {sample_idx} -- "
                f"Step {current_step_idx + 1} exceeds plan length {len(plan)}"
            )
            return state

        current_step = plan[current_step_idx]
        step_type = current_step.get("type")

        if step_type == "CALL_TOOL":
            # The assistant first extracts parameters for the tool call
            # These parameters are then sent to the tool agent for simulation

            retry_count = state.get("retry_count", 0)  # Track retry attempts
            error_msg = state.get("error_msg", None)  # Error message from previous attempt

            if retry_count >= self.max_retries:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"Max retries ({self.max_retries}) reached for step {current_step_idx + 1}. Aborting."
                )
                return {
                    **state,
                    "abort_due_to_error": True,
                    "error_msg": f"Max retries ({self.max_retries}) reached for assistant tool caller."
                }

            tool_name = current_step.get("tool_name")
            tool_parameters = current_step.get("parameters", {})

            tool_schema = next((t for t in self.tools_schema if t.get("function", {}).get("name") == tool_name), None)

            if not tool_schema:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"Tool schema not found for tool: '{tool_name}'"
                )
                return state

            filtered_tool_schema = filter_tool_schemas(
                [tool_schema], tool_parameters, filter_mode="keep_inputs",
            )[0]

            prompt = replace_placeholders(
                self.tool_caller_prompt_template if self.generation_strategy == "generate" else self.tool_caller_user_prompt,
                conversations, memory_cache,
                filtered_tool_schema if self.generation_strategy == "generate" else None,
                tool_parameters, error_msg=error_msg,
                input_schema=filtered_tool_schema.get("function", {}).get("parameters", {}) \
                    if self.generation_strategy != "generate" else None,
            )

            if self.generation_strategy == "generate":
                llm_response_obj = self._invoke_llm(prompt, use_chat_mode=False)
                llm_response_content = llm_response_obj.content.replace("```json", "").replace("```", "").strip()

                assistant_message = parse_assistant_response(llm_response_content, sample_idx)

                try:
                    extracted_tool_parameters = json.loads(assistant_message) if assistant_message else {}
                except json.JSONDecodeError:
                    extracted_tool_parameters = {}

            else:
                prompt = [
                    {"role": "system", "content": self.tool_caller_system_prompt},
                    {"role": "user", "content": prompt}
                ]

                llm_response_obj = self._invoke_llm(
                    prompt, use_chat_mode=True,
                    tools=[tool_schema], tool_choice=tool_schema,
                )

                response_content = llm_response_obj.content
                extracted_tool_parameters = {}  # Start with a safe default

                # 1. First, check if the entire response content is a dictionary (i.e., a tool call)
                if isinstance(response_content, dict):
                    # If it is, we can safely get the 'arguments' key
                    arguments_data = response_content.get("arguments", {})

                    # 2. Now, perform the check from our last conversation to handle
                    #    whether 'arguments' is a string or an already-parsed dictionary.
                    if isinstance(arguments_data, str):
                        try:
                            extracted_tool_parameters = json.loads(arguments_data)
                        except json.JSONDecodeError:
                            print("Warning: 'arguments' was a string but could not be parsed as JSON.")
                            extracted_tool_parameters = {}

                    elif isinstance(arguments_data, dict):
                        extracted_tool_parameters = arguments_data

                    else:
                        print(
                            f"Warning: 'arguments' inside the tool call has an unexpected type: {type(arguments_data)}")
                        extracted_tool_parameters = {}

                else:
                    # The LLM returned a plain string, not a tool call object.
                    # We can log this and proceed gracefully with empty parameters.
                    logger.error(f"Info: LLM did not return a tool call. Got a string instead: '{response_content}'")
                    extracted_tool_parameters = {}

            if extracted_tool_parameters is {}:
                logger.warning(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"Tool call parameters for tool '{tool_name}' are empty or invalid. "
                    f"Retrying ({retry_count + 1}/{self.max_retries})..."
                )
                return self.invoke({
                    **state,
                    "retry_count": retry_count + 1,
                    "error_msg": f"Tool call parameters for tool '{tool_name}' are empty or invalid."
                })

            extracted_tool_parameters = coerce_types_from_schema(
                extracted_tool_parameters, tool_schema.get("function", {}).get("parameters", {})
            )

            validated_params, validation_errors = self._validate_tool_parameters(
                tool_name, extracted_tool_parameters, memory_cache, sample_idx
            )

            if validation_errors:
                if retry_count < self.max_retries:
                    logger.warning(
                        f"[AssistantAgent] Sample {sample_idx} -- "
                        f"Retrying ({retry_count + 1}/{self.max_retries})..."
                    )

                    return self.invoke({
                        **state,
                        "retry_count": retry_count + 1,
                        "error_msg": (
                            "Previous tool call validation failed with the following errors:\n"
                            + "\n".join(validation_errors)
                        ),
                    })

                else:
                    logger.error(
                        f"[AssistantAgent] Sample {sample_idx} -- "
                        f"Max retries ({self.max_retries}) reached. Giving up."
                    )
                    return {
                        **state,
                        "abort_due_to_error": True,
                        "error_msg": f"Max retries ({self.max_retries}) reached for tool calling assistant."
                    }

            tool_call = {"name": tool_name, "arguments": validated_params}

            conversations.append({
                "role": "assistant_tool_call",
                "content": json.dumps([tool_call]),
            })

            logger.debug(f"[AssistantAgent Tool Call]\n{pformat(tool_call)}")
            return {
                **state,
                "tool_call": tool_call,
                "conversations": conversations,
                "retry_count": 0,
                "error_msg": None,
            }


        elif step_type == "ASSISTANT_CLARIFICATION":
            # Ask for specific parameters mentioned in the plan
            parameter_names = current_step.get("parameter_names", [])

            if not parameter_names:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    "No parameter names provided for clarification in the current step."
                )
                return state

            partition_idx = state.get("partition_idx", 0)
            if partition_idx >= len(self.partition_cumsum) - 1:
                logger.warning(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"Partition index {partition_idx} exceeds partition size {len(self.partition_cumsum) - 1}"
                )
                return state

            relevant_tools = self.tools_schema[
                self.partition_cumsum[partition_idx]:self.partition_cumsum[partition_idx + 1]
            ]

            if relevant_tools:
                filtered_tools = filter_tool_schemas(
                    relevant_tools, parameter_names, filter_mode="keep_inputs",
                )
            else:
                filtered_tools = []

            refiner_prompt = replace_placeholders(
                self.clarification_refiner_prompt_template if self.generation_strategy == "generate" \
                    else self.clarification_refiner_user_prompt,
                conversations,
                memory_cache,
                filtered_tools if self.generation_strategy == "generate" else None,
                parameter_names
            )

            if self.generation_strategy == "generate":
                llm_response_obj = self._invoke_llm(refiner_prompt, use_chat_mode=False)
                llm_response_content = llm_response_obj.content.replace("```json", "").replace("```", "").strip()
                refiner_message = parse_assistant_response(llm_response_content, sample_idx)

            else:
                refiner_prompt = [
                    {"role": "system", "content": self.clarification_refiner_system_prompt},
                    {"role": "user", "content": refiner_prompt}
                ]

                llm_response_obj = self._invoke_llm(
                    refiner_prompt, use_chat_mode=True,
                    tools=relevant_tools, tool_choice_option="none",
                )
                refiner_message = llm_response_obj.content.strip()

            parameter_names = refiner_message.split(", ") if refiner_message else []

            if parameter_names == ["N/A"]:
                logger.warning(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"No clarification needed after refinement at step {current_step_idx + 1}. "
                    "Moving to tool call step."
                )

                for i in range(current_step_idx + 2, len(plan)):
                    plan[i]["step"] -= 2

                plan = plan[:current_step_idx] + plan[current_step_idx + 2:]

                return self.invoke({
                    **state,
                    "plan": plan,
                })

            prompt = replace_placeholders(
                self.clarifier_prompt_template if self.generation_strategy == "generate" else self.clarifier_user_prompt,
                conversations,
                memory_cache,
                filtered_tools if self.generation_strategy == "generate" else None,
                parameter_names
            )

            if self.generation_strategy == "generate":
                llm_response_obj = self._invoke_llm(prompt, use_chat_mode=False)
                clarification_message = parse_assistant_response(llm_response_obj.content.strip(), sample_idx)

            else:
                prompt = [
                    {"role": "system", "content": self.clarifier_system_prompt},
                    {"role": "user", "content": prompt}
                ]

                llm_response_obj = self._invoke_llm(
                    prompt, use_chat_mode=True,
                    tools=relevant_tools, tool_choice_option="none",
                )
                clarification_message = llm_response_obj.content.strip()

            if not clarification_message:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    "Clarification message is empty or invalid."
                )
                return state

            conversations.append({
                "role": "assistant",
                "content": clarification_message
            })

            logger.debug(f"[AssistantAgent Clarification] {clarification_message}")
            logger.info(
                f"[AssistantAgent] Sample {sample_idx} -- "
                f"Step {current_step_idx + 1}/{len(plan)} - {step_type} - DONE"
            )
            return {
                **state,
                "conversations": conversations,
                "current_step_idx": current_step_idx + 1  # Advance step after clarification
            }


        elif step_type == "ASSISTANT_RESPONSE_TOOL":
            # Summarize tool outputs based on the plan
            tools_to_summarize = current_step.get("summarizes_tools", [])
            outputs_to_provide = current_step.get("outputs_provided", [])

            if not tools_to_summarize:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    "No tools to summarize in the current step."
                )
                return state

            if not outputs_to_provide:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    "No outputs to provide in the current step."
                )
                return state

            partition_idx = state.get("partition_idx", 0)
            if partition_idx >= len(self.partition_cumsum) - 1:
                logger.warning(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    f"Partition index {partition_idx} exceeds partition size {len(self.partition_cumsum) - 1}"
                )
                return state

            relevant_tools = self.tools_schema[
                self.partition_cumsum[partition_idx]:self.partition_cumsum[partition_idx + 1]
            ]

            if relevant_tools:
                filtered_tools = filter_tool_schemas(
                    relevant_tools, outputs_to_provide, filter_mode="keep_outputs",
                )
            else:
                filtered_tools = []

            prompt = replace_placeholders(
                self.tool_response_summarizer_prompt_template if self.generation_strategy == "generate" \
                    else self.tool_response_summarizer_user_prompt,
                conversations,
                memory_cache,
                filtered_tools if self.generation_strategy == "generate" else None,
                outputs_to_provide
            )

            if self.generation_strategy == "generate":
                llm_response_obj = self._invoke_llm(prompt, use_chat_mode=False)
                assistant_response = parse_assistant_response(llm_response_obj.content.strip(), sample_idx)

            else:
                prompt = [
                    {"role": "system", "content": self.tool_response_summarizer_system_prompt},
                    {"role": "user", "content": prompt}
                ]

                llm_response_obj = self._invoke_llm(
                    prompt, use_chat_mode=True,
                    tools=relevant_tools, tool_choice_option="none",
                )
                assistant_response = llm_response_obj.content.strip()

            if not assistant_response:
                logger.error(
                    f"[AssistantAgent] Sample {sample_idx} -- "
                    "Assistant response is empty or invalid."
                )
                return state

            conversations.append({
                "role": "assistant",
                "content": assistant_response
            })

            logger.debug(f"[AssistantAgent Tool Summary] {assistant_response}")
            logger.info(
                f"[AssistantAgent] Sample {sample_idx} -- "
                f"Step {current_step_idx + 1}/{len(plan)} - {step_type} - DONE"
            )
            return {
                **state,
                "conversations": conversations,
                "current_step_idx": current_step_idx + 1,  # Advance step after response
                "partition_idx": state.get("partition_idx", 0) + 1  # Advance partition index
            }


        else:
            # Handle unexpected step types or user utterances (should not be processed by assistant)
            logger.warning(
                f"[AssistantAgent] Sample {sample_idx} -- Unexpected step type: {step_type}"
            )
            return state
