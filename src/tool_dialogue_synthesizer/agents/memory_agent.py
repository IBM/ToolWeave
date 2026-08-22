import json
import logging
import os
from typing import Any
from pprint import pformat

from langchain_core.runnables import Runnable

from src.tool_dialogue_synthesizer.llm.vllm_llm import LLMResponse as VLLMResponse, VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import LLMResponse as WatsonxLLMResponse, WatsonxLLM
from src.tool_dialogue_synthesizer.schema import DialogueState
from src.tool_dialogue_synthesizer.utils.prompts import load_prompt
from src.tool_dialogue_synthesizer.utils.tool_call_schema import coerce_types_from_schema


logger = logging.getLogger('dialogue_generator')


def replace_placeholders(
    template: str,
    conversations: list[dict],
    memory_cache: dict,
    latest_user_message: str | None,
    latest_tool_call: dict | None,
    tools_schema: list[dict] | None = None
) -> str:
    """
    Replace placeholder variables in a template string with actual values.

    This function processes a template string and substitutes placeholder markers
    with JSON-formatted data from the conversation history, memory cache, and
    latest interaction details.

    Args:
        template: The template string containing placeholder markers
        conversations: List of conversation turn dictionaries representing the dialogue history
        memory_cache: Dictionary containing cached memory state
        latest_user_message: The most recent message from the user, or None
        latest_tool_call: Dictionary containing the most recent tool call information, or None
        tools_schema: Optional list of tool schema dictionaries

    Returns:
        The template string with all placeholders replaced by their corresponding values in JSON format
    """
    if isinstance(conversations, list):
        history = json.dumps(conversations, indent=2)

    mod_template = template.replace("{{history}}", history)
    mod_template = mod_template.replace("{{memory_cache}}", json.dumps(memory_cache, indent=2))

    if latest_user_message != "":
        mod_template = mod_template.replace("{{latest_user_message}}", latest_user_message)
    else:
        mod_template = mod_template.replace("{{latest_user_message}}", "N/A")

    if latest_tool_call:
        mod_template = mod_template.replace(
            "{{latest_tool_call}}",
            json.dumps(latest_tool_call, indent=2)
        )
    else:
        mod_template = mod_template.replace("{{latest_tool_call}}", "N/A")

    if tools_schema:
        mod_template = mod_template.replace(
            "{{tools_schema}}",
            json.dumps(tools_schema, indent=2)
        )

    return mod_template


class MemoryAgent(Runnable):
    """
    Agent responsible for maintaining and updating a memory cache during dialogue generation.

    The MemoryAgent uses an LLM to intelligently update a structured memory cache based on
    conversation history, user messages, and tool execution results. It validates LLM-suggested
    memory updates against tool schemas to ensure only valid parameters are stored.

    The agent supports two generation strategies:
    - "generate": Uses a single prompt template for direct text generation
    - "chat": Uses separate system and user prompts for chat-based generation

    Memory is organized by tool names as top-level keys, with validated parameter names
    and values stored underneath. The agent performs strict validation to filter out
    invalid parameters that don't match the tool schema definitions.

    Attributes:
        llm: The language model client (VLLM or Watsonx) used for memory updates
        tools_schema: List of tool schema dictionaries defining valid tools and parameters
        generation_strategy: Strategy for LLM invocation ("generate" or "chat")
        _valid_tool_to_param_names_map: Internal mapping of tool names to their valid
            parameter names for validation purposes
    """
    def __init__(
        self, llm: VLLMClient | WatsonxLLM, tools_schema: list[dict],
        prompt_path: str, generation_strategy: str = "chat",
    ) -> None:
        """
        Configure the memory agent with an LLM client and load appropriate prompt templates.

        Sets up the agent by storing the LLM client and tool schemas, loading prompt templates
        based on the generation strategy, and building an internal map of valid tool parameters
        for validation purposes.

        Args:
            llm: Language model client for generating memory updates (VLLMClient or WatsonxLLM)
            tools_schema: List of tool schema dictionaries defining valid tools and their parameters
            prompt_path: Path to prompt template file(s) - single file for "generate" strategy,
                or directory containing "system.txt" and "user.txt" for "chat" strategy
            generation_strategy: Approach for LLM invocation - "generate" for completion
                or "chat" for message-based generation (default: "chat")
        """
        self.llm = llm
        self.tools_schema = tools_schema # Needed for context in prompt
        self.generation_strategy = generation_strategy

        if generation_strategy == "generate":
            self.memory_update_prompt_template = load_prompt(prompt_path)

        else:
            self.system_prompt = load_prompt(
                os.path.join(prompt_path, "system.txt")
            )
            self.user_prompt = load_prompt(
                os.path.join(prompt_path, "user.txt")
            )

        self._valid_tool_to_param_names_map = self._extract_tool_param_map(tools_schema)


    @staticmethod
    def _get_return_attributes(result_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Extract property definitions from a tool's results schema.

        Handles different result schema types (object, array) and extracts the
        relevant property definitions that can be stored in the memory cache.

        Args:
            result_schema: The results schema dictionary from a tool definition,
                which may have type "object", "array", or other types

        Returns:
            Dictionary of property definitions if the schema is an object or array
            of objects, the original schema if it's a primitive type, or an empty
            dictionary if properties cannot be extracted
        """
        if result_schema.get("type") == "object":
            return result_schema.get("properties", {})

        elif result_schema.get("type") == "array":
            items_schema = result_schema.get("items", {})

            if items_schema.get("type") == "object":
                return items_schema.get("properties", {})

            else:
                return {}

        else:
            return result_schema


    def _extract_tool_param_map(self, tools_schema: list) -> dict[str, set[str]]:
        """
        Extract a mapping of tool names to their valid parameter names from the tools schema.

        Builds a dictionary where each key is a tool name and the value is a set containing
        all valid parameter names for that tool, including both input parameters and result
        attributes. This mapping is used to validate LLM-suggested memory updates against
        the tool schema definitions.

        Args:
            tools_schema: List of tool schema dictionaries, where each tool entry follows
                the OpenAI tool format with a "function" definition containing
                "parameters" and optional "results"

        Returns:
            Dictionary mapping tool names to sets of valid parameter names,
            or an empty dictionary if tools_schema is not a list or contains no valid tools
        """
        tool_param_map = {}
        if not isinstance(tools_schema, list):
            logger.error(
                "[MemoryAgent] tools_schema is not a list "
                f"(type: {type(tools_schema)}). Cannot extract parameter map."
            )
            return tool_param_map

        for tool_entry in tools_schema:
            if isinstance(tool_entry, dict) and \
                    tool_entry.get('type') == 'function' and \
                    isinstance(tool_entry.get('function'), dict):

                function_details = tool_entry['function']
                tool_name = function_details.get('name')
                if not tool_name:
                    logger.warning(
                        f"[MemoryAgent] Found a tool entry without a name: {tool_entry}"
                    )
                    continue

                current_tool_params_set = set()
                parameters_obj = function_details.get('parameters')

                if isinstance(parameters_obj, dict) and \
                    parameters_obj.get('type') == 'object' and \
                    isinstance(parameters_obj.get('properties'), dict):

                    for param_name in parameters_obj['properties'].keys():
                        current_tool_params_set.add(param_name)

                results_obj = function_details.get('results')
                if isinstance(results_obj, dict):
                    return_attributes = self._get_return_attributes(results_obj)
                    if isinstance(return_attributes, dict):
                        for return_param_name in return_attributes.keys():
                            current_tool_params_set.add(return_param_name)

                # Even if a tool has no parameters, it's a valid tool.
                # Storing it with an empty set allows validation if LLM tries to add params to it.
                tool_param_map[tool_name] = current_tool_params_set

        return tool_param_map


    def _invoke_llm(self, prompt_content: str, **kwargs) -> VLLMResponse | WatsonxLLMResponse:
        """
        Invoke the LLM with the appropriate generation strategy.

        Routes the LLM invocation based on the configured generation strategy,
        using either direct text generation or chat-based generation with
        system and user prompts.

        Args:
            prompt_content: The prompt content to send to the LLM
            **kwargs: Additional keyword arguments to pass to the LLM invoke
                method (e.g., tools, tool_choice_option for chat mode)

        Returns:
            LLM response object containing the generated content, either
            VLLMResponse or WatsonxLLMResponse depending on the LLM client type
        """
        if self.generation_strategy == "generate":
            return self.llm.invoke(prompt_content, use_chat_mode=False)

        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_content}
            ]
            return self.llm.invoke(messages, use_chat_mode=True, **kwargs)


    def invoke(self, state: DialogueState, config=None) -> DialogueState:
        """
        Process the current dialogue state and update the memory cache based on LLM suggestions.

        This method is the main entry point for the MemoryAgent. It extracts conversation history,
        user messages, and tool execution results from the state, prompts the LLM to suggest memory
        updates, validates the suggestions against tool schemas, and returns an updated state
        with the new memory cache.

        Args:
            state: The current dialogue state containing conversations, memory_cache,
                user_message, tool_call, and sample_idx
            config: Optional configuration parameter (unused, for compatibility with Runnable interface)

        Returns:
            Updated DialogueState with the new memory_cache and cleared user_message and tool_call fields
        """
        conversations = state.get("conversations", [])
        memory_cache_current = state.get('memory_cache', {})
        user_message = state.get('user_message', "")
        tool_call = state.get("tool_call", None)
        sample_idx = state.get("sample_idx", 0)

        if user_message == "" and not tool_call:
            logger.warning(
                f"[MemoryAgent] Sample {sample_idx} -- No user message "
                "or tool call to process. Returning current state."
            )
            return state

        if tool_call and "response" not in tool_call:
            logger.warning(
                f"[MemoryAgent] Sample {sample_idx} -- Tool call failed. Returning control "
                f"to the assistant agent in the tool calling mode."
            )
            return state

        memory_prompt = replace_placeholders(
            self.memory_update_prompt_template if self.generation_strategy == "generate" else self.user_prompt,
            conversations,
            memory_cache_current,
            user_message,
            tool_call,
            tools_schema=self.tools_schema if self.generation_strategy == "generate" else None,
        )

        if self.generation_strategy == "generate":
            memory_response = self._invoke_llm(memory_prompt)

        else:
            memory_response = self._invoke_llm(
                memory_prompt, tools=self.tools_schema, tool_choice_option="none",
            )

        llm_memory_output = memory_response.content.strip()

        cleaned_content = llm_memory_output.replace("```json", "").replace("```", "").strip()
        try:
            parsed_memory = json.loads(cleaned_content)
        except json.JSONDecodeError:
            logger.exception(
                f"[MemoryAgent] Sample {sample_idx} -- Failed to parse LLM output as JSON. "
                f"Raw output: {cleaned_content}"
            )
            return {
                **state,
                "memory_cache": memory_cache_current,
                "tool_call": None,
                "user_message": "",
            }

        if self.tools_schema:
            for tool_entry in self.tools_schema:
                if isinstance(tool_entry, dict) and \
                    tool_entry.get('type') == 'function' and \
                    isinstance(tool_entry.get('function'), dict):

                    function_details = tool_entry['function']
                    tool_name = function_details.get('name')

                    if not tool_name:
                        logger.warning(
                            f"[MemoryAgent] Sample {sample_idx} -- Found a tool entry without a name: {tool_entry}"
                        )
                        continue

                    if tool_name not in self._valid_tool_to_param_names_map:
                        logger.warning(
                            f"[MemoryAgent] Sample {sample_idx} -- Tool '{tool_name}' not found in valid tools map. "
                            "Skipping validation for this tool."
                        )
                        continue

                    if not self._valid_tool_to_param_names_map[tool_name]:
                        continue

                    parsed_memory = coerce_types_from_schema(
                        parsed_memory,
                        function_details.get('parameters', {})
                    )

                    results_schema = function_details.get('results', {})
                    if results_schema.get('type') == 'object':
                        parsed_memory = coerce_types_from_schema(parsed_memory, results_schema)
                    elif results_schema.get('type') == 'array':
                        parsed_memory = coerce_types_from_schema(parsed_memory, results_schema.get('items', {}))

        if isinstance(parsed_memory, dict):
            # This will be the new cache, strictly built from LLM's tool-scoped suggestions
            # that pass validation.
            strictly_validated_new_cache = {}
            global_cache = {}

            for tool_key, params_dict in parsed_memory.items():
                # Check if the top-level key from LLM is a known tool name
                if tool_key in self._valid_tool_to_param_names_map:
                    if not isinstance(params_dict, (dict, list)):
                        logger.warning(
                            f"[MemoryAgent] Sample {sample_idx} -- LLM suggested non-dictionary value for tool "
                            f"scope '{tool_key}'. Value: {params_dict}. Skipping this tool scope."
                        )
                        continue

                    if isinstance(params_dict, list):
                        logger.warning(
                            f"[MemoryAgent] Sample {sample_idx} -- LLM suggested a list for tool "
                            f"scope '{tool_key}'. Converting to dict with each key having list of values."
                        )

                        param_keys = set()
                        for d in params_dict:
                            if isinstance(d, dict):
                                param_keys.update(d.keys())
                            else:
                                logger.warning(
                                    f"[MemoryAgent] Sample {sample_idx} -- LLM suggested a list "
                                    f"with non-dict item: {d}. Skipping this item."
                                )

                        if not param_keys:
                            logger.warning(
                                f"[MemoryAgent] Sample {sample_idx} -- LLM suggested a list with "
                                f"no valid dict items for tool scope '{tool_key}'. Skipping this tool scope."
                            )
                            continue

                        params_dict = {k: [d[k] for d in params_dict] for k in param_keys}

                    valid_params_for_this_tool_schema = self._valid_tool_to_param_names_map[tool_key]
                    actual_params_to_store_for_tool = {}

                    for param_name, param_value in params_dict.items():
                        if param_name in valid_params_for_this_tool_schema:
                            actual_params_to_store_for_tool[param_name] = param_value
                        else:
                            logger.warning(
                                f"[MemoryAgent] Sample {sample_idx} -- Filtering out param key '{param_name}' "
                                f"under tool '{tool_key}'. Not a defined input for this tool. Value: '{param_value}'"
                            )

                    if actual_params_to_store_for_tool:  # Only add tool scope if it has valid params
                        strictly_validated_new_cache[tool_key] = actual_params_to_store_for_tool
                    elif not valid_params_for_this_tool_schema and not params_dict:  # Tool has no params by schema, LLM gave empty dict
                        strictly_validated_new_cache[tool_key] = {}  # Store empty dict for tools with no params
                    elif params_dict:  # LLM provided params but all were invalid for this tool
                        logger.warning(
                            f"[MemoryAgent] Sample {sample_idx} -- No valid parameters kept for tool "
                            f"'{tool_key}' after filtering LLM suggestion. Original LLM params for tool: {params_dict}"
                        )

                else:
                    global_cache[tool_key] = params_dict  # Store global cache entries

            updated_memory_cache = strictly_validated_new_cache  # The new cache is what the LLM provided, after filtering
            updated_memory_cache.update(global_cache)
            logger.debug(f"[MemoryAgent] Memory cache updated\n{pformat(updated_memory_cache)}")

        else:
            logger.error(
                f"[MemoryAgent] Sample {sample_idx} -- LLM output for memory update "
                f"was not a dictionary. Type: {type(parsed_memory)}. Raw: {cleaned_content}"
            )

        return {
            **state,
            "memory_cache": updated_memory_cache,
            "tool_call": None,
            "user_message": "",
        }
