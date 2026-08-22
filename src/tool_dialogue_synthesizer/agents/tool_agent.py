import json
import logging
import os
from pprint import pformat
from typing import Any

from langchain_core.runnables import Runnable

from src.tool_dialogue_synthesizer.llm.vllm_llm import LLMResponse as VLLMResponse, VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import LLMResponse as WatsonxLLMResponse, WatsonxLLM
from src.tool_dialogue_synthesizer.schema import DialogueState
from src.tool_dialogue_synthesizer.utils.prompts import load_prompt
from src.tool_dialogue_synthesizer.utils.tool_call_schema import (
    coerce_types_from_schema,
    validate_value_against_schema,
)


logger = logging.getLogger('dialogue_generator')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def replace_placeholders(
    template: str, dialogue_history: list[dict], tool_params: dict,
    tool_definition: dict | None, decision_variables: dict | None,
    results_schema: dict | None, error_msg: str | None = None,
) -> str:
    """
    Replace placeholder variables in a template string with actual tool execution context values.

    This function processes a template string and substitutes placeholder markers with
    JSON-formatted data related to tool execution, including dialogue history, tool parameters,
    tool definitions, and validation schemas.

    Args:
        template: The template string containing placeholder markers
        dialogue_history: List of conversation turn dictionaries with 'role' and 'content' keys
        tool_params: Dictionary containing the parameters for the tool call
        tool_definition: Optional tool schema definition in OpenAI function format
        decision_variables: Optional dictionary containing decision variables for tool execution
            which are used in the case of conditional dialogues to control simulated output values
        results_schema: Optional schema defining the expected structure of tool results
        error_msg: Optional error message from a previous failed attempt

    Returns:
        The template string with all placeholders replaced by their corresponding values
        in JSON format
    """
    tool_params = json.dumps(tool_params, indent=2)
    template = template.replace("{{tool_params}}", tool_params)

    dialogue_history_str = json.dumps(dialogue_history, indent=2)
    template = template.replace("{{dialogue_history}}", dialogue_history_str)

    if tool_definition:
        tool_definition = json.dumps(tool_definition, indent=2)
        template = template.replace("{{tool_definition}}", tool_definition)

    if decision_variables:
        decision_variables = json.dumps(decision_variables, indent=2)
        template = template.replace("{{decision_variables}}", decision_variables)
    else:
        template = template.replace("{{decision_variables}}", "")

    if results_schema:
        template = template.replace("{{results_schema}}", json.dumps(results_schema, indent=2))
    else:
        template = template.replace("{{results_schema}}", "")

    if error_msg:
        template = template.replace("{{error_msg}}", error_msg)
    else:
        template = template.replace("{{error_msg}}", "")

    return template


def parse_tool_response(llm_output_content: str, sample_idx: int) -> dict | list[dict]:
    """
    Clean and parse the JSON response from the tool simulation LLM.

    Removes markdown code block formatting and attempts to parse the content
    as JSON. If parsing fails, returns an error dictionary instead.

    Args:
        llm_output_content: Raw string output from the LLM containing the tool response
        sample_idx: The index of the current sample for logging purposes

    Returns:
        Parsed tool response as a dictionary or list of dictionaries, or an error
        dictionary with "error" key if JSON parsing fails
    """
    cleaned_content = llm_output_content.replace("```json", "").replace("```", "").strip()

    try:
        tool_response = json.loads(cleaned_content)

    except (json.JSONDecodeError, ValueError) as e:
        logger.exception(
            f"[ToolAgent] Sample {sample_idx} -- Failed to parse "
            f"tool response JSON. Raw content: {llm_output_content}"
        )

        tool_response = {"error": f"Invalid JSON response from ToolAgent LLM: {e}"}

    return tool_response


class ToolAgent(Runnable):
    """
    Agent responsible for simulating tool execution and generating tool responses.

    The ToolAgent uses an LLM to simulate tool execution by generating realistic tool
    responses based on the tool call parameters, dialogue context, and tool schema.
    It validates the generated responses against the tool's results schema and supports
    retry logic for handling validation failures.

    The agent supports two generation strategies:
    - "generate": Uses a single prompt template for direct text generation
    - "chat": Uses separate system and user prompts for chat-based generation

    Attributes:
        llm: The language model client (VLLM or Watsonx) used for tool simulation
        tools_schema: List of tool schema dictionaries defining available tools
        generation_strategy: Strategy for LLM invocation ("generate" or "chat")
        max_retries: Maximum number of retry attempts for failed validations
    """
    def __init__(
        self, llm: VLLMClient | WatsonxLLM, tools_schema: list[dict], prompt_path: str,
        generation_strategy: str = "chat", max_retries: int = 2,
    ) -> None:
        """
        Configure the tool agent with LLM client, tool schemas, and prompt templates.

        Initializes the agent by storing the LLM client and tool definitions, loading prompt
        templates based on the generation strategy, and setting the maximum retry limit for
        handling validation failures during tool response generation.

        Args:
            llm: Language model client for simulating tool execution (VLLMClient or WatsonxLLM)
            tools_schema: List of tool schema dictionaries defining available tools and their results schemas
            prompt_path: Path to prompt template file(s) - single file for "generate" strategy,
                or directory containing "system.txt" and "user.txt" for "chat" strategy
            generation_strategy: LLM invocation mode - "generate" for completion or "chat"
                for message-based generation (default: "chat")
            max_retries: Maximum retry attempts when tool response validation fails (default: 2)
        """
        # Store the tools schema and LLM (Large Language Model) instance
        self.llm = llm
        self.tools_schema = tools_schema
        self.generation_strategy = generation_strategy
        self.max_retries = max_retries

        if generation_strategy != "chat":
            self.prompt_template = load_prompt(prompt_path)

        else:
            self.system_prompt = load_prompt(
                os.path.join(prompt_path, "system.txt")
            )
            self.user_prompt = load_prompt(
                os.path.join(prompt_path, "user.txt")
            )


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
        if self.generation_strategy != "chat":
            return self.llm.invoke(prompt_content, use_chat_mode=False)

        else:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_content}
            ]
            return self.llm.invoke(messages, use_chat_mode=True, **kwargs)


    @staticmethod
    def _get_return_attributes(result_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Extract property definitions from a tool's results schema.

        Handles different result schema types (object, array) and extracts the
        relevant property definitions for validation purposes.

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


    def _validate_single_response(
        self, tool_name: str, response: dict[str, Any],
        results_schema: dict[str, Any], sample_idx: int, response_idx: int = 0,
    ) -> tuple[dict[str, Any] | None, list[str] | None]:
        """
        Validate a single tool response against its results schema.

        Checks that all required fields are present, non-null, and conform to the
        schema definitions. Filters out extra fields not defined in the schema and
        logs appropriate warnings and errors.

        Args:
            tool_name: Name of the tool being validated
            response: The tool response dictionary to validate
            results_schema: The results schema dictionary from the tool definition
            sample_idx: The index of the current sample for logging purposes
            response_idx: Index of the response item if validating a list of responses

        Returns:
            Tuple of (validated_response, error_logs) where validated_response is a
            dictionary containing only valid fields or None if validation failed, and
            error_logs is a list of error message strings or None if validation succeeded
        """
        properties = self._get_return_attributes(results_schema)

        validated_response = {}
        missing_or_invalid_fields_logs = []

        for field_name, field_schema in properties.items():
            if field_name not in response:
                missing_or_invalid_fields_logs.append(f"  - Field '{field_name}' is missing")
                continue

            field_value = response.get(field_name, None)
            if field_value is None:
                missing_or_invalid_fields_logs.append(f"  - Field '{field_name}' is null or undefined")
                continue

            is_value_valid, error_msg = validate_value_against_schema(field_value, field_schema, field_name)
            if not is_value_valid:
                missing_or_invalid_fields_logs.append(f"  - Field '{field_name}' failed schema validation: {error_msg}")
                continue

            validated_response[field_name] = field_value

        extra_fields = [field for field in response.keys() if field not in properties]
        if extra_fields:
            logger.warning(
                f"[ToolAgent Validator] Sample {sample_idx} -- "
                f"Response item {response_idx} for tool '{tool_name}' contains extra fields "
                f"not in schema: {pformat(extra_fields)}"
            )

        if missing_or_invalid_fields_logs:
            logger.error(
                f"[ToolAgent Validator Error] Sample {sample_idx} -- "
                f"Response item {response_idx} validation failed for tool '{tool_name}':\n"
                + '\n'.join(missing_or_invalid_fields_logs)
            )
            return None, missing_or_invalid_fields_logs

        return validated_response, None


    def _validate_tool_response(
        self, tool_name: str, tool_response: dict | list[dict],
        tool_schema: dict[str, Any], sample_idx: int,
    ) -> tuple[dict[str, Any] | list[dict[str, Any]] | None, str | None]:
        """
        Validate tool response against the schema to ensure all required fields are present and valid.

        Extracts the results schema from the tool definition and validates each response item
        against it. Handles both single dictionary responses and lists of dictionary responses.

        Args:
            tool_name: Name of the tool being validated
            tool_response: The tool response to validate (either a single dictionary or list of dictionaries)
            tool_schema: The complete tool schema dictionary containing the results schema
            sample_idx: The index of the current sample for logging purposes

        Returns:
            Tuple of (validated_response, error_message) where validated_response is the validated
            response (dict or list of dicts) or None if validation failed, and error_message is a
            string describing the validation error or None if validation succeeded
        """
        results_schema = tool_schema.get("function", {}).get("results", {})
        if not results_schema:
            logger.debug(f"[ToolAgent Validator] No results schema found for tool '{tool_name}'")
            return tool_response, None

        responses_to_validate = tool_response if isinstance(tool_response, list) else [tool_response]
        validated_responses = []

        for idx, response in enumerate(responses_to_validate):
            if not isinstance(response, dict):
                logger.error(
                    f"[ToolAgent Validator Error] Sample {sample_idx} -- "
                    f"Tool '{tool_name}' response item {idx} is not a dictionary: {type(response)}"
                )
                return None, f"Response item {idx} must be a dictionary"

            validated_response, missing_or_invalid_fields = self._validate_single_response(
                tool_name, response, results_schema, sample_idx, idx
            )

            if missing_or_invalid_fields:
                return None, f"Response item {idx} validation failed:\n" + '\n'.join(missing_or_invalid_fields)

            validated_responses.append(validated_response)

        final_response = validated_responses if isinstance(tool_response, list) else validated_responses[0]
        return final_response, None


    def invoke(self, state: DialogueState, config=None) -> DialogueState:
        """
        Process the dialogue state and simulate tool execution with LLM-generated responses.

        This method is the main entry point for the ToolAgent. It extracts tool call information
        from the state, constructs a prompt with tool parameters and context, invokes the LLM
        to generate a simulated tool response, validates the response against the tool's schema,
        and updates the state with the validated response. Supports retry logic for handling
        validation failures.

        Args:
            state: The current dialogue state containing tool_call, plan, conversations,
                sample_idx, current_step_idx, retry_count, and error_msg
            config: Optional configuration parameter (unused, for compatibility with Runnable interface)

        Returns:
            Updated DialogueState with the validated tool response added to conversations,
            updated tool_call with response field, incremented current_step_idx, and reset
            retry_count and error_msg. Returns state with abort_due_to_error flag if max
            retries exceeded or parsing fails
        """
        tool_call = state.get("tool_call", None)
        plan = state.get("plan", [])
        conversations = state.get("conversations", [])
        sample_idx = state.get("sample_idx", 0)
        current_step_idx = state.get("current_step_idx", 0)
        curr_step_plan = plan[current_step_idx] if current_step_idx < len(plan) else None
        retry_count = state.get("retry_count", 0)  # Track retry attempts
        error_msg = state.get("error_msg", None)  # Error message from previous attempt

        if retry_count >= self.max_retries:
            logger.error(
                f"[ToolAgent] Sample {sample_idx} -- Max retries ({self.max_retries}) reached. "
                "Giving up on tool call."
            )
            return {
                **state,
                "abort_due_to_error": True,
                "error_msg": f"Max retries ({self.max_retries}) reached for tool call."
            }

        if tool_call is None:
            logger.debug(f"[ToolAgent] Sample {sample_idx} -- No tool_call found in state. Returning state unchanged.")
            return state

        logger.debug(f"[ToolAgent] Sample {sample_idx} -- Received tool_call:\n{pformat(tool_call)}")
        tool_schema = next((
            t for t in self.tools_schema \
                if t.get("function", {}).get("name") == tool_call.get("name")
            ), None
        )

        results_schema = tool_schema.get("function", {}).get("results", {})

        prompt = replace_placeholders(
            self.prompt_template if self.generation_strategy != "chat" else self.user_prompt,
            conversations,
            tool_call.get("arguments", {}),
            tool_schema if self.generation_strategy != "chat" else None,
            decision_variables=curr_step_plan.get("decision_variables", None) if curr_step_plan else None,
            results_schema=results_schema if (tool_schema and self.generation_strategy == "chat") else None,
            error_msg=error_msg
        )

        if self.generation_strategy != "chat":
            response = self._invoke_llm(prompt)
        else:
            response = self._invoke_llm(
                prompt, tools=[tool_schema], tool_choice_option="none",
            )

        llm_output_content = response.content.strip()
        parsed_tool_response = parse_tool_response(llm_output_content, state['sample_idx'])
        if "error" in parsed_tool_response:
            logger.error(
                f"[ToolAgent] Sample {sample_idx} -- Tool call failed with error: "
                f"{parsed_tool_response['error']}"
            )
            return {
                **state,
                "abort_due_to_error": True,
                "error_msg": parsed_tool_response["error"]
            }

        parsed_tool_response = coerce_types_from_schema(
            parsed_tool_response,
            results_schema if results_schema.get("type") == "object" \
                else results_schema.get("items", {})
        )

        tool_name = tool_call.get("name", "unknown_tool")
        validated_response, missing_or_invalid_fields = self._validate_tool_response(
            tool_name, parsed_tool_response, tool_schema, sample_idx
        )

        if missing_or_invalid_fields:
            if retry_count < self.max_retries:
                logger.warning(
                    f"[ToolAgent] Sample {sample_idx} -- Retrying ({retry_count + 1}/{self.max_retries})..."
                )

            return self.invoke({
                **state,
                "retry_count": retry_count + 1,
                "error_msg": f"Previous attempt failed with the following errors:\n{missing_or_invalid_fields}"
            })

        tool_call.update({"response": validated_response})

        conversations.append({
            "role": "tool",
            "content": json.dumps(validated_response),
        })

        logger.debug(f"[ToolAgent] Validated response:\n{pformat(validated_response)}")
        logger.info(
            f"[ToolAgent] Sample {sample_idx} -- "
            f"Step {state.get('current_step_idx', 0) + 1}/{len(plan)} - CALL_TOOL - DONE"
        )

        return {
            **state,
            "conversations": conversations,
            "tool_call": tool_call,
            "current_step_idx": state.get("current_step_idx", 0) + 1,
            "retry_count": 0,
            "error_msg": None
        }
