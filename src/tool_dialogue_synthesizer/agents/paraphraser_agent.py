import json
import logging
import os
import re

from langchain_core.runnables import Runnable

from scripts.compute_true_multi_step_stats import resolve_param_value
from src.tool_dialogue_synthesizer.llm.vllm_llm import LLMResponse as VLLMResponse, VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import LLMResponse as WatsonxLLMResponse, WatsonxLLM
from src.tool_dialogue_synthesizer.utils.prompts import load_prompt


logger = logging.getLogger('dialogue_refiner')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def replace_placeholders(
    template: str, user_utterance: str,
    dialogue_history: list[dict], params_provided: dict, relevant_tool_names: list[str],
    curr_plan_desc: str | None = None, prev_assistant_utterance: str | None = None
)-> str:
    """
    Replace placeholder variables in a template string with actual dialogue context values.

    This function processes a template string and substitutes placeholder markers with
    JSON-formatted data from the dialogue context, including conversation history,
    user utterances, parameters, and tool information.

    Args:
        template: The template string containing placeholder markers
        user_utterance: The current user message to be paraphrased
        dialogue_history: List of conversation turn dictionaries with 'role' and 'content' keys
        params_provided: Dictionary of parameter names and their values that will be used
            in subsequent tool calls
        relevant_tool_names: List of tool names that are relevant to the current user turn
        curr_plan_desc: Optional description of the current plan step for user utterances
        prev_assistant_utterance: Optional previous assistant message for clarification responses

    Returns:
        The template string with all placeholders replaced by their corresponding values
        in JSON format
    """
    filtered_dialogue_history = []
    for turn in dialogue_history:
        if turn['role'] in ['system', 'user', 'assistant']:
            filtered_dialogue_history.append(
                f"{turn['role'].capitalize()}: {turn['content']}"
            )
    mod_template = template.replace("{{dialogue_history}}", json.dumps(filtered_dialogue_history, indent=2))

    mod_template = mod_template.replace("{{user_utterance}}", user_utterance)

    if curr_plan_desc is not None:
        mod_template = mod_template.replace("{{curr_plan_desc}}", curr_plan_desc)

    if prev_assistant_utterance is not None:
        mod_template = mod_template.replace("{{prev_assistant_utterance}}", prev_assistant_utterance)

    mod_template = mod_template.replace("{{params_provided}}", json.dumps(params_provided, indent=2))

    mod_template = mod_template.replace("{{relevant_tool_names}}", json.dumps(relevant_tool_names, indent=2))

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


class ParaphraserAgent(Runnable):
    """
    Agent responsible for paraphrasing user messages in dialogue samples.

    The ParaphraserAgent processes dialogue samples to generate more natural and varied
    user utterances while preserving the semantic intent and parameter information. It
    handles two types of user turns: initial user utterances and user responses to
    clarification requests.

    The agent supports two generation strategies:
    - "generate": Uses single prompt templates for direct text generation
    - "chat": Uses separate system and user prompts for chat-based generation

    For each user turn, the agent extracts context including dialogue history, provided
    parameters, and relevant tool names to generate contextually appropriate paraphrases
    that maintain coherence with the dialogue flow.

    Attributes:
        llm: The language model client (VLLM or Watsonx) used for paraphrasing
        generation_strategy: Strategy for LLM invocation ("generate" or "chat")
    """
    def __init__(
        self, llm: VLLMClient | WatsonxLLM, 
        utterer_prompt_path: str, clarifier_prompt_path: str,
        generation_strategy: str = "chat",
    ) -> None:
        """
        Set up the paraphraser agent with LLM client and prompt templates for both user turn types.

        Initializes the agent by loading prompt templates for paraphrasing initial user utterances
        and user responses to clarification requests. Template loading varies by generation strategy:
        single files for "generate" mode, or separate system/user files for "chat" mode.

        Args:
            llm: Language model client instance for generating paraphrases (VLLMClient or WatsonxLLM)
            utterer_prompt_path: Path to prompt template(s) for paraphrasing initial user utterances
            clarifier_prompt_path: Path to prompt template(s) for paraphrasing clarification responses
            generation_strategy: LLM invocation approach - "generate" for completion or "chat" 
                for message-based generation (default: "chat")

        Note:
            For "chat" strategy, each path should be a directory containing "system.txt" and 
            "user.txt" files. For "generate" strategy, paths should point to single template files.
        """
        self.llm = llm
        self.generation_strategy = generation_strategy

        if generation_strategy == "generate":
            self.utterer_prompt_template = load_prompt(utterer_prompt_path)
            self.clarifier_prompt_template = load_prompt(clarifier_prompt_path)

        else:
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
            prompt_content: The prompt to send to the LLM
            **kwargs: Additional keyword arguments to pass to the LLM invoke method
                (e.g., use_chat_mode, etc.)

        Returns:
            LLM response object containing the generated content, either
            VLLMResponse or WatsonxLLMResponse depending on the LLM client type
        """
        return self.llm.invoke(prompt_content, **kwargs)


    def invoke(self, sample: dict) -> dict:
        """
        Process a dialogue sample and paraphrase all user messages.

        Iterates through the conversation in the sample, identifying user turns that need
        paraphrasing based on the plan. Extracts context from subsequent turns to determine
        relevant parameters and tools, then generates paraphrased versions while preserving
        semantic intent.

        The method handles two types of user turns:
        - USER_UTTERANCE: Initial user messages with plan descriptions
        - USER_RESPONSE_TO_CLARIFICATION: User responses to assistant clarification requests

        Args:
            sample: Dictionary containing 'dialogue_id', 'plan', and 'conversations' keys,
                where conversations is a list of turn dictionaries with 'role' and 'content'

        Returns:
            Updated sample dictionary with paraphrased user messages in the conversations list
        """
        sample_idx = sample['dialogue_id']
        plan = sample['plan']
        conversation = sample['conversations']

        result_conversation = []
        curr_step_idx = 0

        i = 0
        while i < len(conversation):
            turn = conversation[i]
            role = turn['role']
            content = turn['content']

            result_turn = turn.copy()

            if role == 'user' and curr_step_idx < len(plan):
                curr_plan_step = plan[curr_step_idx]
                step_type = curr_plan_step.get('type')

                if step_type not in ['USER_UTTERANCE', 'USER_RESPONSE_TO_CLARIFICATION']:
                    curr_step_idx += 1
                    i += 1
                    result_conversation.append(result_turn)
                    continue

                if step_type == 'USER_UTTERANCE':
                    params_keys = curr_plan_step.get('provided_params', {}).keys()
                    utterance = curr_plan_step.get('utterance', '')
                    prev_assistant_utterance = None

                elif step_type == 'USER_RESPONSE_TO_CLARIFICATION':
                    params_keys = curr_plan_step.get('provides_params', {}).keys()
                    utterance = None
                    prev_assistant_utterance = result_conversation[curr_step_idx - 1]['content'] \
                        if curr_step_idx > 0 else None

                params_provided = {}
                relevant_tool_names = []
                j = i + 1
                while j < len(conversation):
                    next_turn = conversation[j]

                    if next_turn['role'] == 'assistant_tool_call':
                        next_content = json.loads(next_turn['content'])[0]

                        tool_name = next_content.get('name', '')
                        relevant_tool_names.append(tool_name)

                        arguments = next_content.get('arguments', {})
                        for param_key in params_keys:
                            param_tool_name, param_key_name = param_key.split('.', 1)
                            if param_tool_name == tool_name:
                                params_provided[param_key_name] = resolve_param_value(param_key_name, arguments)

                    elif next_turn['role'] == 'assistant' and len(relevant_tool_names) > 0:
                        break

                    j += 1

                prompt_template = None
                if self.generation_strategy == "generate":
                    prompt_template = self.utterer_prompt_template if step_type == 'USER_UTTERANCE' \
                        else self.clarifier_prompt_template
                else:
                    prompt_template = self.utterer_user_prompt if step_type == 'USER_UTTERANCE' \
                        else self.clarifier_user_prompt

                paraphraser_prompt = replace_placeholders(
                    prompt_template, content,
                    result_conversation, params_provided,
                    relevant_tool_names=relevant_tool_names,
                    curr_plan_desc=utterance,
                    prev_assistant_utterance=prev_assistant_utterance,
                )

                if self.generation_strategy == "generate":
                    llm_response = self._invoke_llm(paraphraser_prompt, use_chat_mode=False)
                    llm_output_content = llm_response.content.strip()
                    paraphrased_message = parse_user_response(llm_output_content, sample_idx)

                else:
                    prompt = [
                        {
                            "role": "system",
                            "content": self.utterer_system_prompt if step_type == 'USER_UTTERANCE' \
                                else self.clarifier_system_prompt
                        },
                        {
                            "role": "user",
                            "content": paraphraser_prompt
                        }
                    ]

                    llm_response = self._invoke_llm(prompt, use_chat_mode=True)
                    llm_output_content = llm_response.content.strip()
                    paraphrased_message = llm_output_content

                if step_type == 'USER_UTTERANCE':
                    logger.debug(
                        f"[ParaphraserAgent] Sample {sample_idx} -- "
                        f"Paraphrased user utterance at index {i + 1}/{len(conversation)}\n"
                        f"Original: {content}\n"
                        f"Paraphrased: {paraphrased_message}\n"
                    )

                elif step_type == 'USER_RESPONSE_TO_CLARIFICATION':
                    logger.debug(
                        f"[ParaphraserAgent] Sample {sample_idx} -- "
                        f"Paraphrased user clarification at index {i + 1}/{len(conversation)}\n"
                        f"Original: {content}\n"
                        f"Paraphrased: {paraphrased_message}\n"
                    )

                result_turn['content'] = paraphrased_message

            if role in ['user', 'tool', 'assistant']:
                curr_step_idx += 1
            i += 1
            result_conversation.append(result_turn)

        return {
            **sample,
            'conversations': result_conversation,
        }
