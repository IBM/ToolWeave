#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import json
import yaml
from time import sleep
from typing import Literal

import httpx
from openai import OpenAI


class LLMResponse:
    """Container for LLM response content."""
    def __init__(self, content: str):
        self.content = content


class VLLMClient:
    """
    A client for interacting with a Large Language Model hosted on a vLLM server.
    This client is designed to be a drop-in replacement for other LLM clients
    by providing a similar `invoke` method.

    Attributes:
        base_url: The base URL of the vLLM server
        model_id: The ID of the model to use for generation
        max_retries: Maximum number of retries on failure
        client: The OpenAI client configured to communicate with the vLLM server
        model_params: A dictionary of parameters to pass to the generation API
    """
    def __init__(self, config_path: str, max_retries: int = 5, timeout: int = 120) -> None:
        """
        Initialize the vLLM client by loading configuration and setting up the connection.

        Args:
            config_path: Path to the YAML configuration file containing server and model settings
            max_retries: Number of times to retry failed API requests (default: 5)
            timeout: Request timeout in seconds (default: 120)

        Raises:
            ValueError: If BASE_URL or MODEL are missing from the configuration file
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is not valid YAML
        """
        # Load config YAML
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.base_url = self.config.get("BASE_URL", "").strip()
        if not self.base_url:
            raise ValueError("`BASE_URL` not found in vLLM config file.")

        self.model_id = self.config.get("MODEL")
        if not self.model_id:
            raise ValueError("`MODEL` not found in vLLM config file.")

        self.max_retries = max_retries

        # Instantiate the OpenAI client to point to the vLLM server
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.config.get("API_KEY", "vllm-no-key"),  # Default key for vLLM
            http_client=httpx.Client(timeout=timeout)  # Set a timeout
        )

        # Store generation parameters
        self.model_params = {
            "max_tokens": self.config.get("MAX_NEW_TOKENS", 8192),
            "temperature": self.config.get("TEMPERATURE", 0.0),
            "repetition_penalty": self.config.get("repetition_penalty"),
            "top_p": self.config.get("TOP_P"),  # optional,
        }

        # Filter out None values to avoid sending them in the API request
        self.model_params = {k: v for k, v in self.model_params.items() if v is not None}

        print(f"VLLMClient initialized for model: {self.model_id} at {self.base_url}")

    def invoke(
        self,
        prompt_or_messages: str | list,
        use_chat_mode: bool = False,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
    ) -> LLMResponse:
        """
        Invokes the LLM for a single prompt. Supports both text completion and chat mode.

        Args:
            prompt_or_messages: A string prompt or list of message objects.
            use_chat_mode: Whether to format the prompt as a conversation for chat-based models
            tools: Optional list of tools for function calling (only in chat mode)
            tool_choice: Optional tool choice to force the model to call a specific tool (only in chat mode)
            tool_choice_option: Optional tool choice option to control tool calling behavior (only in chat mode)
                - "none" disables tool calling
                - "auto" lets the model decide whether to call a tool or respond with a natural language response

        Returns:
            LLMResponse object containing the model's response.
        """
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                # --- TEXT COMPLETION MODE ---
                if not use_chat_mode:
                    if isinstance(prompt_or_messages, list):
                        prompt_str = "\n".join([
                            m.content if hasattr(m, "content") else str(m)
                            for m in prompt_or_messages
                        ])
                    else:
                        prompt_str = str(prompt_or_messages)

                    response = self.client.completions.create(
                        model=self.model_id,
                        prompt=prompt_str,
                        **self.model_params
                    )
                    generated_text = response.choices[0].text
                    return LLMResponse(content=generated_text.strip())

                # --- CHAT COMPLETION MODE ---
                else:
                    if isinstance(prompt_or_messages, str):
                        prompt_or_messages = [{"role": "user", "content": prompt_or_messages}]

                    elif not isinstance(prompt_or_messages, list):
                        raise ValueError("For chat mode, prompt_or_messages should be a list of message-like objects.")

                    api_tool_choice = None
                    if tool_choice and isinstance(tool_choice, dict):
                        function_name = tool_choice.get('function', {}).get('name')
                        if function_name:
                            api_tool_choice = {"type": "function", "function": {"name": function_name}}
                    elif tool_choice_option in ["none", "auto", "required"]:
                        api_tool_choice = tool_choice_option

                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=prompt_or_messages,
                        tools=tools,
                        tool_choice=api_tool_choice,
                        **self.model_params
                    )
                    message = response.choices[0].message

                    # 1. Check for tool calls in the response
                    if message.tool_calls:
                        tool_call = message.tool_calls[0].function
                        tool_call_obj = {
                            "name": tool_call.name,
                            "arguments": json.loads(tool_call.arguments)
                        }

                        return LLMResponse(content=tool_call_obj)

                    # 2. If no tool_calls, check if the model returned a JSON string in the content field
                    #    This handles instruction-following models when a tool is forced.
                    elif message.content and api_tool_choice and api_tool_choice != "none":
                        raw_content = message.content.strip()

                        # Sometimes models output JSON within markdown code blocks (```json ... ```)
                        if raw_content.startswith("```json"):
                            raw_content = raw_content[7:]
                        if raw_content.startswith("```"):
                            raw_content = raw_content[3:]
                        if raw_content.endswith("```"):
                            raw_content = raw_content[:-3]

                        # Strip again to be safe
                        cleaned_content = raw_content.strip()
                        try:
                            # Get the function name from the tool_choice you passed in
                            forced_tool_name = api_tool_choice.get('function', {}).get('name')

                            if forced_tool_name:
                                # Parse the JSON string from the content
                                arguments = json.loads(cleaned_content)

                                # Construct the tool_call_obj manually
                                tool_call_obj = {
                                    "name": forced_tool_name,
                                    "arguments": arguments
                                }
                                return LLMResponse(content=tool_call_obj)

                        except json.JSONDecodeError:
                            # The content was not valid JSON, so fall through and return it as plain text.
                            print(f"[DEBUG] JSONDecodeError: Could not parse message.content: {repr(message.content)}")
                            pass

                    # 3. If neither of the above, return the text content as is (or an empty string)
                    return LLMResponse(content=message.content.strip() if message.content else "")

            except Exception as e:
                retry_count += 1
                print(f"API Error during LLM invocation: {e}")
                if retry_count >= self.max_retries:
                    return LLMResponse(content=f"Error: Failed after {self.max_retries} retries: {e}")
                delay = 2 ** retry_count
                print(f"Retrying in {delay} seconds... (Attempt {retry_count + 1}/{self.max_retries})")
                sleep(delay)

        return LLMResponse(content=f"Error: Invocation failed after {self.max_retries} retries.")
