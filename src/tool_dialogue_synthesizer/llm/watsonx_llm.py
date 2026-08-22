import os
import yaml
from time import sleep
from typing import Literal

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


class LLMResponse:
    """Container for LLM response content."""
    def __init__(self, content: str):
        self.content = content


class WatsonxLLM:
    """
    Wrapper for IBM watsonx.ai foundation models.

    Provides LLM invocation with automatic retry logic and support for both
    standard text generation and chat mode with optional function calling.
    Configuration is loaded from a YAML file, and credentials are read from
    environment variables.

    Attributes:
        model_id: The ID of the model to use for generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature for generation
        max_retries: Maximum number of retries on failure
        repetition_penalty: Penalty for repeated tokens
        decoding_method: Decoding method to use (greedy or sample)
        credentials: Credentials object for authenticating with the IBM Cloud
        model: The ModelInference instance for making API calls
    """
    def __init__(self, config_path: str, max_retries: int = 5) -> None:
        """
        Set up IBM Cloud credentials, load model configuration, and create the ModelInference client.

        Reads API key and project ID from environment variables, parses the YAML config file
        to extract model parameters, and instantiates the watsonx.ai client ready for inference.

        Args:
            config_path: Path to YAML file with model settings (MODEL, MAX_NEW_TOKENS, etc.)
            max_retries: Number of retry attempts for failed API calls (default: 5)

        Raises:
            ValueError: If IBM_CLOUD_API_KEY is missing from environment
            FileNotFoundError: If config_path does not exist
            yaml.YAMLError: If the YAML file is malformed
        """
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("IBM_CLOUD_API_KEY")
        self.project_id = os.getenv("IBM_PROJECT_ID")
        self.region = os.getenv("WATSONX_REGION", "us-south")  # Default region if not set

        if not self.api_key:
            raise ValueError("Missing IBM_CLOUD_API_KEY in .env or environment variables.")

        # Load config YAML
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.max_retries = max_retries

        self.model_id = self.config.get("MODEL", "mistralai/mistral-large")
        self.max_new_tokens = self.config.get("MAX_NEW_TOKENS", 8192)
        self.min_new_tokens = self.config.get("MIN_NEW_TOKENS", 1)
        self.repetition_penalty = self.config.get("REPETITION_PENALTY", 1.0)
        self.temperature = self.config.get("TEMPERATURE", 0.00)

        decoding_method_str = self.config.get("DECODING_METHOD", "greedy").lower()
        if decoding_method_str == "sample":
            self.decoding_method = DecodingMethods.SAMPLE
        else:  # Default to greedy
            self.decoding_method = DecodingMethods.GREEDY

        self.credentials = Credentials(
            url=f"https://{self.region}.ml.cloud.ibm.com",
            api_key=self.api_key,
        )

        # Model parameters for single and batch calls
        self.model_params = {
            GenParams.MAX_NEW_TOKENS: self.max_new_tokens,
            GenParams.MIN_NEW_TOKENS: self.min_new_tokens,
            GenParams.DECODING_METHOD: self.decoding_method,
            GenParams.REPETITION_PENALTY: self.repetition_penalty,
            GenParams.TEMPERATURE: self.temperature,
            # Add other parameters like TOP_K, TOP_P if using DecodingMethods.SAMPLE
        }
        if self.decoding_method == DecodingMethods.SAMPLE:
            self.model_params[GenParams.TOP_K] = self.config.get("TOP_K", 50)
            self.model_params[GenParams.TOP_P] = self.config.get("TOP_P", 0.9)

        # Instantiate the ModelInference object once
        self.model = ModelInference(
            model_id=self.model_id,
            credentials=self.credentials,
            params=self.model_params,
            project_id=self.project_id
        )
        print(f"WatsonxLLM initialized with model: {self.model_id} in region: {self.region}")


    def invoke(
        self, prompt_or_messages: str | list, use_chat_mode: bool = False,
        tools: list | None = None, tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
    ) -> LLMResponse:
        """
        Invokes the LLM for a single prompt.
        prompt_or_messages can be a string or a list of message-like objects.

        Args:
            prompt_or_messages: A string prompt or list of message objects
            use_chat_mode: Whether to format the prompt as a conversation for chat-based models
            tools: Optional list of tools for function calling (only in chat mode)
            tool_choice: Optional tool choice to force the model to call a specific tool (only in chat mode)
            tool_choice_option: Optional tool choice option to control tool calling behavior (only in chat mode)
                - "none" disables tool calling
                - "auto" lets the model decide whether to call a tool or respond with a natural language response

        Returns:
            LLMResponse object containing the model's response
        """
        if not use_chat_mode:
            if isinstance(prompt_or_messages, list):
                # Assuming a list of messages for a single conversational prompt
                prompt_str = "\n".join([
                    m.content if hasattr(m, "content") else str(m)
                    for m in prompt_or_messages
                ])
            else:
                prompt_str = str(prompt_or_messages)

            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    model_output = self.model.generate(prompt=prompt_str)
                    generated_text = model_output["results"][0]["generated_text"]
                    return LLMResponse(content=generated_text)

                except Exception as e:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        print(f"Error during single LLM invocation after {self.max_retries}: {e}")
                        return LLMResponse(content=f"Error: {e}")

                    # Exponential backoff: 2^retry_count seconds (1s, 2s, 4s, 8s, 16s...)
                    delay = 2 ** retry_count
                    print(f"LLM invocation failed, retrying in {delay} seconds... (Attempt {retry_count}/{self.max_retries})")
                    sleep(delay)

        else:
            if isinstance(prompt_or_messages, str):
                prompt_or_messages = [{"role": "user", "content": prompt_or_messages}]

            elif not isinstance(prompt_or_messages, list):
                raise ValueError("For chat mode, prompt_or_messages should be a list of message-like objects.")

            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    model_output = self.model.chat(
                        messages=prompt_or_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        tool_choice_option=tool_choice_option,
                    )

                    model_response = model_output["choices"][0]["message"]
                    if "tool_calls" in model_response:
                        tool_calls = model_response["tool_calls"]
                        if tool_calls:
                            tool_call_obj = tool_calls[0].get("function", {})
                            return LLMResponse(content=tool_call_obj)
                        else:
                            return LLMResponse(content="No tool calls made.")

                    elif "content" in model_response:
                        generated_text = model_response["content"]
                        return LLMResponse(content=generated_text)

                    else:
                        return LLMResponse(content=f"Error: Malformed response from LLM for chat mode: {model_response}")

                except Exception as e:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        print(f"Error during chat LLM invocation after {self.max_retries}: {e}")
                        return LLMResponse(content=f"Error: {e}")

                    # Exponential backoff: 2^retry_count seconds (1s, 2s, 4s, 8s, 16s...)
                    delay = 2 ** retry_count
                    print(f"LLM invocation failed, retrying in {delay} seconds... (Attempt {retry_count}/{self.max_retries})")
                    sleep(delay)
