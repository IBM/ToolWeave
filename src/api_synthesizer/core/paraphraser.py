import logging
import random
import re

from src.api_synthesizer.utils.io import robust_json_loads
from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('api_synthesizer')


def clean_string_with_regex(text: str) -> str:
    """Removes markdown patterns and extra whitespace from the beginning and end of a string"""
    pattern = r"^((---\s*)|(\*\*.*?\*\*:?\s*))|((---\s*)|(\s*---$))+$"
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.MULTILINE).strip()
    return cleaned_text


def get_nested_property(props: dict, path: str) -> dict | None:
    """
    Retrieves a nested property from a dictionary using a dot-notation path.

    Args:
        props: Dictionary containing property definitions
        path: Dot-notation path (e.g., 'filters.product_categories' or 'inventory_data[].product_id')

    Returns:
        The property dictionary if found, None otherwise
    """
    keys = path.replace('[]', '.items.properties').split('.')
    current_level = props
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return None

    return current_level


def set_nested_property(props: dict, path: str, new_name: str, new_desc: str) -> bool:
    """
    Updates a nested property's name and description using its dot-notation path.

    Args:
        props: Dictionary containing property definitions to modify
        path: Dot-notation path to the property to update
        new_name: New name for the property
        new_desc: New description for the property

    Returns:
        True if the property was successfully updated, False if path not found
    """
    keys = path.replace('[]', '.items.properties').split('.')
    base_name = keys[-1]
    parent_keys = keys[:-1]

    parent_level = props
    for key in parent_keys:
        if isinstance(parent_level, dict) and key in parent_level:
            parent_level = parent_level[key]
        else:
            return False  # Path not found

    if isinstance(parent_level, dict) and base_name in parent_level:
        # Pop the old property and add the new one
        original_prop = parent_level.pop(base_name)
        original_prop['description'] = new_desc
        parent_level[new_name] = original_prop
        return True

    return False


class APIParaphraser:
    """
    Paraphrases API definitions to create linguistic variations while preserving semantic meaning.

    This class provides functionality to paraphrase both API descriptions and parameter
    names/descriptions using an LLM. It supports multiple paraphrasing modes (verbose/crisp)
    for descriptions and handles nested parameter structures. The paraphraser prioritizes
    parameters involved in connections and maintains consistency across the API pool and
    connection map during the renaming process.

    The paraphrasing process helps create diverse API seed data for dialogue synthesizers by
    introducing natural language variations while keeping the underlying API functionality
    intact.

    Attributes:
        llm: Language model client used for generating paraphrases
        domain: Domain name for logging and context
    """
    def __init__(self, llm: VLLMClient | WatsonxLLM, domain_name: str) -> None:
        """
        Initializes the APIParaphraser with an LLM client and domain name.

        Args:
            llm: Language model client for generating paraphrases
            domain_name: Name of the domain for logging purposes
        """
        self.llm = llm
        self.domain = domain_name


    def run(
        self, all_apis: list[dict], connection_map: list[dict],
        description_prompt_path: str, param_prompt_path: str, fraction: float = 0.5
    ) -> tuple[list[dict], list[dict], dict]:
        """
        Paraphrases API descriptions and parameter names/descriptions using LLM.

        Paraphrases API descriptions in random modes (verbose/crisp) and renames a fraction
        of parameters, prioritizing those involved in connections. Updates the connection map
        to reflect renamed parameters and maintains consistency across the API pool.

        Args:
            all_apis: List of all API definitions to paraphrase
            connection_map: List of connection mappings between APIs
            description_prompt_path: Path to prompt template for API description paraphrasing
            param_prompt_path: Path to prompt template for parameter paraphrasing
            fraction: Fraction of total parameters to paraphrase (in addition to connected ones)

        Returns:
            Tuple of (updated APIs list, updated connection map, paraphrase log dictionary)
        """
        logger.info(f"[{self.domain}] --- Running Paraphrasing Stage ---")
        if not connection_map:
            logger.warning(f"[{self.domain}] No connections to paraphrase.")
            return all_apis, connection_map, {}

        apis_dict = {api['function']['name']: api for api in all_apis}
        paraphrase_log = {}

        with open(description_prompt_path, 'r') as f:
            desc_prompt_template = f.read()
        with open(param_prompt_path, 'r') as f:
            param_prompt_template = f.read()

        for api in all_apis:
            api_name = api['function']['name']
            original_description = api['function'].get('description', '')
            mode = random.choice(["verbose", "crisp"])
            try:
                logger.debug(f"[{self.domain}] Paraphrasing description for '{api_name}' in mode '{mode}'")
                desc_prompt = desc_prompt_template.format(original_description=original_description, mode=mode)
                if isinstance(self.llm, VLLMClient):
                    message = [{'role': 'user', 'content': desc_prompt}]
                    response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
                else:
                    response = self.llm.invoke(desc_prompt)
                new_description = response.content.strip().replace('"', '')
                new_description = clean_string_with_regex(new_description)
                api['function']['description'] = new_description
            except Exception as e:
                logger.exception(f"[{self.domain}] Failed to paraphrase API description for '{api_name}': {e}")

            try:
                api_description = api['function'].get('description', '')
                props = api['function']['parameters']['properties']

                # Get all possible parameter paths, not just top-level keys
                all_param_paths = list(self._get_all_param_paths(props))

                connected_params = {
                    conn['target_param']
                    for conn in connection_map
                    if conn['target_api'] == api_name
                }

                num_additional = max(0, int(len(all_param_paths) * fraction) - len(connected_params))
                remaining_params = list(set(all_param_paths) - connected_params)
                sampled_params = random.sample(remaining_params, min(num_additional, len(remaining_params)))

                params_to_paraphrase = list(connected_params) + sampled_params

                if not params_to_paraphrase:
                    logger.debug(f"[{self.domain}] No parameters to paraphrase for '{api_name}'. Skipping.")
                    continue

                logger.debug(f"[{self.domain}] Paraphrasing parameters for '{api_name}': {params_to_paraphrase}")

                param_block_list = []
                for p_path in params_to_paraphrase:
                    prop_details = get_nested_property(props, p_path)
                    if prop_details:
                        param_block_list.append(f"{p_path}: {prop_details.get('description', '')}")

                param_block = "\n".join(param_block_list)

                param_prompt = param_prompt_template.format(
                    api_name=api_name,
                    api_description=api_description,
                    parameter_block=param_block
                )

                if isinstance(self.llm, VLLMClient):
                    message = [{'role': 'user', 'content': param_prompt}]
                    response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
                else:
                    response = self.llm.invoke(param_prompt)
                param_paraphrases_raw = robust_json_loads(response.content.strip())

                if isinstance(param_paraphrases_raw, list):
                    param_paraphrases = {}
                    for item in param_paraphrases_raw:
                        if isinstance(item, dict):
                            param_paraphrases.update(item)
                else:
                    param_paraphrases = param_paraphrases_raw

                paraphrase_log[api_name] = {}
                for original_param_path, mapping in param_paraphrases.items():
                    if original_param_path not in params_to_paraphrase:
                        continue

                    new_name_base = mapping.get("new_name")
                    new_desc = mapping.get("new_description")
                    if not new_name_base or not new_desc:
                        continue

                    # Construct the new full path
                    path_parts = original_param_path.split('.')
                    if len(path_parts) > 1:
                        new_full_path = ".".join(path_parts[:-1]) + "." + new_name_base
                    else:
                        new_full_path = new_name_base

                    # Use the helper to set the property
                    success = set_nested_property(props, original_param_path, new_name_base, new_desc)

                    if success:
                        # Update required list if necessary
                        if 'required' in api['function']['parameters']:
                            # This part is tricky with nested params and might need more robust logic
                            # For now, we handle top-level required params
                            if original_param_path in api['function']['parameters']['required']:
                                api['function']['parameters']['required'].remove(original_param_path)
                                api['function']['parameters']['required'].append(new_name_base)

                        # Update connection map
                        for conn in connection_map:
                            if conn['target_api'] == api_name and conn['target_param'] == original_param_path:
                                conn['target_param'] = new_full_path

                        paraphrase_log[api_name][original_param_path] = {
                            "new_name": new_full_path,
                            "new_description": new_desc
                        }
            except Exception as e:
                logger.exception(f"[{self.domain}] Failed to paraphrase parameters for '{api_name}': {e}")

        return list(apis_dict.values()), connection_map, paraphrase_log


    def _get_all_param_paths(self, properties: dict, current_path: str = "") -> set[str]:
        """
        Recursively finds all parameter paths in a nested properties object.

        Args:
            properties: Dictionary containing parameter schema properties
            current_path: Accumulated path prefix for nested parameters

        Returns:
            Set of all parameter paths in dot-notation format
        """
        paths = set()
        if not isinstance(properties, dict):
            return paths

        for name, details in properties.items():
            new_path = f"{current_path}.{name}" if current_path else name

            if isinstance(details, dict):
                if details.get('type') == 'object' and 'properties' in details:
                    paths.update(self._get_all_param_paths(details['properties'], new_path))
                elif details.get('type') == 'array' and details.get('items', {}).get('type') == 'object':
                    if 'properties' in details.get('items', {}):
                        array_path = f"{new_path}[]"
                        paths.update(self._get_all_param_paths(details['items']['properties'], array_path))
                else:
                    paths.add(new_path)

        return paths
