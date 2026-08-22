import copy
import json
import logging

from langchain_core.messages import HumanMessage
from rapidfuzz.fuzz import partial_ratio
from sentence_transformers import SentenceTransformer, util

from src.api_synthesizer.utils.io import (
    load_prompt,
    robust_json_list_of_strings,
    robust_json_loads,
)
from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('api_synthesizer')
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


class APISynthesizer:
    """
    Synthesizes API definitions for a given domain using LLM-based generation and refinement.

    This class orchestrates the complete API synthesis pipeline, including:
    - Multi-step API generation following a predefined synthesis plan
    - Duplicate detection using fuzzy name matching, structural, and semantic similarity
    - Automatic refinement of parameter properties (enums, defaults, required fields, formats)
    - Connection tracking between APIs based on parameter matching (semantic or LLM-based)
    - Echo parameter removal to avoid redundant input/output fields

    The synthesizer maintains an API pool that grows through iterative generation steps,
    with each step building upon previously generated APIs to create interconnected
    tool definitions suitable for multi-turn dialogue scenarios.

    Attributes:
        domain: The domain name for which APIs are being synthesized
        llm: The language model client used for generation
        api_pool: Collection of generated and refined API definitions
        connection_map: Tracked connections between API parameters
        connection_mode: Mode for connection detection ('semantic' or 'llm')
        similarity_threshold: Threshold for semantic similarity comparisons
    """
    def __init__(
        self, domain_name: str, llm: VLLMClient | WatsonxLLM, synthesis_plan_path: str, incontext_examples_path: str,
        enum_refinement_prompt_path: str, default_value_refinement_prompt_path: str,
        required_params_prompt_path: str, connection_mode: str = 'semantic', connection_prompt_path: str = None,
    ) -> None:
        """
        Initializes the APISynthesizer with configuration paths and settings.

        Loads all required prompt templates and sets up the synthesis environment including
        the LLM client, domain context, and connection tracking mode. The in-context examples
        are loaded once during initialization for efficiency.

        Args:
            domain_name: Name of the domain for which APIs will be synthesized
            llm: Language model client for generating and refining API definitions
            synthesis_plan_path: Path to JSON file containing the multi-step synthesis plan
            incontext_examples_path: Path to file containing example API definitions for prompts
            enum_refinement_prompt_path: Path to prompt template for generating enum values
            default_value_refinement_prompt_path: Path to prompt template for generating default values
            required_params_prompt_path: Path to prompt template for determining required parameters
            connection_mode: Mode for detecting parameter connections between APIs ('semantic' or 'llm')
            connection_prompt_path: Path to prompt template for LLM-based connection verification (required if connection_mode is 'llm')

        Raises:
            ValueError: If connection_mode is 'llm' but connection_prompt_path is not provided
        """
        self.domain = domain_name
        self.llm = llm
        self.api_pool = []
        self.connection_map = []

        self.synthesis_plan_path = synthesis_plan_path
        self.incontext_examples_path = incontext_examples_path
        self.enum_prompt_template = load_prompt(enum_refinement_prompt_path)
        self.default_value_prompt_template = load_prompt(default_value_refinement_prompt_path)
        self.required_params_prompt_template = load_prompt(required_params_prompt_path)

        # Load the in-context examples once during initialization
        self.incontext_examples_content = load_prompt(self.incontext_examples_path)
        self.connection_mode = connection_mode
        if self.connection_mode == 'llm':
            if not connection_prompt_path:
                raise ValueError("connection_prompt_path is required for LLM connection mode.")
            self.connection_prompt_template = load_prompt(connection_prompt_path)


    def _are_parameters_semantically_similar(self, desc1: str, desc2: str) -> bool:
        """
        Compares two parameter descriptions using semantic similarity via sentence embeddings.

        Uses a sentence transformer model to encode the descriptions and computes cosine
        similarity between their embeddings. Returns True if the similarity exceeds the
        configured threshold, indicating the parameters likely represent the same concept.

        Args:
            desc1: Description of the first parameter
            desc2: Description of the second parameter

        Returns:
            True if semantic similarity exceeds the threshold, False otherwise or if either
            description is empty or an error occurs during encoding
        """
        if not desc1 or not desc2:
            return False

        try:
            emb1 = self.model.encode(desc1, convert_to_tensor=True, device='cpu')
            emb2 = self.model.encode(desc2, convert_to_tensor=True, device='cpu')
            cosine_similarity = util.pytorch_cos_sim(emb1, emb2).item()
            return cosine_similarity > self.similarity_threshold
        except Exception as e:
            logger.warning(f"[{self.domain}] Could not compare parameter descriptions due to an error: {e}")
            return False


    def _is_connection_llm_verified(self, source_api: dict, target_api: dict, param_name: str) -> bool:
        """
        Verifies whether a parameter connection between two APIs is logically valid using LLM judgment.

        Constructs a prompt with context about both APIs and the shared parameter, then queries
        the LLM to determine if the connection makes sense. The connection is confirmed if the
        LLM response contains 'yes'.

        Args:
            source_api: The API definition containing the output parameter
            target_api: The API definition containing the input parameter
            param_name: Name of the parameter that appears in both APIs

        Returns:
            True if the LLM confirms the connection is valid, False if rejected by LLM or on error
        """
        source_param_desc = source_api['function'].get('results', {}).get('properties', {}).get(param_name, {}).get(
            'description', '')
        target_param_desc = target_api['function'].get('parameters', {}).get('properties', {}).get(param_name, {}).get(
            'description', '')

        prompt = self.connection_prompt_template.format(
            source_api_name=source_api['function']['name'],
            source_api_desc=source_api['function']['description'],
            param_name=param_name,
            source_param_desc=source_param_desc,
            target_api_name=target_api['function']['name'],
            target_api_desc=target_api['function']['description'],
            target_param_desc=target_param_desc
        )

        try:
            if isinstance(self.llm, VLLMClient):
                message = [{'role': 'user', 'content': prompt}]
                response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = response.content.strip().lower()
            return 'yes' in decision
        except Exception as e:
            logger.warning(f"[{self.domain}] LLM connection check failed. Defaulting to no connection. Error: {e}")
            return False


    def _get_all_nested_params(self, properties: dict, current_path: str = "") -> dict:
        """
        Recursively extracts all parameters from a properties schema including nested structures.

        Traverses through object properties and array items to collect parameter names and their
        descriptions, building full dot-notation paths for nested fields (e.g., 'address.city' or
        'items[].product_id').

        Args:
            properties: Dictionary containing parameter schema properties
            current_path: Accumulated path prefix for nested parameters

        Returns:
            Dictionary mapping full parameter paths to their descriptions
        """
        params = {}
        if not isinstance(properties, dict):
            return params

        for name, details in properties.items():
            if isinstance(details, dict):
                new_path = f"{current_path}.{name}" if current_path else name

                if details.get('type') == 'object' and 'properties' in details:
                    params.update(self._get_all_nested_params(details['properties'], new_path))

                elif details.get('type') == 'array' and details.get('items', {}).get('type') == 'object':
                    if 'properties' in details.get('items', {}):
                        array_path = f"{new_path}[]"
                        nested_params = self._get_all_nested_params(details['items']['properties'], array_path)
                        for nested_name, nested_desc in nested_params.items():
                            params[nested_name] = nested_desc

                else:
                    params[new_path] = details.get('description', '')

        return params


    def track_connections(self, new_apis: list, context_apis: list) -> None:
        """
        Identifies and records connections between API parameters based on name and semantic similarity.

        Compares output parameters from existing APIs with input parameters of newly generated APIs
        to establish potential data flow connections. Uses either semantic similarity or LLM-based
        verification depending on the configured connection mode. Connections are stored in the
        connection_map for later use in dialogue generation.

        Args:
            new_apis: List of newly generated API definitions to check for connections
            context_apis: List of existing API definitions that may provide outputs
        """
        all_potential_sources = context_apis + new_apis
        for new_api in new_apis:
            target_name = new_api['function']['name']
            target_params_schema = new_api['function'].get('parameters', {}).get('properties', {})
            if not target_params_schema: continue

            target_params = self._get_all_nested_params(target_params_schema)

            for source_api in all_potential_sources:
                source_name = source_api['function']['name']
                if source_name == target_name: continue
                source_outputs_schema = source_api['function'].get('results', {}).get('properties', {})
                if not source_outputs_schema: continue

                source_outputs = self._get_all_nested_params(source_outputs_schema)

                # Find parameters with the same base name (ignoring the path for the initial match)
                source_base_names = {path.split('.')[-1].replace('[]', ''): path for path in source_outputs.keys()}
                target_base_names = {path.split('.')[-1].replace('[]', ''): path for path in target_params.keys()}

                shared_base_names = set(source_base_names.keys()) & set(target_base_names.keys())

                for base_name in shared_base_names:
                    source_full_path = source_base_names[base_name]
                    target_full_path = target_base_names[base_name]

                    is_connected = False
                    if self.connection_mode == 'semantic':
                        source_param_desc = source_outputs.get(source_full_path, '')
                        target_param_desc = target_params.get(target_full_path, '')
                        if self._are_parameters_semantically_similar(source_param_desc, target_param_desc):
                            is_connected = True
                        else:
                            logger.debug(
                                f"[{self.domain}] Rejected potential connection for '{base_name}' due to low semantic similarity."
                            )

                    elif self.connection_mode == 'llm':
                        if self._is_connection_llm_verified(source_api, new_api, base_name):
                            is_connected = True
                        else:
                            logger.debug(
                                f"[{self.domain}] Rejected potential connection for '{base_name}' based on LLM decision."
                            )

                    if is_connected:
                        conn = {
                            "source_api": source_name,
                            "source_param": source_full_path,
                            "target_api": target_name,
                            "target_param": target_full_path
                        }
                        if conn not in self.connection_map:
                            self.connection_map.append(conn)
                            logger.debug(
                                f"[{self.domain}] Tracked Connection ({self.connection_mode}): "
                                f"{source_name} (out: {source_full_path}) -> {target_name} (in: {target_full_path})"
                            )


    def _extract_available_outputs(self) -> list[str]:
        """Extracts all unique output parameter names from APIs in the pool."""
        outputs = []
        for api in self.api_pool:
            results = api.get("function", {}).get("results", {}).get("properties", {})
            outputs.extend(results.keys())
        return list(set(outputs))


    def _is_structurally_similar(self, api1: dict, api2: dict) -> bool:
        """
        Checks if two APIs have similar parameter structures based on parameter overlap.

        Uses Jaccard similarity on parameter sets with a threshold of 0.7 to determine
        if APIs are structurally similar enough to be considered duplicates.

        Args:
            api1: First API definition to compare
            api2: Second API definition to compare

        Returns:
            True if parameter overlap ratio exceeds 0.7
        """
        p1 = set(api1['function']['parameters']['properties'].keys())
        p2 = set(api2['function']['parameters']['properties'].keys())
        return len(p1 & p2) / max(len(p1 | p2), 1) > 0.7


    def _is_semantically_similar(self, desc1: str, desc2: str) -> bool:
        """Compares two descriptions using semantic similarity with a threshold of 0.85."""
        if not desc1 or not desc2:
            return False
        emb1 = model.encode(desc1, convert_to_tensor=True, device='cpu')
        emb2 = model.encode(desc2, convert_to_tensor=True, device='cpu')
        return util.pytorch_cos_sim(emb1, emb2).item() > 0.85


    def _is_fuzzy_name_match(self, name1: str, name2: str) -> bool:
        """Performs fuzzy string matching on two names using partial ratio."""
        return partial_ratio(name1, name2) > 85


    def _is_duplicate(self, new_api: dict) -> bool:
        """
        Determines if an API is a duplicate based on fuzzy name matching, structural, and semantic similarity.

        Checks against all APIs in the pool using multiple strategies to detect duplicates,
        logging the detection method when a match is found.

        Args:
            new_api: API definition to check for duplication

        Returns:
            True if the API matches an existing one by any similarity metric
        """
        new_fn = new_api.get("function", {})
        new_name = new_fn.get("name", "")
        new_desc = new_fn.get("description", "")
        for api in self.api_pool:
            fn = api.get("function", {})
            existing_name = fn.get("name", "")
            existing_desc = fn.get("description", "")

            if new_name == existing_name or self._is_fuzzy_name_match(new_name, existing_name):
                logger.debug(f"[{self.domain}] Fuzzy name match detected: '{new_name}' similar to '{existing_name}'")
                return True

            if self._is_structurally_similar(new_api, api):
                logger.debug(f"[{self.domain}] Structural similarity detected between '{new_name}' and '{existing_name}'")
                return True

            if self._is_semantically_similar(new_desc, existing_desc):
                logger.debug(f"[{self.domain}] Semantic similarity detected between API descriptions")
                return True

        return False


    def _get_all_property_names(self, properties: dict) -> set[str]:
        """
        Recursively collects all property names from a schema including nested structures.

        Args:
            properties: Dictionary containing parameter schema properties

        Returns:
            Set of all property names found at any nesting level
        """
        names = set()
        if not isinstance(properties, dict):
            return names

        for name, details in properties.items():
            names.add(name)
            if isinstance(details, dict):
                if details.get('type') == 'object' and 'properties' in details:
                    names.update(self._get_all_property_names(details['properties']))
                elif details.get('type') == 'array' and details.get('items', {}).get('type') == 'object':
                    if 'properties' in details.get('items', {}):
                        names.update(self._get_all_property_names(details['items']['properties']))

        return names


    def _recursively_remove_keys(self, properties: dict, keys_to_remove: set) -> None:
        """
        Recursively removes specified keys from a properties dictionary including nested structures.

        Modifies the properties dictionary in-place, traversing through object properties and
        array items to remove all occurrences of the specified keys.

        Args:
            properties: Dictionary containing parameter schema properties to modify
            keys_to_remove: Set of property names to remove
        """
        if not isinstance(properties, dict):
            return

        for key in list(properties.keys()):
            if key in keys_to_remove:
                del properties[key]

            else:
                details = properties.get(key)
                if isinstance(details, dict):
                    if details.get('type') == 'object' and 'properties' in details:
                        self._recursively_remove_keys(details['properties'], keys_to_remove)
                    elif details.get('type') == 'array' and details.get('items', {}).get('type') == 'object':
                        if 'properties' in details.get('items', {}):
                            self._recursively_remove_keys(details['items']['properties'], keys_to_remove)


    def _remove_echo_parameters(self, api: dict) -> dict:
        """
        Removes echo parameters that appear in both input and output schemas.

        Creates a deep copy of the API and removes any parameters from the results section
        that also appear in the parameters section, as these represent redundant echoing
        of input values.

        Args:
            api: API definition to clean

        Returns:
            Cleaned API definition with echo parameters removed from results
        """
        if not isinstance(api, dict) or 'function' not in api:
            return api

        cleaned_api = copy.deepcopy(api)
        function_def = cleaned_api.get('function', {})
        input_params = function_def.get('parameters', {}).get('properties', {})
        output_params = function_def.get('results', {}).get('properties', {})
        if not input_params or not output_params:
            return cleaned_api

        input_param_names = self._get_all_property_names(input_params)
        output_param_names = self._get_all_property_names(output_params)
        echo_params = input_param_names.intersection(output_param_names)
        if echo_params:
            logger.debug(
                f"[{self.domain}] Found echo parameters in '{function_def.get('name')}': "
                f"{list(echo_params)}. Removing from results."
            )
            self._recursively_remove_keys(function_def['results']['properties'], echo_params)

        return cleaned_api


    def _refine_required_parameters(self, function_def: dict) -> None:
        """
        Determines and populates the 'required' field for API parameters using LLM guidance and fallback logic.

        Queries the LLM to identify which parameters should be required based on the API's purpose and
        description. If the LLM query fails or returns empty results, applies heuristic fallback logic
        for action-oriented APIs by selecting common identifier parameters. Also ensures that required
        parameters do not have default values.

        Args:
            function_def: API function definition containing parameters to refine
        """
        params_obj = function_def.get('parameters')
        api_name = function_def.get('name', '')
        has_changed = False

        # Exit if there are no parameters to process.
        if not isinstance(params_obj, dict) or 'properties' not in params_obj or not params_obj.get('properties'):
            if isinstance(params_obj, dict) and 'required' not in params_obj:
                params_obj['required'] = []
                has_changed = True

            if has_changed:
                logger.info(f"Final 'required' field for '{api_name}': []")
            return

        # Store the original list to detect changes.
        original_required = params_obj.get('required', []).copy()

        # If 'required' is missing or empty, then we attempt to determine the parameters.
        if not original_required:
            determined_required_params = []
            logger.debug(f"Querying LLM for required params for '{api_name}'.")
            prompt = self.required_params_prompt_template.format(
                api_name=api_name,
                api_description=function_def.get('description', 'N/A'),
                parameters_json=json.dumps(params_obj.get('properties', {}), indent=2)
            )

            try:
                if isinstance(self.llm, VLLMClient):
                    message = [{'role': 'user', 'content': prompt}]
                    response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
                else:
                    response = self.llm.invoke([HumanMessage(content=prompt)])
                required_params = robust_json_list_of_strings(response.content.strip())
                all_param_names = params_obj.get('properties', {}).keys()
                if isinstance(required_params, list) and all(p in all_param_names for p in required_params):
                    determined_required_params = required_params
            except Exception as e:
                logger.warning(f"LLM query for required params failed for '{api_name}': {e}")

            # --- FALLBACK LOGIC ---
            action_keywords = [
                'search', 'get', 'list', 'find', 'lookup', 'query', 'track', 'update',
                'cancel', 'delete', 'submit', 'book', 'schedule', 'assign', 'retrieve',
                'fetch', 'post', 'manage', 'analyze', 'calculate', 'enroll', 'generate',
                'process', 'add', 'check', 'scan', 'monitor', 'pay'
            ]
            is_action_api = any(keyword in api_name.lower() for keyword in action_keywords)

            if not determined_required_params and is_action_api:
                logger.warning(f"Applying fallback to find a required parameter for '{api_name}'.")
                available_params = list(params_obj.get('properties', {}).keys())

                # Simplified candidate search for brevity
                primary_candidates = [
                    'query', 'id', 'user_id', 'customer_id', 'order_id',
                    'product_id', 'shipment_id', 'ticket_id',
                ]
                found_candidate = False
                for candidate in primary_candidates:
                    if candidate in available_params:
                        determined_required_params = [candidate]
                        found_candidate = True
                        break

                if not found_candidate and available_params:
                    determined_required_params = [available_params[0]]

            params_obj['required'] = determined_required_params

        # Check if the 'required' list itself has changed.
        if original_required != params_obj.get('required', []):
            has_changed = True

        # --- FINALIZATION ---
        # Clean up default values from the final list of required parameters.
        final_required_params = params_obj.get('required', [])
        if final_required_params:
            for param_name in final_required_params:
                param_details = params_obj.get('properties', {}).get(param_name, {})
                if 'default' in param_details:
                    del param_details['default']
                    logger.info(f"Removed 'default' value from required parameter '{param_name}' in '{api_name}'.")
                    has_changed = True  # Removing a default is a change.

        # Only log the final state if a change was made.
        if has_changed:
            logger.info(f"Final 'required' field for '{api_name}': {params_obj.get('required', [])}")


    def _refine_api_properties(self, function_def: dict, properties: dict, add_defaults: bool = False) -> None:
        """
        Refines API parameter properties by adding enums, formats, and default values using LLM and heuristics.

        Processes parameter properties to enhance their definitions through multiple refinement strategies:
        - Adds date/timestamp formats based on parameter name patterns
        - Generates enum values for categorical parameters using LLM
        - Applies rule-based defaults for common parameters like 'limit', 'offset', 'page'
        - Queries LLM for default values of important optional parameters
        - Recursively processes nested object properties

        Args:
            function_def: API function definition containing metadata for refinement context
            properties: Dictionary of parameter properties to refine
            add_defaults: Whether to add default values to optional parameters
        """
        if not isinstance(properties, dict):
            return

        required_params = function_def.get('parameters', {}).get('required', [])
        date_keywords = ['date', 'timestamp', 'created_at', 'updated_at', 'modified_on', 'last_login', 'harvested_on']
        enum_keywords = [
            'status', 'type', 'role', 'category', 'level', 'priority', 'method',
            'unit', 'gender', 'mode', 'condition', 'product_category', 'region',
        ]

        rule_based_defaults = {
            'limit': 3, 'offset': 0, 'page': 1,
            'sort_order': 'desc', 'include_details': False
        }
        # --- NEW: List of important optional parameters that should have a default ---
        default_candidate_keywords = ['status', 'category', 'priority', 'region', 'type', 'format', 'sort_by']

        for param_name, param_details in properties.items():
            if not isinstance(param_details, dict):
                continue

            if param_details.get('type') == 'object' and 'properties' in param_details:
                self._refine_api_properties(function_def, param_details['properties'], add_defaults=add_defaults)

            if add_defaults and param_name not in required_params and 'default' not in param_details:
                if param_name in rule_based_defaults:
                    param_details['default'] = rule_based_defaults[param_name]
                    logger.info(
                        f"[{self.domain}] Applied rule-based default for '{param_name}': {param_details['default']}")

                # --- MODIFIED: Only query LLM for candidates in our curated list ---
                elif any(keyword in param_name.lower() for keyword in default_candidate_keywords):
                    logger.debug(
                        f"[{self.domain}] Parameter '{param_name}' is a candidate for a default value. Querying LLM."
                    )
                    default_prompt = self.default_value_prompt_template.format(
                        api_name=function_def.get('name', 'N/A'),
                        api_description=function_def.get('description', 'N/A'),
                        parameter_name=param_name,
                        parameter_description=param_details.get('description', 'N/A'),
                        parameter_type=param_details.get('type'),
                        enum_values=param_details.get('enum', 'N/A')
                    )

                    try:
                        if isinstance(self.llm, VLLMClient):
                            message = [{'role': 'user', 'content': default_prompt}]
                            response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
                        else:
                            response = self.llm.invoke([HumanMessage(content=default_prompt)])
                        default_value_str = response.content.strip().replace('"', '')

                        # --- NEW: Check for empty response before assigning ---
                        if default_value_str:
                            param_type = param_details.get('type')
                            if param_type == 'integer':
                                default_value = int(default_value_str)
                            elif param_type == 'number':
                                default_value = float(default_value_str)
                            elif param_type == 'boolean':
                                default_value = default_value_str.lower() in ['true', '1', 't', 'y', 'yes']
                            else:
                                default_value = default_value_str

                            param_details['default'] = default_value
                            logger.info(
                                f"[{self.domain}] Successfully generated default for '{param_name}': {default_value}"
                            )

                        else:
                            logger.warning(
                                f"[{self.domain}] LLM returned an empty default for '{param_name}'. Skipping."
                            )

                    except Exception as e:
                        logger.warning(f"[{self.domain}] Could not generate a default for '{param_name}'. Error: {e}")

            if param_details.get('type') == 'string':
                if 'format' not in param_details and any(keyword in param_name.lower() for keyword in date_keywords):
                    param_details['format'] = 'date-time' if 'time' in param_name.lower() or \
                                                'timestamp' in param_name.lower() else 'date'

                if 'enum' not in param_details and any(keyword in param_name.lower() for keyword in enum_keywords):
                    logger.debug(
                        f"[{self.domain}] Keyword '{param_name}' matched for enum refinement. Querying LLM for values."
                    )
                    enum_prompt = self.enum_prompt_template.format(
                        api_name=function_def.get('name', 'N/A'),
                        api_description=function_def.get('description', 'N/A'),
                        parameter_name=param_name,
                        parameter_description=param_details.get('description', 'N/A')
                    )

                    try:
                        if isinstance(self.llm, VLLMClient):
                            message = [{'role': 'user', 'content': enum_prompt}]
                            response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
                        else:
                            response = self.llm.invoke([HumanMessage(content=enum_prompt)])
                        llm_output = response.content.strip()
                        enum_values = robust_json_list_of_strings(llm_output)

                        if isinstance(enum_values, list) and enum_values and all(
                                isinstance(i, str) for i in enum_values):
                            param_details['enum'] = enum_values
                            logger.info(
                                f"[{self.domain}] Successfully generated enum values for '{param_name}': {enum_values}")
                        else:
                            raise ValueError("LLM did not return a valid list of strings.")

                    except Exception as e:
                        logger.warning(
                            f"[{self.domain}] Failed to get enum values for '{param_name}' from LLM, "
                            f"falling back to placeholder. Error: {e}"
                        )
                        param_details['enum'] = ['value_1', 'value_2', 'value_3']


    def _refine_single_api(self, api: dict) -> dict:
        """
        Refines a single API definition by processing required parameters and property attributes.

        Applies refinement to both input parameters (with defaults) and output results
        (without defaults). The refinement includes determining required fields,
        adding enums, formats, and default values where appropriate.

        Args:
            api: API definition to refine

        Returns:
            Refined copy of the API definition with enhanced parameter properties
        """
        if not isinstance(api, dict) or 'function' not in api:
            return api
        refined_api = copy.deepcopy(api)
        function_def = refined_api['function']

        self._refine_required_parameters(function_def)

        if 'parameters' in function_def and 'properties' in function_def.get('parameters', {}):
            self._refine_api_properties(function_def, function_def['parameters']['properties'], add_defaults=True)

        if 'results' in function_def and 'properties' in function_def.get('results', {}):
            self._refine_api_properties(function_def, function_def['results']['properties'], add_defaults=False)

        return refined_api


    def run(self, domain_info: str) -> list[dict]:
        """
        Executes the complete API synthesis pipeline following the configured synthesis plan.

        Orchestrates the multi-step synthesis process by iterating through planned steps, generating
        APIs using LLM prompts, filtering duplicates, refining properties, tracking connections, and
        building up the API pool. Each step builds upon previously generated APIs to create an
        interconnected set of tool definitions.

        Args:
            domain_info: Descriptive information about the domain for which APIs are being synthesized

        Returns:
            List of all generated and refined API definitions accumulated across all synthesis steps
        """
        try:
            with open(self.synthesis_plan_path) as f:
                plan = json.load(f)
        except:
            logger.exception(f"[{self.domain}] Fatal Error: Could not load synthesis plan from '{self.synthesis_plan_path}'")
            return self.api_pool

        logger.info(f"[{self.domain}] Successfully loaded synthesis plan")

        for step_idx, step in enumerate(plan["steps"]):
            logger.info(f"[{self.domain}] --- Running Step {step_idx + 1}/{len(plan['steps'])}: {step['name']} ---")

            # Use stored path for the step's prompt
            prompt_template = load_prompt(step["prompt_path"])

            filled_prompt = prompt_template.format(
                domain_name=self.domain,
                domain_info=domain_info,
                context_apis_json=json.dumps(self.api_pool, indent=2),
                EXAMPLES_SECTION=self.incontext_examples_content, # Use pre-loaded examples
                NUMBER_OF_APIS_TO_GENERATE=step.get("num_to_generate", 2),
                available_output_params=self._extract_available_outputs()
            )

            if isinstance(self.llm, VLLMClient):
                message = [{'role': 'user', 'content': filled_prompt}]
                llm_response = self.llm.invoke(prompt_or_messages = message, use_chat_mode=True)
            else:
                llm_response = self.llm.invoke([HumanMessage(content=filled_prompt)])

            decoded = llm_response.content
            new_apis = robust_json_loads(decoded)
            if not new_apis:
                logger.error(f"[{self.domain}] Failed to parse any valid APIs from LLM response in step '{step['name']}'")
                continue

            cleaned_apis = [self._remove_echo_parameters(api) for api in new_apis]
            logger.debug(f"[{self.domain}] Received {len(cleaned_apis)} potential APIs from LLM.")

            valid_apis = [
                api for api in cleaned_apis
                if isinstance(api, dict) and "function" in api and not self._is_duplicate(api)
            ]
            if not valid_apis:
                logger.warning(f"[{self.domain}] No valid APIs generated in step '{step['name']}'. Skipping to next step.")
                continue

            logger.debug(f"[{self.domain}] Refining {len(valid_apis)} unique APIs...")
            refined_apis = [self._refine_single_api(api) for api in valid_apis]

            self.track_connections(refined_apis, self.api_pool)
            self.api_pool.extend(refined_apis)
            logger.info(
                f"[{self.domain}] Added {len(valid_apis)} new valid APIs to the pool "
                f"in step '{step['name']}'. Total APIs: {len(self.api_pool)}"
            )

        return self.api_pool
