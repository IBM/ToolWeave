def evaluate_api_set(apis: list[dict]) -> dict:
    """
    Evaluate API quality and complexity across multiple dimensions.

    Analyzes a collection of API definitions to compute metrics including interconnectivity
    between APIs, parameter diversity, complexity of parameter types, documentation coverage,
    and other structural characteristics.

    Args:
        apis: List of API schema dictionaries, each in the OpenAI function definition format

    Returns:
        Dictionary containing the following metrics:
            - Interconnectivity: Average number of parameters per API that match output
                fields from other APIs in the set
            - Entity Diversity: Total count of unique parameter names across all APIs
            - Complex API Use: Proportion of APIs using complex parameter types (object/array)
            - Total APIs: Number of valid API definitions in the input
            - Average Parameter Count: Mean number of parameters per API
            - Required Parameter Ratio: Proportion of parameters marked as required
            - Description Richness: Proportion of documentable items with descriptions
    """
    clean_apis = [api for api in apis if isinstance(api, dict) and "function" in api]
    if not clean_apis:
        return {
            "Interconnectivity": 0, "Entity Diversity": 0, "Complex API Use": 0,
            "Total APIs": 0, "Average Parameter Count": 0, "Required Parameter Ratio": 0,
            "Description Richness": 0
        }

    # --- Interconnectivity ---
    all_outputs = {k for api in clean_apis for k in api['function'].get('results', {}).get('properties', {})}
    total_inputs_from_outputs = sum(1 for api in clean_apis for k in api['function']['parameters']['properties'] if k in all_outputs)
    interconnectivity_score = total_inputs_from_outputs / len(clean_apis)

    # --- Entity Diversity ---
    all_params = {p for api in clean_apis for p in api['function']['parameters']['properties']}
    entity_diversity_score = len(all_params)

    # --- Complex API Use ---
    complex_apis = sum(1 for api in clean_apis if any(v.get('type') in ('object', 'array') for v in api['function']['parameters']['properties'].values()))
    complex_api_use_score = complex_apis / len(clean_apis)

    # --- DESCRIPTION AND PARAMETER METRICS ---
    total_params = 0
    total_required_params = 0
    documentable_items = 0
    documented_items = 0

    for api in clean_apis:
        func = api['function']
        
        # Count API description
        documentable_items += 1
        if func.get('description', '').strip():
            documented_items += 1
        
        params = func.get('parameters', {}).get('properties', {})
        total_params += len(params)
        total_required_params += len(func.get('parameters', {}).get('required', []))

        # Recursively check for parameter descriptions
        def check_descriptions(properties: dict) -> None:
            """
            Recursively count documentable items and documented items in nested parameter structures.

            Traverses parameter schemas including nested objects and arrays to identify all
            non-container type parameters and check whether they have descriptions. Updates
            the outer scope counters for documentable_items and documented_items.

            Args:
                properties: Dictionary mapping parameter names to their schema definitions
            """
            nonlocal documentable_items, documented_items
            for param_details in properties.values():
                # Only require descriptions for non-container types
                if param_details.get('type') not in ('object', 'array'):
                    documentable_items += 1
                    if param_details.get('description', '').strip():
                        documented_items += 1
                # Recurse into nested objects
                if 'properties' in param_details:
                    check_descriptions(param_details['properties'])
                # Recurse into arrays of objects
                if 'items' in param_details and 'properties' in param_details['items']:
                    check_descriptions(param_details['items']['properties'])

        check_descriptions(params)

    average_param_count = total_params / len(clean_apis) if clean_apis else 0
    required_param_ratio = total_required_params / total_params if total_params > 0 else 0
    description_richness = documented_items / documentable_items if documentable_items > 0 else 0

    return {
        "Interconnectivity": interconnectivity_score,
        "Entity Diversity": entity_diversity_score,
        "Complex API Use": complex_api_use_score,
        "Total APIs": len(clean_apis),
        "Average Parameter Count": average_param_count,
        "Required Parameter Ratio": required_param_ratio,
        "Description Richness": description_richness
    }
