import json
import logging
import re
from datetime import datetime
from typing import Any, Literal


logger = logging.getLogger('dialogue_generator')


def is_value_a_likely_placeholder(value: Any) -> bool:
    """
    Checks if a given value is likely a placeholder or dummy value.
    This is a heuristic check and can be extended.

    Args:
        value: The value to check.

    Returns:
        True if the value is considered a placeholder, False otherwise.
    """
    if not isinstance(value, str):
        return False  # Placeholders are typically strings

    # 1. Common generic placeholders (case-insensitive)
    common_placeholders = [
        "your_", "insert ", "enter ", "[your", "[insert", "[enter",
        "xxx", "---", "...", "placeholder", "dummy", "temp ", "value"
    ]
    value_lower = value.lower()
    for placeholder in common_placeholders:
        if placeholder in value_lower:  # Using 'in' for broader match, e.g., "your_api_key"
            logger.debug(f"[Placeholder Check] Value '{value}' matched common placeholder heuristic: '{placeholder}'")
            return True

    # 2. Specific known bad patterns (can be extended)
    known_bad_patterns = [
        r"\[.*placeholder.*\]",  # e.g., [API Key Placeholder]
        r"api_key_here",
        r"secret_here",
        r"token_here",
    ]
    for pattern in known_bad_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            logger.debug(f"[Placeholder Check] Value '{value}' matched known bad pattern: '{pattern}'")
            return True

    return False


def validate_value_against_schema(
        value: Any, param_schema: dict[str, Any],
        param_name: str | None = None,
    ) -> tuple[bool, str | None]:
    """
    Validates a given value against its parameter schema, including checks for placeholders,
    enum values, patterns, formats, and length/range constraints.

    Args:
        value: The value to validate.
        param_schema: The JSON schema for the parameter.
        param_name: Optional name of the parameter (used for error messages).

    Returns:
        A tuple where the first element is a boolean indicating if the value is valid,
        and the second element is an error message if invalid, or None if valid.
    """
    if not param_schema:
        return True, None

    if isinstance(value, str) and is_value_a_likely_placeholder(value, param_name):
        error_msg = f"Value '{value}' for parameter '{param_name}' is likely a placeholder."
        return False, error_msg

    # Enum Check
    if "enum" in param_schema:
        enum_values = param_schema["enum"]
        if isinstance(value, str):
            enum_lower = [str(e).lower() for e in enum_values if isinstance(e, str)]
            if value.lower() not in enum_lower:
                error_msg = f"Value '{value}' for parameter '{param_name}' is not one of the allowed enum values: {param_schema['enum']}"
                return False, error_msg
        else:
            if value not in enum_values:
                error_msg = f"Value '{value}' for parameter '{param_name}' is not one of the allowed enum values: {param_schema['enum']}"
                return False, error_msg

    # Pattern Check
    if isinstance(value, str) and "pattern" in param_schema:
        if not re.fullmatch(param_schema["pattern"], value):
            error_msg = f"Value '{value}' for parameter '{param_name}' does not match the required pattern: {param_schema['pattern']}"
            return False, error_msg

    # Format Check
    if isinstance(value, str) and param_schema.get("format") == "date":
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            error_msg = f"Value '{value}' for parameter '{param_name}' is not in the required date format (YYYY-MM-DD)."
            return False, error_msg

    if isinstance(value, str) and param_schema.get("format") == "date-time":
        try:
            # Try parsing with microseconds
            try:
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            except ValueError:
                # Try parsing without microseconds
                datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            error_msg = f"Value '{value}' for parameter '{param_name}' is not in the required date-time format (ISO 8601)."
            return False, error_msg

    # Length/Range Checks
    if isinstance(value, str):
        if "minLength" in param_schema and len(value) < param_schema["minLength"]:
            error_msg = f"Value '{value}' for parameter '{param_name}' is shorter than the minimum length of {param_schema['minLength']}."
            return False, error_msg

        if "maxLength" in param_schema and len(value) > param_schema["maxLength"]:
            error_msg = f"Value '{value}' for parameter '{param_name}' is longer than the maximum length of {param_schema['maxLength']}."
            return False, error_msg

    if isinstance(value, (int, float)):
        if "minimum" in param_schema and value < param_schema["minimum"]:
            error_msg = f"Value '{value}' for parameter '{param_name}' is less than the minimum value of {param_schema['minimum']}."
            return False, error_msg

        if "maximum" in param_schema and value > param_schema["maximum"]:
            error_msg = f"Value '{value}' for parameter '{param_name}' is greater than the maximum value of {param_schema['maximum']}."
            return False, error_msg

    return True, None


def coerce_types_from_schema(data: dict, schema: dict) -> dict:
    """
    Recursively coerce the types of values in `data` according to the OpenAI tool schema.

    Args:
        data: The dictionary of values to coerce.
        schema: The JSON schema defining the expected types for the values.

    Returns:
        A new dictionary with values coerced to the types defined in the schema where possible.
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return data

    properties = schema.get("properties", {})
    for key, value in data.items():
        prop_schema = properties.get(key, {})
        if not prop_schema:
            continue

        if "type" not in prop_schema:
            continue

        prop_type = prop_schema.get("type")

        if prop_type == "object" and isinstance(value, dict):
            data[key] = coerce_types_from_schema(value, prop_schema)

        elif prop_type == "array" and isinstance(value, list):
            item_schema = prop_schema.get("items", {})
            data[key] = [coerce_types_from_schema(item, item_schema) if isinstance(item, dict) else item for item in value]

        elif prop_type == "number":
            try:
                if isinstance(value, str):
                    data[key] = float(value) if ("." in value or "e" in value.lower()) else int(value)
            except Exception:
                pass

        elif prop_type == "integer":
            try:
                if isinstance(value, str):
                    data[key] = int(value)
            except Exception:
                pass

        elif prop_type == "boolean":
            if isinstance(value, str):
                if value.lower() in ("true", "1"):
                    data[key] = True
                elif value.lower() in ("false", "0"):
                    data[key] = False

        elif prop_type == "string":
            if not isinstance(value, str):
                data[key] = str(value)

    return data


def filter_schema_by_paths(schema: dict, paths: list[str]) -> None:
    """
    Modifies schema in-place to only keep properties specified in paths.

    Args:
        schema: JSON schema to modify
        paths: List of property paths (without function name prefix)
    """
    if not isinstance(schema, dict) or "properties" not in schema:
        return

    if schema.get("type") == "array" and "items" in schema:
        item_paths = []
        for path in paths:
            if path.startswith('[].'):
                item_paths.append(path[3:])

            else:
                item_paths.append(path)

        if item_paths and isinstance(schema["items"], dict):
            filter_schema_by_paths(schema["items"], item_paths)

        return

    top_level_props = set()
    nested_paths = {}

    for path in paths:
        if not path:
            continue

        parts = path.split('.', 1)
        prop_name = parts[0]

        if '[]' in prop_name:
            prop_name = prop_name.split('[]')[0]

        top_level_props.add(prop_name)

        if len(parts) > 1:
            if prop_name not in nested_paths:
                nested_paths[prop_name] = []
            nested_paths[prop_name].append(parts[1])

    properties = schema["properties"]
    filtered_properties = {}

    for name, prop_schema in properties.items():
        if name in top_level_props:
            filtered_properties[name] = prop_schema.copy()
            if name in nested_paths:
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    filter_schema_by_paths(filtered_properties[name], nested_paths[name])

                elif prop_schema.get("type") == "array" and "items" in prop_schema:
                    array_paths = []
                    for sub_path in nested_paths[name]:
                        if sub_path.startswith('[].'):
                            array_paths.append(sub_path[3:])
                        else:
                            array_paths.append(sub_path)

                    if isinstance(prop_schema["items"], dict):
                        filter_schema_by_paths(filtered_properties[name]["items"], array_paths)

    schema["properties"] = filtered_properties

    if "required" in schema and isinstance(schema["required"], list):
        schema["required"] = [r for r in schema["required"] if r in top_level_props]


def filter_tool_schemas(
        tool_schemas: list[dict],
        params_to_keep: list[str] | None = None,
        filter_mode: Literal["keep_inputs", "keep_outputs"] = "keep_inputs"
    ) -> list[dict]:
    """
    Filters tool schemas to retain only specified parameters and filter based on mode.

    Args:
        tool_schemas: List of tool schema definitions
        params_to_keep: List of parameter names to keep in the filtered schemas
        filter_mode: Mode determining what to keep in the schema:
            - "keep_inputs": Keep input parameters, remove results/outputs
            - "keep_outputs": Keep results/outputs, remove input parameters

    Returns:
        List of filtered tool schemas with only the required parameters
    """
    if not tool_schemas:
        return tool_schemas

    filtered_schemas = []

    for schema in tool_schemas:
        # Create a deep copy to avoid modifying the original
        filtered_schema = json.loads(json.dumps(schema))

        if "function" not in filtered_schema:
            filtered_schemas.append(filtered_schema)
            continue

        function_name = filtered_schema["function"].get("name", "")

        if filter_mode == "keep_inputs":
            if "results" in filtered_schema["function"]:
                del filtered_schema["function"]["results"]

            if params_to_keep and "parameters" in filtered_schema["function"]:
                if "properties" in filtered_schema["function"]["parameters"]:
                    param_paths = [
                        p[len(function_name) + 1:] for p in params_to_keep
                        if p.startswith(f"{function_name}.")
                    ]

                    if param_paths:
                        filter_schema_by_paths(filtered_schema["function"]["parameters"], param_paths)

                    else:
                        filtered_schema["function"]["parameters"] = {}
                        continue

            elif params_to_keep is None:
                filtered_schema["function"]["parameters"] = {}
                continue

        elif filter_mode == "keep_outputs":
            if "parameters" in filtered_schema["function"]:
                del filtered_schema["function"]["parameters"]

            if params_to_keep and "results" in filtered_schema["function"]:
                param_paths = [
                    p[len(function_name) + 1:] for p in params_to_keep
                    if p.startswith(f"{function_name}.")
                ]

                if param_paths:
                    filter_schema_by_paths(filtered_schema["function"]["results"], param_paths)

                else:
                    filtered_schema["function"]["results"] = {}
                    continue

            elif params_to_keep is None:
                filtered_schema["function"]["results"] = {}
                continue

        filtered_schemas.append(filtered_schema)

    return filtered_schemas
