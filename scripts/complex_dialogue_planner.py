import argparse
import json
import logging
import os
import random
import re
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from pprint import pformat

from natsort import natsorted

from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('complex_dialogue_planner')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def setup_logging(debug_mode: bool = False, log_file: str = None) -> logging.Logger:
    """Setup logging configuration with console and optional file handlers.

    Args:
        debug_mode: Whether to enable debug level logging
        log_file: Optional path to log file for output

    Returns:
        Configured logger instance
    """
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if debug_mode:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def load_prompt(filename: str, prompts_dir: str) -> str:
    """Load a prompt template from a file in the prompts directory."""
    try:
        with open(os.path.join(prompts_dir, filename), 'r') as f:
            return f.read()

    except FileNotFoundError:
        logger.exception(f"Prompt file not found at {os.path.join(prompts_dir, filename)}")
        raise


def extract_json_from_response(raw_text: str) -> str | None:
    """Extract JSON content from an LLM response, handling code blocks and finding the outermost JSON object or array."""
    if not isinstance(raw_text, str): return None
    text = raw_text.strip()

    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
    if match:
        text = match.group(1).strip()
    else:
        text = raw_text

    first_brace, first_bracket = text.find('{'), text.find('[')
    if first_brace == -1 and first_bracket == -1: return None

    start_char = '{' if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket) else '['
    end_char = '}' if start_char == '{' else ']'
    start_index = text.find(start_char)

    open_count = 0
    for i in range(start_index, len(text)):
        if text[i] == start_char:
            open_count += 1

        elif text[i] == end_char:
            open_count -= 1

        if open_count == 0: return text[start_index:i + 1]

    return None


def load_api_definitions(api_definitions_path: str, synthetic_apis: bool = False) -> dict[str, dict]:
    """Load API definitions from a JSON file and return a mapping of function names to their definitions."""
    with open(api_definitions_path, 'r') as f:
        api_list = json.load(f)

        if synthetic_apis:
            api_list = api_list.get("apis", [])

    return {
        api_obj["function"]["name"]: api_obj["function"] for api_obj in api_list
        if "function" in api_obj and "name" in api_obj["function"]
    }


def load_param_connection_map(api_definitions_path: str, synthetic_apis: bool = False) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Load parameter connection mappings from synthetic API definitions for tracking parameter dependencies between tools."""
    if not synthetic_apis:
        return {}
    
    with open(api_definitions_path, 'r') as f:
        api_sdk = json.load(f)

    param_connection_map = {}
    for connection in api_sdk.get("connection_map", []):
        source_api = connection.get("source_api")
        target_api = connection.get("target_api")
        source_param = connection.get("source_param")
        target_param = connection.get("target_param")

        if source_api and target_api and source_param and target_param:
            # We are setting the lookup in reverse direction because we'll know the 
            # current tool and param while sampling relevant params and we'll need to
            # look up the known params from previous tools which we won't have access to
            if (target_api, target_param) not in param_connection_map:
                param_connection_map[(target_api, target_param)] = []

            param_connection_map[(target_api, target_param)].append((source_api, source_param))

    return param_connection_map


def load_goals_from_jsonl(file_path: str, num_goals_to_process: int = -1) -> tuple[dict, list[dict]]:
    """Load goals from a JSONL file where the first line contains metadata and subsequent lines contain goal objects.

    Args:
        file_path: Path to the JSONL file containing goals
        num_goals_to_process: Maximum number of goals to load, -1 for all goals

    Returns:
        Tuple of (metadata dict, list of goal dicts)
    """
    metadata, goals = {}, []
    try:
        with open(file_path, 'r') as f:
            header_line = f.readline()
            if header_line:
                metadata = json.loads(header_line)

            for line in f:
                if line.strip():
                    goals.append(json.loads(line.strip()))

    except FileNotFoundError:
        logger.exception(f"Goals file not found at {file_path}")
        return {}, []

    except json.JSONDecodeError as e:
        logger.exception(f"Error decoding JSON from {file_path}: {e}")
        return {}, []

    if num_goals_to_process > 0:
        goals = goals[:num_goals_to_process]

    return metadata, goals


def flatten_property(path: str, schema: dict) -> dict:
    """Recursively flattens a property and its nested properties."""
    if not isinstance(schema, dict):
        return {path: schema}

    if schema.get("type") != "object" and not (
        schema.get("type") == "array" and 
        isinstance(schema.get("items"), dict) and 
        schema.get("items", {}).get("type") == "object"
    ):
        return {path: schema}

    result = {}

    if schema.get("type") == "object" and "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            nested_path = f"{path}.{prop_name}"
            nested_result = flatten_property(nested_path, prop_schema)
            result.update(nested_result)

    elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        items_schema = schema.get("items", {})
        if items_schema.get("type") == "object" and "properties" in items_schema:
            for prop_name, prop_schema in items_schema["properties"].items():
                nested_path = f"{path}[].{prop_name}"
                nested_result = flatten_property(nested_path, prop_schema)
                result.update(nested_result)

    return result


def get_nested_required_props(parent_path: str, schema: dict) -> list[str]:
    """Gets the flattened required properties from nested objects/arrays."""
    required = []

    if schema.get("type") == "object" and "properties" in schema:
        for req in schema.get("required", []):
            prop_schema = schema["properties"].get(req, {})

            if isinstance(prop_schema, dict) and prop_schema.get("type") in ["object", "array"]:
                nested_req = get_nested_required_props(f"{parent_path}.{req}", prop_schema)
                required.extend(nested_req)

            else:
                required.append(f"{parent_path}.{req}")

    elif schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        items_schema = schema.get("items", {})
        if items_schema.get("type") == "object" and "properties" in items_schema:
            for req in items_schema.get("required", []):
                prop_schema = items_schema["properties"].get(req, {})

                if isinstance(prop_schema, dict) and prop_schema.get("type") in ["object", "array"]:
                    nested_req = get_nested_required_props(f"{parent_path}[].{req}", prop_schema)
                    required.extend(nested_req)

                else:
                    required.append(f"{parent_path}[].{req}")

    return required


def flatten_object_schema(schema: dict) -> dict:
    """Flattens a JSON Schema object by bringing all nested properties to the top level."""
    result = schema.copy()
    if "properties" not in schema or not isinstance(schema["properties"], dict):
        return result

    flattened_properties = {}
    flattened_required = []

    original_required = schema.get("required", [])

    for prop_name, prop_schema in schema["properties"].items():
        flat_props = flatten_property(prop_name, prop_schema)

        flattened_properties.update(flat_props)

        if prop_name in original_required:
            if isinstance(prop_schema, dict) and prop_schema.get("type") in ["object", "array"]:
                nested_required = get_nested_required_props(prop_name, prop_schema)
                flattened_required.extend(nested_required)

            else:
                flattened_required.append(prop_name)

    result["properties"] = flattened_properties
    if flattened_required:
        result["required"] = flattened_required

    return result


def flatten_schema(tool_def: dict) -> dict:
    """Flattens the parameter and result schemas in a tool definition."""
    flattened_tool = tool_def.copy()

    if "parameters" in flattened_tool:
        flattened_tool["parameters"] = flatten_object_schema(flattened_tool["parameters"])

    if "results" in flattened_tool:
        flattened_tool["results"] = flatten_object_schema(flattened_tool["results"])

    return flattened_tool


def get_semantic_tool_partition(llm: VLLMClient | WatsonxLLM, goal: str, tool_path: list[str], prompts_dir: str) -> list[int]:
    """Use LLM to partition a tool sequence into semantic groups based on the goal.

    Args:
        llm: The language model client to use for generation
        goal: The high-level goal text
        tool_path: List of tool names in the sequence
        prompts_dir: Directory containing prompt templates

    Returns:
        List of integers representing the size of each partition group
    """
    prompt_template = load_prompt("partition_goal_prompt.txt", prompts_dir)
    tools_str = ", ".join(tool_path)
    prompt = prompt_template.format(goal=goal, tool_path=tools_str, tool_count=len(tool_path))

    if isinstance(llm, VLLMClient):
        message = [{'role': 'user', 'content': prompt}]
        response = llm.invoke(prompt_or_messages=message, use_chat_mode=True)
    else:
        response = llm.invoke(prompt)
    json_str = extract_json_from_response(response.content)
    try:
        if json_str:
            partition = json.loads(json_str)
            if isinstance(partition, list) and all(isinstance(x, int) for x in partition) \
                and sum(partition) == len(tool_path) and 0 not in partition:
                return partition

    except:
        logger.exception(f"Could not parse partition from LLM")

    return [len(tool_path)]


def generate_sub_goals_for_partition(
    llm: VLLMClient | WatsonxLLM,
    high_level_goal: str, partition: list[list[str]],
    api_mapping: dict, prompts_dir: str,
) -> list[str]:
    """Generate sub-goal utterances for each partition using an LLM.

    Args:
        llm: The language model client to use for generation
        high_level_goal: The overall goal text
        partition: List of tool groups, where each group is a list of tool names
        api_mapping: Mapping of tool names to their definitions
        prompts_dir: Directory containing prompt templates

    Returns:
        List of sub-goal utterances, one per partition group
    """
    prompt_template = load_prompt("generate_sub_goals_prompt.txt", prompts_dir)
    tool_groups_str = []
    for i, group in enumerate(partition):
        group_descs = "\n".join([
            f"  - {tool_name}: {api_mapping.get(tool_name, {}).get('description', '')}"
            for tool_name in group
        ])
        tool_groups_str.append(f"- Tools for sub-task {i + 1}:\n{group_descs}")

    tools_str = "\n".join(tool_groups_str)
    prompt = prompt_template.format(high_level_goal=high_level_goal, tools_str=tools_str, num_groups=len(partition))

    if isinstance(llm, VLLMClient):
        message = [{'role': 'user', 'content': prompt}]
        response = llm.invoke(prompt_or_messages=message, use_chat_mode=True)
    else:
        response = llm.invoke(prompt)

    json_str = extract_json_from_response(response.content)
    try:
        if json_str:
            sub_goals = json.loads(json_str)
            if isinstance(sub_goals, list) and len(sub_goals) == len(partition):
                return sub_goals

    except:
        logger.exception(f"Warning: Could not parse sub-goals from LLM")

    return [f"Complete step {i + 1} of the process." for i in range(len(partition))]


def generate_conditional_utterance(
    llm: VLLMClient | WatsonxLLM,
    high_level_goal: str, first_utterance: str, condition_text: str,
    tool_group: list[str], api_mapping: dict, prompts_dir: str,
) -> str:
    """Generate a natural-sounding utterance for the second step of a conditional plan.

    Args:
        llm: The language model client to use for generation
        high_level_goal: The overall goal text
        first_utterance: The initial user utterance that started the conversation
        condition_text: Description of the condition that triggered this branch
        tool_group: List of tool names to be executed in this conditional branch
        api_mapping: Mapping of tool names to their definitions
        prompts_dir: Directory containing prompt templates

    Returns:
        Generated utterance text for the conditional branch
    """
    prompt_template = load_prompt("generate_conditional_utterance_prompt.txt", prompts_dir)
    group_descs = "\n".join([
        f"    - {tool_name}: {api_mapping.get(tool_name, {}).get('description', '')}"
        for tool_name in tool_group
    ])
    prompt = prompt_template.format(
        high_level_goal=high_level_goal,
        first_utterance=first_utterance,
        condition_text=condition_text,
        tool_group_str=group_descs
    )

    if isinstance(llm, VLLMClient):
        message = [{'role': 'user', 'content': prompt}]
        response = llm.invoke(prompt_or_messages=message, use_chat_mode=True)
    else:
        response = llm.invoke(prompt)
    return response.content.strip()


def generate_dialogue_plan(
    goal_obj: dict, api_mapping: dict[str, dict],
    param_connection_map: dict[tuple[str, str], list[tuple[str, str]]],
    partition_sizes: list[int], sub_goal_utterances: list[str],
    optional_param_sampling_prob: float, clarification_prob: float,
    decision_context: dict = None,
) -> list[dict]:
    """Generate a structured dialogue plan from a goal object with tool calls, user utterances, and assistant responses.

    The function processes tools in semantic groups based on partition sizes, 
    handling parameter dependencies and optional parameters. For each tool group, 
    it determines which parameters can be inferred from previous tool outputs via the 
    connection map, which should be clarified with the user, and which can be 
    assumed from the user's utterance. The plan includes user utterances, 
    assistant clarifications when needed, tool calls with parameter mappings, 
    and assistant responses summarizing tool outputs.

    Args:
        goal_obj: Goal object containing the tool path and metadata
        api_mapping: Mapping of tool names to their flattened definitions
        param_connection_map: Maps (tool, param) tuples to their source dependencies
        partition_sizes: List of integers defining how many tools in each semantic group
        sub_goal_utterances: User utterances corresponding to each partition group
        optional_param_sampling_prob: Probability of sampling optional parameters
        clarification_prob: Probability of requesting clarification for a required parameter
        decision_context: Optional context for conditional branching with decision variables

    Returns:
        List of plan step dictionaries representing the complete dialogue flow
    """
    tool_path = goal_obj.get('path', goal_obj.get('tools', []))
    tool_queue = deque(tool_path)
    utterance_queue = deque(sub_goal_utterances)
    plan, step, known_params, param_sources = [], 1, set(), {}

    for _, group_size in enumerate(partition_sizes):
        utterance_step_index = -1
        if utterance_queue:
            plan.append({"step": step, "type": "USER_UTTERANCE", "utterance": utterance_queue.popleft()})
            utterance_step_index = len(plan) - 1
            step += 1

        tools_for_this_group = [tool_queue.popleft() for _ in range(group_size) if tool_queue]

        all_outputs_for_group = []
        params_to_clarify = []
        group_plan = []
        orig_step = step

        for tool_name in tools_for_this_group:
            tool_def = flatten_schema(api_mapping.get(tool_name, {}))
            required = tool_def.get("parameters", {}).get("required", [])
            properties = tool_def.get("parameters", {}).get("properties", {})

            tool_params_to_clarify, params_assumed = [], {}
            sampled_optional_params = []

            if utterance_step_index != -1 and (random.random() < optional_param_sampling_prob or "update" in tool_name.lower()):
                optional_params_available = []
                for param_name, param_def in properties.items():
                    if param_name not in required and "default" not in param_def:
                        optional_params_available.append(param_name)

                if optional_params_available:
                    num_to_sample = random.randint(1, min(2, len(optional_params_available)))
                    sampled_optional_params = random.sample(optional_params_available, num_to_sample)

            for param_name in required:
                prev_connections = param_connection_map.get((tool_name, param_name), [])
                if prev_connections and any(
                    f"{prev_tool_name}.{prev_connected_param}" in known_params
                    for prev_tool_name, prev_connected_param in prev_connections
                    if prev_tool_name in tool_path[:tool_path.index(tool_name)]
                ):
                    continue

                if random.random() < clarification_prob:
                    tool_params_to_clarify.append(f"{tool_name}.{param_name}")

                else:
                    placeholder = f"$user_provided_${tool_name}.{param_name}"
                    params_assumed[f"{tool_name}.{param_name}"] = {"placeholder": placeholder, "simulation_hint": "VALID_DATA"}

            is_param_for_update = "update" in tool_name.lower()
            for param_name in sampled_optional_params:
                if not is_param_for_update:
                    prev_connections = param_connection_map.get((tool_name, param_name), [])
                    if prev_connections and any(
                        f"{prev_tool_name}.{prev_connected_param}" in known_params
                        for prev_tool_name, prev_connected_param in prev_connections
                        if prev_tool_name in tool_path[:tool_path.index(tool_name)]
                    ):
                        continue

                placeholder = f"$user_provided_${tool_name}.{param_name}"
                params_assumed[f"{tool_name}.{param_name}"] = {"placeholder": placeholder, "simulation_hint": "VALID_DATA"}

            if utterance_step_index != -1 and params_assumed:
                clean_assumed = {k: v['placeholder'] for k, v in params_assumed.items()}
                plan[utterance_step_index].setdefault('provided_params', {}).update(clean_assumed)

            for param_name in properties.keys():
                if param_name not in params_assumed:
                    prev_connections = param_connection_map.get((tool_name, param_name), [])
                    if prev_connections:
                        prev_tool_names = tool_path[:tool_path.index(tool_name)]
                        for prev_tool_name, prev_connected_param in prev_connections:
                            prev_connected_param_name = f"{prev_tool_name}.{prev_connected_param}"
                            if prev_connected_param_name in known_params and prev_tool_name in prev_tool_names:
                                params_assumed[f"{tool_name}.{param_name}"] = {
                                    "placeholder": f"${prev_connected_param_name}",
                                    "simulation_hint": "VALID_DATA"
                                }
                                break

            for param, source_info in params_assumed.items():
                if source_info["simulation_hint"] == "VALID_DATA":
                    known_params.add(f"{tool_name}.{param}")

                else:
                    # If an error was injected, the user will still need to clarify it later
                    tool_params_to_clarify.append(f"{tool_name}.{param}")

                param_sources[param] = source_info["placeholder"]

            params_to_clarify.extend(tool_params_to_clarify)
            if tool_params_to_clarify:
                for param_name in tool_params_to_clarify:
                    placeholder = f"$user_provided_${param_name}"
                    known_params.add(param_name)
                    param_sources[param_name] = placeholder

            prefix = f"{tool_name}."
            params_for_call = {
                key: param_sources[key]
                for key in (prefix + p for p in properties.keys())
                if key in param_sources
            }

            plan_step = {"step": step, "type": "CALL_TOOL", "tool_name": tool_name, "parameters": params_for_call}
            if decision_context and tool_name == decision_context.get("tool_name"):
                plan_step["decision_variables"] = {
                    decision_context["variable"]: decision_context["trigger_value"]
                }
            group_plan.append(plan_step)
            step += 1

            outputs = [f"{tool_name}.{out_name}" for out_name in tool_def.get("results", {}).get("properties", {}).keys()]
            all_outputs_for_group.extend(outputs)
            known_params.update(outputs)

        if params_to_clarify:
            step = orig_step
            unique_clarifications = list(set(params_to_clarify))
            plan.append({"step": step, "type": "ASSISTANT_CLARIFICATION", "parameter_names": unique_clarifications})
            step += 1
            provides_map = {}

            for param_name in unique_clarifications:
                placeholder = f"$user_provided_${param_name}"
                # The user's response to clarification is assumed to be valid
                provides_map[param_name] = placeholder

            plan.append({"step": step, "type": "USER_RESPONSE_TO_CLARIFICATION", "provides_params": provides_map})
            step += 1

            for plan_step in group_plan:
                plan.append({**plan_step, "step": step})
                step += 1

        else:
            plan.extend(group_plan)

        if tools_for_this_group:
            plan.append({
                "step": step,
                "type": "ASSISTANT_RESPONSE_TOOL",
                "summarizes_tools": tools_for_this_group,
                "outputs_provided": all_outputs_for_group
            })
            step += 1

    return plan


def process_goal(
    llm: VLLMClient | WatsonxLLM, goal_obj: dict,
    api_mapping: dict[str, dict],
    param_connection_map: dict[tuple[str, str], list[tuple[str, str]]],
    args: argparse.Namespace,
) -> list[dict]:
    """Process a single complex goal and generate dialogue plan variants based on its type.

    Handles different goal types (linear_chain, random_walk, fan_out_fan_in, conditional) 
    by partitioning tools into semantic groups, generating sub-goal utterances, and 
    creating multiple dialogue plan variants. For fan_out_fan_in goals, generates plans 
    for various partition strategies. For conditional goals, generates separate plans 
    for if and else branches with appropriate decision contexts.

    Args:
        llm: The language model client to use for generation
        goal_obj: Goal object containing goal_data, goal_id, and generation_context
        api_mapping: Mapping of tool names to their flattened definitions
        param_connection_map: Maps (tool, param) tuples to their source dependencies
        args: Command-line arguments containing probabilities and configuration settings

    Returns:
        List of dialogue plan objects with metadata, partitions, and execution steps
    """
    goal_data = goal_obj.get("goal_data", {})
    goal_id = goal_obj.get("goal_id")
    generation_context = goal_obj.get("generation_context", {})
    goal_type = goal_data.get("type")
    goal_text = goal_data.get('goal_text') or goal_data.get('goal')

    logger.info(f"Processing {goal_id} (Type: '{goal_type}')")

    plan_objects = []

    base_plan_info = {
        "source_goal_id": goal_id,
        "goal_type": goal_type,
        "goal_score_breakdown": goal_data.get("score_breakdown"),
        "source_goal_text": goal_text,
        "generation_context": generation_context
    }

    if goal_type in ["linear_chain", "random_walk"]:
        path = goal_data.get("path", [])
        if not path: return []

        partition_sizes = get_semantic_tool_partition(llm, goal_text, path, args.prompts_dir)
        partition_groups = []
        tool_deque = deque(path)
        for size in partition_sizes:
            partition_groups.append([tool_deque.popleft() for _ in range(size)])

        sub_goals = generate_sub_goals_for_partition(llm, goal_text, partition_groups, api_mapping, args.prompts_dir)

        logger.debug(
            f"  - Generating {args.num_plan_variants} dialogue plan "
            f"variants for partition: {partition_sizes}"
        )

        for i in range(args.num_plan_variants):
            dialogue_plan = generate_dialogue_plan(
                goal_data, api_mapping, param_connection_map,
                partition_sizes, sub_goals,
                args.optional_param_sampling_prob,
                args.clarification_prob, args.chitchat_prob,
                args.error_injection_prob, args.simulate_errors
            )

            plan_obj = {
                "dialogue_plan_id": f"{goal_id}_variant_{i + 1}",
                **base_plan_info,
                "partition": list(partition_sizes),
                "path": path,
                "sub_goal_utterances": sub_goals,
                "plan": dialogue_plan
            }
            plan_objects.append(plan_obj)

    elif goal_type == "fan_out_fan_in":
        start = goal_data.get('start_tool')
        branch = goal_data.get('branch_tools')
        end = goal_data.get('end_tool')
        if not all([start, branch, end]): return []

        full_path = [start] + branch + [end]
        temp_goal_data_with_path = goal_data.copy()
        temp_goal_data_with_path['path'] = full_path

        strategies = {
            "Single_Turn": [full_path],
            "Fan_Out_Full": [[start] + branch, [end]],
            "Fan_In_Full": [[start], branch + [end]],
        }

        for i in range(1, len(branch)):
            for subset in combinations(range(len(branch)), i):
                out_subset_indices = list(subset)
                in_subset_indices = [j for j in range(len(branch)) if j not in out_subset_indices]

                out_subset = [branch[idx] for idx in out_subset_indices]
                in_subset = [branch[idx] for idx in in_subset_indices]

                out_indices_str = "_".join(f"T{idx + 1}" for idx in out_subset_indices)
                in_indices_str = "_".join(f"T{idx + 1}" for idx in in_subset_indices)

                strategy_name = f"Fan_Out_{out_indices_str}_Fan_In_{in_indices_str}"
                if strategy_name not in strategies:
                    strategies[strategy_name] = [[start] + out_subset, in_subset + [end]]

        for strategy_name, partition_groups in strategies.items():
            logger.debug(
                f"  - Generating {args.num_plan_variants} "
                f"variants for partition strategy: {strategy_name}"
            )
            sub_goals = generate_sub_goals_for_partition(
                llm, goal_text, partition_groups, api_mapping, args.prompts_dir
            )
            partition_sizes = [len(group) for group in partition_groups]

            for i in range(args.num_plan_variants):
                dialogue_plan = generate_dialogue_plan(
                    temp_goal_data_with_path, api_mapping, param_connection_map,
                    partition_sizes, sub_goals,
                    args.optional_param_sampling_prob, args.clarification_prob,
                    args.chitchat_prob, args.error_injection_prob, args.simulate_errors
                )

                plan_obj = {
                    "dialogue_plan_id": f"{goal_id}_{strategy_name.lower()}_v{i + 1}",
                    **base_plan_info,
                    "partition_strategy": strategy_name,
                    "partition": partition_sizes,
                    "path": temp_goal_data_with_path['path'],
                    "sub_goal_utterances": sub_goals,
                    "plan": dialogue_plan
                }
                plan_objects.append(plan_obj)

            if (
                args.max_fan_out_patterns != -1 and
                len(plan_objects) >= args.max_fan_out_patterns * args.num_plan_variants
            ):
                break

    elif goal_type == "conditional":
        start = goal_data.get('start_tool')
        if_tool = goal_data.get('if_branch_tool')
        else_tool = goal_data.get('else_branch_tool')
        if not all([start, if_tool, else_tool]):
            return []

        condition_details = goal_data.get("condition_details", {})
        variable = condition_details.get("variable")
        operator = condition_details.get("operator")
        value = condition_details.get("value")

        if not all([variable, operator, value]):
            logger.warning(
                f"Skipping conditional goal {goal_id} due to "
                "missing or incomplete 'condition_details'."
            )
            return []

        branch_partition_size = [2]

        logger.debug(f"  - Generating {args.num_plan_variants} variants for the 'IF' branch...")
        if_path = [start, if_tool]

        if_goal_data = goal_data.copy()
        if_goal_data['path'] = if_path

        if_sub_goals = [goal_text]
        decision_context_if = {"tool_name": start, "variable": variable, "trigger_value": value}

        for i in range(args.num_plan_variants):
            dialogue_plan = generate_dialogue_plan(
                if_goal_data, api_mapping, param_connection_map,
                branch_partition_size, if_sub_goals,
                args.optional_param_sampling_prob, args.clarification_prob, args.chitchat_prob,
                args.error_injection_prob, args.simulate_errors, decision_context=decision_context_if
            )
            final_plan_obj = {
                "dialogue_plan_id": f"{goal_id}_if_v{i + 1}",
                **base_plan_info,
                "goal_type": "conditional_if",  # Specify the branch type
                "partition": branch_partition_size,
                "path": if_path,
                "sub_goal_utterances": if_sub_goals,
                "plan": dialogue_plan
            }
            plan_objects.append(final_plan_obj)

        logger.debug(f"  - Generating {args.num_plan_variants} variants for the 'ELSE' branch...")
        else_path = [start, else_tool]

        else_goal_data = goal_data.copy()
        else_goal_data['path'] = else_path

        else_sub_goals = [goal_text]
        decision_context_else = {"tool_name": start, "variable": variable, "trigger_value": f"NOT {value}"}

        for i in range(args.num_plan_variants):
            dialogue_plan = generate_dialogue_plan(
                else_goal_data, api_mapping, param_connection_map,
                branch_partition_size, else_sub_goals,
                args.optional_param_sampling_prob, args.clarification_prob, args.chitchat_prob,
                args.error_injection_prob, args.simulate_errors, decision_context=decision_context_else
            )

            final_plan_obj = {
                "dialogue_plan_id": f"{goal_id}_else_v{i + 1}",
                **base_plan_info,
                "goal_type": "conditional_else",  # Specify the branch type
                "partition": branch_partition_size,
                "path": else_path,
                "sub_goal_utterances": else_sub_goals,
                "plan": dialogue_plan
            }
            plan_objects.append(final_plan_obj)

    return plan_objects


def main():
    parser = argparse.ArgumentParser(description="Generate structural dialogue plans from complex goal patterns.")
    parser.add_argument("--goals_file_path", type=str, required=True)
    parser.add_argument("--num_goals_to_process", type=int, default=-1,
                        help="Number of goals to process from the input file.")
    parser.add_argument("--synthetic_apis", action="store_true",
                        help="If set, uses processing related to synthetic APIs.")
    parser.add_argument("--api_definitions_path", type=str, required=True)
    parser.add_argument("--prompts_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--optional_param_sampling_prob", type=float, default=0.1,
                        help="Probability of sampling optional parameters for a tool call.")
    parser.add_argument("--clarification_prob", type=float, default=0.3)
    parser.add_argument("--num_plan_variants", type=int, default=2)
    parser.add_argument("--max_fan_out_patterns", type=int, default=-1,
                        help="Maximum number of fan-out patterns to generate. `max_fan_out_patterns * num_plan_variants`" + \
                            " plans will be generated for each goal. If set to -1, there is no limit.")
    parser.add_argument("--max_workers", type=int, default=60,
                        help="Maximum number of worker threads for parallel processing.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for detailed logging.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to save the log file. If not provided, logs will only be printed to console.")
    parser.add_argument("--llm_type", type=str, choices=['watsonx', 'vllm'], default='watsonx', help="The type of LLM to use for generation.")

    args = parser.parse_args()

    setup_logging(args.debug, log_file=args.log_file)

    if args.debug:
        logger.debug(f"Debug mode enabled. Setting max_workers to 1 for debugging.")
        args.max_workers = 1

    if args.llm_type == 'watsonx':
        try:
            watsonx_config_path = os.path.join(project_root, 'watsonx_llm_config.yml')
            llm = WatsonxLLM(watsonx_config_path)
        except:
            logger.exception(f"Failed to initialize WatsonxLLM. Please check your config.")
            return
    elif args.llm_type == "vllm":
        try:
            vllm_config_path = os.path.join(project_root, 'vllm_llm_config.yml')
            llm = VLLMClient(vllm_config_path)
        except:
            logger.exception(f"Failed to initialize VLLM. Please check your config.")
            return
    else:
        raise ValueError(f"Unsupported LLM type: {args.llm_type}")



    api_mapping = load_api_definitions(args.api_definitions_path, args.synthetic_apis)
    param_connection_map = load_param_connection_map(args.api_definitions_path, args.synthetic_apis)
    run_metadata, complex_goals = load_goals_from_jsonl(args.goals_file_path, args.num_goals_to_process)
    if not complex_goals:
        logger.warning("No goals were loaded from the file. Exiting.")
        return

    logger.info(f"Loaded {len(complex_goals)} goals. Associated run metadata:\n{pformat(run_metadata)}\n")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize statistics counters
    plan_counts = {}
    goals_processed_count = 0
    total_plans_generated = 0

    with open(output_path, 'w') as f:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_goal, llm, goal_obj, api_mapping,
                    param_connection_map, args
                ): goal_obj.get("goal_data").get("type")
                for goal_obj in complex_goals
            }

            for future in as_completed(futures):
                try:
                    plan_objects = future.result()
                    for plan_obj in plan_objects:
                        f.write(json.dumps(plan_obj) + "\n")
                        goal_type = plan_obj.get("goal_type")
                        if goal_type not in plan_counts:
                            plan_counts[goal_type] = 0
                        plan_counts[goal_type] += 1
                        total_plans_generated += 1

                    goals_processed_count += 1
                    logger.info(
                        f"Generated {len(plan_objects)} plans for goal ID {plan_obj['source_goal_id']} "
                        f"with type '{futures[future]}' ({goals_processed_count}/{len(complex_goals)})"
                    )

                except:
                    logger.exception("Error processing goal object.")

    logger.info("Rewriting the plans sorted by dialogue_plan_id...")
    with open(output_path, 'r') as f:
        plans = [json.loads(line.strip()) for line in f if line.strip()]

    plans = natsorted(plans, key=lambda x: x.get("dialogue_plan_id", ""))
    with open(output_path, 'w') as f:
        for plan in plans:
            f.write(json.dumps(plan) + "\n")

        final_metadata = {
            "run_summary_metadata": {
                "source_goal_run_metadata": run_metadata.get("metadata", {}),
                "dialogue_plan_run_parameters": vars(args),
                "generation_summary": {
                    "total_plans_generated": total_plans_generated,
                    "plan_counts_by_type": plan_counts
                }
            }
        }
        f.write(json.dumps(final_metadata) + "\n")

    logger.info(f"Process complete. Saved {total_plans_generated} total dialogue plans to {args.output_path}")


if __name__ == "__main__":
    main()
