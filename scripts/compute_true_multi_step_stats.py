#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import os
import traceback
from typing import Any


def build_tool_io_maps(conversations: list[dict]) -> tuple[dict, dict]:
    """Build mappings of tool calls and their outputs from conversation history.

    Args:
        conversations: List of conversation turns containing tool calls and executions

    Returns:
        Tuple of tool call map (name to arguments) and tool output map (name to outputs)
    """
    tool_call_map = {}
    tool_output_map = {}

    for i, convo_step in enumerate(conversations):
        if convo_step.get("role") == "assistant_tool_call":
            tool_call = json.loads(convo_step.get("content"))[0]
            tool_call_map[tool_call["name"]] = tool_call["arguments"]
        elif convo_step.get("role") == "tool":
            prev_step = None
            for prev_convo_step in conversations[:i][::-1]:
                if prev_convo_step.get("role") == "user":
                    break
                if prev_convo_step.get("role") == "assistant_tool_call":
                    prev_step = prev_convo_step
                    break

            if prev_step and prev_step.get("role") == "assistant_tool_call":
                tool_call = json.loads(prev_step.get("content"))[0]

                tool_output = json.loads(convo_step.get("content"))
                tool_output_map[tool_call["name"]] = tool_output

    return tool_call_map, tool_output_map


def get_multi_step_sequences(plan: list[dict]) -> list[list[dict]]:
    """Extract all multi-step tool call sequences from a plan.

    Args:
        plan: List of plan steps containing tool calls and other actions

    Returns:
        List of sequences where each sequence contains consecutive tool call steps
    """
    multi_step_sequences = []
    current_sequence = []

    for step in plan:
        if step.get("type") == "CALL_TOOL":
            current_sequence.append(step)

        else:
            if len(current_sequence) > 1:  # We need at least 2 tool calls for a multi-step sequence
                multi_step_sequences.append(current_sequence.copy())
            current_sequence = []

    if len(current_sequence) > 1:
        multi_step_sequences.append(current_sequence.copy())

    return multi_step_sequences


def resolve_param_value(param_name: str, curr_tool_data: dict | list) -> any:
    """Resolve a parameter value by traversing nested structures using dot notation.

    Handles dot notation for nested access and array notation for iterating over arrays.

    Args:
        param_name: Flattened parameter name with dots for nested access
        curr_tool_data: Dictionary or list containing tool data to traverse

    Returns:
        The resolved value(s) or None if not found
    """
    if isinstance(curr_tool_data, list):
        results = []
        for item in curr_tool_data:
            # Recursively call on each item in the list with the same parameter name
            resolved = resolve_param_value(param_name, item)
            if resolved is not None:
                if isinstance(resolved, list):
                    results.extend(resolved)
                else:
                    results.append(resolved)
        return results if results else None
        # ---> END NEW BLOCK <---

    if not curr_tool_data or not isinstance(curr_tool_data, dict) or not param_name:
        return None

    parts = param_name.split('.', 1)
    current_key = parts[0]
    remaining_path = parts[1] if len(parts) > 1 else None

    is_array = False
    if current_key.endswith('[]'):
        is_array = True
        current_key = current_key[:-2]  # Remove the [] suffix

    current_value = curr_tool_data.get(current_key)
    if current_value is None:
        return None

    if not remaining_path:
        return current_value

    if is_array and isinstance(current_value, list):
        results = []
        for item in current_value:
            if isinstance(item, dict):
                resolved = resolve_param_value(remaining_path, item)
                if resolved is not None:
                    if isinstance(resolved, list):
                        results.extend(resolved)
                    else:
                        results.append(resolved)

        return results if results else None

    elif isinstance(current_value, dict):
        return resolve_param_value(remaining_path, current_value)

    return None


def value_exists_in_dict(target_value: Any, data: dict) -> bool:
    """Recursively checks if a value exists within a dictionary or its nested structures.

    Args:
        target_value: The value to search for
        data: Dictionary, list, or primitive value to search within

    Returns:
        True if the value is found, False otherwise
    """
    if data == target_value:
        return True

    if isinstance(data, dict):
        for value in data.values():
            if value_exists_in_dict(target_value, value):
                return True

    elif isinstance(data, list):
        for item in data:
            if value_exists_in_dict(target_value, item):
                return True

    return False


def is_parameter_from_previous_tool(
    param_value: str, param_name: str, curr_tool_name: str, prev_tool_names: list[str],
    tool_call_map: dict[str, dict], tool_output_map: dict[str, dict],
    all_tool_names: list[str], sequence_tool_names: list[str],
) -> bool:
    """Check if a parameter value comes from a previous tool's output.

    Args:
        param_value: The parameter value to check
        param_name: The parameter name
        curr_tool_name: Current tool name
        prev_tool_names: List of previous tool names in the sequence
        tool_call_map: Map of tool calls
        tool_output_map: Map of tool outputs
        all_tool_names: List of all tool names
        sequence_tool_names: List of tool names in the current sequence

    Returns:
        True if the parameter is from a previous tool output, False otherwise
    """
    if not param_value.startswith("$user_provided"):
        param_name = param_name[len(curr_tool_name) + 1:]  # Remove tool name prefix

        extracted_param_values = resolve_param_value(param_name, tool_call_map.get(curr_tool_name, {}))
        if not extracted_param_values:
            return False

        if not isinstance(extracted_param_values, list):
            extracted_param_values = [extracted_param_values]

        for extracted_value in extracted_param_values:
            # Skip if value exists in previous tool calls
            if any(
                value_exists_in_dict(extracted_value, tool_call_map.get(prev_tool_name, {}))
                for prev_tool_name in all_tool_names[:all_tool_names.index(curr_tool_name)]
            ):
                continue

            # Skip if value exists in outputs before the sequence
            if any(
                value_exists_in_dict(extracted_value, tool_output_map.get(prev_tool_name, {}))
                for prev_tool_name in all_tool_names[:all_tool_names.index(sequence_tool_names[0])]
            ):
                continue

            # Check if the value actually comes from a previous tool in this sequence
            prev_tool_name = param_value.split(".", 1)[0][1:]
            if prev_tool_name in prev_tool_names:
                prev_tool_output_param_name = param_value.split(".", 1)[1]
                prev_tool_outputs = resolve_param_value(
                    prev_tool_output_param_name,
                    tool_output_map.get(prev_tool_name, {})
                )

                if not prev_tool_outputs:
                    continue

                if not isinstance(prev_tool_outputs, list):
                    prev_tool_outputs = [prev_tool_outputs]

                if any(
                    prev_tool_output == extracted_value
                    for prev_tool_output in prev_tool_outputs
                ):
                    return True

    return False


def is_sequence_true_multi_step(
    sequence: list[dict], sequence_tool_names: list[str],
    tool_call_map: dict[str, dict], tool_output_map: dict[str, dict],
    all_tool_names: list[str],
) -> bool:
    """Determine if a sequence is a true multi-step sequence (where output from one tool is used as input to another).

    Args:
        sequence: The sequence of tool calls
        sequence_tool_names: List of tool names in the sequence
        tool_call_map: Map of tool calls
        tool_output_map: Map of tool outputs
        all_tool_names: List of all tool names

    Returns:
        True if it's a true multi-step sequence, False otherwise
    """
    for i in range(1, len(sequence)):
        curr_tool_name = sequence_tool_names[i]
        prev_tool_names = sequence_tool_names[:i]
        target_tool_params = sequence[i]["parameters"]

        for param_name, param_value in target_tool_params.items():
            if is_parameter_from_previous_tool(
                param_value, param_name, curr_tool_name, prev_tool_names,
                tool_call_map, tool_output_map, all_tool_names, sequence_tool_names
            ):
                return True

    return False

def get_true_multi_step_stats(sample: dict) -> tuple[int, int]:
    """Compute statistics on true multi-step sequences from a dialogue sample.

    Args:
        sample: The dialogue sample

    Returns:
        A tuple containing the number of true multi-step sequences and the total number of multi-step sequences
    """
    tool_call_map, tool_output_map = build_tool_io_maps(sample.get("conversations", []))
    all_tool_names = list(tool_call_map.keys())

    multi_step_sequences = get_multi_step_sequences(sample.get("plan", []))

    num_true_multi_step = 0

    for sequence in multi_step_sequences:
        sequence_tool_names = [tool["tool_name"] for tool in sequence]

        if is_sequence_true_multi_step(sequence, sequence_tool_names, tool_call_map, tool_output_map, all_tool_names):
            num_true_multi_step += 1

    return num_true_multi_step, len(multi_step_sequences)


def get_true_multi_step_sequences(sample: dict) -> list[list[dict]]:
    """Extract all true multi-step sequences from a dialogue sample.

    Args:
        sample: The dialogue sample containing plan and conversations

    Returns:
        List of sequences that are true multi-step (tool outputs used as inputs)
    """
    tool_call_map, tool_output_map = build_tool_io_maps(sample.get("conversations", []))
    all_tool_names = list(tool_call_map.keys())
    multi_step_sequences = get_multi_step_sequences(sample.get("plan", []))

    true_sequences = []
    for sequence in multi_step_sequences:
        sequence_tool_names = [tool["tool_name"] for tool in sequence]
        if is_sequence_true_multi_step(sequence, sequence_tool_names, tool_call_map, tool_output_map, all_tool_names):
            true_sequences.append(sequence)  # Appends the sequence instead of counting

    return true_sequences


def main():
    parser = argparse.ArgumentParser(description='Compute statistics on dialogue samples across domains.')
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue JSONL files')
    parser.add_argument('--output_file', required=True, help='Output JSON file for true multi-step statistics')

    args = parser.parse_args()

    jsonl_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    domain_stats = {}

    with open(args.output_file, 'w') as outfile:
        for dialogue_file in jsonl_files:
            dialogue_filepath = os.path.join(args.input_dir, dialogue_file)
            domain_name = os.path.splitext(dialogue_file)[0]

            try:
                domain_stats[domain_name] = {
                    "true_multi_step": 0,
                    "total_multi_step": 0
                }
                with open(dialogue_filepath, 'r') as file:
                    for line in file:
                        try:
                            sample = json.loads(line.strip())

                        except json.JSONDecodeError:
                            print(f"Error decoding JSON line in {dialogue_filepath}")
                            continue

                        if 'plan' not in sample:
                            print(f"No plan found in sample from {dialogue_filepath}")
                            continue

                        num_true_multi_step, num_multi_step = get_true_multi_step_stats(sample)
                        domain_stats[domain_name]['true_multi_step'] += num_true_multi_step
                        domain_stats[domain_name]['total_multi_step'] += num_multi_step

            except Exception as e:
                print(f"Error processing {dialogue_filepath}: {str(e)}")
                traceback.print_exc()

        domain_stats['overall'] = {
            "true_multi_step": sum(v['true_multi_step'] for v in domain_stats.values()),
            "total_multi_step": sum(v['total_multi_step'] for v in domain_stats.values())
        }

        json.dump(domain_stats, outfile, indent=2)

    print(f"Statistics written to {args.output_file}")


if __name__ == "__main__":
    main()
