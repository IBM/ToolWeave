#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import os
import traceback

from scripts.compute_true_multi_step_stats import (
    get_multi_step_sequences,
    get_true_multi_step_sequences,
)


def group_tool_calls_and_executions(sample: dict) -> tuple[dict, bool]:
    """Convert false multi-step sequences by grouping tool calls and executions together.

    Identifies false multi-step sequences and reorders conversations so that all
    assistant tool calls are grouped together followed by their corresponding tool
    executions.

    Args:
        sample: The dialogue sample to modify

    Returns:
        Tuple of modified sample with grouped tool calls/executions and boolean
        indicating if any modifications were made
    """
    multi_step_sequences = get_multi_step_sequences(sample.get("plan", []))
    true_sequences = get_true_multi_step_sequences(sample)

    false_sequences = [seq for seq in multi_step_sequences if seq not in true_sequences]

    if not false_sequences:
        return sample, False

    sample["dialogue_id"] += "_parallel"

    false_sequence_tool_names = set()
    for sequence in false_sequences:
        for tool in sequence:
            false_sequence_tool_names.add(tool["tool_name"])

    new_conversations = []
    i = 0
    while i < len(sample["conversations"]):
        convo = sample["conversations"][i]

        if convo.get("role") == "assistant_tool_call":
            tool_call = json.loads(convo.get("content"))[0]
            tool_name = tool_call["name"]

            if tool_name in false_sequence_tool_names:
                assistant_calls = []
                tool_executions = []

                j = i
                while j < len(sample["conversations"]):
                    curr_convo = sample["conversations"][j]
                    
                    if curr_convo.get("role") == "assistant_tool_call":
                        curr_tool_call = json.loads(curr_convo.get("content"))[0]
                        if curr_tool_call["name"] not in false_sequence_tool_names:
                            break

                        assistant_calls.append(curr_convo)
                        j += 1

                    elif curr_convo.get("role") == "tool":
                        tool_executions.append(curr_convo)
                        j += 1

                    else:
                        break

                new_conversations.extend(assistant_calls)
                new_conversations.extend(tool_executions)
                i = j

            else:
                new_conversations.append(convo)
                i += 1

        else:
            new_conversations.append(convo)
            i += 1

    sample["conversations"] = new_conversations

    return sample, True


def process_jsonl_file(filepath: str) -> int:
    """Process a single JSONL file in-place and convert false multi-step sequences.

    Args:
        filepath: Path to JSONL file to modify in-place

    Returns:
        Number of samples modified in the file
    """
    modified_lines = []
    num_modified = 0

    with open(filepath, 'r') as infile:
        for line_num, line in enumerate(infile, 1):
            try:
                sample = json.loads(line.strip())

                if 'plan' not in sample or 'conversations' not in sample:
                    print(f"Skipping line {line_num} in {filepath}: missing 'plan' or 'conversations'")
                    modified_lines.append(line)
                    continue

                modified_sample, modified = group_tool_calls_and_executions(sample)
                modified_lines.append(json.dumps(modified_sample) + '\n')
                num_modified += int(modified)

            except json.JSONDecodeError:
                print(f"Error decoding JSON line {line_num} in {filepath}")
                modified_lines.append(line)

            except Exception as e:
                print(f"Error processing line {line_num} in {filepath}: {str(e)}")
                traceback.print_exc()
                modified_lines.append(line)

    with open(filepath, 'w') as outfile:
        outfile.writelines(modified_lines)

    return num_modified


def main():
    parser = argparse.ArgumentParser(
        description='Convert false multi-step sequences by grouping tool calls and executions together (modifies files in-place).'
    )
    parser.add_argument('--input_dir', required=True, help='Directory containing JSONL files to modify in-place')

    args = parser.parse_args()

    jsonl_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to process")
    print("WARNING: Files will be modified in-place!")

    for jsonl_file in jsonl_files:
        filepath = os.path.join(args.input_dir, jsonl_file)

        try:
            num_false_multi_step = process_jsonl_file(filepath)
            print(f"Converted {num_false_multi_step} false multi-step samples in {jsonl_file}")

        except Exception as e:
            print(f"Error processing {jsonl_file}: {str(e)}")
            traceback.print_exc()

    print(f"\nAll files processed in {args.input_dir}")


if __name__ == "__main__":
    main()
