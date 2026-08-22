#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import os
from collections import defaultdict
from statistics import mean

from scripts.compute_true_multi_step_stats import get_true_multi_step_stats


def count_user_turns(conversations: list[dict]) -> int:
    """Count the number of user turns in a conversation."""
    return sum(1 for msg in conversations if msg.get("role") == "user")


def count_assistant_clarifications(plan: list[dict]) -> int:
    """Count the number of assistant clarifications in a plan."""
    return sum(1 for step in plan if step.get("type") == "ASSISTANT_CLARIFICATION")


def count_tool_calls(plan: list[dict] = None, conversations: list[dict] = None) -> int:
    """Count the number of tool calls in a plan or conversations."""
    if plan:
        return sum(1 for step in plan if step.get("type") == "CALL_TOOL")
    elif conversations:
        return sum(
            1 for msg in conversations
            if msg.get("role") == "assistant_tool_call"
        )
    return 0


def analyze_multi_step_tool_calls(conversations: list[dict]) -> dict:
    """Extract all true multi-step sequences from a dialogue sample.

    Args:
        sample: The dialogue sample containing plan and conversations

    Returns:
        List of sequences that are true multi-step (tool outputs used as inputs)
    """
    result = {
        "count": 0,
        "multi_step_sequences": []  # Lengths of each multi-step sequence
    }

    user_indices = [i for i, msg in enumerate(conversations) if msg.get("role") == "user"]
    user_indices.append(len(conversations))  # Add end of list as a boundary

    tool_count_distribution = defaultdict(int)

    for i in range(len(user_indices) - 1):
        start_idx = user_indices[i]
        end_idx = user_indices[i+1]

        tool_calls = sum(
            1 for j in range(start_idx + 1, end_idx)
            if conversations[j].get("role") == "assistant_tool_call"
        )

        if tool_calls > 1:
            result["count"] += 1
            result["multi_step_sequences"].append(tool_calls)
            tool_count_distribution[tool_calls] += 1

    result["tool_count_distribution"] = {str(k): v for k, v in tool_count_distribution.items()}
    return result


def compute_statistics(samples: list[dict]) -> dict:
    """Compute comprehensive statistics from dialogue samples.

    Calculates average, min, and max values for turns, clarifications, tool calls,
    and multi-step sequences, along with dialogue IDs for extremes.

    Args:
        samples: List of dialogue samples to analyze

    Returns:
        Dictionary containing computed statistics across all metrics
    """
    if not samples:
        return {"error": "No samples found"}

    stats = {
        "total_samples": len(samples),
        "turns": {
            "avg": 0,
            "min": float('inf'),
            "max": 0,
            "min_dialogue_ids": [],
            "max_dialogue_ids": []
        },
        "clarifications": {
            "avg": 0,
            "min": float('inf'),
            "max": 0,
            "min_dialogue_ids": [],
            "max_dialogue_ids": []
        },
        "tool_calls": {
            "avg": 0,
            "min": float('inf'),
            "max": 0,
            "min_dialogue_ids": [],
            "max_dialogue_ids": []
        },
        "multi_step_tool_calls": {
            "samples_with_multi_step": 0,
            "true_multi_step_sequences": 0,
            "total_multi_step_sequences": 0,
            "avg": 0,
            "min": float('inf'),
            "max": 0,
            "min_dialogue_ids": [],
            "max_dialogue_ids": [],
            "tool_count_distribution": {}
        }
    }

    turns_data = []
    clarifications_data = []
    tool_calls_data = []
    multi_step_data = []
    tool_count_distribution = defaultdict(int)

    for sample in samples:
        dialogue_id = sample.get("dialogue_id", "unknown")

        turns = count_user_turns(sample.get("conversations", []))
        turns_data.append(turns)

        if turns < stats["turns"]["min"]:
            stats["turns"]["min"] = turns
            stats["turns"]["min_dialogue_ids"] = [dialogue_id]
        elif turns == stats["turns"]["min"]:
            stats["turns"]["min_dialogue_ids"].append(dialogue_id)

        if turns > stats["turns"]["max"]:
            stats["turns"]["max"] = turns
            stats["turns"]["max_dialogue_ids"] = [dialogue_id]
        elif turns == stats["turns"]["max"]:
            stats["turns"]["max_dialogue_ids"].append(dialogue_id)

        clarifications = count_assistant_clarifications(sample.get("plan", []))
        clarifications_data.append(clarifications)

        if clarifications < stats["clarifications"]["min"]:
            stats["clarifications"]["min"] = clarifications
            stats["clarifications"]["min_dialogue_ids"] = [dialogue_id]
        elif clarifications == stats["clarifications"]["min"]:
            stats["clarifications"]["min_dialogue_ids"].append(dialogue_id)

        if clarifications > stats["clarifications"]["max"]:
            stats["clarifications"]["max"] = clarifications
            stats["clarifications"]["max_dialogue_ids"] = [dialogue_id]
        elif clarifications == stats["clarifications"]["max"]:
            stats["clarifications"]["max_dialogue_ids"].append(dialogue_id)

        tool_calls = count_tool_calls(conversations=sample.get("conversations", []))
        tool_calls_data.append(tool_calls)

        if tool_calls < stats["tool_calls"]["min"]:
            stats["tool_calls"]["min"] = tool_calls
            stats["tool_calls"]["min_dialogue_ids"] = [dialogue_id]
        elif tool_calls == stats["tool_calls"]["min"]:
            stats["tool_calls"]["min_dialogue_ids"].append(dialogue_id)

        if tool_calls > stats["tool_calls"]["max"]:
            stats["tool_calls"]["max"] = tool_calls
            stats["tool_calls"]["max_dialogue_ids"] = [dialogue_id]
        elif tool_calls == stats["tool_calls"]["max"]:
            stats["tool_calls"]["max_dialogue_ids"].append(dialogue_id)

        multi_step_result = analyze_multi_step_tool_calls(sample.get("conversations", []))

        if multi_step_result["count"] > 0:
            stats["multi_step_tool_calls"]["samples_with_multi_step"] += 1

            for sequence_length, count in multi_step_result["tool_count_distribution"].items():
                tool_count_distribution[sequence_length] += count

            num_sequences = len(multi_step_result["multi_step_sequences"])
            multi_step_data.append(num_sequences)

            if num_sequences < stats["multi_step_tool_calls"]["min"]:
                stats["multi_step_tool_calls"]["min"] = num_sequences
                stats["multi_step_tool_calls"]["min_dialogue_ids"] = [dialogue_id]
            elif num_sequences == stats["multi_step_tool_calls"]["min"]:
                stats["multi_step_tool_calls"]["min_dialogue_ids"].append(dialogue_id)

            if num_sequences > stats["multi_step_tool_calls"]["max"]:
                stats["multi_step_tool_calls"]["max"] = num_sequences
                stats["multi_step_tool_calls"]["max_dialogue_ids"] = [dialogue_id]
            elif num_sequences == stats["multi_step_tool_calls"]["max"]:
                stats["multi_step_tool_calls"]["max_dialogue_ids"].append(dialogue_id)

            try:
                stats["multi_step_tool_calls"]["true_multi_step_sequences"] += get_true_multi_step_stats(sample)[0]
            except Exception as e:
                print(f"Error computing true multi-step stats for dialogue {dialogue_id}: {str(e)}")
                raise e

            stats["multi_step_tool_calls"]["total_multi_step_sequences"] += multi_step_result["count"]

    stats["turns"]["avg"] = mean(turns_data) if turns_data else 0
    stats["clarifications"]["avg"] = mean(clarifications_data) if clarifications_data else 0
    stats["tool_calls"]["avg"] = mean(tool_calls_data) if tool_calls_data else 0
    stats["multi_step_tool_calls"]["avg"] = mean(multi_step_data) if multi_step_data else 0

    stats["multi_step_tool_calls"]["tool_count_distribution"] = \
        {str(k): v for k, v in tool_count_distribution.items()}

    if stats["turns"]["min"] == float('inf'):
        stats["turns"]["min"] = 0
    if stats["clarifications"]["min"] == float('inf'):
        stats["clarifications"]["min"] = 0
    if stats["tool_calls"]["min"] == float('inf'):
        stats["tool_calls"]["min"] = 0
    if stats["multi_step_tool_calls"]["min"] == float('inf'):
        stats["multi_step_tool_calls"]["min"] = 0
        stats["multi_step_tool_calls"]["min_dialogue_ids"] = []

    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute statistics on dialogue samples across domains.')
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue JSONL files')
    parser.add_argument('--output_file', required=True, help='Output JSONL file for statistics')

    args = parser.parse_args()

    jsonl_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])

    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return

    all_samples = []
    domain_stats = {}

    for filename in jsonl_files:
        filepath = os.path.join(args.input_dir, filename)
        domain_name = os.path.splitext(filename)[0]
        samples = []

        try:
            with open(filepath, 'r') as file:
                for line in file:
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)

                        overall_sample = sample.copy()
                        overall_sample['dialogue_id'] = \
                            f"{domain_name}_{sample.get('dialogue_id', 'unknown')}"
                        all_samples.append(overall_sample)  # Collect for overall stats

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON line in {filepath}")
                        continue

            domain_stats[domain_name] = compute_statistics(samples)
            print(f"Processed {len(samples)} samples from {domain_name}")

        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")

    overall_stats = compute_statistics(all_samples)

    with open(args.output_file, 'w') as outfile:
        for domain, stats in domain_stats.items():
            result = {
                "domain": domain,
                "stats": stats
            }
            outfile.write(json.dumps(result) + '\n')

        result = {
            "domain": "overall",
            "stats": overall_stats
        }
        outfile.write(json.dumps(result) + '\n')

    print(f"Statistics written to {args.output_file}")


if __name__ == "__main__":
    main()
