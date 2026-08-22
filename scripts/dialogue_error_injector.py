#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import copy
import json
import os
import random

from difflib import SequenceMatcher
from natsort import natsorted
from sentence_transformers import SentenceTransformer, util

from scripts.compute_true_multi_step_stats import get_true_multi_step_sequences


class HybridSimilarityMatcher:
    """Matcher that combines semantic and lexical similarity for text comparison.

    Uses a Sentence Transformer model for semantic embeddings and SequenceMatcher
    for lexical similarity to determine if two texts are similar

    Attributes:
        model: SentenceTransformer model for semantic encoding
        device: Device to run the model on (cpu or cuda)
    """
    def __init__(self, semantic_model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu') -> None:
        """Initialize the matcher with a Sentence Transformer model.

        Args:
            semantic_model_name: Name of the SentenceTransformer model to use
            device: Device to run the model on (cpu or cuda)
        """
        self.model = SentenceTransformer(semantic_model_name, device=device)
        self.device = device


    def are_similar(
        self, text1: str, text2: str,
        semantic_threshold: float = 0.7, lexical_threshold: float = 0.6
    ) -> bool:
        """Check if two texts are similar based on both semantic and lexical scores.

        Args:
            text1: First text to compare
            text2: Second text to compare
            semantic_threshold: Minimum cosine similarity score for semantic match
            lexical_threshold: Minimum SequenceMatcher ratio for lexical match

        Returns:
            True if both semantic and lexical thresholds are met, False otherwise
        """
        if not text1 or not text2:
            return False

        # 1. Semantic Check
        embeddings = self.model.encode([text1, text2], convert_to_tensor=True, device=self.device)
        semantic_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        if semantic_score < semantic_threshold:
            return False

        # 2. Lexical Check
        lexical_score = SequenceMatcher(None, text1, text2).ratio()
        if lexical_score < lexical_threshold:
            return False

        return True


def inject_out_of_order_recovery_error(
    dialogue_data: dict, tool_schemas: list[dict],
    matcher: HybridSimilarityMatcher = None
) -> tuple[dict, bool, list[str]]:
    """Inject an out-of-order error where a dependent tool call is made prematurely.

    Selects a multi-step sequence, makes a premature call to a dependent tool with
    missing parameters, receives an error, then executes the prerequisite tool call
    followed by a retry of the originally failed call.

    Args:
        dialogue_data: Dialogue dictionary containing plan, conversations, and tools
        tool_schemas: List of tool schema definitions
        matcher: Optional similarity matcher (unused in this function)

    Returns:
        Tuple of modified dialogue, success flag, and list of applied error types
    """
    plan = dialogue_data.get("plan", [])
    conversations = dialogue_data.get("conversations", [])

    true_multi_step_sequences = get_true_multi_step_sequences(dialogue_data)
    valid_sequences = [seq for seq in true_multi_step_sequences if len(seq) >= 2]
    if not valid_sequences:
        return dialogue_data, False, []

    sequence = random.choice(valid_sequences)
    j = random.randint(1, len(sequence) - 1)
    step_A, step_B = sequence[j - 1], sequence[j]
    tool_A_name, tool_B_name = step_A["tool_name"], step_B["tool_name"]

    try:
        dependent_param_key = None
        for key, value in step_B.get("parameters", {}).items():
            if isinstance(value, str) and value.startswith(f"${tool_A_name}."):
                dependent_param_key = key.split('.')[-1]
                break
        if not dependent_param_key:
            return dialogue_data, False, []  # Dependency not found, abort

        plan_idx_A = next(i for i, s in enumerate(plan) if s == step_A)
        plan_idx_B = next(i for i, s in enumerate(plan) if s == step_B)

        plan_tool_indices = {
            idx: i for i, idx in enumerate([
                p_idx for p_idx, s in enumerate(plan) if s.get("type") == "CALL_TOOL"
            ])
        }
        conv_tool_indices = [
            c_idx for c_idx, turn in enumerate(conversations) \
                if turn.get("role") == "assistant_tool_call"
        ]
        conv_idx_A = conv_tool_indices[plan_tool_indices[plan_idx_A]]
        conv_idx_B = conv_tool_indices[plan_tool_indices[plan_idx_B]]

        premature_call_turn = conversations[conv_idx_B].copy()
        call_content = json.loads(premature_call_turn['content'])

        # Remove the key that we don't have the value for yet
        if dependent_param_key in call_content[0]['arguments']:
            del call_content[0]['arguments'][dependent_param_key]

        premature_call_turn['content'] = json.dumps(call_content)
        premature_call_turn['role'] = 'ignored_assistant_tool_call'

        # Get original turns for recovery and retry
        recovery_call_turn = conversations[conv_idx_A]
        recovery_tool_response = conversations[conv_idx_A + 1]
        final_retry_turn = conversations[conv_idx_B]
        final_retry_response = conversations[conv_idx_B + 1]

        error_message = f"Missing required parameter: '{dependent_param_key}'"
        error_prefix = random.choice(["Error", "error", "ERROR"])
        error_response = {
            "function_name": tool_B_name, "status": "error",
            "error_type": error_prefix, "error_message": error_message
        }
        error_turn = {"role": "tool", "content": json.dumps(error_response)}

        # Re-assemble conversation with full recovery
        new_conversations = conversations[:conv_idx_A]
        new_conversations.extend([
            premature_call_turn, error_turn,
            recovery_call_turn, recovery_tool_response,
            final_retry_turn, final_retry_response
        ])

        final_assistant_response_idx = next(
            (i for i in range(len(conversations) - 1, -1, -1) \
                if conversations[i].get("role") == "assistant"
            ),
            -1
        )
        if final_assistant_response_idx != -1:
            new_conversations.append(conversations[final_assistant_response_idx])

        dialogue_data["conversations"] = new_conversations

        # Update the plan (this part remains largely the same)
        new_plan = plan[:plan_idx_A]
        original_step_num = step_A['step']

        new_plan.append({'step': original_step_num, 'type': 'ASSISTANT_OUT_OF_ORDER_CALL', 'tool_name': tool_B_name})
        new_plan.append({'step': original_step_num + 1, 'type': 'TOOL_MISSING_INPUT_ERROR'})

        step_A['step'] = original_step_num + 2
        step_B['step'] = original_step_num + 3
        new_plan.extend([step_A, step_B])

        original_end_idx = max(plan_idx_A, plan_idx_B) + 1
        for i in range(original_end_idx, len(plan)):
            step = plan[i]
            step['step'] += 2
            new_plan.append(step)

        dialogue_data["plan"] = new_plan
        return dialogue_data, True, ["out_of_order_errors"]

    except Exception:
        return dialogue_data, False, []


def inject_cascading_failure_error(
    dialogue_data: dict, tool_schemas: list[dict],
    matcher: HybridSimilarityMatcher = None,
) -> tuple[dict, bool, list[str]]:
    """Inject cascading failures by attempting a multi-step sequence in reverse order.

    Targets sequences of 3+ steps and attempts to execute them in reverse order,
    generating errors for each premature call before executing the correct sequence.

    Args:
        dialogue_data: Dialogue dictionary containing plan, conversations, and tools
        tool_schemas: List of tool schema definitions
        matcher: Optional similarity matcher (unused in this function)

    Returns:
        Tuple of modified dialogue, success flag, and list of applied error types
    """
    plan = dialogue_data.get("plan", [])
    conversations = dialogue_data.get("conversations", [])

    true_multi_step_sequences = get_true_multi_step_sequences(dialogue_data)
    # Target longer sequences for a more dramatic failure chain
    long_sequences = [seq for seq in true_multi_step_sequences if len(seq) >= 3]
    if not long_sequences:
        return dialogue_data, False, []

    sequence = random.choice(long_sequences)

    try:
        # Get plan and conversation indices for all steps in the sequence
        plan_indices = [next(i for i, s in enumerate(plan) if s == step) for step in sequence]

        plan_to_conv_map = {
            p_idx: c_idx for p_idx, c_idx in zip(
                [p[0] for p in enumerate(plan) if p[1].get("type") == "CALL_TOOL"],
                [c[0] for c in enumerate(conversations) if c[1].get("role") == "assistant_tool_call"]
            )
        }
        conv_indices = [plan_to_conv_map[pi] for pi in plan_indices]

        # --- Build the new conversation turns ---
        new_conv_turns = []
        # 1. Add premature calls in reverse order, each resulting in an error
        for i in range(len(sequence) - 1, 0, -1):
            premature_call_step = sequence[i]
            prerequisite_step = sequence[i - 1]
            premature_conv_idx = conv_indices[i]

            premature_call_turn = conversations[premature_conv_idx].copy()
            # Modify call to remove dependency
            call_content = json.loads(premature_call_turn['content'])
            dependent_param_key = next(
                (
                    k.split('.')[-1] for k, v in premature_call_step.get("parameters", {}).items() \
                        if isinstance(v, str) and v.startswith(f"${prerequisite_step['tool_name']}.")
                ),
                None
            )
            if dependent_param_key and dependent_param_key in call_content[0]['arguments']:
                del call_content[0]['arguments'][dependent_param_key]
            premature_call_turn['content'] = json.dumps(call_content)
            premature_call_turn['role'] = 'ignored_assistant_tool_call'

            error_messages = [
                "Missing prerequisite data.",
                "A required input is missing.",
                "Prerequisite step not completed. Cannot proceed.",
                "Dependency error: required input not found."
            ]
            error_prefix = random.choice(["Error", "error", "ERROR"])

            error_response = {
                "function_name": premature_call_step['tool_name'],
                "status": "error",
                "error_type": error_prefix,
                "error_message": random.choice(error_messages)
            }
            error_turn = {"role": "tool", "content": json.dumps(error_response)}

            new_conv_turns.extend([premature_call_turn, error_turn])

        # 2. Add the full, correct recovery sequence
        for i in range(len(sequence)):
            correct_conv_idx = conv_indices[i]
            new_conv_turns.extend(conversations[correct_conv_idx: correct_conv_idx + 2])

        # 3. Assemble the final conversation
        start_conv_idx = conv_indices[0]
        final_summary_idx = next(
            (
                i for i in range(len(conversations) - 1, -1, -1) \
                    if conversations[i]['role'] == 'assistant'
            ),
            -1
        )

        final_conversations = conversations[:start_conv_idx] + new_conv_turns
        if final_summary_idx != -1:
            final_conversations.append(conversations[final_summary_idx])

        dialogue_data['conversations'] = final_conversations

        # --- Rebuild the Plan ---
        start_plan_idx = plan_indices[0]
        new_plan_segment = []
        step_num = plan[start_plan_idx]['step']

        for i in range(len(sequence) - 1, 0, -1):
            new_plan_segment.append({
                'step': 0,
                'type': 'ASSISTANT_OUT_OF_ORDER_CALL',
                'tool_name': sequence[i]['tool_name']
            })
            new_plan_segment.append({'step': 0, 'type': 'TOOL_MISSING_INPUT_ERROR'})

        new_plan_segment.extend(copy.deepcopy(sequence))

        # Append the rest of the plan and re-number everything
        end_of_original_sequence_idx = plan_indices[-1] + 1

        final_plan = plan[:start_plan_idx] + new_plan_segment + plan[end_of_original_sequence_idx:]
        for i in range(start_plan_idx, len(final_plan)):
            if 'step' in final_plan[i]:
                final_plan[i]['step'] = step_num + (i - start_plan_idx)

        dialogue_data['plan'] = final_plan
        return dialogue_data, True, ["cascading_failure_errors"]

    except Exception:
        return dialogue_data, False, []


def compose_errors(
    dialogue_data: dict, tool_schemas: list[dict],
    matcher: HybridSimilarityMatcher,
) -> tuple[dict, bool, list[str]]:
    """Apply a sequence of different error injectors to create layered errors.

    Attempts a logical error first (cascading, out-of-order, or wrong tool), then
    optionally layers a parameter error on top with 50% probability.

    Args:
        dialogue_data: Dialogue dictionary containing plan, conversations, and tools
        tool_schemas: List of tool schema definitions
        matcher: Similarity matcher for finding confusable tool names

    Returns:
        Tuple of modified dialogue, success flag, and list of applied error types
    """
    applied_errors = []

    # 1. Attempt a major logical error first
    logical_injectors = [
        inject_cascading_failure_error,
        inject_out_of_order_recovery_error,
        inject_wrong_tool_self_correct_error
    ]
    random.shuffle(logical_injectors)

    modified_dialogue, modified, logical_errors_applied = logical_injectors[0](dialogue_data, tool_schemas, matcher)
    if modified:
        applied_errors.extend(logical_errors_applied)

    # 2. If a logical error was injected, 50% chance to layer a parameter error on top
    if modified and random.random() < 0.5:
        final_dialogue, final_modified, param_errors_applied = \
            inject_schema_based_error(modified_dialogue, tool_schemas, matcher)

        if final_modified:
            applied_errors.extend(param_errors_applied)
            return final_dialogue, True, applied_errors

    return modified_dialogue, modified, applied_errors


# --- INJECTION AND PROCESSING LOGIC ---

def parse_tool_schemas(tools_list: list[dict]) -> dict[str, dict]:
    """Parse tool definitions into schema dictionaries.

    Args:
        tools_list: List of tool definitions containing function schemas

    Returns:
        Dictionary mapping tool names to their parameter schemas
    """
    schemas = {}
    for tool_definition in tools_list:
        if tool_definition.get("type") == "function":
            func = tool_definition.get("function", {})
            tool_name, params = func.get("name"), func.get("parameters", {})
            if tool_name and params:
                schemas[tool_name] = {
                    "properties": params.get("properties", {}),
                    "required": params.get("required", [])
                }

    return schemas


def inject_wrong_tool_self_correct_error(
    dialogue_data: dict, tool_schemas: dict[str, dict],
    matcher: HybridSimilarityMatcher,
) -> tuple[dict, bool, list[str]]:
    """Inject an error where a similar but incorrect tool is called before correction.

    Uses semantic and lexical similarity to identify confusable tool names, makes an
    initial call to a wrong but similar tool, then proceeds with the correct tool call.

    Args:
        dialogue_data: Dialogue dictionary containing plan, conversations, and tools
        tool_schemas: Dictionary mapping tool names to their parameter schemas
        matcher: Similarity matcher for finding confusable tool names

    Returns:
        Tuple of modified dialogue, success flag, and list of applied error types
    """
    conversations = dialogue_data.get("conversations", [])
    plan = dialogue_data.get("plan", [])
    tool_names = list(tool_schemas.keys())

    plan_tool_call_indices = [i for i, step in enumerate(plan) if step.get("type") == "CALL_TOOL"]
    conv_tool_call_indices = [i for i, turn in enumerate(conversations) if turn.get("role") == "assistant_tool_call"]

    if len(plan_tool_call_indices) != len(conv_tool_call_indices):
        return dialogue_data, False, []

    possible_injection_points = []
    for i, plan_idx in enumerate(plan_tool_call_indices):
        correct_tool_name = plan[plan_idx].get("tool_name")
        # Find other tools that are similar using the hybrid matcher
        confusable_alternatives = [
            other for other in tool_names
            if correct_tool_name != other and matcher.are_similar(correct_tool_name, other)
        ]
        if confusable_alternatives:
            possible_injection_points.append({
                "plan_idx": plan_idx,
                "conv_idx": conv_tool_call_indices[i],
                "confusable_tools": confusable_alternatives
            })

    if not possible_injection_points:
        return dialogue_data, False, []

    injection_point = random.choice(possible_injection_points)
    plan_idx, conv_idx = injection_point["plan_idx"], injection_point["conv_idx"]

    try:
        correct_plan_step = plan[plan_idx]
        correct_tool_name = correct_plan_step.get("tool_name")
        correct_call_turn = conversations[conv_idx]
        correct_call_args = json.loads(correct_call_turn['content'])[0]['arguments']

        wrong_tool_name = random.choice(injection_point["confusable_tools"])
        wrong_tool_schema = tool_schemas.get(wrong_tool_name, {})
        wrong_tool_params = list(wrong_tool_schema.get("properties", {}).keys())

        if not wrong_tool_params or not correct_call_args:
            return dialogue_data, False, []

        wrong_call_args = {wrong_tool_params[0]: list(correct_call_args.values())[0]}
        wrong_tool_call = {"name": wrong_tool_name, "arguments": wrong_call_args}
        wrong_call_turn = {
            "role": "ignored_assistant_tool_call",
            "content": json.dumps([wrong_tool_call])
        }
        unhelpful_response = {
            "status": "success",
            "result": f"Executed {wrong_tool_name}, but the output may not be what you intended."
        }
        wrong_response_turn = {"role": "tool", "content": json.dumps(unhelpful_response)}

        new_conversations = conversations[:conv_idx] + [wrong_call_turn, wrong_response_turn] + conversations[conv_idx:]
        dialogue_data["conversations"] = new_conversations

        original_step_num = correct_plan_step.get('step')
        wrong_tool_plan_step = {
            'step': original_step_num,
            'type': 'ASSISTANT_WRONG_TOOL_CALL',
            'tool_name': wrong_tool_name
        }
        plan.insert(plan_idx, wrong_tool_plan_step)

        for j in range(plan_idx + 1, len(plan)):
            if 'step' in plan[j]: plan[j]['step'] += 1
        dialogue_data["plan"] = plan

        return dialogue_data, True, ["wrong_tool_errors"]

    except Exception:
        return dialogue_data, False, []


def inject_schema_based_error(
    dialogue_data: dict, tool_schemas: dict[str, dict],
    matcher: HybridSimilarityMatcher = None,
) -> tuple[dict, bool, list[str]]:
    """Inject parameter-based errors such as type mismatches, missing params, or invalid values.

    Randomly selects a tool call and injects one of three error types: incorrect type,
    missing required parameter, or invalid enum value, then retries with correct parameters.

    Args:
        dialogue_data: Dialogue dictionary containing plan, conversations, and tools
        tool_schemas: Dictionary mapping tool names to their parameter schemas
        matcher: Optional similarity matcher (unused in this function)

    Returns:
        Tuple of modified dialogue, success flag, and list of applied error types
    """
    conversations = dialogue_data.get("conversations", [])
    plan = dialogue_data.get("plan", [])
    conv_tool_call_indices = [
        i for i, turn in enumerate(conversations) \
            if turn.get("role") in ["assistant_tool_call", "ignored_assistant_tool_call"]
    ]

    plan_tool_call_indices = [i for i, step in enumerate(plan) if step.get("type") == "CALL_TOOL"]
    if not conv_tool_call_indices or len(conv_tool_call_indices) != len(
        plan_tool_call_indices):
        return dialogue_data, False, []

    injection_point_map = dict(zip(conv_tool_call_indices, plan_tool_call_indices))
    shuffled_conv_indices = random.sample(list(injection_point_map.keys()), len(injection_point_map))

    for conv_idx in shuffled_conv_indices:
        if (conv_idx + 1) >= len(conversations) or conversations[conv_idx + 1].get("role") != "tool": continue
        turn = conversations[conv_idx]

        try:
            original_tool_call = json.loads(turn['content'])[0]
            tool_name, args = original_tool_call.get("name"), original_tool_call.get("arguments", {})
            if tool_name not in tool_schemas: continue

            schema = tool_schemas[tool_name]
            type_error_params = [
                p for p, props in schema["properties"].items() \
                    if props.get("type") == "string" and p in args \
                    and isinstance(args.get(p), str) and args[p].isdigit()
            ]
            missing_param_params = [p for p in schema.get("required", []) if p in args]
            invalid_value_params = [p for p, props in schema["properties"].items() if "enum" in props and p in args]
            available_error_types = {}

            if type_error_params: available_error_types["INCORRECT_TYPE"] = type_error_params
            if missing_param_params: available_error_types["MISSING_PARAMETER"] = missing_param_params
            if invalid_value_params: available_error_types["INVALID_VALUE"] = invalid_value_params
            if not available_error_types: continue

            error_types = list(available_error_types.keys())
            weights = [3 if et != "MISSING_PARAMETER" else 1 for et in list(available_error_types.keys())]
            chosen_err_type = random.choices(error_types, weights=weights, k=1)[0]
            param_key = random.choice(available_error_types[chosen_err_type])
            erroneous_call, error_message_core = copy.deepcopy(original_tool_call), ""

            if chosen_err_type == "INCORRECT_TYPE":
                erroneous_call["arguments"][param_key] = int(args[param_key])
                error_message_core = f"Invalid parameter type for '{param_key}'. Expected string, got number."

            elif chosen_err_type == "MISSING_PARAMETER":
                del erroneous_call["arguments"][param_key]
                error_message_core = f"Missing required parameter: '{param_key}'"

            elif chosen_err_type == "INVALID_VALUE":
                valid_options = schema["properties"][param_key]["enum"]
                invalid_value = random.choice(["unknown", "critical", "other", "N/A"])
                erroneous_call["arguments"][param_key] = invalid_value
                error_message_core = f"Invalid value for '{param_key}'. Expected one of {valid_options}, got '{invalid_value}'."

            error_prefix = random.choice(["Error:", "error:", "ERROR:"])
            error_response_dict = {
                "function_name": tool_name,
                "error_message": f"{error_prefix} {error_message_core}",
                "status": "error"
            }

            erroneous_conv_turn = {"role": "ignored_assistant_tool_call", "content": json.dumps([erroneous_call])}
            error_conv_turn = {"role": "tool", "content": json.dumps(error_response_dict)}

            new_conversations = conversations[:conv_idx] + \
                                [erroneous_conv_turn, error_conv_turn] + \
                                conversations[conv_idx:]
            dialogue_data["conversations"] = new_conversations

            plan_idx, original_step_num = injection_point_map[conv_idx], plan[injection_point_map[conv_idx]].get('step')
            if original_step_num is None: continue

            error_call_plan_step = {
                'step': original_step_num,
                'type': 'ASSISTANT_ERROR_TOOL_CALL',
                'tool_name': tool_name
            }
            tool_error_plan_step = {'step': original_step_num + 1, 'type': 'TOOL_ERROR'}
            plan.insert(plan_idx, tool_error_plan_step)
            plan.insert(plan_idx, error_call_plan_step)

            for j in range(plan_idx + 2, len(plan)):
                if 'step' in plan[j]: plan[j]['step'] += 2
            dialogue_data["plan"] = plan

            return dialogue_data, True, ["parameter_errors"]

        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    return dialogue_data, False, []


def process_file(
    input_path: str, output_path: str,
    probability: float, matcher: HybridSimilarityMatcher,
) -> tuple[dict[str, int], int]:
    """Process a JSONL file and inject errors into dialogues based on probability.

    Reads dialogues from input file, randomly injects either composed or simple errors
    based on configuration, and writes modified dialogues to output file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file for modified dialogues
        probability: Probability of injecting errors into a dialogue (0.0 to 1.0)
        matcher: Similarity matcher for finding confusable tool names

    Returns:
        Tuple of error type counts dictionary and total modified dialogue count
    """
    total_lines = 0
    counts = {
        "parameter_errors": 0,
        "wrong_tool_errors": 0,
        "out_of_order_errors": 0,
        "cascading_failure_errors": 0,
        "composed_dialogues": 0
    }

    modified_dialogue_count = 0

    simple_injectors = {
        inject_schema_based_error: 0.2,
        inject_wrong_tool_self_correct_error: 0.4,
        inject_out_of_order_recovery_error: 0.4
    }
    simple_injector_list, simple_weights = list(simple_injectors.keys()), list(simple_injectors.values())

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            total_lines += 1
            try:
                dialogue_data = json.loads(line)
                if random.random() < probability:
                    tool_schemas = parse_tool_schemas(dialogue_data.get("tools", []))
                    if not (tool_schemas and dialogue_data.get("conversations") and dialogue_data.get("plan")): continue

                    modified_dialogue, modified = None, False

                    # 30% chance to attempt a complex/composite error
                    if random.random() < 0.7:
                        modified_dialogue, modified, applied_errors = compose_errors(
                            copy.deepcopy(dialogue_data), tool_schemas, matcher
                        )
                        if modified:
                            for error_name in applied_errors: counts[error_name] += 1
                            if len(applied_errors) > 1: counts["composed_dialogues"] += 1

                    else:  # 70% chance for a single, simple error
                        chosen_injector = random.choices(simple_injector_list, weights=simple_weights, k=1)[0]
                        modified_dialogue, modified, applied_errors = chosen_injector(
                            copy.deepcopy(dialogue_data), tool_schemas, matcher
                        )
                        if modified:
                            for error_name in applied_errors:
                                counts[error_name] += 1

                    if modified:
                        modified_dialogue['dialogue_id'] += '_error'
                        outfile.write(json.dumps(modified_dialogue) + '\n')
                        modified_dialogue_count += 1

            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line in {input_path}")

    total_injected = sum(v for k, v in counts.items() if k != "composed_dialogues")
    total_dialogues_mod = sum(counts.values()) - counts["composed_dialogues"]
    print(f"  Scanned: {total_lines} dialogues")
    print(f"  Injected {total_injected} total errors into {total_dialogues_mod} dialogues")
    print(f"  Saved: {modified_dialogue_count} modified dialogues")

    return counts, modified_dialogue_count


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the error injection script."""
    parser = argparse.ArgumentParser(description="Injects logical errors into dialogue .jsonl files.")
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue JSONL files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed JSONL files')
    parser.add_argument("-p", "--probability", type=float, default=0.2,
                        help="The probability (0.0 to 1.0) of injecting an error.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Hybrid Similarity Matcher model...")
    matcher = HybridSimilarityMatcher()
    print("Model loaded.")

    grand_total_counts = {
        "parameter_errors": 0,
        "wrong_tool_errors": 0,
        "out_of_order_errors": 0,
        "cascading_failure_errors": 0,
        "composed_dialogues": 0
    }
    total_files_processed = 0
    grand_total_modified_dialogues = 0
    file_list = natsorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])

    for file_name in file_list:
        total_files_processed += 1
        input_path, output_path = os.path.join(args.input_dir, file_name), os.path.join(args.output_dir, file_name)
        print(f"\nProcessing file: {file_name}")

        counts_for_file, modified_in_file = process_file(input_path, output_path, args.probability, matcher)
        grand_total_modified_dialogues += modified_in_file

        for key, value in counts_for_file.items(): grand_total_counts[key] += value

    total_errors = sum(v for k, v in grand_total_counts.items() if k != "composed_dialogues")

    print("\n========================================")
    print("           All Files Processed           ")
    print("-----------------------------------------")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total dialogues modified: {grand_total_modified_dialogues}")
    print(f"Total individual errors injected: {total_errors}\n")

    print("Error Type Breakdown:")
    print(f"  - Parameter Errors: {grand_total_counts['parameter_errors']}")
    print(f"  - Wrong Tool Errors: {grand_total_counts['wrong_tool_errors']}")
    print(f"  - Out-of-Order (2-step): {grand_total_counts['out_of_order_errors']}")
    print(f"  - Cascading Failures (3+ step): {grand_total_counts['cascading_failure_errors']}")
    print(f"  - Dialogues with Composed (layered) Errors: {grand_total_counts['composed_dialogues']}")

    print(f"\nOutput directory: {args.output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()