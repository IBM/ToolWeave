import argparse
import json
import os
import random

from natsort import natsorted


def process_dialogue(data: dict) -> dict:
    """Process a dialogue by adding missing function elements to the plan and conversation.

    Selects a random position between a user utterance and tool call, inserts
    missing function plan steps, and adds corresponding conversation turns

    Args:
        data: Dictionary containing dialogue data with plan, conversations, and tools

    Returns:
        Modified dialogue dictionary with missing function elements added, or None
        if no valid insertion points exist
    """
    result = data.copy()
    result['dialogue_id'] = result['dialogue_id'] + '_missing_func'

    plan = result['plan']
    candidate_indices = []

    for i in range(len(plan) - 1):
        if plan[i].get('type') == 'USER_UTTERANCE' and plan[i+1].get('type') == 'CALL_TOOL':
            candidate_indices.append(i)

    if not candidate_indices:
        return None

    chosen_idx = random.choice(candidate_indices)

    user_step = plan[chosen_idx].get('step', 0)
    assistant_missing_func_plan = {
        'step': user_step + 1,
        'type': 'ASSISTANT_MISSING_FUNCTION_MENTION'
    }
    user_missing_func_plan = {
        'step': user_step + 2,
        'type': 'USER_MISSING_FUNCTION_ADDITION'
    }

    plan.insert(chosen_idx + 1, assistant_missing_func_plan)
    plan.insert(chosen_idx + 2, user_missing_func_plan)

    for i in range(chosen_idx + 3, len(plan)):
        if 'step' in plan[i]:
            plan[i]['step'] += 2

    possible_missing_tool_names = []
    for plan_step in range(chosen_idx + 3, len(plan)):
        if plan[plan_step].get('type') == 'ASSISTANT_RESPONSE_TOOL':
            break

        elif plan[plan_step].get('type') == 'CALL_TOOL':
            possible_missing_tool_names.append(plan[plan_step].get('tool_name', 'unknown_tool'))

    chosen_tool = random.choice(possible_missing_tool_names)
    chosen_tool_schema = next((
        tool for tool in result['tools'] if tool['function']['name'] == chosen_tool
    ), {})

    assistant_missing_func_dialogue = {
        "role": "assistant",
        "content": "I do not have the necessary tools to complete this task."
    }
    user_missing_func_dialogue = {
        "role": "user",
        "content": f"{json.dumps(chosen_tool_schema, indent=2)}\nI have updated some more functions you can choose from. What about now?"
    }

    conversations = result['conversations']
    conversation_idx = sum(2 if step.get('type') == 'CALL_TOOL' else 1 for step in plan[:chosen_idx + 1])
    conversations.insert(conversation_idx + 1, assistant_missing_func_dialogue)
    conversations.insert(conversation_idx + 2, user_missing_func_dialogue)

    return {
        'dialogue_id': result['dialogue_id'],
        'goal_type': result['goal_type'],
        'overall_goal': result['overall_goal'],
        'partition': result['partition'],
        'goal_score_breakdown': result['goal_score_breakdown'],
        'missing_function': chosen_tool,
        'plan': plan,
        'tools': result['tools'],
        'conversations': conversations,
    }


def process_file(input_path: str, output_path: str, missing_func_fraction: float) -> None:
    """Process a single JSONL file to add missing function mentions.

    Filters valid dialogues, randomly samples based on the specified fraction,
    and writes processed dialogues to the output file

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        missing_func_fraction: Fraction of valid dialogues to process (0.0 to 1.0)
    """
    valid_dialogues = []

    with open(input_path, 'r') as infile:
        for line in infile:
            data = json.loads(line.strip())

            if 'plan' in data and 'dialogue_id' in data and 'conversations' in data:
                plan = data['plan']
                for i in range(len(plan) - 1):
                    if plan[i].get('type') == 'USER_UTTERANCE' and plan[i+1].get('type') == 'CALL_TOOL':
                        valid_dialogues.append(data)
                        break

    total_valid = len(valid_dialogues)
    num_to_process = min(max(1, int(total_valid * missing_func_fraction)), total_valid)

    dialogues_to_process = natsorted(random.sample(valid_dialogues, num_to_process), key=lambda x: x['dialogue_id'])

    with open(output_path, 'w') as outfile:
        for dialogue in dialogues_to_process:
            processed_data = process_dialogue(dialogue)
            if processed_data:
                outfile.write(json.dumps(processed_data) + '\n')


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dialogue processing configuration"""
    parser = argparse.ArgumentParser(description='Add missing function mentions to dialogue JSON files')
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue JSONL files')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed JSONL files')
    parser.add_argument('--missing_func_fraction', type=float, default=0.15,
                        help='Fraction of samples that will have missing function mentions (default: 0.15)')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for file_name in sorted(os.listdir(args.input_dir)):
        if file_name.endswith('.jsonl'):
            domain = file_name.split('.')[0]

            input_path = os.path.join(args.input_dir, file_name)
            output_path = os.path.join(args.output_dir, file_name)
            process_file(input_path, output_path, args.missing_func_fraction)
            print(f"Processed {domain} domain. Saved to {output_path}")


if __name__ == "__main__":
    main()
