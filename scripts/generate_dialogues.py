import argparse
import datetime
import itertools
import json
import logging
import os
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from filelock import FileLock
from pathlib import Path
from pprint import pformat

from natsort import natsorted

from src.tool_dialogue_synthesizer.schema import DialogueState
from src.tool_dialogue_synthesizer.agent_workflow import build_graph
from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('dialogue_generator')


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


def load_plans_from_jsonl(plans_file_path: str, num_total_plans: int = -1) -> list[dict]:
    """Load dialogue plans from a JSONL file.

    Args:
        plans_file_path: Path to the JSONL file containing plans
        num_total_plans: Maximum number of plans to load (-1 for all)

    Returns:
        List of plan dictionaries with dialogue_plan_id fields
    """
    all_plans_data = []
    try:
        with open(plans_file_path, "r") as f:
            for line_num, line in enumerate(f):
                if num_total_plans != -1 and len(all_plans_data) >= num_total_plans:
                    break
                try:
                    plan_object = json.loads(line.strip())
                    if "dialogue_plan_id" in plan_object:
                        all_plans_data.append(plan_object)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping corrupted line {line_num} in {plans_file_path}: {e}")
                except Exception as e:
                    logger.exception(f"An unexpected error occurred while processing line {line_num} in {plans_file_path}")
        return all_plans_data

    except FileNotFoundError:
        logger.error(f"Plans file not found at '{plans_file_path}'")
        return []

    except Exception as e:
        logger.exception(f"Unexpected error opening {plans_file_path}")
        return []


def load_processed_ids(filepath: str) -> set[str]:
    """Load dialogue IDs that have already been processed from output file.

    Args:
        filepath: Path to the output file containing processed dialogues

    Returns:
        Set of dialogue IDs that have been previously processed
    """
    ids = set()
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if 'dialogue_id' in data and isinstance(data['dialogue_id'], str):
                            ids.add(data['dialogue_id'])
                        else:
                            logger.warning(f"Missing or invalid 'dialogue_id' in line {line_num} of {filepath}")

                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line {line_num} in {filepath}")

        except Exception as e:
            logger.exception(f"Error reading processed dialogue IDs file {filepath}")

    return ids


def load_full_tool_list(json_path: str, synthetic_apis: bool = False) -> list[dict]:
    """Load the complete list of available tools from a JSON file.

    Args:
        json_path: Path to JSON file with tool definitions
        synthetic_apis: Whether the APIs are synthetically generated

    Returns:
        List of tool schema definitions
    """
    with open(json_path, "r") as f:
        apis = json.load(f)

        if synthetic_apis:
            apis = apis.get("apis", [])

    return apis


def build_tools_dict(schema: list[dict]) -> dict[str, dict]:
    """Build a dictionary mapping tool names to their schema definitions.

    Args:
        schema: List of tool schema definitions

    Returns:
        Dictionary mapping tool names to tool schemas
    """
    return {
        tool["function"]["name"]: tool
        for tool in schema
        if "function" in tool and "name" in tool["function"]
    }


def get_tools_list_from_names(schema_dict: dict[str, dict], sampled_tool_names: list[str]) -> list[dict]:
    """Retrieve tool schemas for a given list of tool names.

    Args:
        schema_dict: Dictionary mapping tool names to schemas
        sampled_tool_names: List of tool names to retrieve

    Returns:
        List of tool schemas matching the provided names
    """
    return [schema_dict[name] for name in sampled_tool_names if name in schema_dict]


def modify_tool_types(tools: list[dict]) -> list[dict]:
    """Modify tool parameter types to ensure compatibility with the LLM.

    Converts credentials to string, and list/select/enum types to array.

    Args:
        tools: List of tool schemas to modify

    Returns:
        Modified list of tool schemas
    """
    for tool in tools:
        if (tool.get('function') and 
            tool['function'].get('parameters') and 
            tool['function']['parameters'].get('properties')):

            properties = tool['function']['parameters']['properties']

            for key, property_info in properties.items():
                if property_info.get('type') == 'credentials':
                    property_info['type'] = 'string'
                elif property_info.get('type') in ['list', 'select', 'enum']:
                    property_info['type'] = 'array'

    return tools


def generate_dialogue_for_plan(
    llm: VLLMClient | WatsonxLLM, sample_idx: int,
    plan: list[dict], tools_subset: list[dict],
    partition_cumsum: list[int], prompts_config_path: str,
    generation_strategy: str = "chat", max_retries: int = 2,
) -> dict:
    """Generate a complete dialogue by executing a plan through the agent workflow.

    Args:
        llm: LLM client for generation
        sample_idx: Index of the plan being processed
        plan: List of plan steps to execute
        tools_subset: List of available tools for this dialogue
        partition_cumsum: Cumulative sum of partition boundaries
        prompts_config_path: Path to prompts configuration file
        generation_strategy: Strategy for LLM generation (chat, generate, or hybrid)
        max_retries: Maximum number of retries for LLM calls

    Returns:
        Dictionary containing generated dialogue state with conversations
    """
    initial_state: DialogueState = {
        "sample_idx": sample_idx,
        "plan": plan,
        "current_step_idx": 0,
        "partition_idx": 0,
        "user_message": "",
        "tool_call": None,
        "memory_cache": {},
        "conversations": [{
            "role": "system",
            "content": f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
        }],
        "abort_due_to_error": False,
        "retry_count": 0,
        "error_msg": None,
    }

    logger.debug(f"Initial State:\n{pformat(initial_state)}\n")

    graph = build_graph(
        llm, tools_subset, partition_cumsum,
        prompts_config_path, generation_strategy, max_retries
    )
    return graph.invoke(initial_state, config={'recursion_limit': 50})


def process_plan_entry(
    entry: dict, llm: VLLMClient | WatsonxLLM,
    tools_dict: dict[str, dict], prompts_config_path: str,
    generation_strategy: str = "chat", max_retries: int = 2,
) -> dict | None:
    """Process a single plan entry to generate a dialogue.

    Args:
        entry: Plan entry containing dialogue plan, tools, and metadata
        llm: LLM client for generation
        tools_dict: Dictionary mapping tool names to their schemas
        prompts_config_path: Path to prompts configuration file
        generation_strategy: Strategy for LLM generation (chat, generate, or hybrid)
        max_retries: Maximum number of retries for LLM calls

    Returns:
        Generated dialogue dictionary with conversations and metadata, or None if failed
    """
    dialogue_plan_id = entry.get("dialogue_plan_id", None)
    goal_type = entry.get("goal_type", None)
    overall_goal = entry.get("source_goal_text", None)

    if not isinstance(entry, dict) or "path" not in entry or "plan" not in entry:
        logger.warning(f"Skipping invalid entry at id {dialogue_plan_id}")
        return None

    plan = entry.get("plan", [])
    sampled_tool_names = entry.get("path", [])
    relevant_tools = get_tools_list_from_names(tools_dict, sampled_tool_names)
    relevant_tools = modify_tool_types(relevant_tools)

    partition = entry.get("partition", [])
    partition_cumsum = [0] + list(itertools.accumulate(partition))

    if not isinstance(plan, list):
        logger.warning(f"Skipping plan at id {dialogue_plan_id} due to invalid format (expected list, got {type(plan)})")
        return None

    logger.info(f"Generating dialogue for Plan #{dialogue_plan_id}")
    logger.debug(f"Sampled Tools for Plan #{dialogue_plan_id}:\n{pformat(sampled_tool_names)}\n")

    try:
        dialogue_result = generate_dialogue_for_plan(
            llm, dialogue_plan_id, plan, relevant_tools,
            partition_cumsum, prompts_config_path,
            generation_strategy, max_retries,
        )

        if dialogue_result is None or 'conversations' not in dialogue_result:
            logger.warning(f"Dialogue generation failed or returned invalid result for Plan #{dialogue_plan_id}")
            return None

        if dialogue_result.get("abort_due_to_error", False):
            logger.error(f"Dialogue generation for Plan #{dialogue_plan_id} aborted due to an error")
            return None

        result_entry = {
            "dialogue_id": dialogue_plan_id,
            "goal_type": goal_type,
            "overall_goal": overall_goal,
            "partition": partition,
            "goal_score_breakdown": entry.get("goal_score_breakdown", {}),
            "plan": dialogue_result['plan'],
            "tools": relevant_tools,
            "conversations": dialogue_result['conversations'],
        }

        logger.info(
            "\n" + "=" * 80 + "\n" +
            f"Dialogue for Plan #{dialogue_plan_id} generated successfully\n" +
            "=" * 80 + "\n"
        )
        return result_entry

    except Exception as e:
        logger.exception(f"Error generating dialogue for Plan #{dialogue_plan_id}")
        return None


def write_result_to_file(result: dict, output_file: str) -> None:
    """Write a single result to the output file with file locking to prevent race conditions.

    Args:
        result: Dialogue result dictionary to write
        output_file: Path to output JSONL file
    """
    lock_file = f"{output_file}.lock"
    with FileLock(lock_file):
        with open(output_file, "a") as f:
            json.dump(result, f)
            f.write("\n")


def sort_and_rewrite_results(output_file: str) -> int:
    """Load all results from the file, sort them by dialogue_id, and rewrite the file.

    Args:
        output_file: Path to output JSONL file

    Returns:
        Total number of entries in the sorted file
    """
    entries = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping malformed JSON line in {output_file}")

    entries = natsorted(entries, key=lambda x: x.get('dialogue_id', 'unknown'))

    with open(output_file, 'w') as f:
        for entry in entries:
            json.dump(entry, f)
            f.write("\n")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description="Generate dialogues for saved plans.")
    parser.add_argument(
        "--plans_file",
        type=str,
        required=True,
        default="output/plans/agriculture.jsonl",
        help="Path to the JSONL file containing generated plans."
    )
    parser.add_argument(
        "--synthetic_apis",
        action="store_true",
        help="If set, uses processing related to synthetic APIs."
    )
    parser.add_argument(
        "--tools_list_path",
        type=str,
        required=True,
        default="output/apis/agriculture/sdk_agriculture.json",
        help="Path to the JSON file with all the tool definitions.",
    )
    parser.add_argument(
        "--prompt_config_file",
        type=str,
        required=True,
        default="prompt_configs/dialogue_generators/chat.yml",
        help="Path to the prompts configuration file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        default="output/dialogues/agriculture.jsonl",
        help="Path to save the generated dialogues."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, the script will not retain previously generated dialogues and will overwrite the output file.",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["watsonx", 'vllm'],
        default="watsonx",
        help="Which LLM to use: 'watsonx' or 'vllm'. Default is 'watsonx'."
    )
    parser.add_argument(
        "--generation_strategy",
        type=str,
        choices=["generate", "chat", "hybrid"],
        default="chat",
        help="Specify the generation strategy for LLM invocation. Default is 'chat'."
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum number of retries for LLM calls in case of failures (especially for tool caller and simulator)."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=60,
        help="Maximum number of worker threads for parallel processing."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging. Forces max_workers=1."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Optional log file path. If not specified, logs only to console."
    )
    parser.add_argument(
        "--num_total_plans",
        type=int,
        default=-1,
        help="Total number of plans to process. Used for debugging and progress tracking."
    )

    args = parser.parse_args()
    setup_logging(debug_mode=args.debug, log_file=args.log_file)

    if args.debug:
        args.max_workers = 1
        logger.info("Debug mode enabled: setting max_workers=1 for sequential processing")

    logger.info(f"Arguments:\n{pformat(vars(args))}\n")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if args.llm_type == "watsonx":
        config_path = os.path.join(project_root, 'watsonx_llm_config.yml')
        llm = WatsonxLLM(config_path)
    elif args.llm_type == "vllm":
        config_path = os.path.join(project_root, 'vllm_llm_config.yml')
        llm = VLLMClient(config_path)
    else:
        raise ValueError(f"Unsupported LLM type: {args.llm_type}")

    prompts_config_path = os.path.join(project_root, args.prompt_config_file)

    full_tool_list = load_full_tool_list(args.tools_list_path, synthetic_apis=args.synthetic_apis)
    tools_dict = build_tools_dict(full_tool_list)

    logger.info(f"Loading plans from: {args.plans_file}")
    all_plans_data = load_plans_from_jsonl(args.plans_file, args.num_total_plans)

    # Load ids of already processed dialogues
    if not args.overwrite:
        processed_ids = load_processed_ids(args.output_file)
        logger.info(f"Found {len(processed_ids)} previously processed ids")
    else:
        with open(args.output_file, 'w') as f:
            f.write("")
        processed_ids = set()
        logger.info("Overwrite mode enabled: Clearing output file and skipping previously generated dialogues")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plans_to_process = [entry for entry in all_plans_data if entry.get('dialogue_plan_id') not in processed_ids]
    total_plans_to_process = len(plans_to_process)

    num_workers = min(args.max_workers, total_plans_to_process)
    if num_workers <= 0:
        logger.error("No plans to process or max_workers is set to 0. Exiting.")
        sys.exit(1)
    logger.info(f"Starting processing with {num_workers} workers for {total_plans_to_process} plans")

    processed_count = 0
    failed_count = 0

    if num_workers == 1:
        # Sequential processing for debug mode
        for entry in plans_to_process:
            result = process_plan_entry(
                entry, llm, tools_dict, prompts_config_path,
                args.generation_strategy, args.max_retries,
            )
            if result:
                write_result_to_file(result, args.output_file)
                processed_count += 1
            else:
                failed_count += 1
                logger.error(f"Failed to process Plan #{entry.get('dialogue_id', 'unknown')}")

    else:
        # Parallel processing for production mode
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_plan_entry, 
                    entry, llm, tools_dict, prompts_config_path,
                    args.generation_strategy, args.max_retries,
                ): entry.get('dialogue_plan_id', 'unknown') for entry in plans_to_process
            }

            for future in as_completed(futures):
                entry_idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        write_result_to_file(result, args.output_file)
                        processed_count += 1
                        logger.info(f"Successfully processed Plan #{entry_idx} -- ({processed_count}/{total_plans_to_process})")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to process Plan #{entry_idx}")

                except Exception as e:
                    logger.exception(f"Plan #{entry_idx} failed with an unhandled exception")
                    failed_count += 1

    logger.info("Sorting and rewriting all entries to ensure they're ordered by index...")
    total_entries = sort_and_rewrite_results(args.output_file)

    logger.info("--- Generation Complete ---")
    logger.info(f"Processed {total_plans_to_process} plans")
    logger.info(f"Successfully generated {processed_count} new dialogues")
    logger.info(f"Failed to generate {failed_count} dialogues")
    logger.info(f"Skipped {len(processed_ids)} previously generated dialogues")
    logger.info(f"Total entries in output file: {total_entries}")
    logger.info(f"Results saved in {args.output_file}")

if __name__ == "__main__":
    main()
