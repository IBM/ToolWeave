#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import logging
import os
import sys
import yaml
from concurrent.futures import as_completed, ThreadPoolExecutor
from filelock import FileLock
from pathlib import Path
from pprint import pformat

from langchain_core.runnables import Runnable
from natsort import natsorted

from src.tool_dialogue_synthesizer.agents.paraphraser_agent import ParaphraserAgent
from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('dialogue_refiner')


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


def initialize_agents(
    refinements: list[str], llm: VLLMClient | WatsonxLLM,
    prompts_config_path: str, generation_strategy: str = "chat",
) -> list[Runnable]:
    """Initialize agents based on the specified refinements.

    Args:
        refinements: List of refinement types to apply
        llm: LLM client for agent operations
        prompts_config_path: Path to prompts configuration file
        generation_strategy: Strategy for LLM generation (chat, generate, or hybrid)

    Returns:
        List of initialized agent instances
    """
    agents = []

    with open(prompts_config_path, 'r') as file:
        prompts_config = yaml.safe_load(file)

    if 'paraphrase' in refinements:
        agents.append(
            ParaphraserAgent(
                llm, prompts_config["USER_UTTERANCE_PARAPHRASER_PROMPT_PATH"],
                prompts_config["USER_CLARIFICATION_PARAPHRASER_PROMPT_PATH"],
                generation_strategy=generation_strategy
            )
        )

    return agents


def load_dialogues_from_jsonl(dialogues_file_path: str, num_total_dialogues: int = -1) -> list[dict]:
    """Load dialogues from a JSONL file.

    Args:
        dialogues_file_path: Path to the JSONL file containing dialogues
        num_total_dialogues: Maximum number of dialogues to load (-1 for all)

    Returns:
        List of dialogue dictionaries
    """
    all_dialogues_data = []
    try:
        with open(dialogues_file_path, "r") as f:
            for line_num, line in enumerate(f):
                if num_total_dialogues != -1 and len(all_dialogues_data) >= num_total_dialogues:
                    break

                try:
                    dialogue_object = json.loads(line.strip())
                    all_dialogues_data.append(dialogue_object)

                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupted line {line_num} in {dialogues_file_path}")

                except:
                    logger.exception(f"An unexpected error occurred while processing line {line_num} in {dialogues_file_path}")

        return all_dialogues_data

    except FileNotFoundError:
        logger.exception(f"Dialogues file not found at '{dialogues_file_path}'")
        return []

    except:
        logger.exception(f"Unexpected error opening {dialogues_file_path}")
        return []


def load_refined_ids(filepath: str) -> set[str]:
    """Load dialogue IDs that have already been refined from output file.

    Args:
        filepath: Path to the output file containing refined dialogues

    Returns:
        Set of dialogue IDs that have been previously refined
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

        except Exception:
            logger.exception(f"Error reading dialogue IDs file {filepath}")

    return ids


def refine_dialogue_entry(entry: dict, agents: list[Runnable]) -> dict | None:
    """Refine a single dialogue entry by sequentially applying all agents.

    Args:
        entry: Dialogue dictionary containing plan and conversations
        agents: List of agent instances to apply in sequence

    Returns:
        Refined dialogue dictionary, or None if refinement failed or made no changes
    """
    dialogue_id = entry.get("dialogue_id", None)

    if not isinstance(entry, dict) or "plan" not in entry or "conversations" not in entry:
        logger.warning(f"Skipping invalid entry at id {dialogue_id}")
        return None

    plan = entry.get("plan", [])
    if not isinstance(plan, list):
        logger.warning(f"Skipping dialogue at id {dialogue_id} due to invalid format (expected list, got {type(plan)})")
        return None

    conversations = entry.get("conversations", [])
    if not isinstance(conversations, list):
        logger.warning(f"Skipping dialogue at id {dialogue_id} due to invalid conversations format (expected list, got {type(conversations)})")
        return None

    logger.info(f"Refining dialogue for sample {dialogue_id}")

    result_entry = entry.copy()

    for agent in agents:
        agent_class_name = agent.__class__.__name__

        try:
            result = agent.invoke(result_entry)

            if not isinstance(result, dict):
                logger.warning(f"Agent {agent_class_name} returned invalid result for dialogue {dialogue_id}")
                continue

            if 'dialogue_id' not in result or result['dialogue_id'] != dialogue_id:
                logger.warning(f"Agent {agent_class_name} modified dialogue_id for {dialogue_id}. Expected {dialogue_id}, got {result.get('dialogue_id')}")
                result['dialogue_id'] = dialogue_id

            if 'plan' not in result or not isinstance(result['plan'], list):
                logger.warning(f"Agent {agent_class_name} returned invalid plan for dialogue {dialogue_id}")
                continue

            if 'conversations' not in result or not isinstance(result['conversations'], list):
                logger.warning(f"Agent {agent_class_name} returned invalid conversations for dialogue {dialogue_id}")
                continue

            logger.info(f"Agent {agent_class_name} successfully refined dialogue {dialogue_id}")
            result_entry = result

        except:
            logger.exception(f"Error refining dialogue {dialogue_id} with agent {agent_class_name}. Continuing with next agent.")
            continue

    if result_entry == entry:
        logger.warning(f"No refinements made for dialogue {dialogue_id}. Not saving this entry.")
        return None

    return result_entry


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
    parser = argparse.ArgumentParser(description="Perform refinement for saved dialogues.")
    parser.add_argument(
        "--dialogues_file",
        type=str,
        required=True,
        default="output/dialogues/agriculture.jsonl",
        help="Path to the JSON file containing generated dialogues.",
    )
    parser.add_argument(
        "--num_total_dialogues",
        type=int,
        default=-1,
        help="Total number of dialogues to process. Used for debugging and progress tracking.",
    )
    parser.add_argument(
        "--prompt_config_file",
        type=str,
        required=True,
        default="prompt_configs/dialogue_refiners/chat.yml",
        help="Path to the prompts configuration file.",
    )
    parser.add_argument(
        "--generation_strategy",
        type=str,
        choices=["generate", "chat", "hybrid"],
        default="chat",
        help="Specify the generation strategy for LLM invocation. Default is 'chat'.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        default="output/refined_dialogues/agriculture.jsonl",
        help="Path to save the refined dialogues.",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["watsonx", "vllm"],
        default="watsonx",
        help="Which LLM to use: 'watsonx' or 'vllm'. Default is 'watsonx'.",
    )
    parser.add_argument(
        "--refinements",
        type=str,
        default=["paraphrase"],
        nargs='+',
        choices=["paraphrase"],
        help="List of refinements to apply. Default is ['paraphrase'].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, the script will not retain previously refined dialogues and will overwrite the output file.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=60,
        help="Maximum number of worker threads for parallel processing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed logging. Forces max_workers=1.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Optional log file path. If not specified, logs only to console.",
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

    agents = initialize_agents(
        args.refinements, llm, prompts_config_path,
        generation_strategy=args.generation_strategy,
    )

    logger.info(f"Loading dialogues from: {args.dialogues_file}")
    all_dialogues_data = load_dialogues_from_jsonl(args.dialogues_file, args.num_total_dialogues)

    # Load ids of already refined dialogues
    if not args.overwrite:
        refined_ids = load_refined_ids(args.output_file)
        logger.info(f"Found {len(refined_ids)} previously refined ids")
    else:
        with open(args.output_file, 'w') as f:
            f.write("")
        refined_ids = set()
        logger.info("Overwrite mode enabled: Clearing output file and skipping previously refined dialogues")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dialogues_to_refine = [entry for entry in all_dialogues_data if entry.get('dialogue_id') not in refined_ids]
    total_dialogues_to_refine = len(dialogues_to_refine)

    num_workers = min(args.max_workers, total_dialogues_to_refine)
    if num_workers <= 0:
        if len(all_dialogues_data) > 0:
            logger.warning(f"Looks like all dialogues have been refined already or max_workers is set to 0. Exiting.")
            sys.exit(0)

        else:
            logger.error("No dialogues to process or max_workers is set to 0. Exiting.")
            sys.exit(1)

    logger.info(f"Starting refinement with {num_workers} workers for {total_dialogues_to_refine} dialogues")

    processed_count = 0
    failed_count = 0

    if num_workers == 1:
        for entry in dialogues_to_refine:
            result = refine_dialogue_entry(entry, agents)
            if result:
                write_result_to_file(result, args.output_file)
                processed_count += 1
                logger.info(
                    f"Successfully refined dialogue {entry.get('dialogue_id', 'unknown')} "
                    f"-- ({processed_count}/{total_dialogues_to_refine})"
                )
            else:
                failed_count += 1
                logger.error(f"Failed to refine dialogue #{entry.get('dialogue_id', 'unknown')}")

    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    refine_dialogue_entry,
                    entry, agents,
                ): entry.get('dialogue_id', 'unknown') for entry in dialogues_to_refine
            }

            for future in as_completed(futures):
                entry_idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        write_result_to_file(result, args.output_file)
                        processed_count += 1
                        logger.info(
                            f"Successfully refined dialogue {entry_idx} -- "
                            f"({processed_count}/{total_dialogues_to_refine})"
                        )
                    else:
                        failed_count += 1
                        logger.error(f"Failed to refine dialogue {entry_idx}")

                except:
                    logger.exception(f"Dialogue {entry_idx} failed with an unhandled exception")
                    failed_count += 1

    logger.info("Sorting and rewriting all entries to ensure they're ordered by index...")
    total_entries = sort_and_rewrite_results(args.output_file)

    logger.info("--- Refinement Complete ---")
    logger.info(f"{total_dialogues_to_refine} dialogues to be refined")
    logger.info(f"Successfully refined {processed_count} new dialogues")
    logger.info(f"Failed to refine {failed_count} dialogues")
    logger.info(f"Skipped {len(refined_ids)} previously refined dialogues")
    logger.info(f"Total entries in output file: {total_entries}")
    logger.info(f"Results saved in {args.output_file}")

if __name__ == "__main__":
    main()
