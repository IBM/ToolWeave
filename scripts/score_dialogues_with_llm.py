#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import json
import logging
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('llm_based_dialogue_scorer')


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


def sample_entries_from_file(input_path: str, num_entries: int) -> list[dict]:
    """Randomly sample entries from a JSONL file.

    Args:
        input_path: Path to input JSONL file
        num_entries: Number of entries to sample

    Returns:
        List of sampled dialogue entries as dictionaries
    """
    sampled_entries = []
    with open(input_path, 'r') as infile:
        lines = infile.readlines()
        lines = [line for line in lines if json.loads(line.strip()).get('conversations')]

        if len(lines) <= num_entries:
            sampled_entries = [json.loads(line.strip()) for line in lines]

        else:
            sampled_lines = random.sample(lines, num_entries)
            sampled_entries = [json.loads(line.strip()) for line in sampled_lines]

    return sampled_entries


def extract_scores_from_llm_output(llm_output_content: str) -> dict:
    """Extract numeric scores for each evaluation category from LLM output text.

    Args:
        llm_output_content: Raw text output from the LLM

    Returns:
        Dictionary mapping category names to extracted scores, or None if not found
    """
    scores = {
        "Naturalness": None,
        "Coherence": None, 
        "Helpfulness": None,
        "Accuracy": None
    }

    try:
        for category in scores.keys():
            # Match both numbered (e.g., "1. Naturalness:") and unnumbered (e.g., "Naturalness:") patterns
            patterns = [
                fr"\d+\.\s*{category}:\s*(\d+(?:\.\d+)?)\s*/\s*\d+",  # "1. Naturalness: 4 / 5" or "1. Naturalness: 4.5 / 5"
                fr"{category}:\s*(\d+(?:\.\d+)?)\s*/\s*\d+"           # "Naturalness: 4 / 5" or "Naturalness: 4.5 / 5"
            ]

            for pattern in patterns:
                match = re.search(pattern, llm_output_content)
                if match:
                    # Convert to float first, then to int if it's a whole number
                    score_value = float(match.group(1))
                    scores[category] = score_value
                    break

    except Exception:
        logger.exception("Error extracting scores from LLM output.")

    return scores


def score_dialogue(entry: dict, llm: WatsonxLLM, scoring_prompt: str) -> tuple[str, dict]:
    """Score a single dialogue using the LLM with the provided scoring prompt.

    Args:
        entry: Dialogue entry containing conversation and metadata
        llm: WatsonxLLM instance for generating scores
        scoring_prompt: System prompt with scoring instructions

    Returns:
        Tuple of dialogue ID and dictionary of extracted scores
    """
    if 'conversations' not in entry:
        return None

    dialogue_id = entry.get('dialogue_id')
    conversation = entry.get('conversations')

    chat_template_prompts = [
        {
            "role": "system",
            "content": scoring_prompt
        },
        {
            "role": "user",
            "content": f"The dialogue you need to evaluate is as follows:\n```json\n{json.dumps(conversation, indent=2)}\n```\n"
        },
    ]

    response = llm.invoke(chat_template_prompts, use_chat_mode=True)
    llm_output_content = response.content.strip()
    logger.debug(f"LLM response content: {llm_output_content}")

    scores = extract_scores_from_llm_output(llm_output_content)
    logger.debug(f"Extracted scores: {json.dumps(scores, indent=2)}")

    return dialogue_id, scores


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dialogue scoring configuration"""
    parser = argparse.ArgumentParser(description='Process dialogue JSON files')
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue JSONL files')
    parser.add_argument('--output_file', required=True, help='Path of JSON file to save scores of dialogues')
    parser.add_argument('--prompt_file', required=True, help='Path to the scoring prompt file')
    parser.add_argument('--num_entries_per_file', type=int, default=10, help='Number of entries to process per file')
    parser.add_argument('--num_workers', type=int, default=60, help='Number of parallel workers to use')
    parser.add_argument('--log_file', help='Path to log file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    llm_config_path = os.path.join(project_root, "llama3_405b_watsonx_llm_config.yml")
    llm = WatsonxLLM(llm_config_path)

    setup_logging(debug_mode=args.debug, log_file=args.log_file)

    with open(args.prompt_file, 'r') as pf:
        scoring_prompt = pf.read()

    all_domain_names = []
    all_dialogue_samples = []

    for file_name in sorted(os.listdir(args.input_dir)):
        if file_name.endswith('.jsonl'):
            domain_name = file_name.replace('.jsonl', '')
            all_domain_names.append(domain_name)

            input_path = os.path.join(args.input_dir, file_name)
            domain_samples = sample_entries_from_file(input_path, args.num_entries_per_file)
            for sample in domain_samples:
                sample['dialogue_id'] = f"{domain_name}_{sample['dialogue_id']}"

            all_dialogue_samples.extend(domain_samples)

    all_dialogue_scores = {
        "Naturalness": [],
        "Coherence": [], 
        "Helpfulness": [],
        "Accuracy": []
    }

    total_samples = len(all_dialogue_samples)
    processed_count = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []

        for entry in all_dialogue_samples:
            logger.info(f"Submitting sample {entry.get('dialogue_id')} for processing")
            futures.append(executor.submit(score_dialogue, entry, llm, scoring_prompt))

        for future in as_completed(futures):
            processed_count += 1
            dialogue_id, scores = future.result()
            logger.info(f"Sample {dialogue_id} processed ({processed_count}/{total_samples})")

            for category, score in scores.items():
                if score is not None:
                    all_dialogue_scores[category].append((dialogue_id, score))

    domain_wise_mean_scores = {
        domain: {category: None for category in all_dialogue_scores.keys()}
        for domain in all_domain_names
    }

    for category, scores in all_dialogue_scores.items():
        domain_scores = {domain: [] for domain in all_domain_names}

        for dialogue_id, score in scores:
            for domain in domain_scores.keys():
                if dialogue_id.startswith(f"{domain}_"):
                    domain_scores[domain].append(score)
                    break

        for domain, dscores in domain_scores.items():
            if dscores:
                mean_score = sum(dscores) / len(dscores)
                domain_wise_mean_scores[domain][category] = round(mean_score, 2)

    overall_mean_scores = {}
    for category, scores in all_dialogue_scores.items():
        if scores:
            overall_mean = sum(score for _, score in scores) / len(scores)
            overall_mean_scores[category] = round(overall_mean, 2)
        else:
            overall_mean_scores[category] = None

    domain_wise_mean_scores['Overall'] = overall_mean_scores

    with open(args.output_file, 'w') as outfile:
        json.dump(domain_wise_mean_scores, outfile, indent=2)

    logger.info(f"Scores saved to {args.output_file}")


if __name__ == "__main__":
    main()
