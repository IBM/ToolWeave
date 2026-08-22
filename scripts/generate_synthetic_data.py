#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import os
import subprocess
import time


def print_command(cmd: list[str]) -> None:
    """Print the command that will be executed."""
    print(">>", " ".join(cmd))


def create_directories(output_dir: str, apis_exist: bool = False) -> None:
    """Create all necessary output directories.

    Args:
        output_dir: Root directory for outputs
        apis_exist: Whether API directory already exists
    """
    directories = ["logs", "goals", "plans", "dialogues", "refined_dialogues"]
    for directory in directories:
        os.makedirs(os.path.join(output_dir, directory), exist_ok=True)

    if not apis_exist:
        os.makedirs(os.path.join(output_dir, "apis"), exist_ok=True)


def generate_apis(output_dir: str, llm_type: str) -> str:
    """Generate API definitions for all domains.

    Args:
        output_dir: Root directory for outputs
        llm_type: Type of LLM to use (watsonx or vllm)

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        cmd = [
            "python", "-m", "scripts.domain_api_synthesizer",
            "--output_dir", os.path.join(output_dir, "apis"),
            "--log_file", os.path.join(output_dir, "logs", "api_synthesizer.log"),
            "--llm_type", llm_type
        ]
        print_command(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            return f"API generation failed (see {os.path.join(output_dir, 'logs', 'api_synthesizer.log')})"

        elapsed = time.time() - start_time
        return f"API generation completed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"API generation error - {str(e)}"


def generate_domain_goals_and_plans(output_dir: str, domain: str, domain_apis_dir: str, llm_type: str) -> str:
    """Process a single domain by running the goal and plan generation scripts.

    Args:
        output_dir: Root directory for outputs
        domain: Domain name to process
        domain_apis_dir: Directory containing API definitions for the domain
        llm_type: Type of LLM to use (watsonx or vllm)

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        json_path = os.path.join(domain_apis_dir, f"sdk_{domain}.json")
        graph_path = os.path.join(domain_apis_dir, f"graph_{domain}.pkl")
        goals_path = os.path.join(output_dir, "goals", f"{domain}.jsonl")
        plans_path = os.path.join(output_dir, "plans", f"{domain}.jsonl")

        os.makedirs(os.path.join(output_dir, "logs", domain), exist_ok=True)

        print(f"[{domain}] Running goal generator...")
        cmd1 = [
            "python", "-m", "scripts.complex_goal_generator",
            "--graph_path", graph_path,
            "--api_definitions_path", json_path,
            "--output_goals_file_path", goals_path,
            "--synthetic_apis",
            "--log_file", os.path.join(output_dir, "logs", domain, "goal_generator.log"),
            "--llm_type", llm_type
        ]
        print_command(cmd1)
        result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if result1.returncode != 0:
            return f"[{domain}] Goal generator failed (see {os.path.join(output_dir, 'logs', domain, 'goal_generator.log')})"

        print(f"[{domain}] Running plan generator...")
        cmd2 = [
            "python", "-m", "scripts.complex_dialogue_planner",
            "--goals_file_path", goals_path,
            "--api_definitions_path", json_path,
            "--prompts_dir", "prompts/prompts_for_partitioning_goal_new",
            "--output_path", plans_path,
            "--synthetic_apis",
            "--log_file", os.path.join(output_dir, "logs", domain, "plan_generator.log"),
            "--llm_type", llm_type
        ]
        print_command(cmd2)
        result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        if result2.returncode != 0:
            return f"[{domain}] Dialogue planner failed (see {os.path.join(output_dir, 'logs', domain, 'plan_generator.log')})"

        elapsed = time.time() - start_time
        return f"[{domain}] Successfully processed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"[{domain}] Error - {str(e)}"


def generate_domain_dialogues(output_dir: str, domain: str, domain_apis_dir: str, llm_type: str) -> str:
    """Generate dialogues for a single domain sequentially.

    Args:
        output_dir: Root directory for outputs
        domain: Domain name to process
        domain_apis_dir: Directory containing API definitions for the domain
        llm_type: Type of LLM to use (watsonx or vllm)

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        json_path = os.path.join(domain_apis_dir, f"sdk_{domain}.json")
        plans_path = os.path.join(output_dir, "plans", f"{domain}.jsonl")
        dialogues_path = os.path.join(output_dir, "dialogues", f"{domain}.jsonl")

        os.makedirs(os.path.join(output_dir, "logs", domain), exist_ok=True)

        if llm_type == 'vllm':
            # Run only the specified 'chat' command for vLLM
            print(f"[{domain}] Running dialogue generator in chat mode for vLLM...")
            cmd1 = [
                "python", "-m", "scripts.generate_dialogues",
                "--plans_file", plans_path,
                "--tools_list_path", json_path,
                "--prompt_config_file", "prompt_configs/dialogue_generators/chat.yml",
                "--output_file", dialogues_path,
                "--generation_strategy", "chat",
                "--synthetic_apis",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_generator.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd1)
            result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result1.returncode != 0:
                return f"[{domain}] Dialogue generator failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_generator.log')})"
        else:
            print(f"[{domain}] Running dialogue generator...")
            cmd1 = [
                "python", "-m", "scripts.generate_dialogues",
                "--plans_file", plans_path,
                "--tools_list_path", json_path,
                "--prompt_config_file", "prompt_configs/dialogue_generators/hybrid.yml",
                "--output_file", dialogues_path,
                "--generation_strategy", "hybrid",
                "--synthetic_apis",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_generator.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd1)
            result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result1.returncode != 0:
                return f"[{domain}] Dialogue generator failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_generator.log')})"

            print(f"[{domain}] Running dialogue generator again looking for the possibility of few more successful generations...")
            cmd2 = [
                "python", "-m", "scripts.generate_dialogues",
                "--plans_file", plans_path,
                "--tools_list_path", json_path,
                "--prompt_config_file", "prompt_configs/dialogue_generators/generate.yml",
                "--output_file", dialogues_path,
                "--generation_strategy", "generate",
                "--synthetic_apis",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_generator_run_2.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd2)
            result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result2.returncode != 0:
                return f"[{domain}] Dialogue generator failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_generator_run_2.log')})"

        elapsed = time.time() - start_time
        return f"[{domain}] Dialogue generation completed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"[{domain}] Dialogue generation error - {str(e)}"


def convert_false_multi_step_to_parallel(output_dir: str) -> str:
    """Convert false multi-step sequences to parallel format for all domains.

    Args:
        output_dir: Root directory containing dialogues

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        dialogues_dir = os.path.join(output_dir, "dialogues")

        cmd = [
            "python", "-m", "scripts.convert_false_multi_step_to_parallel",
            "--input_dir", dialogues_dir
        ]
        print_command(cmd)
        result_log_file = os.path.join(output_dir, "logs", "false_multi_step_converter.log")
        with open(result_log_file, "w") as log_file:
            result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        if result.returncode != 0:
            return f"False multi-step conversion failed (see {result_log_file})"

        elapsed = time.time() - start_time
        return f"False multi-step conversion completed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"False multi-step conversion error - {str(e)}"


def refine_dialogues(output_dir: str, dialogue_file_name: str, llm_type: str) -> str:
    """Refine dialogues for a single domain.

    Args:
        output_dir: Root directory for outputs
        dialogue_file_name: Name of the dialogue file to refine
        llm_type: Type of LLM to use (watsonx or vllm)

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        dialogues_path = os.path.join(output_dir, "dialogues", dialogue_file_name)
        refined_dialogues_path = os.path.join(output_dir, "refined_dialogues", dialogue_file_name)

        domain = dialogue_file_name.split(".")[0]
        os.makedirs(os.path.join(output_dir, "logs", domain), exist_ok=True)

        if llm_type == 'vllm':
            # Run only a single 'chat' command for vLLM
            print(f"[{domain}] Running dialogue refinement in chat mode for vLLM...")
            cmd1 = [
                "python", "-m", "scripts.refine_dialogues",
                "--dialogues_file", dialogues_path,
                "--prompt_config_file", "prompt_configs/dialogue_refiners/chat.yml",
                "--generation_strategy", "chat",
                "--output_file", refined_dialogues_path,
                "--refinements", "paraphrase",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_refiner.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd1)
            result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result1.returncode != 0:
                return f"[{domain}] Dialogue refinement failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_refiner.log')})"
        else:
            print(f"[{domain}] Running dialogue refinement...")
            cmd1 = [
                "python", "-m", "scripts.refine_dialogues",
                "--dialogues_file", dialogues_path,
                "--prompt_config_file", "prompt_configs/dialogue_refiners/hybrid.yml",
                "--generation_strategy", "hybrid",
                "--output_file", refined_dialogues_path,
                "--refinements", "paraphrase",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_refiner.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd1)
            result1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result1.returncode != 0:
                return f"[{domain}] Dialogue refinement failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_refiner.log')})"

            print(f"[{domain}] Running dialogue refinement again looking for the possibility of few more successful refinements...")
            cmd2 = [
                "python", "-m", "scripts.refine_dialogues",
                "--dialogues_file", dialogues_path,
                "--prompt_config_file", "prompt_configs/dialogue_refiners/generate.yml",
                "--generation_strategy", "generate",
                "--output_file", refined_dialogues_path,
                "--refinements", "paraphrase",
                "--log_file", os.path.join(output_dir, "logs", domain, "dialogue_refiner_run_2.log"),
                "--llm_type", llm_type
            ]
            print_command(cmd2)
            result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            if result2.returncode != 0:
                return f"[{domain}] Dialogue refinement failed (see {os.path.join(output_dir, 'logs', domain, 'dialogue_refiner_run_2.log')})"

        elapsed = time.time() - start_time
        return f"[{domain}] Dialogue refinement completed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"[{domain}] Dialogue refinement error - {str(e)}"


def compute_dialogue_stats(output_dir: str, dialogues_dir: str = "refined_dialogues") -> str:
    """Compute statistics for the generated dialogues.

    Args:
        output_dir: Root directory containing dialogues
        dialogues_dir: Name of the directory containing dialogue files

    Returns:
        Status message indicating success or failure
    """
    start_time = time.time()
    try:
        cmd1 = [
            "python", "-m", "scripts.compute_dialogue_statistics",
            "--input_dir", os.path.join(output_dir, dialogues_dir),
            "--output_file", os.path.join(output_dir, "dialogue_stats.jsonl")
        ]
        print_command(cmd1)
        result1_log_file = os.path.join(output_dir, "logs", "dialogue_statistics.log")
        with open(result1_log_file, "w") as log_file:
            result1 = subprocess.run(cmd1, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        if result1.returncode != 0:
            return f"Dialogue statistics computation failed (see {result1_log_file})"

        cmd2 = [
            "python", "-m", "scripts.compute_true_multi_step_stats",
            "--input_dir", os.path.join(output_dir, dialogues_dir),
            "--output_file", os.path.join(output_dir, "true_multi_step_stats.json")
        ]
        print_command(cmd2)
        result2_log_file = os.path.join(output_dir, "logs", "true_multi_step_stats.log")
        with open(result2_log_file, "w") as log_file:
            result2 = subprocess.run(cmd2, stdout=log_file, stderr=subprocess.STDOUT, text=True)

        if result2.returncode != 0:
            return f"Dialogue statistics computation failed (see {result2_log_file})"

        elapsed = time.time() - start_time
        return f"Dialogue statistics computation completed in {elapsed:.2f} seconds"

    except Exception as e:
        return f"Dialogue statistics computation error - {str(e)}"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate synthetic data for tool dialogue synthesizer.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--apis_dir", type=str, help="Optional directory with subdirectories for all domains. "
                        "If this does not exist, APIs will be generated at `<output_dir>/apis`.")
    parser.add_argument("--generate_apis", action="store_true", help="Generate API definitions for all domains.")
    parser.add_argument("--generate_goals_and_plans", action="store_true", help="Generate goals and plans for all domains.")
    parser.add_argument("--generate_dialogues", action="store_true", help="Generate dialogues for all domains.")
    parser.add_argument("--refine_dialogues", action="store_true", help="Refine dialogues for all domains.")
    parser.add_argument("--compute_dialogue_stats", action="store_true", help="Compute statistics for the generated dialogues.")
    parser.add_argument("--llm_type", type=str, choices=['watsonx', 'vllm'], default='watsonx', help="The type of LLM to use for generation.")

    args = parser.parse_args()

    apis_exist = args.apis_dir is not None and os.path.exists(args.apis_dir) and os.path.isdir(args.apis_dir)

    if apis_exist or args.generate_apis:
        # All stages except stats computation depend on existence of APIs
        create_directories(args.output_dir, apis_exist)

        if args.generate_apis or not apis_exist:
            print("\n=== Generating API definitions for all domains ===")
            message = generate_apis(args.output_dir, args.llm_type)
            print(message)

        if apis_exist:
            domains = sorted([d for d in os.listdir(args.apis_dir) if os.path.isdir(os.path.join(args.apis_dir, d))])
        else:
            apis_dir = os.path.join(args.output_dir, "apis")
            domains = sorted([d for d in os.listdir(apis_dir) if os.path.isdir(os.path.join(apis_dir, d))])

        if not domains:
            print("No domains found. Please ensure that API definitions are available.")
            return

        if args.generate_goals_and_plans:
            for domain in domains:
                print(f"\n=== Generating goals and plans for domain: {domain} ===")

                if apis_exist:
                    domain_apis_dir = os.path.join(args.apis_dir, domain)
                else:
                    domain_apis_dir = os.path.join(args.output_dir, "apis", domain)

                message = generate_domain_goals_and_plans(args.output_dir, domain, domain_apis_dir, args.llm_type)
                print(message)

        if args.generate_dialogues:
            for domain in domains:
                print(f"\n=== Generating dialogues for domain: {domain} ===")

                if apis_exist:
                    domain_apis_dir = os.path.join(args.apis_dir, domain)
                else:
                    domain_apis_dir = os.path.join(args.output_dir, "apis", domain)

                message = generate_domain_dialogues(args.output_dir, domain, domain_apis_dir, args.llm_type)
                print(message)

    if args.refine_dialogues:
        dialogue_files = [
            f for f in os.listdir(os.path.join(args.output_dir, "dialogues"))
            if f.endswith(".jsonl")
        ]

        for dialogue_file in dialogue_files:
            domain = dialogue_file.split(".")[0]
            print(f"\n=== Refining dialogues for domain: {domain} ===")

            message = refine_dialogues(args.output_dir, dialogue_file, args.llm_type)
            print(message)

    if args.compute_dialogue_stats:
        print("\n=== Computing dialogue statistics ===")

        dialogues_dir = None
        for dir_name in ["refined_dialogues", "dialogues"]:
            dir_path = os.path.join(args.output_dir, dir_name)
            if os.path.isdir(dir_path) and any(f.endswith(".jsonl") for f in os.listdir(dir_path)):
                dialogues_dir = dir_name
                break

        if not dialogues_dir:
            raise FileNotFoundError("No dialogues found in the expected directories.")

        message = compute_dialogue_stats(args.output_dir, dialogues_dir)
        print(message)

    if args.generate_dialogues or args.refine_dialogues:
        print(f"\n=== Converting false multi-step sequences to parallel format ===")

        dialogues_dir = None
        for dir_name in ["refined_dialogues", "dialogues"]:
            dir_path = os.path.join(args.output_dir, dir_name)
            if os.path.isdir(dir_path) and any(f.endswith(".jsonl") for f in os.listdir(dir_path)):
                dialogues_dir = dir_name
                break

        if not dialogues_dir:
            raise FileNotFoundError("No dialogues found in the expected directories.")

        message = convert_false_multi_step_to_parallel(args.output_dir)
        print(message)

    print("\nAll processing completed.")

if __name__ == "__main__":
    main()
