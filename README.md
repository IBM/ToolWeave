# ToolWeave: Structured Synthesis of Complex Multi-Turn Tool-Calling Dialogues

This repository contains tools and scripts to generate **synthetic function calling data** for fine-tuning large language models (LLMs).  
The goal is to simulate realistic tool-usage scenarios that can help models learn how to correctly call functions or use APIs.

## 🔐 Environment Setup for using watsonx.ai Models

If you plan to use models from **IBM watsonx.ai**, you must create a `.env` file in the **root folder** of the repository (`ToolWeave`).

Create a `.env` file with the following contents:

```env
IBM_CLOUD_API_KEY="Your IBM API key required to authenticate with watsonx.ai"
IBM_PROJECT_ID="The Project ID for watsonx.ai, where the requests will be forwarded."
WATSONX_REGION="The ProjectRegion for watsonx.ai (\"us-south\", \"eu-gb\", \"jp-tok\", \"eu-de\")"
```

### Adding Model Parameters

To configure a model, add its generation parameters to the `watsonx_llm_config.yml` file. Each model should be defined using the following keys:

- `MODEL`: Name or path of the model (e.g., `"openai/gpt-oss-120b"`)
- `MAX_NEW_TOKENS`: Maximum number of new tokens to generate
- `TEMPERATURE`: Sampling temperature (set to `0` for deterministic output)
- `DECODING_METHOD`: Generation decoding strategy (e.g., `"greedy"`, `"beam_search"`, `"top_k"`, etc.)
- `REPETITION_PENALTY`: Penalty for repeated phrases (typically `1.0` for no penalty)

### Sample Configuration

```yaml
MODEL: "openai/gpt-oss-120b"
MAX_NEW_TOKENS: 8192
TEMPERATURE: 0
DECODING_METHOD: "greedy"
REPETITION_PENALTY: 1.0
```

## 🚀 Running the Pipeline
For the command to run the main script which runs all individual components of the pipeline, go to the end of this section.

To generate tool graphs, sample tools, create dialogue plans, and synthesize dialogues, follow these steps:

1.  Ensure you are in the **main project root directory** (e.g., `ToolWeave`, the one containing the `src`, `scripts`, and `data` folders). All `python -m ...` commands should be run from this directory.

2. To run the API synthesizer:
```bash
python -m scripts.domain_api_synthesizer \
  --domains_file domains.txt \
  --output_dir output/apis/ \
  --connection_mode llm \
  --max_workers 20
```
- The `connection_mode` argument is used to control what algorithm is used to create connections between APIs. `llm` queries an LLM for each parameter pair while `semantic` uses embedding-based similarity to determine connections.
- Adding a `#` before domain names in the domain file will skip API synthesis for those domains.
- Detailed info about all supported arguments can be found by running `python -m scripts.domain_api_synthesizer --help`.

3. To generate goals for each domain:
```bash
python -m scripts.complex_goal_generator \
  --graph_path output/apis/agriculture/graph_agriculture.pkl \
  --api_definitions_path output/apis/agriculture/sdk_agriculture.json \
  --output_goals_file_path output/goals/agriculture.jsonl \
  --synthetic_apis --algorithm all_patterns
```
- The hyperparameters for various algorithms that perform traversal over the tool graph have already been set to optimal values.
- If you want to change them, take a look at all the possible arguments to the script by running `python -m scripts.complex_goal_generator --help`.

4. To generate dialogue plans:
```bash
python -m scripts.complex_dialogue_planner \
  --goals_file_path output/goals/agriculture.jsonl \
  --api_definitions_path output/apis/agriculture/sdk_agriculture.json \
  --prompts_dir prompts/prompts_for_partitioning_goal \
  --output_path output/plans/agriculture.jsonl \
  --num_plan_variants 2 \
  --max_fan_out_patterns -1 \
  --synthetic_apis --max_workers 60
```
- More info about all supported arguments can be found by running `python -m scripts.complex_dialogue_planner --help`.

5. To generate dialogues:
```bash
python -m scripts.generate_dialogues \
  --plans_file output/plans/agriculture.jsonl \
  --tools_list_path output/apis/agriculture/sdk_agriculture.json \
  --output_file output/dialogues/agriculture.jsonl \
  --generation_strategy chat --prompt_config_file prompt_configs/dialogue_generators/chat.yml \
  --synthetic_apis --max_workers 60
```
- The options available for generation strategy are `generate` and `chat` which use all the models in `model.generate` and `model.chat` modes, respectively.
- Information about other relevant parameters can be found by running `python -m scripts.generate_dialogues --help`.

6. Due to the inherent randomness in LLM generations, few multi-step dialogues might inadvertently end up being false multi-step dialogues. To address this, we provide a script that converts such false multi-step sequences into parallel tool call sequences:
```bash
python -m scripts.convert_false_multi_step_to_parallel \
  --input_dir output/dialogues/
```
- **Note**: This script modifies dialogues IN-PLACE. It is recommended to run this script at this point (before the following post-processing steps) to avoid repeated conversions.

7. To perform dialogue refinement:
```bash
python -m scripts.refine_dialogues \
  --dialogues_file output/dialogues/agriculture.jsonl \
  --output_file output/refined_dialogues/agriculture.jsonl \
  --refinements paraphrase \
  --generation_strategy chat --prompt_config_file prompt_configs/dialogue_refiners/chat.yml \
  --max_workers 60
```
- The `--refinements` argument takes in various types of refinements to apply to the dialogues. For now, only `paraphrase` is available.
- For more information on all supported arguments, you can run `python -m scripts.refine_dialogues --help`.

8. To compute overall dialogue statistics:
```bash
python -m scripts.compute_dialogue_statistics \
  --input_dir output/refined_dialogues/ \
  --output_file output/dialogue_statistics.jsonl
```
- Each line in the output file will contain statistics for a single domain whose dialogue file is located in the `--input_dir`.
- The final line in the file contains the overall statistics for all domains.

9. To compute true multi-step statistics (a much leaner version of overall statistics that focuses on multi-step turns):
```bash
python -m scripts.compute_true_multi_step_stats \
  --input_dir output/refined_dialogues/ \
  --output_file output/true_multi_step_stats.json
```
- There is a single json file with true vs total multi-step turns for each domain along with overall statistics.

10. To run the entire pipeline in one go:
```bash
python -m scripts.generate_synthetic_data \
  --output_dir output/ \
  --generate_apis \
  --generate_goals_and_plans \
  --generate_dialogues \
  --refine_dialogues \
  --compute_dialogue_stats
```
- Note that this main script also runs the false multi-step to parallel conversion step mentioned in point 6.

11. To add missing function data after running the full pipeline:
```bash
python -m scripts.add_missing_functions_to_dialogues \
  --input_dir output/refined_dialogues/ \
  --output_dir output/missing_func_dialogues/ \
  --missing_func_fraction 0.15
```

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## 📙 Citation

If you use this repo or our paper in your research, please cite:

```bibtex
@article{khandelwal2026toolweave,
  title={ToolWeave: Structured Synthesis of Complex Multi-Turn Tool-Calling Dialogues},
  author={Khandelwal, Dinesh and Punnavajhala, Gnana Prakash and Bhargav, GPS and Pandey, Gaurav and Joshi, Sachin and Karanam, Hima and Raghu, Dinesh},
  journal={arXiv preprint arXiv:2605.12521},
  year={2026}
}
```
