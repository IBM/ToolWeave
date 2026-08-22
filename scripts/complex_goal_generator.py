#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import random
import re
import sys
from concurrent.futures import as_completed, CancelledError, ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from pprint import pformat

import networkx as nx
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('complex_goal_generator')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

COMPLEX_PROMPTS_DIR = os.path.join(project_root, "prompts", "prompts_for_goals", "complex_patterns")
BEAM_SEARCH_PROMPTS_DIR = os.path.join(project_root, "prompts", "prompts_for_goals", "beam_search")


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


def load_prompt(prompt_dir: str, filename: str) -> str:
    """Load a prompt template from a file.

    Args:
        prompt_dir: Directory containing the prompt template files
        filename: Name of the prompt template file to load

    Returns:
        Contents of the prompt template file as a string

    Raises:
        FileNotFoundError: If the prompt file does not exist at the specified path
    """
    try:
        with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
            return f.read()

    except FileNotFoundError:
        logger.error(f"Error: Prompt file not found at {os.path.join(prompt_dir, filename)}")
        raise


def extract_json_from_response(raw_text: str) -> str | None:
    """Extract JSON content from an LLM response that may contain markdown code blocks.

    Args:
        raw_text: Raw text response from the LLM that may contain JSON

    Returns:
        Extracted JSON string (object or array), or None if no valid JSON structure found
    """
    if not isinstance(raw_text, str): return None
    text = raw_text.strip()

    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
    if match:
        text = match.group(1).strip()
    else:
        text = raw_text

    first_brace, first_bracket = text.find('{'), text.find('[')
    if first_brace == -1 and first_bracket == -1: return None

    start_char = '{' if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket) else '['
    end_char = '}' if start_char == '{' else ']'

    start_index = text.find(start_char)

    open_count = 0
    for i in range(start_index, len(text)):
        if text[i] == start_char:
            open_count += 1

        elif text[i] == end_char:
            open_count -= 1

        if open_count == 0: return text[start_index:i + 1]

    return None


def create_tool_summary_with_io(tool_definition: dict) -> str:
    """Create a formatted string summary of a tool's inputs and outputs.

    Generates a human-readable summary including the tool's name, description,
    input parameters, and output parameters for use in LLM prompts.

    Args:
        tool_definition: Dictionary containing the tool's API definition with the following fields:
            - 'name': The name of the tool
            - 'description': A brief description of the tool's functionality
            - 'parameters': A dictionary defining the tool's input parameters (under 'properties')
            - 'results': A dictionary defining the tool's output parameters (under 'properties')

    Returns:
        Formatted multi-line string summarizing the tool's I/O specification.
        Format: "- Tool: <name>\n  Description: <desc>\n  Inputs: <list>\n  Outputs: <list>"
    """
    name = tool_definition.get("name", "N/A")
    description = tool_definition.get("description", "No description.")

    try:
        inputs = list(tool_definition.get("parameters", {}).get("properties", {}).keys())
        inputs_str = f"Inputs: {inputs if inputs else ['None']}"

    except Exception:
        inputs_str = "Inputs: Error"

    try:
        results_schema = tool_definition.get("results", {}).get("properties", {})
        outputs = list(results_schema.keys())

        outputs_str = f"Outputs: {outputs if outputs else ['None']}"

    except Exception:
        outputs_str = "Outputs: Error"

    return f"- Tool: {name}\n  Description: {description}\n  {inputs_str}\n  {outputs_str}"


def load_api_definitions(api_definitions_path: str, synthetic_apis: bool = True) -> dict[str, dict]:
    """Load and parse API definitions from a JSON file.

    Args:
        api_definitions_path: Path to the JSON file containing API definitions
        synthetic_apis: If True, expects API definitions under an 'apis' key in the JSON.
            If False, expects a flat list of API definitions. Defaults to True.

    Returns:
        Dictionary mapping API function names to their function definitions.
        Only includes entries with valid 'function' and 'name' fields.
    """
    with open(api_definitions_path, 'r') as f:
        api_list = json.load(f)
        if synthetic_apis:
            api_list = api_list.get("apis", api_list)

    return {
        api_obj["function"]["name"]: api_obj["function"]
        for api_obj in api_list
        if "function" in api_obj and "name" in api_obj["function"]
    }


def load_param_connection_map(api_definitions_path: str, synthetic_apis: bool = True) -> dict:
    """Load the parameter connection map for synthetic APIs.

    Creates a reverse lookup map that connects target API parameters to their potential
    source API parameters, enabling efficient discovery of parameter dataflow during
    goal generation.

    Args:
        api_definitions_path: Path to the JSON file containing API SDK with connection map
        synthetic_apis: If True, loads and processes the connection map from the API SDK.
            If False, returns an empty dictionary. Defaults to True.

    Returns:
        Dictionary mapping (target_api, target_param) tuples to lists of 
        (source_api, source_param) tuples. Returns empty dict if synthetic_apis is False.

    Note:
        The map is stored in reverse direction (target -> sources) to facilitate
        looking up available source parameters when processing a target tool parameter.
    """
    if not synthetic_apis:
        return {}
    
    with open(api_definitions_path, 'r') as f:
        api_sdk = json.load(f)

    param_connection_map = {}
    for connection in api_sdk.get("connection_map", []):
        source_api = connection.get("source_api")
        target_api = connection.get("target_api")
        source_param = connection.get("source_param")
        target_param = connection.get("target_param")

        if source_api and target_api and source_param and target_param:
            # We are setting the lookup in reverse direction because we'll know the 
            # current tool and param while sampling relevant params and we'll need to
            # look up the known params from previous tools which we won't have access to
            if (target_api, target_param) not in param_connection_map:
                param_connection_map[(target_api, target_param)] = []

            param_connection_map[(target_api, target_param)].append((source_api, source_param))

    return param_connection_map


def load_tool_graphs(tool_graph_path: str, synthetic_apis: bool = True) -> nx.DiGraph:
    """Load a tool graph from a pickled file.

    Args:
        tool_graph_path: Path to the pickled graph file
        synthetic_apis: If True, expects the graph as the only object in the pickle.
            If False, expects a tuple where the second element is the graph.
            Defaults to True.

    Returns:
        NetworkX DiGraph representing the tool dependency graph

    Raises:
        ValueError: If the loaded object is not a DiGraph, or if the expected structure
            doesn't match the synthetic_apis parameter
    """
    with open(tool_graph_path, 'rb') as f:
        if synthetic_apis:
            directed_graph = pickle.load(f)
        else:
            _, directed_graph = pickle.load(f)

    if not isinstance(directed_graph, nx.DiGraph):
        if synthetic_apis:
            raise ValueError("The loaded graph is not a DiGraph")
        else:
            raise ValueError("The second element in the pickle file is not a DiGraph")

    return directed_graph


def select_top_start_nodes_by_connectivity(
    graph: nx.DiGraph, api_mapping: dict[str, dict],
    top_k: int, min_children: int, max_children: int,
) -> list[tuple[str, list[str]]]:
    """Select top candidate start nodes based on their graph connectivity.

    Identifies and ranks nodes in the graph by their number of outgoing edges (successors),
    filtering by specified child count constraints. Useful for selecting diverse and
    well-connected starting points for goal generation.

    Args:
        graph: NetworkX directed graph of tool relationships
        api_mapping: Dictionary mapping tool names to their API definitions
        top_k: Number of top candidates to return. Use -1 to return all candidates.
        min_children: Minimum number of successors required for a node to be a candidate
        max_children: Maximum number of successors allowed for a node to be a candidate

    Returns:
        List of tuples, where each tuple contains:
            - node name
            - list of child node names
        Sorted in descending order by number of children. Limited to top_k items
        unless top_k is -1.
    """
    candidates = []
    for node in graph.nodes:
        if node not in api_mapping:
            continue
        children = list(graph.successors(node))
        if min_children <= len(children) <= max_children:
            candidates.append((node, children))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    if top_k == -1:
        return candidates
    return candidates[:top_k]


class ComplexGoalGenerator:
    """Complex goal generator for synthesizing diverse multi-tool interaction patterns.

    This class implements multiple algorithms for generating synthetic user goals that span
    multiple tools in a tool dependency graph. It supports various interaction patterns
    including linear chains, fan-out/fan-in, and conditional branching.

    The generator uses a combination of:
    - Graph traversal algorithms (beam search, path enumeration)
    - LLM-based goal synthesis and scoring
    - Semantic similarity analysis for parameter dataflow
    - Maximum Marginal Relevance (MMR) for diversity

    Attributes:
        graph: NetworkX directed graph representing tool relationships and dependencies
        apis: Dictionary mapping tool names to their API definitions
        llm: LLM instance (VLLMClient or WatsonxLLM) for generating and scoring goals
        use_cache: Whether to cache generated goals and scores for efficiency
        synthetic_apis: Whether working with synthetic vs real-world APIs
        param_connection_map: Reverse lookup map from (target_api, target_param) to
            list of (source_api, source_param) tuples
        tool_descriptions: Dictionary mapping tool names to their descriptions
        all_tool_descs_str: Formatted string of all tool descriptions
        embedding_model: SentenceTransformer model for semantic similarity computation
        goal_cache: Cache of generated goals keyed by tool path tuples (if use_cache=True)
        score_cache: Cache of goal scores keyed by tool path tuples (if use_cache=True)
        param_embeddings: Pre-computed embeddings for all tool parameters (real APIs only)
        embedding_cache_path: Path to cached parameter embeddings file (real APIs only)

    Supported Goal Patterns:
        - Linear chains: Sequential tool invocations discovered via beam search
        - Fan-out/Fan-in: Parallel tool branches converging to a single endpoint
        - Conditional: If/else branching based on output parameter conditions
        - Maximal paths: Complete paths from start node to leaf nodes
    """
    def __init__(
        self, tool_graph: nx.DiGraph, api_mapping: dict[str, dict],
        param_connection_map: dict[tuple[str, str], tuple[str, str]],
        llm_instance: VLLMClient | WatsonxLLM, use_cache: bool = True,
        synthetic_apis: bool = True, embedding_cache_path: str | None = None
    ) -> None:
        """Initialize the ComplexGoalGenerator with tool graph, APIs, and LLM configuration.

        Sets up the goal generation system by loading tool relationships, API definitions,
        and initializing caching and embedding systems. For real-world APIs, pre-computes
        or loads parameter embeddings for semantic similarity analysis.

        Args:
            tool_graph: NetworkX directed graph representing tool dependencies and relationships.
                Each edge contains a 'weight' attribute for synthetic APIs.
            api_mapping: Dictionary mapping tool names to their API definitions.
                Each definition should contain 'name', 'description', 'parameters', and 'results' fields.
            param_connection_map: Reverse lookup map from (target_api, target_param) tuples
                to lists of (source_api, source_param) tuples. Used to identify 
                valid parameter dataflow connections.
            llm_instance: Language model instance (VLLMClient or WatsonxLLM) used for
                generating goal text and scoring goal quality.
            use_cache: Whether to enable in-memory caching of generated goals and scores
                to avoid redundant LLM calls. Defaults to True.
            synthetic_apis: Whether the APIs are synthetic (True) or real-world (False).
                Affects how semantic dataflow is calculated and whether embeddings are used. 
                Defaults to True.
            embedding_cache_path: Path to save/load pre-computed parameter embeddings for
                real APIs. Only used when synthetic_apis=False. If the file exists, 
                embeddings are loaded; otherwise, they are computed and saved. Defaults to None.

        Raises:
            FileNotFoundError: If embedding_cache_path parent directory cannot be created

        Note:
            For real-world APIs, the initialization process includes pre-computing or loading
            parameter embeddings using SentenceTransformer ('all-MiniLM-L6-v2'). This can
            take significant time for large API sets on first run, but embeddings are cached
            for future use.
        """
        self.graph = tool_graph
        self.apis = api_mapping
        self.llm = llm_instance
        self.use_cache = use_cache
        self.synthetic_apis = synthetic_apis
        self.param_connection_map = param_connection_map

        self.tool_descriptions = {k: v.get("description", "") for k, v in self.apis.items()}
        self.all_tool_descs_str = "\n".join([f"- {name}: {desc}" for name, desc in self.tool_descriptions.items()])

        # This model is also used to create embeddings for goal texts
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        if self.use_cache:
            self.goal_cache = {}
            self.score_cache = {}

        if not self.synthetic_apis:
            self.embedding_cache_path = Path(embedding_cache_path)

            if self.embedding_cache_path.exists():
                logger.info(f"Loading pre-computed parameter embeddings from {self.embedding_cache_path}...")
                with open(self.embedding_cache_path, 'rb') as f:
                    self.param_embeddings = pickle.load(f)

            else:
                logger.info("No embedding cache found. Pre-computing parameter embeddings...")
                self.param_embeddings = self._precompute_parameter_embeddings()

                logger.info(f"Saving embeddings to {self.embedding_cache_path} for future runs...")
                self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.embedding_cache_path, 'wb') as f:
                    pickle.dump(self.param_embeddings, f)

            logger.info("Embeddings are ready.")


    def _simulate_goal_update(self, path: list[str]) -> str:
        """Generate a natural language goal description for a given tool path using the LLM.

        Args:
            path: Ordered list of tool names representing a tool invocation sequence

        Returns:
            Natural language goal text describing what the user wants to accomplish
            using the specified tool sequence
        """
        if self.use_cache:
            path_tuple = tuple(path)
            if path_tuple in self.goal_cache:
                return self.goal_cache[path_tuple]

        prompt_template = load_prompt(BEAM_SEARCH_PROMPTS_DIR, "generate_goal_prompt.txt")
        tool_list = [f"- {tool}: {self.tool_descriptions.get(tool, 'No description')}" for tool in path]

        tools_str = "\n".join(tool_list)
        prompt = prompt_template.format(tools_str=tools_str)

        if isinstance(self.llm, VLLMClient):
            message = [{'role': 'user', 'content': prompt}]
            llm_response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
        else:
            llm_response = self.llm.invoke(prompt, use_chat_mode=False)

        goal_text = llm_response.content.strip()

        if self.use_cache:
            self.goal_cache[tuple(path)] = goal_text

        return goal_text


    def _precompute_parameter_embeddings(self) -> dict:
        """Pre-compute semantic embeddings for all input and output parameters across all tools.

        Returns:
            Nested dictionary with structure:
            {
                tool_name: {
                    'inputs': {param_name: embedding_vector, ...},
                    'outputs': {param_name: embedding_vector, ...}
                },
                ...
            }

        Note:
            Only called for real-world APIs (synthetic_apis=False). Embeddings are cached
            to disk to avoid recomputation on subsequent runs. Uses 'all-MiniLM-L6-v2' model.
        """
        param_embeddings = {}
        key_map = {'inputs': 'parameters', 'outputs': 'results'}

        for tool_name, tool_def in tqdm(self.apis.items(), desc="Computing Parameter Embeddings"):
            param_embeddings[tool_name] = {'inputs': {}, 'outputs': {}}

            for param_type in key_map.keys():
                params = tool_def.get(key_map[param_type], {}).get("properties", {})
                for param_name, param_def in params.items():
                    desc = param_def.get('description', '')
                    text_to_embed = f"{param_name}: {desc}"
                    embedding = self.embedding_model.encode(text_to_embed)
                    param_embeddings[tool_name][param_type][param_name] = embedding

        return param_embeddings


    def _calculate_semantic_dataflow(self, path: list[str], threshold: float | int) -> float:
        """Calculate the semantic dataflow score for a tool path.

        Measures how well consecutive tools in the path are connected by analyzing whether
        output parameters from one tool can semantically or structurally flow into input
        parameters of the next tool.

        For real APIs: Uses cosine similarity between parameter embeddings.
        For synthetic APIs: Uses edge weights from the tool graph.

        Args:
            path: Ordered list of tool names representing a tool invocation sequence
            threshold: For real APIs (float, 0-1): minimum cosine similarity to count as
                a strong connection. For synthetic APIs (int, >=0): minimum edge weight 
                to count as a strong connection.

        Returns:
            Float between 0.0 and 1.0 representing the ratio of strong connections
            to total consecutive tool pairs in the path. Returns 0.0 for paths with
            length <= 1.

        Raises:
            AssertionError: If threshold is out of valid range for the API type
        """
        path_length = len(path)
        if path_length <= 1: return 0.0

        strong_connections = 0
        for i in range(path_length - 1):
            if not self.synthetic_apis:
                assert 0 <= threshold <= 1, "Similarity threshold must be between 0 and 1"
                threshold = float(threshold)

                outputs_A = self.param_embeddings.get(path[i], {}).get('outputs', {})
                inputs_B = self.param_embeddings.get(path[i + 1], {}).get('inputs', {})
                if not outputs_A or not inputs_B: continue

                if any(
                    util.cos_sim(out_emb, in_emb).item() > threshold
                    for out_emb in outputs_A.values() for in_emb in inputs_B.values()
                ):
                    strong_connections += 1

            else:
                assert isinstance(threshold, int) and threshold >= 0, \
                    "For synthetic APIs, threshold must be a non-negative integer " \
                    "which denotes the threshold for edge weights."

                if threshold == 0:
                    strong_connections += 1
                    continue

                edge_weight = self.graph.get_edge_data(path[i], path[i + 1], {}).get('weight', 0)
                if edge_weight >= threshold:
                    strong_connections += 1

        return strong_connections / (path_length - 1)


    def _score_goal(self, goal: str, path: list[str], config: dict) -> dict:
        """Score a generated goal using LLM-based evaluation and semantic dataflow analysis.

        Evaluates the quality of a generated goal by combining multiple scoring components:
        - LLM ratings for coherence and relevance (scaled to [-2, 2])
        - Semantic dataflow score (0.0 to 1.0)
        - Path length bonus

        Args:
            goal: Natural language goal text to evaluate
            path: Ordered list of tool names that the goal describes
            config: Configuration dictionary containing scoring weights and thresholds:
                - 'llm_rating_weight': Weight for LLM coherence/relevance ratings
                - 'dataflow_weight': Weight for semantic dataflow score
                - 'length_bonus_weight': Weight applied per tool in path
                - 'similarity_threshold' or 'edge_weight_threshold': Dataflow threshold
                - 'synthetic_apis': Whether using synthetic or real APIs

        Returns:
            Dictionary with score breakdown:
            {
                'final_score': float (>= 0.0),
                'coherence': int (-2 to 2),
                'relevance': int (-2 to 2),
                'dataflow': float (0.0 to 1.0),
                'length_bonus': float
            }
        """
        path_tuple = tuple(path)
        if self.use_cache and path_tuple in self.score_cache:
            return self.score_cache[path_tuple]

        prompt_template = load_prompt(BEAM_SEARCH_PROMPTS_DIR, "score_goal_prompt.txt")
        tool_summaries = [create_tool_summary_with_io(self.apis[tool]) for tool in path if tool in self.apis]
        tools_str = "\n\n".join(tool_summaries)
        prompt = prompt_template.format(tools_str=tools_str, goal=goal)

        if isinstance(self.llm, VLLMClient):
            message = [{'role': 'user', 'content': prompt}]
            response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
        else:
            response = self.llm.invoke(prompt, use_chat_mode=False)
        json_str = extract_json_from_response(response.content)

        try:
            scores = json.loads(json_str) if json_str else {}
            # Convert ratings to [-2, 2] range to have negative final scores to denote bad goals
            coherence_rating = scores.get('coherence_rating', 3) - 3
            relevance_rating = scores.get('relevance_rating', 3) - 3

            coherence_rating = max(-2, min(2, coherence_rating))
            relevance_rating = max(-2, min(2, relevance_rating))

        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning(f"Failed to parse JSON from goal scorer LLM for path: {pformat(path)}")
            coherence_rating = 0
            relevance_rating = 0

        if not config['synthetic_apis']:
            semantic_dataflow_score = self._calculate_semantic_dataflow(path, config['similarity_threshold'])
        else:
            semantic_dataflow_score = self._calculate_semantic_dataflow(path, config['edge_weight_threshold'])

        length_bonus = len(path) * config['length_bonus_weight']
        final_score = (coherence_rating + relevance_rating) * config['llm_rating_weight'] + \
                        semantic_dataflow_score * config['dataflow_weight'] + length_bonus

        score_breakdown = {
            "final_score": max(0.0, final_score),
            "coherence": coherence_rating,
            "relevance": relevance_rating,
            "dataflow": semantic_dataflow_score,
            "length_bonus": length_bonus
        }
        logger.info(f"Path: {pformat(path)}\nScores: {pformat(score_breakdown)}\n")

        if self.use_cache:
            self.score_cache[path_tuple] = score_breakdown

        return score_breakdown


    def _call_llm_for_text(self, prompt_template_name: str, replacements: dict) -> str:
        """Invoke the LLM with a prompt template and return the text response.

        Args:
            prompt_template_name: Name of the prompt template file in COMPLEX_PROMPTS_DIR
            replacements: Dictionary of key-value pairs to format the prompt template

        Returns:
            Stripped text content from the LLM response
        """
        prompt_template = load_prompt(COMPLEX_PROMPTS_DIR, prompt_template_name)
        prompt = prompt_template.format(**replacements)
        if isinstance(self.llm, VLLMClient):
            message = [{'role': 'user', 'content': prompt}]
            response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
        else:
            response = self.llm.invoke(prompt, use_chat_mode=False)
        return response.content.strip()


    def _call_llm_for_json(self, prompt_template_name: str, replacements: dict) -> dict | None:
        """Invoke the LLM with a prompt template and return the parsed JSON response.

        Args:
            prompt_template_name: Name of the prompt template file in COMPLEX_PROMPTS_DIR
            replacements: Dictionary of key-value pairs to format the prompt template

        Returns:
            Parsed JSON dictionary if extraction and parsing succeed, None otherwise
        """
        prompt_template = load_prompt(COMPLEX_PROMPTS_DIR, prompt_template_name)
        prompt = prompt_template.format(**replacements)
        if isinstance(self.llm, VLLMClient):
            message = [{'role': 'user', 'content': prompt}]
            response = self.llm.invoke(prompt_or_messages=message, use_chat_mode=True)
        else:
            response = self.llm.invoke(prompt, use_chat_mode=False)
        json_str = extract_json_from_response(response.content)

        try:
            return json.loads(json_str) if json_str else None
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM for prompt {prompt_template_name}")
            return None


    def _filter_subsumed_paths(self, results: list[dict]) -> list[dict]:
        """Filter out paths that are strict prefixes of longer paths in the result set.

        Args:
            results: List of result dictionaries, each containing a 'path' key with
                a list of tool names

        Returns:
            Filtered list of result dictionaries containing only maximal paths
        """
        path_tuples = {tuple(res['path']) for res in results}
        filtered_results = []
        for res in results:
            current_path = tuple(res['path'])
            is_prefix_of_another = any(
                len(current_path) < len(other_path) and other_path[:len(current_path)] == current_path
                for other_path in path_tuples
            )
            if not is_prefix_of_another:
                filtered_results.append(res)

        logger.info(f"Filtered out {len(results) - len(filtered_results)} subsumed paths.")
        return filtered_results


    def _select_top_k_with_mmr(self, candidates: list[dict], top_k: int, lambda_param: float) -> list[dict]:
        """Select top-k candidates using Maximum Marginal Relevance (MMR) for diversity.

        Implements MMR to balance relevance (final score) and diversity (semantic
        dissimilarity) when selecting candidates. Iteratively selects candidates
        that maximize a weighted combination of their score and their distance from
        already-selected candidates.

        Args:
            candidates: List of candidate dictionaries, each containing:
                - 'score_breakdown': dict with 'final_score' key
                - 'goal_txt_embedding': embedding vector for the goal text
            top_k: Number of candidates to select
            lambda_param: MMR trade-off parameter (0.0 to 1.0):
                - 1.0 = pure relevance (highest scores)
                - 0.0 = pure diversity (most dissimilar)
                - intermediate values balance both

        Returns:
            List of selected candidate dictionaries (up to top_k items)
        """
        if not candidates or top_k <= 0: return []
        if len(candidates) <= top_k: return candidates

        candidates.sort(key=lambda x: x['score_breakdown']['final_score'], reverse=True)
        best_candidate = candidates.pop(0)
        selected_candidates = [best_candidate]
        selected_embeddings = [best_candidate['goal_txt_embedding']]

        while len(selected_candidates) < top_k and candidates:
            mmr_scores = [
                lambda_param * c['score_breakdown']['final_score'] - (1 - lambda_param) * max(
                        util.cos_sim(c['goal_txt_embedding'], s_emb).item()
                        for s_emb in selected_embeddings
                    )
                for c in candidates
            ]

            best_idx = mmr_scores.index(max(mmr_scores))
            next_best_candidate = candidates.pop(best_idx)
            selected_candidates.append(next_best_candidate)
            selected_embeddings.append(next_best_candidate['goal_txt_embedding'])

        return selected_candidates


    def _flatten_schema_properties(self, params_obj: dict, prefix: str = '') -> list[str]:
        """Recursively flatten nested JSON schema properties into a list of dot-notation paths.

        Args:
            params_obj: Dictionary representing a JSON schema or its properties
            prefix: Current path prefix for nested recursion (internal use)

        Returns:
            List of flattened parameter paths in dot notation. Examples:
            - 'user.name' for nested objects
            - 'items[].price' for array items
            - 'config.settings.enabled' for deeply nested properties
        """
        flattened_params = []

        if not params_obj or not isinstance(params_obj, dict):
            return flattened_params

        if "properties" in params_obj:
            return self._flatten_schema_properties(params_obj["properties"], prefix)

        for key, value in params_obj.items():
            if isinstance(value, dict):
                if value.get("type") == "object" and "properties" in value:
                    nested_params = self._flatten_schema_properties(value["properties"], prefix + key + '.')
                    flattened_params.extend(nested_params)

                elif value.get("type") == "array" and isinstance(value.get("items"), dict) and "properties" in value.get("items", {}):
                    nested_params = self._flatten_schema_properties(value["items"]["properties"], prefix + key + '[].')
                    flattened_params.extend(nested_params)

                elif "properties" in value:
                    nested_params = self._flatten_schema_properties(value["properties"], prefix + key + '.')
                    flattened_params.extend(nested_params)

                else:
                    flattened_params.append(prefix + key)

            else:
                flattened_params.append(prefix + key)

        return flattened_params


    ### --- PATTERN GENERATION METHODS ---


    def generate_goals_with_beam_search(self, start_tool: str, config: dict) -> list[dict]:
        """Generate linear chain goals using beam search with MMR and advanced scoring.

        Implements a beam search algorithm that explores tool paths by iteratively expanding
        the most promising candidates. Uses Maximum Marginal Relevance (MMR) to balance
        relevance (score) and diversity (semantic dissimilarity) when selecting beams at
        each depth level.

        Args:
            start_tool: Name of the tool to begin the search from
            config: Configuration dictionary containing:
                - 'max_depth': Maximum path length to explore
                - 'beam_width': Number of top candidates to keep at each depth
                - 'min_score_threshold': Minimum score for a path to be considered
                - 'mmr_lambda': MMR trade-off parameter (0.0-1.0)
                - 'max_workers': Number of parallel workers for exploration
                - Plus scoring configuration (weights, thresholds, etc.)

        Returns:
            List of goal dictionaries, each containing:
            {
                'type': 'linear_chain',
                'path': list of tool names,
                'goal': natural language goal text,
                'score_breakdown': scoring details dict
            }
            Sorted by final_score in descending order, with subsumed paths filtered out.
        """
        logger.info(f"--- Starting Enhanced Beam Search for '{start_tool}' ---")

        initial_goal = self._simulate_goal_update([start_tool])
        score_breakdown = self._score_goal(initial_goal, [start_tool], config)

        initial_entry = {
            "path": [start_tool],
            "score_breakdown": score_breakdown,
            "goal": initial_goal,
            "type": "linear_chain"
        }
        beams = [initial_entry]
        discovered_paths = {tuple([start_tool]): initial_entry}

        for depth in range(1, config['max_depth']):
            all_candidates = []

            def explore_neighbor(beam: dict, neighbor: str) -> dict | None:
                """Explore a neighboring tool by extending the current beam's path and evaluating the new goal.

                Args:
                    beam: Current beam dictionary containing 'path' and other info
                    neighbor: Name of the neighboring tool to explore

                Returns:
                    A new candidate dictionary if the extended path meets the score threshold, None otherwise
                """
                if neighbor in beam['path']:
                    return None

                new_path = beam['path'] + [neighbor]
                goal_text = self._simulate_goal_update(new_path)
                score_breakdown = self._score_goal(goal_text, new_path, config)

                if score_breakdown['final_score'] > config['min_score_threshold']:
                    goal_txt_embedding = self.embedding_model.encode(goal_text)
                    return {
                        "path": new_path,
                        "goal": goal_text,
                        "score_breakdown": score_breakdown,
                        "goal_txt_embedding": goal_txt_embedding,
                        "type": "linear_chain"
                    }
                return None

            tasks = []
            for beam in beams:
                last_tool = beam['path'][-1]
                for neighbor in self.graph.successors(last_tool):
                    tasks.append((beam, neighbor))

            max_workers = config.get('max_workers', len(tasks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_args = {
                    executor.submit(explore_neighbor, beam, neighbor): (beam, neighbor)
                    for beam, neighbor in tasks
                }

                for future in as_completed(future_to_args):
                    result = future.result()
                    if result is not None:
                        all_candidates.append(result)

            if not all_candidates:
                logger.warning(f"No new candidates found at depth {depth + 1}. Stopping search.")
                break

            beams = self._select_top_k_with_mmr(
                all_candidates, config['beam_width'], config['mmr_lambda']
            )
    
            # Remove 'goal_txt_embedding' key from each beam for space efficiency
            for b in beams:
                b.pop('goal_txt_embedding', None)

            for beam in beams:
                discovered_paths[tuple(beam['path'])] = beam

        final_discovered_results = list(discovered_paths.values())
        final_maximal_results = self._filter_subsumed_paths(final_discovered_results)
        final_maximal_results.sort(key=lambda x: x['score_breakdown']['final_score'], reverse=True)
        return final_maximal_results


    def generate_fan_out_goals(self, start_node: str, config: dict) -> list[dict]:
        """Generate fan-out/fan-in pattern goals where parallel branches converge.

        Identifies scenarios where a start tool branches into multiple parallel tool paths
        that eventually converge at a common successor tool. Optionally enforces strict
        fan-out where each branch tool must provide values for different parameters of the
        end tool.

        Args:
            start_node: Name of the starting tool that begins the fan-out
            config: Configuration dictionary containing:
                - 'max_branch': Maximum number of parallel branches to consider
                - 'strict_fan_out': If True, requires each branch to contribute to
                    different parameters of the end tool
                - 'max_workers': Number of parallel workers
                - Plus scoring configuration

        Returns:
            List of fan-out goal dictionaries, each containing:
            {
                'type': 'fan_out_fan_in',
                'start_tool': start node name,
                'branch_tools': list of parallel branch tool names,
                'end_tool': convergence point tool name,
                'goal_text': natural language goal description,
                'score_breakdown': scoring details dict
            }
        """
        logger.info(f"--- Searching for Fan-Out, Fan-In patterns from: {start_node} ---")

        successors = list(self.graph.successors(start_node))
        if len(successors) < 2: return []

        def process_branch_combination(branch_nodes):
            try:
                successor_sets = [set(list(self.graph.successors(bn))) for bn in branch_nodes]
                common_successors = set.intersection(*successor_sets)

            except (nx.NetworkXError, nx.NodeNotFound):
                return None

            if common_successors:
                if config['strict_fan_out']:
                    suitable_end_node = None
                    for candidate_end_node in common_successors:
                        end_node_api_def = self.apis.get(candidate_end_node, {})
                        input_params = self._flatten_schema_properties(end_node_api_def.get("parameters", {}))

                        has_multiple_prev_tools = False
                        for param_name in input_params:
                            if len(self.param_connection_map.get((candidate_end_node, param_name), [])) > 1:
                                has_multiple_prev_tools = True
                                break

                        if not has_multiple_prev_tools:
                            suitable_end_node = candidate_end_node
                            break

                    if not suitable_end_node:
                        logger.warning(f"No suitable end node found for branch nodes: {branch_nodes}")
                        return None

                    end_node = suitable_end_node

                else:
                    end_node = random.choice(list(common_successors))

                branch_descs = "\n".join([f"- {b}: {self.tool_descriptions.get(b, '')}" for b in branch_nodes])
                replacements = {
                    "start_tool_desc": self.tool_descriptions.get(start_node, ''),
                    "branch_tools_descs": branch_descs,
                    "end_tool_desc": self.tool_descriptions.get(end_node, '')
                }

                goal_text = self._call_llm_for_text("generate_fan_out_goal_prompt.txt", replacements)
                score_breakdown = self._score_goal(
                    goal_text, [start_node] + list(branch_nodes) + [end_node], config
                )

                return {
                    "type": "fan_out_fan_in",
                    "start_tool": start_node,
                    "branch_tools": list(branch_nodes),
                    "end_tool": end_node,
                    "goal_text": goal_text,
                    "score_breakdown": score_breakdown
                }
            return None

        tasks = []
        for i in range(2, min(len(successors), config['max_branch']) + 1):
            tasks.extend((combinations(successors, i)))

        generated_goals = []
        with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
            future_to_result = {
                executor.submit(process_branch_combination, branch_nodes): branch_nodes
                for branch_nodes in tasks
            }

            for future in as_completed(future_to_result):
                result = future.result()
                if result:
                    generated_goals.append(result)

        return generated_goals


    def generate_conditional_goals(self, start_node: str, config: dict) -> list[dict]:
        """Generate conditional (if/else) pattern goals based on output parameter values.

        Creates goals where the execution path branches based on a condition evaluated on
        the start tool's output parameters. Filters output parameters to only include
        suitable decision variables (non-ID fields with primitive types).

        Args:
            start_node: Name of the tool whose outputs determine the condition
            config: Configuration dictionary containing:
                - 'max_goals_per_node': Maximum conditional goals to generate
                - 'max_workers': Number of parallel workers
                - Plus scoring configuration

        Returns:
            List of conditional goal dictionaries, each containing:
            {
                'type': 'conditional',
                'start_tool': start node name,
                'if_branch_tool': tool to invoke if condition is true,
                'else_branch_tool': tool to invoke if condition is false,
                'condition_details': dict with condition specification,
                'goal_text': natural language goal description,
                'score_breakdown': scoring details dict
            }
        """
        logger.info(f"--- Searching for Conditional patterns from: {start_node} ---")
        successors = list(self.graph.successors(start_node))
        if len(successors) < 2: return []

        start_node_results_schema = self.apis.get(start_node, {}).get("results", {})
        tool_a_outputs = start_node_results_schema.get("properties", {})

        if not tool_a_outputs:
            return []

        suitable_variables = {}
        for param_name, param_def in tool_a_outputs.items():
            param_type = param_def.get('type', 'string').lower()
            if 'id' in param_name.lower() or 'identifier' in param_name.lower():
                continue
            if param_type in ['string', 'boolean', 'integer', 'number']:
                suitable_variables[param_name] = param_def

        if not suitable_variables:
            return []

        output_lines = []
        for param_name, param_def in suitable_variables.items():
            param_type = param_def.get('type', 'unknown')
            param_desc = param_def.get('description', 'No description.')
            output_lines.append(f"  - **{param_name}** (`{param_type}`): {param_desc}")
        outputs_str = "\n".join(output_lines)

        def process_conditional_goal(tool_c, tool_d):
            replacements = {
                "tool_A_name": start_node,
                "tool_A_outputs_str": outputs_str,
                "tool_C_name": tool_c,
                "tool_C_desc": self.tool_descriptions.get(tool_c, ''),
                "tool_D_name": tool_d,
                "tool_D_desc": self.tool_descriptions.get(tool_d, '')
            }

            response_json = self._call_llm_for_json("generate_conditional_goal_prompt.txt", replacements)

            if response_json and "goal" in response_json and "condition_details" in response_json:
                goal_text = response_json.get("goal")
                score_breakdown = self._score_goal(goal_text, [start_node, tool_c, tool_d], config)
                return {
                    "type": "conditional",
                    "start_tool": start_node,
                    "if_branch_tool": tool_c,
                    "else_branch_tool": tool_d,
                    "condition_details": response_json.get("condition_details"),
                    "goal_text": goal_text,
                    "score_breakdown": score_breakdown
                }
            return None

        tasks = list(combinations(successors, 2))
        generated_goals = []
        with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
            future_to_result = {
                executor.submit(process_conditional_goal, tool_c, tool_d): (tool_c, tool_d)
                for tool_c, tool_d in tasks
            }

            for future in as_completed(future_to_result):
                try:
                    result = future.result()
                    if result is not None:
                        generated_goals.append(result)
                        if len(generated_goals) >= config.get('max_goals_per_node', 5):
                            for pending_future in future_to_result:
                                if not pending_future.done():
                                    pending_future.cancel()
                            break

                except CancelledError:
                    continue

        return generated_goals


    def find_and_generate_all_conditional_goals_in_subtree(self, start_node: str, config: dict) -> list[dict]:
        """Find and generate conditional goals for all branching points in a subtree.

        Args:
            start_node: Root node of the subtree to search
            config: Configuration dictionary containing:
                - 'max_conditional_goals': Maximum total conditional goals to generate
                - Plus parameters passed to generate_conditional_goals()

        Returns:
            List of conditional goal dictionaries (up to max_conditional_goals items).
            See generate_conditional_goals() for dictionary structure.
        """
        logger.info(f"--- Searching for all conditional opportunities in the subtree of '{start_node}'... ---")

        try:
            nodes_in_subtree = nx.descendants(self.graph, start_node)
            nodes_to_check = [start_node] + list(nodes_in_subtree)
        except nx.NodeNotFound:
            nodes_to_check = [start_node]

        all_found_goals = []
        for potential_branch_node in nodes_to_check:
            if self.graph.out_degree(potential_branch_node) >= 2:
                new_goals = self.generate_conditional_goals(potential_branch_node, config)
                if new_goals: all_found_goals.extend(new_goals)

            if len(all_found_goals) >= config.get('max_conditional_goals', 5): break

        return all_found_goals[:config.get('max_conditional_goals', 5)]


    def _find_all_paths_to_leaves(self, start_node: str, max_depth: int) -> list[list[str]]:
        """Find all simple paths from start_node to leaf nodes within maximum depth.

        Args:
            start_node: Starting node for path enumeration
            max_depth: Maximum path length (number of edges) allowed

        Returns:
            List of paths, where each path is a list of tool names from start_node
            to a leaf node
        """
        reachable_nodes = nx.descendants(self.graph, start_node)
        leaf_nodes = {node for node in reachable_nodes if self.graph.out_degree(node) == 0}

        all_paths = []
        for leaf in leaf_nodes:
            for path in nx.all_simple_paths(
                self.graph, source=start_node, target=leaf, cutoff=max_depth
            ):
                all_paths.append(path)

        return all_paths


    def generate_maximal_paths_with_mmr(self, start_node: str, config: dict) -> list[dict]:
        """Generate maximal path goals using exhaustive enumeration and MMR selection.

        Enumerates all paths from start_node to leaf nodes (up to max_depth), scores all
        paths, and selects the top-k most diverse and high-scoring paths using Maximum
        Marginal Relevance (MMR).

        Args:
            start_node: Starting tool for path generation
            config: Configuration dictionary containing:
                - 'max_depth': Maximum path length to explore
                - 'min_score_threshold': Minimum score for a path to be considered
                - 'maximal_path_top_k': Number of diverse paths to select
                - 'mmr_lambda': MMR trade-off parameter (0.0-1.0)
                - Plus scoring configuration

        Returns:
            List of maximal path goal dictionaries, each containing:
            {
                'type': 'maximal_path',
                'path': list of tool names from start to leaf,
                'goal': natural language goal text,
                'score_breakdown': scoring details dict
            }
            Selected for diversity and relevance using MMR, up to maximal_path_top_k items.

        Note:
            This method can be computationally expensive for graphs with many paths.
            All paths are scored before MMR selection, which may involve many LLM calls.
        """
        logger.info(f"--- Starting Bounded Maximal Path Generation for '{start_node}' ---")
        all_paths = self._find_all_paths_to_leaves(start_node, config['max_depth'])
        logger.info(f"Found {len(all_paths)} unique paths up to depth {config['max_depth']}.")

        if not all_paths: return []

        logger.info(f"Scoring and embedding all {len(all_paths)} paths. This may take a while...")
        candidate_pool = []
        for path in tqdm(all_paths, desc="Scoring All Paths"):
            goal_text = self._simulate_goal_update(path)
            score_breakdown = self._score_goal(goal_text, path, config)
            if score_breakdown['final_score'] > config.get('min_score_threshold', 0.1):
                embedding = self.embedding_model.encode(goal_text)
                candidate_pool.append({
                    "type": "maximal_path",
                    "path": path,
                    "goal": goal_text,
                    "score_breakdown": score_breakdown,
                    "embedding": embedding
                })

        logger.info(
            f"Applying MMR to select top {config['maximal_path_top_k']} "
            f"diverse paths from {len(candidate_pool)} candidates."
        )
        final_goals = self._select_top_k_with_mmr(candidate_pool, config['maximal_path_top_k'], config['mmr_lambda'])
        for goal in final_goals: goal.pop('embedding', None)

        return final_goals


def main():
    parser = argparse.ArgumentParser(
        description="Generate complex, multi-pattern user goals from a tool graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--llm_type", type=str, choices=['watsonx', 'vllm'], default='watsonx', help="The type of LLM to use for generation.")

    # --- File & Cache Paths ---
    parser.add_argument("--graph_path", type=str, required=True, help="Pickled graph path")
    parser.add_argument("--api_definitions_path", type=str, required=True, help="Path to converted_apis JSON")
    parser.add_argument(
        "--output_goals_file_path", type=str, required=True,
        help="Output file to save the goals (JSONL)."
    )
    parser.add_argument("--synthetic_apis", action="store_true", 
                        help="If set, uses processing related to synthetic APIs.")
    parser.add_argument("--embedding_cache_path", type=str, default="output/goals/param_embeddings.pkl",
                        help="Path to save/load parameter embeddings.")
    parser.add_argument("--no_cache", action="store_true",
                        help="If set, disables caching for goals and scores for a fresh run.")

    # --- Generation Control ---
    parser.add_argument("--algorithm", type=str, default="all_patterns",
                        choices=["beam_search", "maximal_path", "all_patterns"],
                        help="Which goal generation algorithm to use.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of starting tools to process.")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum length of tool paths to generate.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of parallel workers for goal generation. Default is `max_children * beam_width`.")

    # --- Node Selection ---
    parser.add_argument("--min_children", type=int, default=2, help="Min successors for a tool to be a start node.")
    parser.add_argument("--max_children", type=int, default=20, help="Max successors for a tool to be a start node.")

    # --- Scoring ---
    parser.add_argument("--llm_rating_weight", type=float, default=0.5,
                        help="Weight for the LLM rating components in scoring.")
    parser.add_argument("--dataflow_weight", type=float, default=0.8,
                        help="Weight for the semantic dataflow score component.")
    parser.add_argument("--length_bonus_weight", type=float, default=0.3,
                        help="Weight for the path length bonus in scoring.")
    parser.add_argument("--similarity_threshold", type=float, default=0.85,
                        help="Cosine similarity threshold for semantic dataflow.")
    parser.add_argument("--edge_weight_threshold", type=int, default=1,
                        help="Edge weight threshold for synthetic APIs to consider a strong connection.")
    parser.add_argument("--min_score_threshold", type=float, default=0.2,
                        help="Minimum score for a path to be considered a candidate.")

    # --- Algorithm-Specific ---
    parser.add_argument("--beam_width", type=int, default=20, help="Beam width for Beam Search (k).")
    parser.add_argument("--mmr_lambda", type=float, default=0.7,
                        help="Lambda for MMR (0=max diversity, 1=max relevance).")
    parser.add_argument("--maximal_path_top_k", type=int, default=10,
                        help="Number of top paths to select from the maximal path algorithm.")
    parser.add_argument("--max_goals_per_node", type=int, default=5,
                        help="Max number of patterns (e.g., fan_out) to generate per node.")
    parser.add_argument("--max_branch", type=int, default=5, help="Maximum number of parallel branches for a fan-out goal.")
    parser.add_argument("--strict_fan_out", action="store_true",
                        help="Only generate fan-out goals where each branch tool provides values for different end tool parameters.")

    # --- Misc ---
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to save the log file. If not provided, logs will only be printed to console.")

    args = parser.parse_args()
    if args.max_workers is None:
        args.max_workers = args.max_children * args.beam_width

    run_parameters = vars(args)
    setup_logging(log_file=args.log_file)

    if args.llm_type == 'watsonx':
        try:
            watsonx_config_path = os.path.join(project_root, 'watsonx_llm_config.yml')
            llm = WatsonxLLM(watsonx_config_path)
        except:
            logger.exception(f"Failed to initialize WatsonxLLM. Please check your config.")
            return
    elif args.llm_type == "vllm":
        try:
            vllm_config_path = os.path.join(project_root, 'vllm_llm_config.yml')
            llm = VLLMClient(vllm_config_path)
        except:
            logger.exception(f"Failed to initialize VLLM. Please check your config.")
            return
    else:
        raise ValueError(f"Unsupported LLM type: {args.llm_type}")


    goals_path = Path(args.output_goals_file_path)
    goals_path.parent.mkdir(parents=True, exist_ok=True)


    graph = load_tool_graphs(args.graph_path, args.synthetic_apis)
    api_mapping = load_api_definitions(args.api_definitions_path, args.synthetic_apis)
    param_connection_map = load_param_connection_map(args.api_definitions_path, args.synthetic_apis)

    generator = ComplexGoalGenerator(
        tool_graph=graph, api_mapping=api_mapping, llm_instance=llm,
        param_connection_map=param_connection_map,
        use_cache=not args.no_cache, synthetic_apis=args.synthetic_apis,
        embedding_cache_path=args.embedding_cache_path if not args.synthetic_apis else None,
    )

    start_nodes = select_top_start_nodes_by_connectivity(
        graph, api_mapping, args.num_samples, args.min_children, args.max_children
    )

    goal_id_counter = 0
    config = run_parameters
    sampled_nodes = set()

    with open(goals_path, "w") as f:
        # write metadata as first JSON line
        f.write(json.dumps({"metadata": run_parameters}) + "\n")

        def process_and_add_goals(new_goals: list[dict], start_node: str):
            nonlocal goal_id_counter
            for goal_data in new_goals:
                output_object = {
                    "goal_id": f"goal_{goal_id_counter}",
                    "generation_context": {"start_node": start_node},
                    "goal_data": goal_data
                }
                f.write(json.dumps(output_object) + "\n")
                goal_id_counter += 1

        for idx, (start_node, _) in enumerate(start_nodes):
            logger.info(
                f"--- PROCESSING START NODE {idx + 1}/{len(start_nodes)}: "
                f"{start_node} (Algorithm: {args.algorithm}) ---"
            )

            if args.algorithm == "all_patterns":
                beam_search_goals = generator.generate_goals_with_beam_search(start_node, config)
                sampled_nodes.update(tool for goal in beam_search_goals for tool in goal['path'])
                process_and_add_goals(beam_search_goals, start_node)

                fan_out_goals = generator.generate_fan_out_goals(start_node, config)
                sampled_nodes.update(
                    tool for goal in fan_out_goals
                    for tool in [goal['start_tool']] + goal['branch_tools'] + [goal['end_tool']]
                )
                process_and_add_goals(fan_out_goals, start_node)

                conditional_goals = generator.generate_conditional_goals(start_node, config)
                sampled_nodes.update(
                    tool for goal in conditional_goals
                    for tool in [goal['start_tool'], goal['if_branch_tool'], goal['else_branch_tool']]
                )
                process_and_add_goals(conditional_goals, start_node)

            elif args.algorithm == "beam_search":
                beam_search_goals = generator.generate_goals_with_beam_search(start_node, config)
                sampled_nodes.update(tool for goal in beam_search_goals for tool in goal['path'])
                process_and_add_goals(beam_search_goals, start_node)

            elif args.algorithm == "maximal_path":
                maximal_path_goals = generator.generate_maximal_paths_with_mmr(start_node, config)
                sampled_nodes.update(tool for goal in maximal_path_goals for tool in goal['path'])
                process_and_add_goals(maximal_path_goals, start_node)

            logger.info(
                "\n" + "=" * 80 + "\n" +
                f"Completed processing for start node '{start_node}'. "
                f"Total goals generated: {goal_id_counter}\n" +
                "=" * 80 + "\n"
            )

        pending_nodes = sorted(
            (node for node in graph.nodes if graph.out_degree(node) < args.min_children and node not in sampled_nodes),
            key=lambda x: (graph.out_degree(x), x)
        )
        for idx, node in enumerate(pending_nodes):
            if node in sampled_nodes:
                continue

            logger.info(
                f"--- PROCESSING PENDING NODE {idx + 1}/{len(pending_nodes)}: "
                f"{node} with algorithm: beam_search ---"
            )

            beam_search_goals = generator.generate_goals_with_beam_search(node, config)
            sampled_nodes.update(tool for goal in beam_search_goals for tool in goal['path'])
            process_and_add_goals(beam_search_goals, node)

            logger.info(
                f"Completed processing for pending node '{node}'. "
                f"Total goals generated: {goal_id_counter}\n"
            )

    logger.info(f"\nProcess complete. Saved {goal_id_counter} total goals to {goals_path}")


if __name__ == "__main__":
    main()