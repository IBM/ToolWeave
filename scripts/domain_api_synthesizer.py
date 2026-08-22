import argparse
import logging
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock, Timeout
from pprint import pformat

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
matplotlib.use("Agg")

from src.api_synthesizer.core.generator import APISynthesizer
from src.api_synthesizer.core.paraphraser import APIParaphraser
from src.api_synthesizer.evaluation.metrics import evaluate_api_set
from src.api_synthesizer.utils.io import save_json
from src.api_synthesizer.utils.wikipedia_wikidata import build_domain_prompt, get_wikidata_qid
from src.tool_dialogue_synthesizer.llm.vllm_llm import VLLMClient
from src.tool_dialogue_synthesizer.llm.watsonx_llm import WatsonxLLM


logger = logging.getLogger('api_synthesizer')


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


def save_interactive_graph(graph: nx.DiGraph, domain_name: str, output_dir: str) -> None:
    """Save an interactive HTML visualization of the API connection graph.

    Args:
        graph: Directed graph representing API connections
        domain_name: Name of the domain for labeling
        output_dir: Directory to save the HTML visualization
    """
    vis_path = os.path.join(output_dir, f"graph_{domain_name.lower().replace(' ', '_')}.html")
    net = Network(height="700px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for node in graph.nodes:
        net.add_node(node, label=node)

    for u, v, data in graph.edges(data=True):
        weight = data.get("weight", 1)
        # Updated to use 'params' key to match graph creation
        label = ", ".join(data.get("params", [])) or f"weight={weight}"
        net.add_edge(u, v, value=weight, title=label)

    net.set_options("""
    var options = {
        "edges": { "arrows": { "to": { "enabled": true } }, "font": { "size": 12, "align": "middle" }, "color": { "inherit": true }, "smooth": true },
        "nodes": { "font": { "size": 14 }, "shape": "dot", "scaling": { "min": 10, "max": 30 } },
        "physics": { "barnesHut": { "gravitationalConstant": -8000, "centralGravity": 0.3, "springLength": 100 }, "minVelocity": 0.75 }
    }
    """)
    net.write_html(vis_path)
    logger.info(f"Saved interactive graph visualization to: {vis_path}")


def analyze_and_save_api_graph(
    apis: list[dict], connection_map: list[dict],
    domain_name: str, output_dir: str, visualize: bool = False
) -> dict:
    """Build the API graph, calculate cycle-robust metrics, and save visualizations.

    Constructs a directed graph from APIs and connections, computes graph metrics
    including density, path lengths, and chain analysis, then saves the graph and
    optional visualizations.

    Args:
        apis: List of API function definitions
        connection_map: List of dictionaries defining API-to-API connections
        domain_name: Name of the domain for labeling
        output_dir: Directory to save outputs
        visualize: Whether to generate static PNG visualization

    Returns:
        Dictionary containing computed graph metrics
    """
    G = nx.DiGraph()
    api_names = {api['function']['name'] for api in apis}
    G.add_nodes_from(api_names)

    for conn in connection_map:
        src, tgt = conn['source_api'], conn['target_api']
        if src in api_names and tgt in api_names:
            param = conn.get('source_param', 'link')
            if G.has_edge(src, tgt):
                G[src][tgt]['weight'] += 1
                G[src][tgt]['params'].append(param)
            else:
                G.add_edge(src, tgt, weight=1, params=[param])

    # --- METRIC CALCULATIONS (CYCLE-ROBUST) ---
    graph_metrics = {}
    num_nodes = G.number_of_nodes()

    graph_metrics['graph_density'] = nx.density(G) if num_nodes > 0 else 0
    graph_metrics['isolated_nodes_count'] = len(list(nx.isolates(G)))
    graph_metrics['entry_nodes_count'] = sum(1 for node, degree in G.in_degree() if degree == 0)
    graph_metrics['exit_nodes_count'] = sum(1 for node, degree in G.out_degree() if degree == 0)

    # 1. Calculate Average Path Length on the largest strongly connected component
    try:
        if num_nodes > 0:
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_scc)
            if subgraph.number_of_nodes() > 1:
                graph_metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
            else:
                graph_metrics['average_path_length'] = 0
        else:
            graph_metrics['average_path_length'] = 0
    except (nx.NetworkXError, ValueError):
        graph_metrics['average_path_length'] = "infinite"

    # 2. Find the Longest Simple Path in the entire graph
    longest_simple_path = []
    if num_nodes > 1:
        # This can be computationally expensive on very large graphs, but is fine for this scale.
        for source_node in G.nodes:
            for target_node in G.nodes:
                if source_node != target_node:
                    try:
                        for path in nx.all_simple_paths(G, source=source_node, target=target_node):
                            if len(path) > len(longest_simple_path):
                                longest_simple_path = path
                    except nx.NodeNotFound:
                        continue # Skip if a node somehow doesn't exist

    graph_metrics['longest_chain_length'] = len(longest_simple_path)
    graph_metrics['longest_chain'] = longest_simple_path

    # --- SAVE GRAPH ---
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, f"graph_{domain_name.lower().replace(' ', '_')}.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    logger.info(f"Graph saved to: {graph_path}")
    logger.info(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    save_interactive_graph(G, domain_name, output_dir)
    if visualize:
        _visualize_graph_static(G, domain_name, output_dir)

    return graph_metrics


def _visualize_graph_static(G: nx.DiGraph, domain_name: str, output_dir: str) -> None:
    """Generate and save a static PNG visualization of the API graph.

    Args:
        G: Directed graph to visualize
        domain_name: Name of the domain for labeling
        output_dir: Directory to save the PNG file
    """
    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=25, edge_color='gray', connectionstyle='arc3,rad=0.1')

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(f"API Connection Graph: {domain_name}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

    vis_path = os.path.join(output_dir, f"graph_{domain_name.lower().replace(' ', '_')}.png")
    plt.savefig(vis_path)
    plt.close()

    logger.info(f"Saved static graph visualization to: {vis_path}")


def load_domains(file_path: str) -> list[str]:
    """Load domain names from a text file.

    Args:
        file_path: Path to text file with one domain per line

    Returns:
        Sorted list of domain names, excluding comments and empty lines
    """
    try:
        with open(file_path, "r") as f:
            domains = sorted([line.strip() for line in f if line.strip() and not line.startswith('#')])
        return domains
    except FileNotFoundError:
        logger.exception(f"Domain file not found at '{file_path}'")
        return []


def process_domain(
    domain_name: str, llm: VLLMClient | WatsonxLLM,
    args: argparse.Namespace, output_dir: str,
) -> str | None:
    """Process a single domain to synthesize APIs and generate outputs.

    Resolves domain information, synthesizes APIs, evaluates quality, paraphrases
    connections, analyzes the API graph, and saves all results

    Args:
        domain_name: Name of the domain to process
        llm: LLM client for generation
        args: Command-line arguments with configuration
        output_dir: Root directory for saving outputs

    Returns:
        Domain name if successful, None otherwise
    """
    logger.info(f"{'='*20} Processing domain: {domain_name} {'='*20}")

    sanitized_domain_name = domain_name.replace("/", "_")
    domain_output_dir = os.path.join(output_dir, sanitized_domain_name.lower().replace(' ', '_'))
    os.makedirs(domain_output_dir, exist_ok=True)

    wikidata_qid = get_wikidata_qid(domain_name)
    if not wikidata_qid:
        logger.error(f"Skipped: Could not resolve QID for '{domain_name}'")
        return None
    domain_info = build_domain_prompt(domain_name, wikidata_qid)

    # --- Instantiate synthesizer with paths from args ---
    synthesizer = APISynthesizer(
        domain_name=domain_name, 
        llm=llm,
        synthesis_plan_path=args.synthesis_plan_path,
        incontext_examples_path=args.incontext_examples_path,
        enum_refinement_prompt_path=args.enum_refinement_prompt_path,
        default_value_refinement_prompt_path=args.default_value_prompt_path,
        required_params_prompt_path=args.required_params_prompt_path,
        connection_mode=args.connection_mode,
        connection_prompt_path=args.connection_prompt_path
    )
    apis = synthesizer.run(domain_info)

    if not apis:
        logger.error(f"Skipped: API synthesis failed for '{domain_name}'")
        return None

    evaluation_scores = evaluate_api_set(apis)
    logger.info(f"\n--- Evaluation Report for {domain_name} ---\n{pformat(evaluation_scores)}")

    paraphraser = APIParaphraser(llm, domain_name=domain_name)
    apis, updated_conn_map, paraphrase_log = paraphraser.run(
        all_apis=apis,
        connection_map=synthesizer.connection_map,
        description_prompt_path=args.description_prompt_path,
        param_prompt_path=args.param_prompt_path,
        fraction=args.paraphrase_fraction
    )

    graph_metrics = analyze_and_save_api_graph(
        apis=apis,
        connection_map=updated_conn_map,
        domain_name=sanitized_domain_name,
        output_dir=domain_output_dir,
        visualize=args.visualize_static
    )

    apis_missing_required_field = 0
    apis_with_empty_required_list = 0
    for api in apis:
        params = api.get('function', {}).get('parameters', {})
        if 'required' not in params:
            apis_missing_required_field += 1
        elif isinstance(params.get('required'), list) and not params['required']:
            apis_with_empty_required_list += 1

    required_field_metrics = {
        "apis_missing_required_field": apis_missing_required_field,
        "apis_with_empty_required_list": apis_with_empty_required_list
    }

    # --- MERGE ALL METRICS FOR FINAL OUTPUT ---
    final_scores = {
        **evaluation_scores,  # Includes metrics like Interconnectivity
        **graph_metrics,      # Adds all graph metrics
        **required_field_metrics
    }

    output_payload = {
        "domain": domain_name,
        "wikidata_qid": wikidata_qid,
        "scores": final_scores,
        "num_apis": len(apis),
        "apis": apis,
        "connection_map": updated_conn_map,
        "param_rename_map": paraphrase_log
    }

    sdk_path = os.path.join(
        domain_output_dir,
        f"sdk_{sanitized_domain_name.lower().replace(' ', '_')}.json"
    )
    lock_path = f"{sdk_path}.lock"

    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            save_json(output_payload, sdk_path)
            logger.info(f"Saved tool schema for {domain_name} to {sdk_path}")
    except Timeout:
        logger.error(f"Failed to acquire lock for {sdk_path}. Another process may be writing to it.")
        return None
    except:
        logger.exception(f"Error saving tool schema for {domain_name} to {sdk_path}")

    return domain_name


def main(args: argparse.Namespace):
    # --- LLM Initialization ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.llm_type == 'watsonx':
        config_path = os.path.join(project_root, 'watsonx_llm_config.yml')
        llm = WatsonxLLM(config_path)
    elif args.llm_type == "vllm":
        config_path = os.path.join(project_root, 'vllm_llm_config.yml')
        llm = VLLMClient(config_path)
    else:
        raise ValueError(f"Unsupported LLM type: {args.llm_type}")

    # --- Load and process domains ---
    domains = load_domains(args.domains_file)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    max_workers = min(args.max_workers, len(domains))

    # Use ThreadPoolExecutor for parallel processing
    logger.info(f"Starting parallel processing of {len(domains)} domains with {max_workers} workers")
    successful_domains = 0
    failed_domains = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all domain processing tasks
        future_to_domain = {
            executor.submit(process_domain, domain, llm, args, output_dir): domain
            for domain in domains
        }

        # Process results as they complete
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                result = future.result()
                if result:
                    successful_domains += 1
                    logger.info(f"Successfully processed domain: {domain} ({successful_domains}/{len(domains)})")

                else:
                    failed_domains += 1
                    logger.warning(f"Failed to process domain: {domain}")

            except:
                failed_domains += 1
                logger.exception(f"Error processing domain {domain}")

    # Summary
    logger.info(
        "\n" + "=" * 80 + "\n" +
        f"Processing complete: {successful_domains} successful, {failed_domains} failed"
        + "\n" + "=" * 80
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool to synthesize a complete API SDK for a given domain.")

    parser.add_argument("--domains_file", type=str, default="domains.txt", help="Path to the text file containing a list of domains to process.")
    parser.add_argument("--output_dir", type=str, default="output/apis/", help="Root directory to save all generated outputs.")
    parser.add_argument("--llm_type", type=str, choices=['watsonx', 'vllm'], default='watsonx', help="The type of LLM to use for generation.")

    # --- ADDED ARGUMENTS ---
    parser.add_argument("--synthesis_plan_path", type=str, default="prompts/api_synthesizer/synthesis_plan.json", help="Path to the JSON file defining the generation steps.")
    parser.add_argument("--incontext_examples_path", type=str, default="prompts/api_synthesizer/incontext_examples.json", help="Path to the JSON file with in-context API examples.")

    parser.add_argument("--description_prompt_path", type=str, default="prompts/api_synthesizer/api_description_paraphrase_prompt.txt", help="Path to the prompt for paraphrasing API descriptions.")
    parser.add_argument("--param_prompt_path", type=str, default="prompts/api_synthesizer/param_paraphrase_prompt.txt", help="Path to the prompt for paraphrasing API parameters.")
    parser.add_argument("--enum_refinement_prompt_path", type=str, default="prompts/api_synthesizer/enum_refinement_prompt.txt", help="Path to the prompt for refining enums.")
    parser.add_argument("--default_value_prompt_path", type=str, default="prompts/api_synthesizer/default_value_prompt.txt", help="Path to the prompt for generating default values.")
    parser.add_argument("--connection_prompt_path", type=str, default="prompts/api_synthesizer/batch_api_connection_prompt.txt", help="Path to the prompt for LLM-based connection tracking.")
    parser.add_argument("--required_params_prompt_path", type=str, default="prompts/api_synthesizer/required_params_prompt.txt", help="Path to the prompt for identifying required parameters.")


    parser.add_argument("--connection_mode", type=str, choices=['semantic', 'llm'], default='semantic', help="The method to use for tracking API connections.")
    parser.add_argument("--paraphrase_fraction", type=float, default=0.5, help="Fraction of API connections to paraphrase for added realism (0.0 to 1.0).")
    parser.add_argument("--visualize_static", action='store_true', help="If set, generates a static PNG image of the API graph in addition to the interactive HTML.")

    parser.add_argument("--max_workers", type=int, default=20, help="Maximum number of parallel workers for API synthesis.")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode for detailed logging.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to the log file for debug output.")

    args = parser.parse_args()
    if args.debug:
        logger.info("Setting max workers to 1 in debug mode for easier debugging.")
        args.max_workers = 1

    logger.info(f"Arguments:\n{pformat(vars(args))}")

    setup_logging(debug_mode=args.debug, log_file=args.log_file)

    main(args)