#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Smoke test used by CI to verify a dependency upgrade did not break anything.

Checks, in order:
  1. every project module imports,
  2. every documented CLI entry point responds to --help,
  3. the LangGraph dialogue workflow runs end to end through the real
     routing functions in agent_workflow.

Run locally with: uv run python .github/scripts/smoke_test.py
"""

import inspect
import importlib
import pathlib
import subprocess
import sys

# The script lives in .github/scripts/, so put the repo root on sys.path and
# use it as the working directory for the CLI checks.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CLI_SCRIPTS = [
    "domain_api_synthesizer",
    "complex_goal_generator",
    "complex_dialogue_planner",
    "generate_dialogues",
    "convert_false_multi_step_to_parallel",
    "refine_dialogues",
    "compute_dialogue_statistics",
    "compute_true_multi_step_stats",
    "generate_synthetic_data",
    "add_missing_functions_to_dialogues",
]


def check_imports() -> list[str]:
    """Import every tracked module and return a list of failures."""
    files = subprocess.check_output(["git", "ls-files", "*.py"], text=True, cwd=REPO_ROOT).split()
    modules = [f[:-3].replace("/", ".") for f in files if not f.endswith("__init__.py")]
    failures = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report, don't mask
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    print(f"imports : {len(modules) - len(failures)}/{len(modules)} modules OK")
    return failures


def check_clis() -> list[str]:
    """Run each documented entry point with --help and return failures."""
    failures = []
    for script in CLI_SCRIPTS:
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.{script}", "--help"],
            capture_output=True,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            failures.append(f"scripts.{script}: exit {result.returncode}")
    print(f"clis    : {len(CLI_SCRIPTS) - len(failures)}/{len(CLI_SCRIPTS)} entry points OK")
    return failures


def check_workflow() -> list[str]:
    """Build and run the dialogue graph with stub agents and the real routers."""
    from langchain_core.runnables import Runnable
    from langgraph.graph import END, START, StateGraph

    import src.tool_dialogue_synthesizer.agent_workflow as agent_workflow
    from src.tool_dialogue_synthesizer.schema import DialogueState

    routers = {
        name: fn
        for name, fn in inspect.getmembers(agent_workflow, inspect.isfunction)
        if fn.__module__ == agent_workflow.__name__ and name != "build_graph"
    }
    for required in ("should_continue_based_on_plan", "should_continue_after_tool"):
        if required not in routers:
            return [f"router {required} not found in agent_workflow"]

    visited: list[str] = []

    class StubAgent(Runnable):
        """Stands in for a real agent; same call shape, no LLM calls."""

        def __init__(self, name: str, fn) -> None:
            self.name = name
            self.fn = fn

        def invoke(self, state: DialogueState, config=None) -> DialogueState:
            visited.append(self.name)
            return self.fn(dict(state))

    def assistant(state: dict) -> dict:
        step = state.get("current_step_idx", 0)
        state["tool_call"] = {"name": "stub"} if step % 2 == 0 else None
        state["current_step_idx"] = step + 1
        return state

    graph = StateGraph(DialogueState)
    graph.add_node("user", StubAgent("user", lambda state: state))
    graph.add_node("memory", StubAgent("memory", lambda state: state))
    graph.add_node("assistant", StubAgent("assistant", assistant))
    graph.add_node("tool", StubAgent("tool", lambda state: {**state, "tool_call": None}))
    graph.add_edge(START, "user")
    graph.add_edge("user", "memory")
    graph.add_edge("memory", "assistant")
    graph.add_conditional_edges(
        "assistant",
        routers["should_continue_based_on_plan"],
        {"tool": "tool", "user": "user", END: END},
    )
    graph.add_conditional_edges(
        "tool",
        routers["should_continue_after_tool"],
        {"assistant": "assistant", "user": "user", "memory": "memory", "tool": "tool", END: END},
    )

    plan_length = 4
    final_state = graph.compile().invoke(
        {"plan": list(range(plan_length)), "current_step_idx": 0, "messages": []},
        config={"recursion_limit": 50},
    )
    if final_state.get("current_step_idx") != plan_length:
        return [f"workflow ended at step {final_state.get('current_step_idx')}, expected {plan_length}"]
    print(f"workflow: graph ran {len(visited)} nodes and terminated correctly")
    return []


def main() -> int:
    failures = check_imports() + check_clis() + check_workflow()
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
