#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

import json
import re


def load_prompt(path: str) -> str:
    """Load and return the contents of a text file.

    Args:
        path: Path to the file to read.

    Returns:
        The contents of the file as a string.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_json(obj: dict | list, path: str) -> None:
    """Save a dictionary or list as JSON to a file.

    Args:
        obj: The dictionary or list to save.
        path: Path where the JSON file will be written.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def robust_json_loads(raw: str) -> list[dict]:
    """Parse JSON from a string, handling markdown code blocks and returning a list of dictionaries.

    Attempts to extract JSON from markdown code blocks (```json...```), then parses it.
    If the parsed data is a dict, wraps it in a list. Filters out non-dict items from lists.

    Args:
        raw: Raw string potentially containing JSON, possibly wrapped in markdown code blocks.

    Returns:
        A list of dictionaries parsed from the input. Returns an empty list if parsing fails.
    """
    match = re.search(r"```json(.*)```", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = match.group(1).strip() if match else raw.strip()
    raw = re.sub(r"^```json", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    return []


def robust_json_list_of_strings(raw: str) -> list[str]:
    """Parse JSON from a string, handling markdown code blocks and returning a list of strings.

    Attempts to extract JSON from markdown code blocks (```json...```), then parses it.
    Only returns the data if it's a list type.

    Args:
        raw: Raw string potentially containing JSON, possibly wrapped in markdown code blocks.

    Returns:
        A list parsed from the input. Returns an empty list if parsing fails or if the 
        parsed data is not a list.
    """
    match = re.search(r"```json(.*)```", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = match.group(1).strip() if match else raw.strip()
    raw = re.sub(r"^```json", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []
