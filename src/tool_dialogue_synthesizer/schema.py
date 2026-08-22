#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#

from typing import TypedDict

class DialogueState(TypedDict):
    """
    Defines the structure and required attributes for a dialogue instance.

    This class serves as a template for dialogue state management during
    synthetic dialogue generation, specifying all necessary fields that
    must be present in a dialogue instance.

    Attributes:
        sample_idx: Unique identifier for the dialogue sample
        plan: List of planned steps/goals for the dialogue
        current_step_idx: Index of the current step being executed in the plan
        partition_idx: Index of the current partition/turn in the dialogue
        user_message: The latest user's message in the dialogue
        tool_call: Dictionary containing the last tool/function call information
        memory_cache: Dictionary storing cached memory state for the dialogue
        conversations: List of conversation turn dictionaries
        abort_due_to_error: Flag indicating if the dialogue was aborted due to an error
        retry_count: Number of retry attempts made for the current operation
        error_msg: Error message if an error occurred, None otherwise
    """
    sample_idx: int
    plan: list[str]
    current_step_idx: int
    partition_idx: int
    user_message: str
    tool_call: dict
    memory_cache: dict
    conversations: list[dict]
    abort_due_to_error: bool
    retry_count: int
    error_msg: str | None
