def load_prompt(prompt_file_path: str) -> str:
    """Load a prompt from a text file."""
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        return file.read()
