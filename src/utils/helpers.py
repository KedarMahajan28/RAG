import json
from typing import Any, Dict

def load_json(file_path: str) -> Dict[str, Any]:
    """Loads a JSON file and returns a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Saves a dictionary to a JSON file with pretty printing."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def format_prompt(template_path: str, **kwargs) -> str:
    """Loads a template file and formats it with kwargs."""
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    return template.format(**kwargs)
