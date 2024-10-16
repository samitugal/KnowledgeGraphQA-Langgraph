import os


def load_prompt(template_name: str) -> str:
    """
    Load a prompt template from the prompts directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    template_path = os.path.join(project_root, "Prompts", f"{template_name}.txt")

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file {template_path} not found.")

    with open(template_path, "r") as file:
        return file.read()


def fill_prompt(template, **kwargs):
    return template.format(**kwargs)
