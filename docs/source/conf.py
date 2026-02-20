import importlib
import inspect
import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Llm Async"
copyright = "2026, Johanderson"
author = "Johanderson"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx_rtd_theme",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_BLOB_URL = "https://github.com/sonic182/llm-async/blob/master"


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    module_name = info.get("module")
    if not module_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None

    obj = module
    fullname = info.get("fullname")
    if fullname:
        for part in fullname.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
    if obj is None:
        obj = module

    if callable(obj):
        try:
            obj = inspect.unwrap(obj)
        except Exception:
            pass

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except Exception:
        try:
            source_file = inspect.getsourcefile(module) or inspect.getfile(module)
        except Exception:
            return None

    file_path = Path(source_file).resolve()
    try:
        relative_path = file_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None

    linespec = ""
    try:
        source_lines, start_line = inspect.getsourcelines(obj)
        end_line = start_line + len(source_lines) - 1
        linespec = f"#L{start_line}-L{end_line}"
    except Exception:
        pass

    return f"{REPO_BLOB_URL}/{relative_path}{linespec}"

autodoc_mock_imports = ["aiosonic"]
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}
