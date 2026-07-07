from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    if ":" in path:
        module_name, attr_path = path.split(":", 1)
    else:
        module_name, attr_path = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj
