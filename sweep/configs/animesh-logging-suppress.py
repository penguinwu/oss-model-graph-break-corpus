"""Setup script: register logging.Logger methods with
torch._dynamo.config.ignore_logging_functions so library-level logger calls
don't show up as graph breaks.

Source: Animesh's gchat snippet, 2026-04-27 user group (spaces/AAQABmB_3Is).
Used for the [animesh-fullgraph] sweep series.
"""
import logging

import torch._dynamo


_METHODS = (
    "debug", "info", "warning", "warn", "error",
    "critical", "fatal", "log", "exception", "warning_once",
)

for method_name in _METHODS:
    method = getattr(logging.Logger, method_name, None)
    if method is not None:
        torch._dynamo.config.ignore_logging_functions.add(method)
