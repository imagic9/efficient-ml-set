"""Shared fixtures.

`RunContext` attaches handlers to the *root* logger, which is process-global: a test
that creates a run and leaves its handler attached makes every later test write into a
tmp_path that has been deleted. Cleaning up here rather than per-file means a new test
that creates a run cannot forget.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _isolate_logging():
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in before:
            handler.close()
            root.removeHandler(handler)
