"""Canonical scenarios that ship with simulacra.

Each scenario module exposes a `run(controller, out, seed)` function so
engineers can use the scenario with their own controller. Scripts run
directly via `uv run -m simulacra.scenarios.<name>` execute with a default
controller declared in each scenario's __main__ block.

This package is empty in v0.1 until Plan 2 ports hopper_short and
hopper_long to the new Controller interface. The current top-level
scenarios/hopper_short.py and scenarios/hopper_long.py continue to work
under the old interface.
"""
