# csdk/ — simulacra C SDK

Write a flight controller in C, run it inside simulacra's Python simulator
via `ctypes`, and target the same `.c` file for the flight computer.

## Quick start

```bash
cd csdk
make                                # builds examples/libpid_hover.{so,dylib}
```

Then in Python (from the repo root):

```python
from simulacra.onboard.c_controller import CController
from simulacra.scenarios import hopper_short

ctrl = CController("csdk/examples/libpid_hover.dylib")
hopper_short.run(ctrl)              # writes NDJSON to stdout
```

## Files

- **`include/simulacra.h`** — the public header. Read this first.
- **`examples/pid_hover.c`** — reference PD altitude controller, ~50 lines.
- **`Makefile`** — builds each example as a shared library for this platform.
- **`tests/`** — reserved for C-side unit tests (none yet).

## Documentation

- **`../docs/c-sdk-spec.md`** — the canonical reference: every type and function.
- **`../docs/c-sdk-manual.md`** — the engineer's manual: how to use the SDK.

## ABI version

`0.1.0`. Forward-binary-compatible within the `0.1.x` series.
