# ktir-cpu-run CLI

Command-line interface for running KTIR MLIR programs through the CPU interpreter. 
This feature gives the means to allow users of ktir-cpu to treat it as a standalone
KTIR execution virtual machine. This CLI interface will need to refine further.

Run a kernel by MLIR file path:
```bash
ktir-cpu-run triton-ktir/vector_add_ktir.mlir --arg BLOCK_SIZE=128
```

Run a kernel by name:
```bash
ktir-cpu-run add_kernel --arg BLOCK_SIZE=256
```

## Installation

The CLI is installed as a console script via `pyproject.toml`:

```bash
pip install -e .
# or
uv pip install -e .
```

This makes `ktir-cpu-run` available on your PATH.

## Usage

```
ktir-cpu-run KERNEL_OR_FILE [--arg NAME=VALUE ...] [--show-latency]
ktir-cpu-run --list
ktir-cpu-run --list_src
ktir-cpu-run --all
ktir-cpu-run --help
```

## Options

| Flag | Description |
|------|-------------|
| `--list` | List all available kernels with parameters and defaults in a table |
| `--list_src` | List all MLIR source files grouped by category with their kernels |
| `--all` | Run all kernels with default parameters, time each, and display results |
| `--arg NAME=VALUE` | Override a kernel argument (can be repeated) |
| `--show-latency` | Print latency report after execution |
| `--help` | Show help message |

## Examples

Run a kernel by name:
```bash
ktir-cpu-run add_kernel --arg BLOCK_SIZE=256
```

Run a kernel by MLIR file path:
```bash
ktir-cpu-run triton-ktir/vector_add_ktir.mlir --arg BLOCK_SIZE=128
```

List all kernels:
```bash
ktir-cpu-run --list
```

List MLIR source files by category:
```bash
ktir-cpu-run --list_src
```

Run all kernels and show timing:
```bash
ktir-cpu-run --all
```

The `--all` output includes a `Time (s)` column with execution time in seconds (3 decimal places). Kernels with known spec gaps (unimplemented features) display `KnownError` instead of timing. Any unexpected failures display `ERROR`.

## Kernel Resolution

The CLI resolves a positional argument as follows:

1. If it ends with `.mlir` or contains `/`, treat it as a file path (relative to `examples/` or absolute).
2. Otherwise, look it up by name in the kernel registry (`EXAMPLE_PARAMS` from `tests/conftest.py`).

## Known Limitations

Some kernels in `examples/rfc/` exercise KTIR features not yet implemented in the interpreter (indirect access tiles, distributed memory views, `linalg.add`, etc.). These are tracked as spec gaps and marked `KnownError` in `--all` output. See `tests/test_spec_gaps.py` for details.
