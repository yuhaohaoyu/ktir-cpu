#!/usr/bin/env python3
"""ktir-cpu-run: Command-line interface for running KTIR MLIR programs.

Usage:
    ktir-cpu-run KERNEL_OR_FILE [--arg NAME=VALUE ...] [--show-latency]
    ktir-cpu-run --list
    ktir-cpu-run --list_src
    ktir-cpu-run --help

Examples:
    ktir-cpu-run add_kernel --arg BLOCK_SIZE=256
    ktir-cpu-run examples/triton-ktir/vector_add_ktir.mlir --arg BLOCK_SIZE=256
    ktir-cpu-run --list
    ktir-cpu-run --list_src
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ktir_cpu import KTIRInterpreter

# Import test fixtures to access EXAMPLE_PARAMS
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from conftest import EXAMPLE_PARAMS, get_test_params


def _get_examples_dir() -> Path:
    """Return path to examples directory."""
    return Path(__file__).parent.parent / "examples"


def list_kernels_tabular():
    """Print all available kernels in a formatted table."""
    print("Available MLIR kernels:")
    print()
    print(
        f"{'Kernel Name':<40} {'Path':<45} {'Parameters':<60} {'Defaults':<60}"
    )
    print("-" * 205)

    for func_name, entries in sorted(EXAMPLE_PARAMS.items()):
        for i, entry in enumerate(entries):
            if "exception_msg" in entry:
                continue  # Skip failure examples

            path = entry["path"]
            kwargs = entry.get("execute_kwargs", {})

            # Build parameter names and defaults strings
            param_names = [k for k, v in kwargs.items() if v is not None and not isinstance(v, list)]
            defaults = {k: v for k, v in kwargs.items() if v is not None and not isinstance(v, list)}

            param_str = ", ".join(param_names) if param_names else "(none)"
            default_str = ", ".join(f"{k}={v}" for k, v in defaults.items()) if defaults else "(none)"

            print(
                f"{func_name:<40} {path:<45} {param_str:<60} {default_str:<60}"
            )


def list_mlir_sources():
    """Print all available MLIR source files with their kernels and parameters."""
    examples_dir = _get_examples_dir()

    print("Available MLIR source files:")
    print()

    # Group kernels by source file path
    mlir_to_kernels = {}
    for func_name, entries in EXAMPLE_PARAMS.items():
        for entry in entries:
            if "exception_msg" in entry:
                continue
            path = entry["path"]
            if path not in mlir_to_kernels:
                mlir_to_kernels[path] = []
            mlir_to_kernels[path].append((func_name, entry.get("execute_kwargs", {})))

    # Print by category
    for category in ["triton-ktir", "latency", "ktir", "rfc"]:
        files = sorted([p for p in mlir_to_kernels.keys() if p.startswith(category)])
        if not files:
            continue

        print(f"\n{category.upper()}:")
        print()
        for file_path in files:
            abs_path = examples_dir / file_path
            exists = "✓" if abs_path.exists() else "✗"
            print(f"  [{exists}] {file_path}")

            for kernel_name, kwargs in mlir_to_kernels[file_path]:
                param_str = ", ".join(
                    f"{k}={v}" for k, v in kwargs.items()
                    if v is not None and not isinstance(v, list)
                ) or "(none)"
                print(f"      → {kernel_name}: {param_str}")


KNOWN_GAP_KERNELS = {
    "add",
    "indirect_access_copy",
    "indirect_scatter",
    "distributed_view_copy",
    "paged_tensor_copy_1core",
    "paged_tensor_write_1core",
    "ring_reduce",
}


def run_all_kernels():
    """Run all kernels with default parameters, time each, and display as a table."""
    examples_dir = _get_examples_dir()
    rows = []

    for func_name, entries in sorted(EXAMPLE_PARAMS.items()):
        for entry in entries:
            if "exception_msg" in entry:
                continue

            path = entry["path"]
            kwargs = entry.get("execute_kwargs", {})
            param_names = [k for k, v in kwargs.items() if v is not None and not isinstance(v, list)]
            defaults = {k: v for k, v in kwargs.items() if v is not None and not isinstance(v, list)}
            param_str = ", ".join(param_names) if param_names else "(none)"
            default_str = ", ".join(f"{k}={v}" for k, v in defaults.items()) if defaults else "(none)"

            abs_path = str(examples_dir / path)
            elapsed = _time_kernel(func_name, abs_path, entry)
            is_known_gap = func_name in KNOWN_GAP_KERNELS
            rows.append((elapsed, func_name, path, param_str, default_str, is_known_gap))

    print("Available MLIR kernels (timed with defaults):")
    print()
    print(
        f"{'Time (s)':<12} {'Kernel Name':<40} {'Path':<45} {'Parameters':<60} {'Defaults':<60}"
    )
    print("-" * 217)
    for elapsed, func_name, path, param_str, default_str, is_known_gap in rows:
        if elapsed is not None:
            time_str = f"{elapsed:.3f}"
        elif is_known_gap:
            time_str = "KnownError"
        else:
            time_str = "ERROR"
        print(
            f"{time_str:<12} {func_name:<40} {path:<45} {param_str:<60} {default_str:<60}"
        )


def _time_kernel(kernel_name: str, abs_path: str, entry: dict) -> Optional[float]:
    """Run a single kernel with its defaults and return elapsed seconds, or None on error."""
    try:
        interp = KTIRInterpreter()
        interp.load(abs_path)

        arg_names = interp.arg_names(kernel_name)
        tensor_sizes = interp.tensor_input_output_sizes(kernel_name)

        exec_kwargs = {}
        for k, v in entry.get("execute_kwargs", {}).items():
            if isinstance(v, list):
                exec_kwargs[k] = v[0]
            else:
                exec_kwargs[k] = v

        scalar_args = set(arg_names) - set(tensor_sizes.keys())
        tensor_args = set(arg_names) & set(tensor_sizes.keys())

        for arg_name in scalar_args:
            if arg_name in exec_kwargs and isinstance(exec_kwargs[arg_name], int):
                exec_kwargs[arg_name] = np.int32(exec_kwargs[arg_name])

        for arg_name in tensor_args:
            shape = tensor_sizes[arg_name]["shape"]
            dtype_str = tensor_sizes[arg_name]["dtype"]
            dtype = _parse_dtype(dtype_str)

            resolved_shape = []
            for dim in shape:
                if isinstance(dim, int):
                    resolved_shape.append(dim)
                elif isinstance(dim, str):
                    sym_name = dim.lstrip("%")
                    val = exec_kwargs.get(sym_name)
                    if val is None:
                        for k, v in exec_kwargs.items():
                            if sym_name in k and isinstance(v, (int, np.integer)):
                                val = v
                                break
                    if val is not None:
                        resolved_shape.append(int(val))
                    else:
                        raise ValueError(f"Cannot resolve symbolic dim '{dim}' for {arg_name}")
                else:
                    resolved_shape.append(int(dim))

            if "float" in dtype_str:
                data = np.random.randn(*resolved_shape).astype(dtype)
            elif "int" in dtype_str:
                data = np.random.randint(0, 100, size=resolved_shape, dtype=dtype)
            else:
                data = np.zeros(resolved_shape, dtype=dtype)
            exec_kwargs[arg_name] = data

        start = time.perf_counter()
        interp.execute_function(kernel_name, **exec_kwargs)
        elapsed = time.perf_counter() - start
        return elapsed
    except Exception as e:
        print(f"  [ERROR] {kernel_name}: {e}", file=sys.stderr)
        return None


def _resolve_mlir_path_or_kernel(spec: str) -> tuple[str, str]:
    """Resolve a kernel name or MLIR file path to (abs_path, kernel_name).

    If spec looks like a file path (.mlir) or exists as a file, treat it as MLIR source.
    Otherwise, treat it as a kernel name and look it up in EXAMPLE_PARAMS.

    Returns:
        (abs_path, kernel_name) tuple

    Raises:
        FileNotFoundError or ValueError if resolution fails.
    """
    examples_dir = _get_examples_dir()

    # Check if spec is a file path (relative or absolute)
    if spec.endswith(".mlir") or "/" in spec or "\\" in spec:
        # Try to resolve as relative path from examples dir
        candidate = examples_dir / spec if not Path(spec).is_absolute() else Path(spec)
        if candidate.exists() and candidate.suffix == ".mlir":
            # Extract kernel name: parse the MLIR to find function definitions
            kernel_name = _extract_first_kernel_name(str(candidate))
            return str(candidate), kernel_name

        raise FileNotFoundError(f"MLIR file not found: {spec}")

    # Treat as kernel name
    if spec not in EXAMPLE_PARAMS:
        raise ValueError(f"Unknown kernel: {spec}")

    entries = [e for e in EXAMPLE_PARAMS[spec] if "exception_msg" not in e]
    if not entries:
        raise ValueError(f"Kernel '{spec}' has no valid (non-failure) examples")

    entry = entries[0]
    abs_path = str(examples_dir / entry["path"])
    return abs_path, spec


def _extract_first_kernel_name(mlir_path: str) -> str:
    """Extract the first function name from an MLIR file."""
    with open(mlir_path) as f:
        content = f.read()

    # Match "func.func @name(...)" or "func @name(...)"
    match = re.search(r'func(?:\.func)?\s+@(\w+)\s*\(', content)
    if match:
        return match.group(1)

    raise ValueError(f"No kernel function found in {mlir_path}")


def _extract_kernel_names(mlir_path: str) -> list[str]:
    """Extract all function names from an MLIR file."""
    with open(mlir_path) as f:
        content = f.read()

    matches = re.findall(r'func(?:\.func)?\s+@(\w+)\s*\(', content)
    return matches or []


def run_kernel(
    kernel_or_file: str,
    overrides: dict[str, str],
    show_latency: bool = False,
) -> int:
    """Load and execute a KTIR kernel.

    Args:
        kernel_or_file: Kernel name (e.g., "add_kernel") or path to MLIR file
        overrides: Dict of argument name -> value overrides
        show_latency: If True, print latency report after execution

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        abs_path, kernel_name = _resolve_mlir_path_or_kernel(kernel_or_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        # Load the MLIR file
        interp = KTIRInterpreter()
        interp.load(abs_path)

        # Get function arguments and sizes
        arg_names = interp.arg_names(kernel_name)
        tensor_sizes = interp.tensor_input_output_sizes(kernel_name)

        # Build execute_kwargs: start with fixture defaults, then apply overrides
        exec_kwargs = {}
        if kernel_or_file in EXAMPLE_PARAMS:
            entries = [e for e in EXAMPLE_PARAMS[kernel_or_file] if "exception_msg" not in e]
            if entries:
                for k, v in entries[0].get("execute_kwargs", {}).items():
                    exec_kwargs[k] = v[0] if isinstance(v, list) else v

        # Classify arguments: scalars vs. tensors
        scalar_args = set(arg_names) - set(tensor_sizes.keys())
        tensor_args = set(arg_names) & set(tensor_sizes.keys())

        # Populate scalar arguments (from fixture or overrides)
        for arg_name in scalar_args:
            if arg_name in overrides:
                exec_kwargs[arg_name] = _parse_value(overrides[arg_name])
            elif arg_name not in exec_kwargs or exec_kwargs[arg_name] is None:
                print(
                    f"Warning: scalar argument '{arg_name}' has no default; "
                    f"use --arg {arg_name}=VALUE to specify",
                    file=sys.stderr,
                )

        # Populate tensor arguments with synthetic data
        for arg_name in tensor_args:
            shape = tensor_sizes[arg_name]["shape"]
            dtype_str = tensor_sizes[arg_name]["dtype"]
            dtype = _parse_dtype(dtype_str)

            if arg_name in overrides:
                print(f"Warning: ignoring override for tensor '{arg_name}'", file=sys.stderr)

            # Resolve symbolic dimensions from scalar kwargs
            resolved_shape = []
            for dim in shape:
                if isinstance(dim, int):
                    resolved_shape.append(dim)
                elif isinstance(dim, str):
                    sym_name = dim.lstrip("%")
                    val = exec_kwargs.get(sym_name)
                    if val is None:
                        for k, v in exec_kwargs.items():
                            if sym_name in k and isinstance(v, (int, np.integer)):
                                val = v
                                break
                    if val is not None:
                        resolved_shape.append(int(val))
                    else:
                        print(
                            f"Warning: cannot resolve symbolic dim '{dim}' for tensor '{arg_name}'",
                            file=sys.stderr,
                        )
                        resolved_shape.append(1)
                else:
                    resolved_shape.append(int(dim))

            # Generate random data
            if "float" in dtype_str:
                data = np.random.randn(*resolved_shape).astype(dtype)
            elif "int" in dtype_str:
                data = np.random.randint(0, 100, size=resolved_shape, dtype=dtype)
            else:
                data = np.zeros(resolved_shape, dtype=dtype)

            exec_kwargs[arg_name] = data

        # Execute the kernel
        print(f"Running {kernel_name} from {Path(abs_path).name}...", file=sys.stderr)
        print(f"  Arg names: {arg_names}", file=sys.stderr)
        print(f"  Execute kwargs keys: {list(exec_kwargs.keys())}", file=sys.stderr)

        start = time.perf_counter()
        results = interp.execute_function(kernel_name, **exec_kwargs)
        elapsed = time.perf_counter() - start
        print(f"\n  Time: {elapsed:.3f}s", file=sys.stderr)

        # Print output shapes
        print(f"\nResults:", file=sys.stderr)
        for out_name, out_data in results.items():
            if isinstance(out_data, np.ndarray):
                print(
                    f"  {out_name}: shape={out_data.shape}, dtype={out_data.dtype}",
                    file=sys.stderr,
                )
            else:
                print(f"  {out_name}: {out_data} (type={type(out_data).__name__})", file=sys.stderr)

        # Print latency report if requested
        if show_latency:
            report = interp.get_latency_report()
            if report:
                print(f"\nLatency Report:", file=sys.stderr)
                print(report, file=sys.stderr)
            else:
                print(f"\nNo latency report available (latency tracking may be disabled)", file=sys.stderr)

        return 0

    except Exception as e:
        print(f"Error executing {kernel_name}: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


def _parse_value(s: str):
    """Parse a command-line value into int, float, or string."""
    # Try int
    try:
        return int(s)
    except ValueError:
        pass
    # Try float
    try:
        return float(s)
    except ValueError:
        pass
    # Fall back to string
    return s


def _parse_dtype(dtype_str: str):
    """Map MLIR dtype string (e.g., 'f16', 'f32', 'i32') to numpy dtype."""
    dtype_map = {
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
    }
    return dtype_map.get(dtype_str, np.float32)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ktir-cpu-run",
        description="Run KTIR MLIR kernels through the CPU interpreter.",
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        help="Kernel function name or path to .mlir file (e.g., add_kernel, examples/triton-ktir/vector_add_ktir.mlir)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available kernels with parameters and defaults",
    )
    parser.add_argument(
        "--list_src",
        action="store_true",
        help="List all available MLIR source files with kernels and parameters",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        help="Override a kernel argument: --arg name=value (can be used multiple times)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all kernels with default parameters, time each, and display results",
    )
    parser.add_argument(
        "--show-latency",
        action="store_true",
        help="Print latency report after execution",
    )

    args = parser.parse_args()

    if args.all:
        run_all_kernels()
        return 0

    if args.list:
        list_kernels_tabular()
        return 0

    if args.list_src:
        list_mlir_sources()
        return 0

    if not args.kernel:
        parser.print_help(file=sys.stderr)
        return 1

    # Parse --arg overrides into a dict
    overrides = {}
    for arg_spec in args.arg:
        if "=" not in arg_spec:
            print(f"Error: invalid --arg format '{arg_spec}'; use --arg name=value", file=sys.stderr)
            return 1
        name, value = arg_spec.split("=", 1)
        overrides[name] = value

    return run_kernel(args.kernel, overrides, show_latency=args.show_latency)


if __name__ == "__main__":
    sys.exit(main())
