"""Unit and functional tests for the ktir-cpu-run CLI.

Only uses kernels: add, add_kernel, add_kernel_dynamic.
"""

import subprocess
import sys
from pathlib import Path

import pytest

CLI_MODULE = str(Path(__file__).parent.parent / "cli" / "ktir_cpu_run.py")


def run_cli(*args):
    """Run ktir-cpu-run with the given arguments and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, CLI_MODULE] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Unit tests: CLI flags, help, error handling
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_flag(self):
        returncode, stdout, _ = run_cli("--help")
        assert returncode == 0
        assert "ktir-cpu-run" in stdout
        assert "--list" in stdout
        assert "--all" in stdout
        assert "--arg" in stdout
        assert "--show-latency" in stdout

    def test_no_args_shows_help(self):
        returncode, _, stderr = run_cli()
        assert returncode != 0
        assert "usage" in stderr.lower() or "kernel" in stderr.lower()


class TestCLIList:
    def test_list_shows_table_with_add_kernel(self):
        returncode, stdout, _ = run_cli("--list")
        assert returncode == 0
        assert "Kernel Name" in stdout
        assert "add_kernel" in stdout
        assert "BLOCK_SIZE" in stdout


class TestCLIErrors:
    def test_unknown_kernel(self):
        returncode, _, stderr = run_cli("unknown_kernel_xyz")
        assert returncode != 0
        assert "unknown" in stderr.lower() or "error" in stderr.lower()

    def test_invalid_arg_format(self):
        returncode, _, stderr = run_cli("add_kernel", "--arg", "BLOCK_SIZE")
        assert returncode != 0
        assert "invalid" in stderr.lower() or "=" in stderr


# ---------------------------------------------------------------------------
# Functional tests: execute add, add_kernel, add_kernel_dynamic
# ---------------------------------------------------------------------------


class TestFunctionalAddKernel:
    """Functional tests for add_kernel (vector add, known working)."""

    def test_run_by_name(self):
        returncode, _, stderr = run_cli("add_kernel")
        assert returncode == 0
        assert "Running add_kernel" in stderr
        assert "Results:" in stderr

    def test_run_by_file_path(self):
        returncode, _, stderr = run_cli(
            "triton-ktir/vector_add_ktir.mlir", "--arg", "BLOCK_SIZE=128"
        )
        assert returncode == 0
        assert "add_kernel" in stderr
        assert "Results:" in stderr


class TestFunctionalAddKernelDynamic:
    """Functional tests for add_kernel_dynamic (symbolic/dynamic shapes)."""

    def test_run_by_name(self):
        returncode, _, stderr = run_cli("add_kernel_dynamic")
        assert returncode == 0
        assert "Running add_kernel_dynamic" in stderr
        assert "Results:" in stderr


class TestFunctionalAdd:
    """Functional tests for add (rfc/add-with-control-flow.mlir, known spec gap)."""

    def test_run_fails_gracefully(self):
        returncode, _, stderr = run_cli("add")
        assert returncode != 0
        assert "error" in stderr.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
