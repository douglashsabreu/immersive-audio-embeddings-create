#!/usr/bin/env python3
"""Script to run comprehensive quality checks on the codebase."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output.

    Args:
        cmd: Command to run as list of strings
        description: Description of the command for output

    Returns:
        Tuple of (success, output)
    """
    print(f"\n🔍 {description}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent, check=False
        )

        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True, result.stdout
        else:
            print(f"❌ {description} failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False, result.stderr

    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e)


def main() -> int:
    """Run all quality checks and return exit code."""
    print("🚀 Running comprehensive quality checks for embeddings-create")

    checks = [
        (["ruff", "check", "embeddings_create/", "main.py"], "Ruff linting"),
        (
            ["black", "--check", "--line-length=100", "embeddings_create/", "main.py"],
            "Black formatting check",
        ),
        (
            ["isort", "--check-only", "--profile=black", "embeddings_create/", "main.py"],
            "Import sorting check",
        ),
        (
            ["mypy", "embeddings_create/", "main.py", "--strict", "--ignore-missing-imports"],
            "MyPy type checking",
        ),
        (
            [
                "lizard",
                "embeddings_create/",
                "main.py",
                "--length",
                "50",
                "--arguments",
                "8",
                "--CCN",
                "10",
            ],
            "Complexity analysis",
        ),
        (
            [
                "bandit",
                "-r",
                "embeddings_create/",
                "main.py",
                "-f",
                "json",
                "-o",
                "bandit-report.json",
            ],
            "Security check",
        ),
    ]

    results = []

    for cmd, description in checks:
        success, output = run_command(cmd, description)
        results.append((description, success, output))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Quality Check Summary")
    print("=" * 60)

    passed = 0
    failed = 0

    for description, success, output in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<30} {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Total checks: {len(results)} | Passed: {passed} | Failed: {failed}")

    if failed > 0:
        print("\n❌ Some quality checks failed. Please fix the issues above.")
        return 1
    else:
        print("\n🎉 All quality checks passed! Your code is ready for production.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
