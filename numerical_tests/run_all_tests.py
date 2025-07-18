import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """
    Run tests accompanying the publication.

    Available test directories:
        - Advection diffusion equation 2D (2 tests, varying $\\epsilon$
          and $N$)
        - Burgers' equation 2D (3 tests, for stability, cpu times
          and convergence)
        - Advection diffusion equation 3D (2 tests, varying $\\epsilon$
          and $N$)

    Usage:
        - Run all tests: python run_all_tests.py
        - Run specific tests: python run_all_tests.py dir1 dir2 ...
          (e.g., python run_all_tests.py advDiff2D burgers2D)
    """
    parser = argparse.ArgumentParser(
        description='Run tests from specified directories.')
    parser.add_argument('dirs', nargs='*',
                        help='List of directories to run tests from.'
                        'If none provided, runs tests from all directories.')
    args = parser.parse_args()
    root = Path(__file__).parent

    # If directories are specified, use them, otherwise run all tests
    if args.dirs:
        subdirs = [root / dir_name for dir_name in args.dirs
                   if (root / dir_name).is_dir()]
        if not subdirs:
            print("None of the specified directories exist.")
            return
    else:
        subdirs = [sub for sub in root.iterdir()
                   if sub.is_dir() and not sub.name.startswith("__")]

    for sub in subdirs:
        for test in sub.glob("test_*.py"):
            print(f"\n=== Running {sub.name}/{test.name} ===")
            subprocess.run([sys.executable, str(test)], check=True)


if __name__ == "__main__":
    main()
