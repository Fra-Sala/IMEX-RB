import sys
import subprocess
from pathlib import Path


def main():
    """
    Run all tests accompanying the publication, i.e.
        - Advection diffusion equation 2D (2 tests, varying $\\epsilon$
          and $N$)
        - Burgers' equation 2D (3 tests, for stability, cpu times
          and convergence)
        - Advection diffusion equation 3D (2 tests, varying $\\epsilon$
          and $N$)
    """
    root = Path(__file__).parent
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith("__"):
            continue
        for test in sub.glob("test_*.py"):
            print(f"\n=== Running {sub.name}/{test.name} ===")
            subprocess.run([sys.executable, str(test)], check=True)


if __name__ == "__main__":
    main()
