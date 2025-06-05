import sys
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).parent
    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith("__"):
            continue
        for test in sub.glob("test_*.py"):
            print(f"\n=== Running {sub.name}/{test.name} ===")
            subprocess.run([sys.executable, str(test)], check=True)


if __name__ == "__main__":
    main()
