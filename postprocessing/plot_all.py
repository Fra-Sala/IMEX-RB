import importlib.util
from pathlib import Path

ADV_DIR = Path(__file__).parent / "AdvDiff"
BURG_DIR = Path(__file__).parent / "Burgers2D"


def _load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """
    Discover and run all scripts in the AdvDiff and Burgers2D directories.
    AdvDiff scripts are run for both the 2D and 3D cases; Burgers2D
    scripts are run without parameters.
    """
    for script in ADV_DIR.glob("*.py"):
        module = _load_module(script)
        for dim_problem in (2, 3):
            module.main(dim_problem)

    for script in BURG_DIR.glob("*.py"):
        module = _load_module(script)
        module.main()


if __name__ == "__main__":
    main()
