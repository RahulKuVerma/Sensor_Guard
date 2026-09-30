from pathlib import Path


def show_project_structure():
    root = Path(__file__).resolve().parent

    directories = [
        "data/raw",
        "data/processed",
        "data/degraded",
        "notebooks",
        "src/data",
        "src/sensors",
        "src/features",
        "src/models",
        "src/uncertainty",
        "src/evaluation",
        "configs",
        "results/figures",
        "results/metrics",
        "results/models",
        "app",
        "tests",
    ]

    print("=" * 60)
    print("SensorGuard Project")
    print("=" * 60)

    for directory in directories:
        path = root / directory

        if path.exists():
            status = "OK"
        else:
            status = "MISSING"

        print(f"[{status}] {directory}")

    print("=" * 60)


if __name__ == "__main__":
    show_project_structure()