"""
SensorGuard - MC-Dropout Experiments for FD002, FD003, and FD004

Reuses the existing, tested MC-Dropout experiment implementation in
src.evaluation.mc_dropout_experiments. FD001 is intentionally excluded
because its experiment has already been run.

This script:
- does not retrain models;
- evaluates six scenarios per subset;
- saves outputs in each subset's existing mc_dropout_experiments folder;
- does not overwrite or modify FD001 results.

Run from the SensorGuard project root:
    python -m src.evaluation.mc_dropout_additional_subsets
"""

from pathlib import Path

from src.evaluation import mc_dropout_experiments as experiment


SUBSETS = ["FD002", "FD003", "FD004"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 78)
    print("SensorGuard - Additional MC-Dropout Experiments")
    print("=" * 78)
    print(f"Subsets to evaluate: {', '.join(SUBSETS)}")
    print("FD001 is unchanged; its experiment has already been completed.")
    print("No models will be retrained.")
    print("")

    failures = []

    for subset in SUBSETS:
        # Configure the existing experiment module for this subset.
        experiment.SUBSET = subset
        experiment.CHECKPOINT_PATH = (
            PROJECT_ROOT
            / "results"
            / "models"
            / subset
            / "baseline_lstm.pt"
        )
        experiment.OUTPUT_DIR = (
            PROJECT_ROOT
            / "results"
            / "metrics"
            / subset
            / "mc_dropout_experiments"
        )

        print("\n" + "#" * 78)
        print(f"STARTING {subset}")
        print("#" * 78)

        try:
            experiment.main()
        except Exception as exc:
            failures.append({
                "subset": subset,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"\nERROR in {subset}: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 78)
    print("ADDITIONAL MC-DROPOUT EXPERIMENTS FINISHED")
    print("=" * 78)

    for subset in SUBSETS:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / "metrics"
            / subset
            / "mc_dropout_experiments"
        )
        print(f"{subset}: {output_dir}")

    if failures:
        print("\nThe following subsets failed:")
        for failure in failures:
            print(f"  {failure['subset']}: {failure['error']}")
        raise RuntimeError(
            f"{len(failures)} subset(s) failed. Review the errors above."
        )

    print("\nAll requested subsets completed successfully.")


if __name__ == "__main__":
    main()
