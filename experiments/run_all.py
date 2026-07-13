"""
run_all.py -- single entry point that reproduces every experiment and
figure in this project. Run from the repository root:

    python experiments/run_all.py

Expected total runtime: roughly 30-50 minutes on the full real datasets
(Experiments 3-5 dominate; see each experiment module's docstring for
individual timing notes and the compute-tractability decisions -- dataset
subsampling, parameter sweep granularity -- made along the way).

Each experiment is wrapped in its own try/except: if one experiment fails
(e.g. a dataset genuinely isn't available), the rest still run, and a
summary at the end reports exactly what succeeded and what didn't. This
is a deliberate design choice for a one-command reproduction script -- a
single missing dataset shouldn't silently prevent every other result from
being generated.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.common import ensure_output_dirs, get_available_datasets


def _run_step(name: str, fn, *args, **kwargs) -> tuple[bool, float]:
    """Run one experiment step, isolated from failures in other steps.
    Returns (succeeded, elapsed_seconds)."""
    print(f"\n{'#'*70}\n# {name}\n{'#'*70}")
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n[DONE] {name} completed in {elapsed:.1f}s")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FAILED] {name} raised {type(e).__name__}: {e}")
        traceback.print_exc()
        return False, elapsed


def main() -> None:
    overall_start = time.time()
    ensure_output_dirs()

    print("Loading datasets...")
    datasets = get_available_datasets()
    print(f"Loaded {len(datasets)} dataset(s): {list(datasets.keys())}")

    if "breast_cancer" not in datasets:
        print("WARNING: breast_cancer is required for Experiment 6 (bias-variance "
              "decomposition) and could not be loaded. Experiment 6 will be skipped.")

    step_results: dict[str, tuple[bool, float]] = {}

    # --- Experiment 1: Baseline ---
    from experiments.scaling import run_experiment_1
    step_results["Experiment 1 (Baseline)"] = _run_step(
        "Experiment 1: Baseline", run_experiment_1, datasets
    )

    # --- Experiment 2: AdaBoost scaling ---
    from experiments.adaboost_scaling import run_experiment_2
    step_results["Experiment 2 (AdaBoost scaling)"] = _run_step(
        "Experiment 2: AdaBoost scaling", run_experiment_2, datasets
    )

    # --- Experiment 3: Random Forest scaling ---
    from experiments.rf_scaling import run_experiment_3
    step_results["Experiment 3 (RF scaling)"] = _run_step(
        "Experiment 3: Random Forest scaling", run_experiment_3, datasets
    )

    # --- Experiment 4: Head-to-head ---
    from experiments.head_to_head import run_experiment_4
    step_results["Experiment 4 (Head-to-head)"] = _run_step(
        "Experiment 4: Head-to-head comparison", run_experiment_4, datasets
    )

    # --- Experiment 5: Noise robustness ---
    from experiments.noise_robustness import run_experiment_5
    step_results["Experiment 5 (Noise robustness)"] = _run_step(
        "Experiment 5: Noise robustness", run_experiment_5, datasets
    )

    # --- Experiment 6: Bias-variance decomposition (breast_cancer only) ---
    if "breast_cancer" in datasets:
        from experiments.bias_variance import run_experiment_6
        step_results["Experiment 6 (Bias-variance)"] = _run_step(
            "Experiment 6: Bias-variance decomposition",
            run_experiment_6, datasets["breast_cancer"],
        )
    else:
        step_results["Experiment 6 (Bias-variance)"] = (False, 0.0)

    # --- Experiment 7: Unsupervised analysis ---
    from experiments.unsupervised_analysis import run_experiment_7
    step_results["Experiment 7 (Unsupervised)"] = _run_step(
        "Experiment 7: Unsupervised analysis", run_experiment_7, datasets
    )

    # ---------------- summary ----------------
    total_elapsed = time.time() - overall_start
    print(f"\n\n{'='*70}\nRUN SUMMARY\n{'='*70}")
    n_succeeded = 0
    for name, (succeeded, elapsed) in step_results.items():
        status = "OK" if succeeded else "FAILED"
        print(f"  [{status:6s}] {name:45s} {elapsed:7.1f}s")
        if succeeded:
            n_succeeded += 1

    print(f"\n{n_succeeded}/{len(step_results)} experiments completed successfully.")
    print(f"Total runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Figures saved to: figures/")
    print(f"Results (JSON) saved to: results/")

    if n_succeeded < len(step_results):
        print("\nSome experiments FAILED -- see the [FAILED] entries and tracebacks "
              "above for details before treating this as a complete run.")
        sys.exit(1)


if __name__ == "__main__":
    main()