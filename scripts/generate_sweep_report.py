"""Generate a Markdown report for a W&B hyperparameter sweep.

The report is generated from sweep metadata and completed run telemetry
retrieved through the Weights & Biases API.

Usage:
    python scripts/generate_sweep_report.py \
        --fine-tune-method sft \
        --sweep-id <sweep-id>
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_FINE_TUNE_METHODS = ["sft", "grpo"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a report for a W&B hyperparameter sweep."
    )

    parser.add_argument(
        "--fine-tune-method",
        required=True,
        choices=SUPPORTED_FINE_TUNE_METHODS,
        help="Fine-tuning method associated with the sweep.",
    )

    parser.add_argument(
        "--sweep-id",
        required=True,
        help="W&B sweep ID.",
    )

    parser.add_argument(
        "--metric-name",
        help="Override metric name to extract from runs.",
    )

    parser.add_argument(
        "--metric-goal",
        choices=["maximize", "minimize"],
        help="Override metric optimization goal.",
    )

    return parser.parse_args()


def get_sweep_parameters(sweep: wandb.apis.public.Sweep) -> list[str]:
    """Return the parameters actually optimized by the sweep."""
    config = getattr(sweep, "config", {}) or {}
    parameters = config.get("parameters", {}) or {}

    return list(parameters.keys())


def get_sweep_metric(
    sweep: wandb.apis.public.Sweep,
    override_name: str | None = None,
    override_goal: str | None = None,
) -> tuple[str | None, str | None]:
    """Return the sweep objective metric and optimization goal."""
    if override_name and override_goal:
        return override_name, override_goal

    config = getattr(sweep, "config", {}) or {}
    metric = config.get("metric", {}) or {}

    if isinstance(metric, dict):
        name = metric.get("name")
        goal = metric.get("goal")
    else:
        name = None
        goal = None

    return name, goal


def format_parameter_name(name: str) -> str:
    """Convert a W&B parameter name into a readable table heading."""
    return name.replace("_", " ").title()


def generate_sweep_report(
    fine_tune_method: str,
    sweep_id: str,
    metric_name_override: str | None = None,
    metric_goal_override: str | None = None,
) -> None:
    """Generate a Markdown report for a W&B hyperparameter sweep.

    Args:
        fine_tune_method: Fine-tuning method associated with the sweep.
        sweep_id: W&B sweep ID.
        metric_name_override: Override metric name to extract from runs.
        metric_goal_override: Override metric goal (maximize/minimize).
    """
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT")

    if not entity:
        raise RuntimeError("WANDB_ENTITY is not set.")

    if not project:
        raise RuntimeError("WANDB_PROJECT is not set.")

    logger.info("Connecting to Weights & Biases API...")

    api = wandb.Api()

    sweep_path = f"{entity}/{project}/{sweep_id}"

    logger.info(f"Fetching sweep: {sweep_path}")

    try:
        sweep = api.sweep(sweep_path)
    except Exception as exc:
        logger.error(f"Error fetching sweep {sweep_path}: {exc}")
        return

    sweep_name = getattr(sweep, "name", None) or f"Sweep {sweep_id}"

    parameter_names = get_sweep_parameters(sweep)
    metric_name, metric_goal = get_sweep_metric(sweep, metric_name_override, metric_goal_override)

    if not metric_name:
        raise RuntimeError(f"Sweep {sweep_id} does not define a metric.")

    if not parameter_names:
        logger.warning(f"Sweep {sweep_id} does not define optimized parameters.")

    logger.info(f"Sweep objective: {metric_name} ({metric_goal})")
    logger.info(f"Sweep parameters: {parameter_names}")

    runs = []

    for run in sweep.runs:
        if run.state != "finished":
            continue

        if metric_name not in run.summary:
            continue

        row = {
            "Run ID": run.id,
            "Name": run.name,
            "DVC Experiment": f"sweep_{run.id}",
        }

        for parameter_name in parameter_names:
            row[format_parameter_name(parameter_name)] = run.config.get(
                parameter_name,
                "N/A",
            )

        row["Objective"] = run.summary.get(
            metric_name,
            float("inf"),
        )

        row["URL"] = run.url

        runs.append(row)

    if not runs:
        logger.info(f"No completed runs with {metric_name} found for sweep {sweep_id}.")
        return

    df = pd.DataFrame(runs)

    sort_ascending = metric_goal != "maximize"

    df = df.sort_values(
        "Objective",
        ascending=sort_ascending,
    )

    os.makedirs("docs/plots", exist_ok=True)

    sns.set_theme(style="whitegrid")

    report_content = [
        "# Parameter Sweep Report",
        "",
        f"**Fine-tuning method:** `{fine_tune_method}`",
        "",
        f"**Sweep:** `{sweep_name}`",
        "",
        f"**Sweep ID:** `{sweep_id}`",
        "",
        f"**W&B:** `{sweep_path}`",
        "",
        f"**Objective:** `{metric_name}` ({metric_goal})",
        "",
        "This report is dynamically generated from Weights & Biases telemetry.",
        "",
        "## Optimized Parameters",
        "",
    ]

    if parameter_names:
        for parameter_name in parameter_names:
            report_content.append(f"- `{parameter_name}`")
    else:
        report_content.append("No optimized parameters were detected.")

    report_content.extend(
        [
            "",
            "## Top Performing Configurations",
            "",
            df.to_markdown(index=False),
            "",
        ]
    )

    # Generate a one-dimensional parameter/objective plot
    # when the sweep has exactly one optimized parameter.
    if len(parameter_names) == 1:
        parameter_name = parameter_names[0]
        column_name = format_parameter_name(parameter_name)

        valid_runs = df[df[column_name] != "N/A"].copy()

        if not valid_runs.empty:
            valid_runs[column_name] = pd.to_numeric(
                valid_runs[column_name],
                errors="coerce",
            )

            valid_runs["Objective"] = pd.to_numeric(
                valid_runs["Objective"],
                errors="coerce",
            )

            valid_runs = valid_runs.dropna(subset=[column_name, "Objective"])

            if not valid_runs.empty:
                plt.figure(figsize=(8, 6))

                sns.scatterplot(
                    data=valid_runs,
                    x=column_name,
                    y="Objective",
                    s=100,
                )

                if valid_runs[column_name].gt(0).all() and parameter_name == "learning_rate":
                    plt.xscale("log")

                plt.title(f"Sweep: {sweep_name}")
                plt.xlabel(column_name)
                plt.ylabel(metric_name)
                plt.tight_layout()

                plot_path = f"docs/plots/sweep_val_curve_{sweep_id}.png"

                plt.savefig(plot_path)
                plt.close()

                report_content.extend(
                    [
                        "### Parameter vs Objective",
                        "",
                        f"![Parameter vs Objective](plots/sweep_val_curve_{sweep_id}.png)",
                        "",
                    ]
                )

    report_path = f"docs/sweep_report_{fine_tune_method}_{sweep_id}.md"

    with open(report_path, "w") as file:
        file.write("\n".join(report_content))

    logger.info(f"Sweep report successfully generated at {report_path}")


if __name__ == "__main__":
    args = parse_args()

    generate_sweep_report(
        fine_tune_method=args.fine_tune_method,
        sweep_id=args.sweep_id,
        metric_name_override=args.metric_name,
        metric_goal_override=args.metric_goal,
    )
