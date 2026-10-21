"""CLI entrypoint for reviewed trajectory collection in Harness-Zero agent distillation.

Collects multi-turn rollouts reviewed at the response boundary by HarnessReviewer
across stratified Spider tasks, formats trajectories as clean ChatML conversations,
and exports data for Supervised Fine-Tuning (SFT).

Usage:
    python scripts/collect_reviewed_trajectories.py --dry-run
    python scripts/collect_reviewed_trajectories.py --n-medium 150 --n-hard 200 --n-extra 150
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ask_the_database.distillation.collector import (
    MockStudentPolicy, ReviewedTrajectoryCollector, StudentPolicy,
    sample_stratified_spider_tasks)
from ask_the_database.distillation.harness_reviewer import HarnessReviewer
from ask_the_database.training.adapters.skyrl_adapter import \
    SkyRLSQLEnvironment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def build_student_policy(
    backend: str, model_name: str, temperature: float = 0.0
) -> StudentPolicy:
    """Instantiate student policy based on requested backend."""
    if backend == "mock":
        return MockStudentPolicy()
    elif backend == "litellm":
        try:
            import litellm

            class LiteLLMStudentPolicy:
                def __init__(self, model: str, temp: float) -> None:
                    self.model = model
                    self.temperature = temp

                def generate(
                    self,
                    observation: str,
                    history: list[dict[str, str]],
                    instance: dict[str, Any],
                    turn: int,
                ) -> str:
                    resp = litellm.completion(
                        model=self.model,
                        messages=history,
                        temperature=self.temperature,
                        max_tokens=512,
                    )
                    content = resp.choices[0].message.content or ""
                    return content.strip()

            return LiteLLMStudentPolicy(model_name, temperature)
        except ImportError:
            logger.warning("litellm not installed; falling back to MockStudentPolicy.")
            return MockStudentPolicy()
    else:
        logger.warning("Backend '%s' not supported; using MockStudentPolicy.", backend)
        return MockStudentPolicy()


def build_teacher_callable(backend: str, model_name: str) -> Any | None:
    """Instantiate optional teacher reviewer callable."""
    if backend == "rule":
        return None  # Pure rule engine
    elif backend == "litellm":
        try:
            import litellm

            def litellm_teacher(teacher_input: dict[str, Any]) -> dict[str, Any]:
                system_prompt = (
                    "You are a database agent harness reviewer enforcing procedural grounding.\n"
                    "Inspect the student proposal. If sound, return JSON with decision='PASS'.\n"
                    "If flawed, return decision='REPLACE' and replacement_action from [inspect_schema, sample_data, execute_sql, final_answer]."
                )
                user_content = json.dumps(teacher_input)
                resp = litellm.completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw_text = resp.choices[0].message.content or "{}"
                return json.loads(raw_text)

            return litellm_teacher
        except Exception as e:
            logger.warning(
                "Failed to configure litellm teacher: %s. Using rule engine.", e
            )
            return None
    return None


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    """Execute trajectory collection workflow."""
    logger.info("Starting Reviewed Trajectory Collection with args: %s", vars(args))

    # Resolve paths
    raw_spider_path = Path(args.raw_spider)
    databases_dir = Path(args.databases_dir)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics_output)
    rules_config_path = Path(args.rules_config) if args.rules_config else None

    # Fallback search if default raw path does not exist
    if not raw_spider_path.exists():
        fallback_processed = Path("data/processed/train.jsonl")
        if fallback_processed.exists():
            logger.info(
                "Raw spider path not found. Falling back to %s", fallback_processed
            )
            raw_spider_path = fallback_processed
        else:
            raise FileNotFoundError(
                f"Spider training data not found at {raw_spider_path}"
            )

    # Dry-run override
    if args.dry_run:
        logger.info("Dry-run mode enabled. Collecting 5 sample trajectories...")
        n_med, n_hrd, n_ext = 2, 2, 1
    else:
        n_med, n_hrd, n_ext = args.n_medium, args.n_hard, args.n_extra

    # 1. Sample stratified tasks
    tasks = sample_stratified_spider_tasks(
        spider_path=raw_spider_path,
        databases_dir=databases_dir,
        n_medium=n_med,
        n_hard=n_hrd,
        n_extra=n_ext,
        seed=args.seed,
    )

    if args.max_samples and len(tasks) > args.max_samples:
        tasks = tasks[: args.max_samples]

    logger.info("Loaded %d stratified tasks for trajectory collection.", len(tasks))

    # 2. Build teacher reviewer and student policy
    teacher_callable = build_teacher_callable(args.teacher_backend, args.teacher_model)
    reviewer = HarnessReviewer(
        config=rules_config_path,
        teacher_llm_callable=teacher_callable,
    )
    student = build_student_policy(args.student_backend, args.student_model)

    env_cfg = OmegaConf.create(
        {
            "max_turns": args.max_turns,
            "step_penalty": -0.05,
        }
    )
    env = SkyRLSQLEnvironment(env_cfg)

    collector = ReviewedTrajectoryCollector(
        reviewer=reviewer,
        env=env,
        student=student,
        max_turns=args.max_turns,
        require_success=not args.dry_run,  # in dry run, accept rollouts regardless of gold execution
    )

    # 3. Collect trajectories
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    records, metrics = collector.collect_batch(
        instances=tasks,
        output_path=output_path,
        progress_bar=not args.no_progress,
    )

    # Save metrics JSON
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved %d reviewed trajectories to: %s", len(records), output_path)
    logger.info("Saved collection metrics to: %s", metrics_path)
    return metrics


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Collect reviewed trajectories for Harness-Zero agent distillation."
    )
    parser.add_argument(
        "--raw-spider",
        type=str,
        default="data/raw/spider_data/train_spider.json",
        help="Path to Spider training split JSON or processed JSONL.",
    )
    parser.add_argument(
        "--databases-dir",
        type=str,
        default="data/databases",
        help="Directory containing SQLite database files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/distillation/reviewed_trajectories.jsonl",
        help="Output path for reviewed ChatML trajectories JSONL.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="data/distillation/collection_metrics.json",
        help="Output path for collection summary metrics JSON.",
    )
    parser.add_argument(
        "--rules-config",
        type=str,
        default="configs/distillation/k_sql_rules.yaml",
        help="Path to K_SQL procedural rules YAML config.",
    )
    parser.add_argument(
        "--n-medium", type=int, default=150, help="Target count for Medium tasks."
    )
    parser.add_argument(
        "--n-hard", type=int, default=200, help="Target count for Hard tasks."
    )
    parser.add_argument(
        "--n-extra", type=int, default=150, help="Target count for Extra Hard tasks."
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed.")
    parser.add_argument(
        "--max-turns", type=int, default=5, help="Turn limit per rollout."
    )
    parser.add_argument(
        "--student-backend",
        type=str,
        default="mock",
        choices=["mock", "hf", "litellm"],
        help="Student policy backend.",
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Student model identifier.",
    )
    parser.add_argument(
        "--teacher-backend",
        type=str,
        default="rule",
        choices=["rule", "litellm", "mock"],
        help="Teacher reviewer backend.",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gemini/gemini-2.0-flash",
        help="Teacher model identifier for LiteLLM (default: gemini/gemini-2.0-flash; alt: anthropic/claude-3-5-sonnet-20241022).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute hermetic 5-sample collection smoke test.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Upper limit on collected samples.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive tqdm progress bar.",
    )

    return parser.parse_args(args)


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_args(sys.argv[1:])
    run_collection(args)


if __name__ == "__main__":
    main()
