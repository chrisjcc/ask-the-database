"""Comprehensive evaluation harness across Spider and its robustness variants."""

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.environment.test_suite_evaluator import TestSuiteEvaluator
from ask_the_database.inference.pipeline import NL2SQLPipeline
from ask_the_database.rewards.execution_reward import ExecutionEquivalenceEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark_evaluation(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Execute evaluation harness across configured benchmark splits."""
    eval_cfg: dict[str, Any] = (
        cfg if isinstance(cfg, dict) else OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    )

    evaluation_section = eval_cfg.get("evaluation", eval_cfg)
    model_path = evaluation_section.get("model_path", "models/rlvr/final")
    max_turns = int(evaluation_section.get("max_turns", 5))
    max_samples = evaluation_section.get("max_samples")
    output_dir = Path(evaluation_section.get("output_dir", "results"))
    summary_file = Path(
        evaluation_section.get("summary_file", output_dir / "evaluation_summary.json")
    )

    benchmarks = evaluation_section.get(
        "benchmarks", ["spider", "spider_dk", "spider_syn", "spider_realistic"]
    )
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]

    logger.info("Initializing NL2SQL Evaluation Harness for model: %s", model_path)
    pipeline = NL2SQLPipeline(model_path=model_path, max_turns=max_turns)
    ts_evaluator = TestSuiteEvaluator()
    ex_evaluator = ExecutionEquivalenceEvaluator()

    leaderboard: dict[str, Any] = {}

    for bench in benchmarks:
        logger.info("Evaluating Benchmark: %s...", bench)

        # Candidate file paths for the benchmark split
        candidate_paths = [
            Path(f"data/processed/{bench}_eval.jsonl"),
            Path(f"data/processed/{bench}_validation.jsonl"),
            Path(f"data/processed/{bench}.jsonl"),
        ]
        dataset_path = None
        for p in candidate_paths:
            if p.exists():
                dataset_path = p
                break

        samples: list[dict[str, Any]] = []
        if dataset_path is not None:
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            # Check if custom samples injected in cfg for testing
            samples = evaluation_section.get("custom_samples", {}).get(bench, [])

        if not samples:
            logger.warning("Benchmark split '%s' not found. Skipping.", bench)
            continue

        if max_samples is not None:
            samples = samples[: int(max_samples)]

        total = len(samples)
        ts_passed, ex_passed, valid_count, recoveries = 0, 0, 0, 0

        for s in tqdm(samples, desc=bench):
            question = s["question"]
            db_path = s["db_path"]
            db_id = s.get("db_id", "default_db")
            gold_sql = s.get("gold_sql", "SELECT 1;")

            trajectory = pipeline.run_interactive(
                question=question,
                db_path=db_path,
                db_id=db_id,
                max_turns=max_turns,
                gold_sql=gold_sql,
            )
            final_sql = trajectory.final_query

            # 1. Valid execution rate (VA)
            if trajectory.final_execution_success:
                valid_count += 1

            # 2. Execution accuracy (EX)
            db_env = SQLiteEnvironment(db_path)
            res_gold = db_env.execute(gold_sql)
            ex_res = ex_evaluator.compare(trajectory.final_execution_result, res_gold)
            if ex_res.is_equivalent:
                ex_passed += 1

            # 3. Test-suite accuracy (TS)
            ts_res = ts_evaluator.evaluate_test_suite(db_path, final_sql, gold_sql)
            if ts_res.passed_all_suites:
                ts_passed += 1

            # 4. Multi-turn recovery rate (MRR)
            if trajectory.had_error_recovery and ts_res.passed_all_suites:
                recoveries += 1

        leaderboard[bench] = {
            "total_samples": total,
            "test_suite_accuracy_ts": round(ts_passed / total * 100, 2) if total else 0.0,
            "execution_accuracy_ex": round(ex_passed / total * 100, 2) if total else 0.0,
            "valid_execution_rate_va": round(valid_count / total * 100, 2) if total else 0.0,
            "recovery_rate_mrr": (
                round(recoveries / max(1, total - ex_passed) * 100, 2) if total else 0.0
            ),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2)

    logger.info("Evaluation Complete. Results saved to %s", summary_file)
    return leaderboard


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI main entrypoint."""
    logger.info("Starting NL2SQL Evaluation Harness with config:\n%s", OmegaConf.to_yaml(cfg))
    run_benchmark_evaluation(cfg)


if __name__ == "__main__":
    main()
