"""Comprehensive evaluation harness across Spider and its robustness variants."""

import ask_the_database  # noqa: F401 - Triggers sanitize_user_site() before any third-party imports

# isort: split

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ask_the_database.environment.sqlite_env import SQLiteEnvironment
from ask_the_database.environment.test_suite_evaluator import TestSuiteEvaluator
from ask_the_database.inference.candidate_sampler import CandidateSampler
from ask_the_database.inference.pipeline import NL2SQLPipeline
from ask_the_database.inference.trajectory_verifier import (
    TrajectoryCandidateSampler,
    TrajectoryVerifierSelector,
)
from ask_the_database.inference.verifier_selector import VerifierSelector
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

    use_verifier = bool(evaluation_section.get("use_verifier", False))
    use_trajectory_verifier = bool(evaluation_section.get("use_trajectory_verifier", False))
    verification_mode = str(
        evaluation_section.get(
            "verification_mode",
            "trajectory" if use_trajectory_verifier else ("query" if use_verifier else "none"),
        )
    ).lower()
    num_candidates = int(evaluation_section.get("num_candidates", 5))
    temperature = float(evaluation_section.get("temperature", 0.7))

    pipeline = NL2SQLPipeline(model_path=model_path, max_turns=max_turns)
    if pipeline.model is None and model_path:
        if pipeline.load_error:
            raise RuntimeError(
                f"Failed to load model weights from '{model_path}': {pipeline.load_error}. "
                "Aborting evaluation to prevent generating misleading fallback results."
            )
        logger.warning(
            "⚠️  WARNING: Could not load trained model weights from '%s'! "
            "Pipeline is running in OFFLINE MOCK/FALLBACK mode. "
            "Results will NOT reflect real model performance.",
            model_path,
        )

    verifier_selector: VerifierSelector | None = None
    trajectory_selector: TrajectoryVerifierSelector | None = None

    if verification_mode == "trajectory":
        logger.info(
            "Initializing TrajectoryVerifierSelector (Best-of-%d, T=%.2f, Max Turns=%d) for model: %s",
            num_candidates,
            temperature,
            max_turns,
            model_path,
        )
        traj_sampler = TrajectoryCandidateSampler(
            pipeline=pipeline,
            num_candidates=num_candidates,
            temperature=temperature,
            max_turns=max_turns,
        )
        trajectory_selector = TrajectoryVerifierSelector(sampler=traj_sampler)
    elif verification_mode == "query" or use_verifier:
        logger.info(
            "Initializing VerifierSelector (Best-of-%d, T=%.2f) for model: %s",
            num_candidates,
            temperature,
            model_path,
        )
        sampler = CandidateSampler(
            model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            model_path=model_path,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        verifier_selector = VerifierSelector(sampler=sampler)
    else:
        logger.info("Initializing NL2SQL Evaluation Harness for model: %s", model_path)

    ts_evaluator = TestSuiteEvaluator()
    ex_evaluator = ExecutionEquivalenceEvaluator()

    leaderboard: dict[str, Any] = {}

    for bench in benchmarks:
        logger.info("Evaluating Benchmark: %s...", bench)

        # Candidate file paths for the benchmark split
        candidate_paths = [
            Path(f"data/processed/{bench}_eval.jsonl"),
            Path(f"data/processed/{bench}_validation.jsonl"),
            Path(f"data/processed/{bench}_val.jsonl"),
            Path(f"data/processed/{bench}.jsonl"),
        ]
        if bench in ("spider", "spider_dev", "val"):
            candidate_paths.append(Path("data/processed/val.jsonl"))
        dataset_path = None
        for p in candidate_paths:
            if p.exists():
                dataset_path = p
                break

        samples: list[dict[str, Any]] = []
        custom_candidates = evaluation_section.get("custom_samples", {}).get(bench)
        if custom_candidates is not None:
            samples = list(custom_candidates)
        elif dataset_path is not None:
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

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

            if trajectory_selector is not None:
                traj_outcome = trajectory_selector.run_pipeline(
                    question=question,
                    db_path=db_path,
                    db_id=db_id,
                    num_candidates=num_candidates,
                    temperature=temperature,
                    gold_sql=gold_sql,
                )
                final_sql = traj_outcome.selected_query
                exec_env = SQLiteEnvironment(db_path)
                final_exec_res = (
                    traj_outcome.selected_trajectory.final_execution_result
                    or exec_env.execute(final_sql)
                )
                final_success = final_exec_res.success
                had_recovery = traj_outcome.selected_trajectory.had_error_recovery
            elif verifier_selector is not None:
                schema_ddl = SQLiteEnvironment(db_path).get_schema_summary()
                outcome = verifier_selector.run_pipeline(
                    question=question,
                    schema_ddl=schema_ddl,
                    db_path=db_path,
                    num_candidates=num_candidates,
                    temperature=temperature,
                )
                final_sql = outcome.selected_query
                exec_env = SQLiteEnvironment(db_path)
                final_exec_res = exec_env.execute(final_sql)
                final_success = final_exec_res.success
                had_recovery = False
            else:
                trajectory = pipeline.run_interactive(
                    question=question,
                    db_path=db_path,
                    db_id=db_id,
                    max_turns=max_turns,
                    gold_sql=gold_sql,
                )
                final_sql = trajectory.final_query
                final_exec_res = trajectory.final_execution_result
                final_success = trajectory.final_execution_success
                had_recovery = trajectory.had_error_recovery

            # 1. Valid execution rate (VA)
            if final_success:
                valid_count += 1

            # 2. Execution accuracy (EX)
            db_env = SQLiteEnvironment(db_path)
            res_gold = db_env.execute(gold_sql)
            ex_res = ex_evaluator.compare(final_exec_res, res_gold)
            if ex_res.is_equivalent:
                ex_passed += 1

            # 3. Test-suite accuracy (TS)
            ts_res = ts_evaluator.evaluate_test_suite(db_path, final_sql, gold_sql)
            if ts_res.passed_all_suites:
                ts_passed += 1

            # 4. Multi-turn recovery rate (MRR)
            if had_recovery and ts_res.passed_all_suites:
                recoveries += 1

        leaderboard[bench] = {
            "total_samples": total,
            "test_suite_accuracy_ts": (round(ts_passed / total * 100, 2) if total else 0.0),
            "execution_accuracy_ex": (round(ex_passed / total * 100, 2) if total else 0.0),
            "valid_execution_rate_va": (round(valid_count / total * 100, 2) if total else 0.0),
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
