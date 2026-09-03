from __future__ import annotations

import tempfile
import unittest
import json
import os
from collections import Counter
from pathlib import Path
from unittest import mock

from sql_debug_agent.agent import DebugAgent
from sql_debug_agent.badcase_analysis import analyze_badcases
from sql_debug_agent.comparison import compare_evaluation_reports
from sql_debug_agent.cloud_training import (
    build_colab_archive,
    build_grpo_v2_patch_archive,
    extract_completion_sql,
    make_sql_execution_reward,
    prepare_cloud_training_data,
    summarize_grpo_signal,
)
from sql_debug_agent.cloud_evaluation import (
    build_final_comparison,
    compare_checkpoint_rows,
    summarize_checkpoint_rows,
)
from sql_debug_agent.data_generation import (
    TARGET_DISTRIBUTION,
    generate_training_tasks,
    summarize_distribution,
    task_fingerprint,
)
from sql_debug_agent.data_optimization import (
    V2_TARGET_DISTRIBUTION,
    generate_v2_tasks,
)
from sql_debug_agent.database import create_demo_database, create_training_database
from sql_debug_agent.dataset import load_tasks
from sql_debug_agent.evaluation import evaluate
from sql_debug_agent.experiment_summary import summarize_sft_experiment
from sql_debug_agent.model_errors import ModelConfigurationError
from sql_debug_agent.mlx_repair import MLXGeneratorClient, extract_sql
from sql_debug_agent.mlx_training import (
    DEFAULT_MLX_MODEL,
    build_mlx_lora_command,
    prepare_mlx_dataset,
)
from sql_debug_agent.holdout import build_final_holdout
from sql_debug_agent.ollama_repair import OllamaChatClient, OllamaRepairer
from sql_debug_agent.repair import RuleBasedRepairer
from sql_debug_agent.rl_data import build_rl_dataset
from sql_debug_agent.rl_reward import RobustSQLReward
from sql_debug_agent.rl_replay import compare_replay_runs, export_hard_preferences, summarize_rollouts
from sql_debug_agent.preparation import prepare_sft_data
from sql_debug_agent.openai_repair import (
    OpenAIRepairer,
    OpenAIResponsesClient,
)
from sql_debug_agent.preference_replay import build_corrective_replay_dataset
from sql_debug_agent.verifier import SQLVerifier
from sql_debug_agent.v3_data import (
    V3_TARGET_DISTRIBUTION,
    build_v3_dataset,
    build_v3_holdout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SQLDebugAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = create_demo_database(Path(self.temp_dir.name) / "demo.db")
        verifier = SQLVerifier(self.database)
        self.agent = DebugAgent(verifier, RuleBasedRepairer(self.database), max_turns=3)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_all_demo_bad_cases_are_repaired(self) -> None:
        tasks = load_tasks(PROJECT_ROOT / "data" / "tasks.jsonl")
        summary, _ = evaluate(self.agent, tasks)
        self.assertEqual(summary.baseline_success_count, 0)
        self.assertEqual(summary.final_success_count, len(tasks))
        self.assertEqual(summary.repaired_count, len(tasks))

    def test_write_query_is_rejected(self) -> None:
        result = self.agent.verifier.execute("DELETE FROM customers")
        self.assertFalse(result.executable)
        self.assertIn("只允许", result.error or "")

    def test_execution_only_mode_is_explicit(self) -> None:
        result = self.agent.verifier.verify("SELECT COUNT(*) FROM customers")
        self.assertTrue(result.passed)
        self.assertIn("没有验证语义正确性", result.feedback)

    def test_execution_match_ignores_equivalent_column_labels(self) -> None:
        result = self.agent.verifier.verify(
            "SELECT AVG(balance) AS avg_balance FROM accounts",
            "SELECT AVG(balance) FROM accounts",
        )
        self.assertTrue(result.passed)
        self.assertIn("列名", result.feedback)

    def test_prepare_sft_data_has_separate_splits(self) -> None:
        tasks = load_tasks(PROJECT_ROOT / "data" / "tasks.jsonl")
        output_dir = Path(self.temp_dir.name) / "processed"
        train_path, eval_path, train_count, eval_count = prepare_sft_data(
            tasks,
            self.agent.verifier,
            self.database,
            output_dir,
        )
        self.assertTrue(train_path.exists())
        self.assertTrue(eval_path.exists())
        self.assertEqual(train_count + eval_count, len(tasks))
        self.assertGreater(train_count, 0)
        self.assertGreater(eval_count, 0)

    def test_30_baseline_tasks_start_as_bad_cases(self) -> None:
        tasks = load_tasks(PROJECT_ROOT / "data" / "baseline_eval.jsonl")
        self.assertEqual(len(tasks), 30)
        for task in tasks:
            with self.subTest(task=task.task_id):
                reference = self.agent.verifier.execute(task.reference_sql)
                self.assertTrue(reference.executable, reference.error)
                verification = self.agent.verifier.verify(
                    task.initial_sql, task.reference_sql
                )
                self.assertFalse(verification.passed)

    def test_generated_training_data_is_valid_balanced_and_leak_free(self) -> None:
        training_database = create_training_database(
            Path(self.temp_dir.name) / "training.db"
        )
        test_tasks = load_tasks(PROJECT_ROOT / "data" / "baseline_eval.jsonl")
        test_fingerprints = {task_fingerprint(task) for task in test_tasks}
        tasks = generate_training_tasks(
            training_database, excluded_fingerprints=test_fingerprints
        )

        self.assertEqual(len(tasks), 200)
        self.assertEqual(summarize_distribution(tasks), dict(sorted(TARGET_DISTRIBUTION.items())))
        fingerprints = [task_fingerprint(task) for task in tasks]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertTrue(set(fingerprints).isdisjoint(test_fingerprints))

        verifier = SQLVerifier(training_database)
        for task in tasks:
            with self.subTest(task=task.task_id):
                self.assertTrue(verifier.execute(task.reference_sql).executable)
                self.assertFalse(
                    verifier.verify(task.initial_sql, task.reference_sql).passed
                )

        output_dir = Path(self.temp_dir.name) / "sft_v1"
        train_path, eval_path, train_count, eval_count = prepare_sft_data(
            tasks, verifier, training_database, output_dir
        )
        self.assertEqual(train_count + eval_count, 200)
        train_records = [json.loads(line) for line in train_path.read_text().splitlines()]
        eval_records = [json.loads(line) for line in eval_path.read_text().splitlines()]
        train_templates = {
            record["metadata"]["template_id"] for record in train_records
        }
        eval_templates = {
            record["metadata"]["template_id"] for record in eval_records
        }
        self.assertTrue(train_templates.isdisjoint(eval_templates))
        expected_types = set(TARGET_DISTRIBUTION)
        self.assertEqual(
            {record["metadata"]["error_type"] for record in train_records},
            expected_types,
        )
        self.assertEqual(
            {record["metadata"]["error_type"] for record in eval_records},
            expected_types,
        )

    def test_badcase_analysis_ranks_failures_and_records_stage_decision(self) -> None:
        source = Path(self.temp_dir.name) / "baseline.json"
        json_output = Path(self.temp_dir.name) / "analysis.json"
        markdown_output = Path(self.temp_dir.name) / "analysis.md"
        source.write_text(
            json.dumps(
                {
                    "summary": {
                        "task_count": 2,
                        "final_success_count": 1,
                        "final_accuracy": 0.5,
                        "by_error_type": {
                            "join_type": {
                                "task_count": 1,
                                "final_success_count": 0,
                                "final_accuracy": 0.0,
                            },
                            "syntax_error": {
                                "task_count": 1,
                                "final_success_count": 1,
                                "final_accuracy": 1.0,
                            },
                        },
                    },
                    "reports": [
                        {
                            "task": {
                                "task_id": "join_1",
                                "question": "包括零交易客户",
                                "initial_sql": "SELECT 1",
                                "reference_sql": "SELECT 2",
                                "error_type": "join_type",
                            },
                            "report": {
                                "success": False,
                                "final_sql": "SELECT 1",
                                "steps": [{"feedback": "SQL 可以执行，但结果不正确。"}],
                            },
                        },
                        {
                            "task": {
                                "task_id": "syntax_1",
                                "question": "修复语法",
                                "initial_sql": "SELEC 1",
                                "reference_sql": "SELECT 1",
                                "error_type": "syntax_error",
                            },
                            "report": {
                                "success": True,
                                "final_sql": "SELECT 1",
                                "steps": [{"feedback": "通过"}],
                            },
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        analysis = analyze_badcases(source, json_output, markdown_output)
        self.assertEqual(analysis["failure_count"], 1)
        self.assertEqual(analysis["priorities"][0]["error_type"], "join_type")
        self.assertEqual(analysis["failure_behaviors"]["unchanged_sql"], 1)
        self.assertEqual(analysis["decision"]["next_stage"], "SFT")
        self.assertTrue(json_output.exists())
        self.assertIn("优先改进方向", markdown_output.read_text(encoding="utf-8"))

    def test_report_comparison_detects_fixed_and_regressed_tasks(self) -> None:
        base_path = Path(self.temp_dir.name) / "base.json"
        candidate_path = Path(self.temp_dir.name) / "candidate.json"
        json_output = Path(self.temp_dir.name) / "comparison.json"
        markdown_output = Path(self.temp_dir.name) / "comparison.md"

        def report(successes):
            return {
                "summary": {
                    "task_count": 2,
                    "final_accuracy": sum(successes) / 2,
                    "by_error_type": {
                        "join_type": {
                            "final_accuracy": sum(successes) / 2,
                        }
                    },
                },
                "reports": [
                    {
                        "task": {"task_id": f"task_{index}"},
                        "report": {"success": success},
                    }
                    for index, success in enumerate(successes, start=1)
                ],
            }

        base_path.write_text(json.dumps(report([False, True])), encoding="utf-8")
        candidate_path.write_text(
            json.dumps(report([True, False])), encoding="utf-8"
        )
        comparison = compare_evaluation_reports(
            base_path, candidate_path, json_output, markdown_output
        )
        self.assertEqual(comparison["fixed_tasks"], ["task_1"])
        self.assertEqual(comparison["regressed_tasks"], ["task_2"])
        self.assertFalse(comparison["decision"]["start_rl"])
        self.assertTrue(markdown_output.exists())

    def test_experiment_summary_blocks_rl_when_sft_regresses(self) -> None:
        report_paths = []

        def report(successes):
            return {
                "summary": {
                    "task_count": 2,
                    "final_success_count": sum(successes),
                    "final_accuracy": sum(successes) / 2,
                    "by_error_type": {
                        "join_type": {"final_accuracy": sum(successes) / 2}
                    },
                },
                "reports": [
                    {
                        "task": {"task_id": f"task_{index}"},
                        "report": {"success": success},
                    }
                    for index, success in enumerate(successes, start=1)
                ],
            }

        for name, successes in (
            ("base", [True, True]),
            ("v1", [True, False]),
            ("v2", [False, True]),
        ):
            path = Path(self.temp_dir.name) / f"{name}.json"
            path.write_text(json.dumps(report(successes)), encoding="utf-8")
            report_paths.append(path)

        summary = summarize_sft_experiment(
            *report_paths,
            Path(self.temp_dir.name) / "summary.json",
            Path(self.temp_dir.name) / "summary.md",
        )
        self.assertEqual(summary["winner"], "MLX Base")
        self.assertFalse(summary["stage_gate"]["start_rl"])
        self.assertFalse(summary["stage_gate"]["checks"]["best_sft_beats_base"])
        self.assertEqual(
            summary["stage_gate"]["next_stage"],
            "SFT_data_quality_and_generalization",
        )

    def test_v3_data_and_new_holdout_are_valid_and_disjoint(self) -> None:
        holdout_path, holdout = build_v3_holdout(
            self.database, Path(self.temp_dir.name) / "holdout_v3.jsonl"
        )
        training_database = create_training_database(Path(self.temp_dir.name) / "train_v3.db")
        forbidden = (
            load_tasks(PROJECT_ROOT / "data" / "baseline_eval.jsonl")
            + load_tasks(PROJECT_ROOT / "data" / "final_holdout.jsonl")
            + load_tasks(PROJECT_ROOT / "data" / "sft_raw.jsonl")
            + load_tasks(PROJECT_ROOT / "data" / "sft_v2_increment.jsonl")
            + holdout
        )
        manifest = build_v3_dataset(
            training_database,
            PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_train.jsonl",
            PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_eval.jsonl",
            Path(self.temp_dir.name) / "v3_increment.jsonl",
            Path(self.temp_dir.name) / "sft_v3",
            forbidden,
        )
        self.assertEqual(manifest["increment_distribution"], dict(sorted(V3_TARGET_DISTRIBUTION.items())))
        self.assertEqual(manifest["train_count"], 158 + sum(V3_TARGET_DISTRIBUTION.values()))
        self.assertEqual(len(load_tasks(holdout_path)), 28)

    def test_robust_reward_blocks_common_reward_hacks(self) -> None:
        training_database = create_training_database(Path(self.temp_dir.name) / "reward_train.db")
        reward = RobustSQLReward([self.database, training_database])
        reference = "SELECT COUNT(*) FROM customers WHERE region='华北'"
        accidental_match = "SELECT COUNT(*) FROM customers WHERE region='西南'"
        self.assertTrue(self.agent.verifier.verify(accidental_match, reference).passed)
        self.assertFalse(reward.score(accidental_match, reference).matches_all_databases)
        self.assertEqual(reward.score("DELETE FROM customers", reference).total, -1.0)
        self.assertLess(
            reward.score("SELECT * FROM customers WHERE 1=0", "SELECT * FROM customers").total,
            reward.score("SELECT * FROM customers", "SELECT * FROM customers", previous_sql="SELECT * FROM customers WHERE 1=0").total,
        )
        alias = reward.score(
            "SELECT COUNT(*) AS customer_count FROM customers",
            "SELECT COUNT(*) FROM customers",
        )
        self.assertTrue(alias.matches_all_databases)
        reduced_columns = reward.score(
            "SELECT id FROM customers ORDER BY id",
            "SELECT id, name FROM customers ORDER BY id",
        )
        self.assertFalse(reduced_columns.matches_all_databases)
        repeated = reward.score(reference, reference, previous_sql=reference)
        plain_correct = reward.score(reference, reference)
        self.assertLess(repeated.total, plain_correct.total)

    def test_rl_data_excludes_holdouts_and_has_positive_margins(self) -> None:
        training_database = create_training_database(Path(self.temp_dir.name) / "rl_train.db")
        manifest = build_rl_dataset(
            PROJECT_ROOT / "data" / "baseline_eval.jsonl",
            PROJECT_ROOT / "data" / "sft_v3_increment.jsonl",
            self.database,
            training_database,
            Path(self.temp_dir.name) / "rl_data",
        )
        self.assertEqual(manifest["source_task_count"], 90)
        self.assertGreater(manifest["preference_pair_count"], 0)
        self.assertGreater(manifest["reward_margin"]["minimum"], 0)
        self.assertFalse(manifest["training_ready"])
        self.assertEqual(len(manifest["excluded_final_holdouts"]), 2)

    def test_rollout_summary_detects_signal_and_keeps_training_blocked(self) -> None:
        def item(task_id, error_type, total, matched, safe, delta, database_matches):
            return {
                "metadata": {"task_id": task_id, "error_type": error_type},
                "candidate_reward": {
                    "total": total,
                    "safe": safe,
                    "executable_on_all_databases": safe,
                    "matches_all_databases": matched,
                    "repeat_penalty": 0.0,
                    "database_matches": database_matches,
                },
                "reward_delta": delta,
            }

        summary = summarize_rollouts(
            [
                item("a", "join_type", 1.5, True, True, 1.5, [True, True]),
                item("b", "join_type", 0.0, False, True, 0.0, [True, False]),
            ]
        )
        self.assertTrue(summary["reward_gate"]["reward_ready"])
        self.assertFalse(summary["reward_gate"]["training_ready"])
        self.assertEqual(summary["database_disagreement_count"], 1)

    def test_failed_rollouts_export_as_hard_preferences(self) -> None:
        prompts_path = Path(self.temp_dir.name) / "prompts.jsonl"
        rollouts_path = Path(self.temp_dir.name) / "rollouts.jsonl"
        output_path = Path(self.temp_dir.name) / "hard.jsonl"
        prompts_path.write_text(
            json.dumps({
                "messages": [{"role": "user", "content": "fix"}],
                "metadata": {"task_id": "task_1", "error_type": "join_type"},
            }) + "\n",
            encoding="utf-8",
        )
        rollouts_path.write_text(
            json.dumps({
                "metadata": {"task_id": "task_1", "error_type": "join_type"},
                "candidate_sql": "SELECT 1",
                "reference_sql": "SELECT 2",
                "candidate_reward": {"total": 0.0, "matches_all_databases": False},
            }) + "\n",
            encoding="utf-8",
        )
        summary = export_hard_preferences(prompts_path, rollouts_path, output_path)
        self.assertEqual(summary["pair_count"], 1)
        pair = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(pair["metadata"]["negative_type"], "real_model_failure")
        self.assertGreater(pair["reward_margin"], 0)

    def test_corrective_replay_balances_hard_and_protection_records(self) -> None:
        hard_path = Path(self.temp_dir.name) / "hard_pairs.jsonl"
        prompts_path = Path(self.temp_dir.name) / "prompts.jsonl"
        rollouts_path = Path(self.temp_dir.name) / "rollouts.jsonl"
        hard_path.write_text(json.dumps({
            "prompt": [{"role": "user", "content": "fix hard"}],
            "chosen_sql": "SELECT 2",
            "metadata": {"task_id": "hard", "error_type": "join_type"},
        }) + "\n", encoding="utf-8")
        prompts_path.write_text(json.dumps({
            "messages": [{"role": "user", "content": "protect"}],
            "metadata": {"task_id": "safe", "error_type": "syntax_error"},
        }) + "\n", encoding="utf-8")
        rollouts_path.write_text(json.dumps({
            "metadata": {"task_id": "safe", "error_type": "syntax_error"},
            "reference_sql": "SELECT 1",
            "candidate_reward": {"matches_all_databases": True},
        }) + "\n", encoding="utf-8")
        manifest = build_corrective_replay_dataset(
            hard_path,
            prompts_path,
            rollouts_path,
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl",
            Path(self.temp_dir.name) / "preference_replay",
        )
        self.assertEqual(manifest["hard_correction_count"], 1)
        self.assertEqual(manifest["capability_protection_count"], 1)
        self.assertEqual(manifest["train_count"], 2)
        self.assertFalse(manifest["is_dpo"])

    def test_replay_comparison_rejects_candidate_without_net_gain(self) -> None:
        base_summary = Path(self.temp_dir.name) / "base_summary.json"
        candidate_summary = Path(self.temp_dir.name) / "candidate_summary.json"
        base_rollouts = Path(self.temp_dir.name) / "base_rollouts.jsonl"
        candidate_rollouts = Path(self.temp_dir.name) / "candidate_rollouts.jsonl"
        for path in (base_summary, candidate_summary):
            path.write_text(json.dumps({
                "correct_on_all_count": 1,
                "repeated_sql_count": 1,
                "reward_distribution": {"mean": 0.5},
            }), encoding="utf-8")
        record = json.dumps({
            "metadata": {"task_id": "task"},
            "candidate_reward": {"matches_all_databases": True},
        }) + "\n"
        base_rollouts.write_text(record, encoding="utf-8")
        candidate_rollouts.write_text(record, encoding="utf-8")
        comparison = compare_replay_runs(
            base_summary,
            candidate_summary,
            base_rollouts,
            candidate_rollouts,
            Path(self.temp_dir.name) / "comparison.json",
            Path(self.temp_dir.name) / "comparison.md",
        )
        self.assertFalse(comparison["decision"]["accept_candidate"])
        self.assertEqual(comparison["decision"]["selected_adapter"], "sft_v3_30")

    def test_openai_repairer_uses_structured_responses_output(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            decision = json.dumps(
                {
                    "error_type": "schema_linking",
                    "change_summary": "将 amounts 改为 amount",
                    "corrected_sql": "SELECT AVG(amount) FROM transactions",
                },
                ensure_ascii=False,
            )
            return json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": decision}],
                        }
                    ]
                }
            ).encode("utf-8")

        client = OpenAIResponsesClient(
            api_key="test-key", model="test-model", transport=fake_transport
        )
        repairer = OpenAIRepairer(self.database, client)
        result = repairer.repair(
            "计算平均交易金额",
            "SELECT AVG(amounts) FROM transactions",
            "no such column: amounts",
        )
        self.assertEqual(result, "SELECT AVG(amount) FROM transactions")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/responses")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")

    def test_openai_client_requires_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelConfigurationError):
                OpenAIResponsesClient.from_env()

    def test_ollama_repairer_uses_local_chat_and_json_schema(self) -> None:
        captured: dict[str, object] = {}

        def fake_transport(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            decision = json.dumps(
                {
                    "error_type": "schema_linking",
                    "change_summary": "将 amounts 改为 amount",
                    "corrected_sql": "SELECT AVG(amount) FROM transactions",
                },
                ensure_ascii=False,
            )
            return json.dumps(
                {"message": {"role": "assistant", "content": decision}}
            ).encode("utf-8")

        client = OllamaChatClient(model="local-test-model", transport=fake_transport)
        repairer = OllamaRepairer(self.database, client)
        result = repairer.repair(
            "计算平均交易金额",
            "SELECT AVG(amounts) FROM transactions",
            "no such column: amounts",
        )
        self.assertEqual(result, "SELECT AVG(amount) FROM transactions")
        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        payload = captured["payload"]
        self.assertEqual(payload["model"], "local-test-model")
        self.assertEqual(payload["format"]["type"], "object")
        self.assertFalse(payload["stream"])

    def test_ollama_client_does_not_require_api_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            client = OllamaChatClient.from_env()
        self.assertEqual(client.model, "qwen2.5-coder:1.5b-instruct")
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_mlx_data_and_8gb_training_command_are_reproducible(self) -> None:
        training_database = create_training_database(
            Path(self.temp_dir.name) / "mlx_training.db"
        )
        test_tasks = load_tasks(PROJECT_ROOT / "data" / "baseline_eval.jsonl")
        tasks = generate_training_tasks(
            training_database,
            excluded_fingerprints={task_fingerprint(task) for task in test_tasks},
        )
        sft_dir = Path(self.temp_dir.name) / "sft_source"
        sft_train, sft_eval, _, _ = prepare_sft_data(
            tasks, SQLVerifier(training_database), training_database, sft_dir
        )
        output_dir = Path(self.temp_dir.name) / "mlx_data"
        manifest = prepare_mlx_dataset(
            sft_train,
            sft_eval,
            PROJECT_ROOT / "data" / "baseline_eval.jsonl",
            self.database,
            output_dir,
            sql_terminator=True,
        )
        self.assertEqual(manifest["train_count"], 158)
        self.assertEqual(manifest["valid_count"], 42)
        self.assertEqual(manifest["test_count"], 30)
        first_record = json.loads(
            (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(set(first_record), {"messages"})
        self.assertTrue(first_record["messages"][-1]["content"].endswith(";"))

        command = build_mlx_lora_command(
            ".venv/bin/python",
            DEFAULT_MLX_MODEL,
            output_dir,
            Path(self.temp_dir.name) / "adapter",
            iters=5,
            smoke=True,
        )
        self.assertIn("--mask-prompt", command)
        self.assertIn("--grad-checkpoint", command)
        self.assertEqual(command[command.index("--batch-size") + 1], "1")
        self.assertEqual(command[command.index("--num-layers") + 1], "4")
        resume_file = Path(self.temp_dir.name) / "previous.safetensors"
        resumed = build_mlx_lora_command(
            "python",
            DEFAULT_MLX_MODEL,
            output_dir,
            Path(self.temp_dir.name) / "continued",
            5,
            False,
            learning_rate=5e-6,
            resume_adapter_file=resume_file,
        )
        self.assertEqual(resumed[resumed.index("--learning-rate") + 1], "5e-06")
        self.assertEqual(
            resumed[resumed.index("--resume-adapter-file") + 1],
            str(resume_file.resolve()),
        )

    def test_mlx_client_loads_adapter_once_and_extracts_sql(self) -> None:
        calls = {"load": 0, "generate": 0}

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "formatted prompt"

        def fake_loader(model_name, adapter_path):
            calls["load"] += 1
            self.assertEqual(model_name, "test-model")
            self.assertTrue(adapter_path.endswith("adapter"))
            return object(), FakeTokenizer()

        def fake_generator(model, tokenizer, **kwargs):
            calls["generate"] += 1
            self.assertEqual(kwargs["prompt"], "formatted prompt")
            return "修正如下：\n```sql\nSELECT amount FROM transactions\n```"

        client = MLXGeneratorClient(
            model_name="test-model",
            adapter_path=Path(self.temp_dir.name) / "adapter",
            loader=fake_loader,
            generator=fake_generator,
        )
        messages = [{"role": "user", "content": "修复 SQL"}]
        self.assertEqual(
            client.create_sql_repair(messages), "SELECT amount FROM transactions"
        )
        client.create_sql_repair(messages)
        self.assertEqual(calls, {"load": 1, "generate": 2})
        self.assertEqual(extract_sql("答案是 SELECT 1"), "SELECT 1")
        self.assertEqual(
            extract_sql("SELECT amount FROM transactions<|im_end|>"),
            "SELECT amount FROM transactions",
        )
        self.assertEqual(
            extract_sql("SELECT amount FROM transactions\n!\n!\n!"),
            "SELECT amount FROM transactions",
        )
        self.assertEqual(
            extract_sql("SELECT amount FROM transactions; trailing"),
            "SELECT amount FROM transactions;",
        )

    def test_v2_increment_and_final_holdout_are_valid_and_disjoint(self) -> None:
        holdout_path, holdout_tasks = build_final_holdout(
            self.database, Path(self.temp_dir.name) / "holdout.jsonl"
        )
        self.assertTrue(holdout_path.exists())
        self.assertEqual(len(holdout_tasks), 35)
        self.assertEqual(
            Counter(task.error_type for task in holdout_tasks),
            Counter(
                {
                    "aggregation": 5,
                    "date": 5,
                    "duplicate_counting": 5,
                    "filter": 5,
                    "join_type": 5,
                    "schema_linking": 5,
                    "syntax_error": 5,
                }
            ),
        )

        forbidden_tasks = (
            load_tasks(PROJECT_ROOT / "data" / "baseline_eval.jsonl")
            + load_tasks(PROJECT_ROOT / "data" / "sft_raw.jsonl")
            + holdout_tasks
        )
        forbidden = {task_fingerprint(task) for task in forbidden_tasks}
        training_database = create_training_database(
            Path(self.temp_dir.name) / "v2.db"
        )
        additions = generate_v2_tasks(training_database, forbidden)
        self.assertEqual(len(additions), 60)
        self.assertEqual(
            Counter(task.error_type for task in additions),
            Counter(V2_TARGET_DISTRIBUTION),
        )
        self.assertTrue(
            {task_fingerprint(task) for task in additions}.isdisjoint(forbidden)
        )

    def test_last_turn_does_not_create_an_unverified_final_sql(self) -> None:
        class AlwaysChangesRepairer:
            def repair(self, question, sql, feedback):
                return f"{sql} changed"

        agent = DebugAgent(
            self.agent.verifier, AlwaysChangesRepairer(), max_turns=1
        )
        report = agent.run(
            "错误 SQL", "SELECT missing FROM customers", "SELECT name FROM customers"
        )
        self.assertFalse(report.success)
        self.assertEqual(report.final_sql, "SELECT missing FROM customers")
        self.assertEqual(len(report.steps), 1)

    def test_cloud_grpo_export_and_dual_database_reward(self) -> None:
        output_dir = Path(self.temp_dir.name) / "cloud_data"
        manifest = prepare_cloud_training_data(
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_train.jsonl",
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl",
            PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl",
            PROJECT_ROOT / "data" / "finance_demo.db",
            PROJECT_ROOT / "data" / "train_finance.db",
            output_dir,
            PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl",
        )
        self.assertEqual(manifest["sft_train_count"], 218)
        self.assertEqual(manifest["sft_eval_count"], 42)
        self.assertEqual(manifest["grpo_prompt_count"], 90)
        self.assertEqual(manifest["grpo_hard_prompt_count"], 26)
        self.assertEqual(
            manifest["grpo_hard_source_distribution"],
            {"development": 9, "v3_increment": 17},
        )
        self.assertFalse(manifest["final_holdouts_included"])
        hard_records = [
            json.loads(line)
            for line in (output_dir / "grpo_hard_train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(hard_records), 26)
        self.assertEqual(len({item["task_id"] for item in hard_records}), 26)
        sft_record = json.loads(
            (output_dir / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        self.assertEqual(sft_record["prompt"][-1]["role"], "user")
        self.assertEqual(sft_record["completion"][-1]["role"], "assistant")
        reward = make_sql_execution_reward(
            [output_dir / "finance_demo.db", output_dir / "train_finance.db"]
        )
        values = reward(
            completions=[[{"role": "assistant", "content": "```sql\nSELECT COUNT(*) FROM customers\n```"}]],
            reference_sql=["SELECT COUNT(*) FROM customers"],
            previous_sql=["SELEC COUNT(*) FROM customers"],
        )
        self.assertEqual(values, [1.5])
        self.assertEqual(
            extract_completion_sql("答案：SELECT COUNT(*) FROM customers"),
            "SELECT COUNT(*) FROM customers",
        )

    def test_colab_archive_excludes_holdouts_and_model_weights(self) -> None:
        data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
        prepare_cloud_training_data(
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_train.jsonl",
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl",
            PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl",
            PROJECT_ROOT / "data" / "finance_demo.db",
            PROJECT_ROOT / "data" / "train_finance.db",
            data_dir,
            PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl",
        )
        archive_path = Path(self.temp_dir.name) / "colab.zip"
        build_colab_archive(PROJECT_ROOT, archive_path)
        import zipfile

        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
        self.assertTrue(any(name.endswith("cloud_grpo/data/grpo_train.jsonl") for name in names))
        self.assertTrue(
            any(name.endswith("cloud_grpo/data/grpo_hard_train.jsonl") for name in names)
        )
        self.assertFalse(any("final_holdout" in name for name in names))
        self.assertFalse(any("eval_data" in name for name in names))
        self.assertFalse(any("evaluate_checkpoints.py" in name for name in names))
        self.assertFalse(any(name.endswith(".safetensors") for name in names))

    def test_grpo_signal_summary_detects_nonzero_advantage_signal(self) -> None:
        summary = summarize_grpo_signal(
            [
                {"loss": 0.0, "reward": 1.5, "reward_std": 0.0},
                {
                    "loss": 0.01,
                    "reward": 0.8,
                    "reward_std": 0.4,
                    "frac_reward_zero_std": 0.0,
                },
                {"train_runtime": 10.0},
            ]
        )
        self.assertEqual(summary["logged_steps"], 2)
        self.assertEqual(summary["steps_with_reward_variance"], 1)
        self.assertEqual(summary["reward_variance_step_rate"], 0.5)
        self.assertTrue(summary["effective_learning_signal"])

    def test_grpo_v2_patch_contains_only_runtime_updates(self) -> None:
        data_dir = PROJECT_ROOT / "cloud_grpo" / "data"
        prepare_cloud_training_data(
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_train.jsonl",
            PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl",
            PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl",
            PROJECT_ROOT / "data" / "finance_demo.db",
            PROJECT_ROOT / "data" / "train_finance.db",
            data_dir,
            PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl",
        )
        archive_path = Path(self.temp_dir.name) / "grpo-v2-patch.zip"
        build_grpo_v2_patch_archive(PROJECT_ROOT, archive_path)
        import zipfile

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        self.assertEqual(
            names,
            {
                "cloud_grpo/train_grpo.py",
                "cloud_grpo/data/grpo_hard_train.jsonl",
                "sql_debug_agent/cloud_training.py",
            },
        )

    def test_final_checkpoint_comparison_detects_grpo_gain_and_regression(self) -> None:
        base = [
            {"task_id": "a", "error_type": "syntax_error", "correct": False},
            {"task_id": "b", "error_type": "join_type", "correct": True},
            {"task_id": "c", "error_type": "join_type", "correct": False},
        ]
        sft = [
            {"task_id": "a", "error_type": "syntax_error", "correct": True},
            {"task_id": "b", "error_type": "join_type", "correct": True},
            {"task_id": "c", "error_type": "join_type", "correct": False},
        ]
        grpo = [
            {"task_id": "a", "error_type": "syntax_error", "correct": True},
            {"task_id": "b", "error_type": "join_type", "correct": False},
            {"task_id": "c", "error_type": "join_type", "correct": True},
        ]
        summary = summarize_checkpoint_rows(sft)
        self.assertEqual(summary["correct_count"], 2)
        change = compare_checkpoint_rows(sft, grpo)
        self.assertEqual(change["fixed_tasks"], ["c"])
        self.assertEqual(change["regressed_tasks"], ["b"])
        comparison = build_final_comparison(
            {"base": base, "sft": sft, "grpo": grpo},
            {"dataset": "frozen.jsonl"},
        )
        self.assertEqual(
            comparison["decision"]["verdict"], "no_measurable_grpo_gain"
        )


if __name__ == "__main__":
    unittest.main()
