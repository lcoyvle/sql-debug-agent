from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import DebugAgent
from .badcase_analysis import analyze_badcases
from .comparison import compare_evaluation_reports
from .data_generation import (
    build_training_dataset,
    summarize_distribution,
    task_fingerprint,
)
from .data_optimization import build_v2_dataset
from .database import (
    DEMO_DB_VERSION,
    create_demo_database,
    create_training_database,
    get_database_version,
    get_schema,
)
from .dataset import DebugTask, load_tasks
from .evaluation import evaluate
from .experiment_summary import summarize_sft_experiment
from .model_errors import ModelAPIError, ModelConfigurationError
from .mlx_repair import MLXGeneratorClient, MLXRepairer
from .mlx_training import (
    DEFAULT_MLX_MODEL,
    prepare_mlx_dataset,
    run_mlx_lora_training,
)
from .ollama_repair import (
    DEFAULT_OLLAMA_MODEL,
    OllamaChatClient,
    OllamaRepairer,
)
from .holdout import build_final_holdout
from .openai_repair import (
    DEFAULT_MODEL,
    OpenAIRepairer,
    OpenAIResponsesClient,
)
from .preparation import prepare_sft_data
from .preference_replay import build_corrective_replay_dataset
from .repair import RuleBasedRepairer
from .rl_data import build_rl_dataset
from .rl_replay import compare_replay_runs, export_hard_preferences, replay_model_rollouts
from .verifier import SQLVerifier
from .v3_data import build_v3_dataset, build_v3_holdout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "finance_demo.db"
DEFAULT_TASKS = PROJECT_ROOT / "data" / "tasks.jsonl"
DEFAULT_BASELINE_TASKS = PROJECT_ROOT / "data" / "baseline_eval.jsonl"
DEFAULT_TRAIN_DB = PROJECT_ROOT / "data" / "train_finance.db"
DEFAULT_TRAIN_TASKS = PROJECT_ROOT / "data" / "sft_raw.jsonl"
DEFAULT_BASELINE_REPORT = PROJECT_ROOT / "artifacts" / "baseline_report.json"
DEFAULT_MLX_DATA = PROJECT_ROOT / "artifacts" / "mlx_data"
DEFAULT_MLX_CACHE = PROJECT_ROOT / "artifacts" / "hf_cache"
DEFAULT_SFT_ADAPTER = PROJECT_ROOT / "artifacts" / "adapters" / "sft_v1"
DEFAULT_SMOKE_ADAPTER = PROJECT_ROOT / "artifacts" / "adapters" / "smoke"
DEFAULT_FINAL_HOLDOUT = PROJECT_ROOT / "data" / "final_holdout.jsonl"
DEFAULT_SFT_V2_DIR = PROJECT_ROOT / "artifacts" / "sft_v2"
DEFAULT_MLX_V2_DATA = PROJECT_ROOT / "artifacts" / "mlx_data_v2"
DEFAULT_SFT_V2_ADAPTER = PROJECT_ROOT / "artifacts" / "adapters" / "sft_v2"
DEFAULT_FINAL_HOLDOUT_V3 = PROJECT_ROOT / "data" / "final_holdout_v3.jsonl"
DEFAULT_SFT_V3_DIR = PROJECT_ROOT / "artifacts" / "sft_v3"


def resolve_model_name(repairer_kind: str, requested_model: str | None) -> str | None:
    if requested_model:
        return requested_model
    if repairer_kind == "ollama":
        return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    if repairer_kind == "openai":
        return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
    if repairer_kind == "mlx":
        return os.environ.get("MLX_MODEL", DEFAULT_MLX_MODEL)
    return None


def make_agent(
    database_path: Path,
    max_turns: int = 3,
    repairer_kind: str = "rule",
    model: str | None = None,
    base_url: str | None = None,
    adapter_path: Path | None = None,
    cache_dir: Path | None = None,
) -> DebugAgent:
    if repairer_kind == "openai":
        repairer = OpenAIRepairer(
            database_path,
            OpenAIResponsesClient.from_env(model=model, base_url=base_url),
        )
    elif repairer_kind == "ollama":
        repairer = OllamaRepairer(
            database_path,
            OllamaChatClient.from_env(model=model, base_url=base_url),
        )
    elif repairer_kind == "mlx":
        repairer = MLXRepairer(
            database_path,
            MLXGeneratorClient(
                model_name=model or os.environ.get("MLX_MODEL", DEFAULT_MLX_MODEL),
                adapter_path=adapter_path,
                cache_dir=cache_dir or DEFAULT_MLX_CACHE,
            ),
        )
    else:
        repairer = RuleBasedRepairer(database_path)
    return DebugAgent(
        verifier=SQLVerifier(database_path),
        repairer=repairer,
        max_turns=max_turns,
    )


def ensure_database(path: Path) -> None:
    if not path.exists():
        create_demo_database(path)
    elif path.resolve() == DEFAULT_DB.resolve() and get_database_version(path) != DEMO_DB_VERSION:
        create_demo_database(path)


def command_init_db(args: argparse.Namespace) -> None:
    path = create_demo_database(args.database)
    print(f"已创建演示数据库：{path}")
    print("\nSchema：")
    print(get_schema(path))


def _run_evaluation(
    args: argparse.Namespace, repairer_kind: str, default_output: Path | None = None
) -> None:
    ensure_database(args.database)
    tasks = load_tasks(args.tasks)
    if getattr(args, "limit", None):
        tasks = tasks[: args.limit]
    agent = make_agent(
        args.database,
        args.max_turns,
        repairer_kind=repairer_kind,
        model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None),
        adapter_path=getattr(args, "adapter_path", None),
        cache_dir=getattr(args, "cache_dir", None),
    )
    summary, reports = evaluate(agent, tasks)

    print(f"任务数：{summary.task_count}")
    print(f"基线准确率：{summary.baseline_accuracy:.1%}")
    print(f"修正后准确率：{summary.final_accuracy:.1%}")
    print(f"成功修复：{summary.repaired_count}")
    print(f"平均轮数：{summary.average_turns:.2f}")
    if summary.by_error_type:
        print("分类准确率：")
        for error_type, metrics in summary.by_error_type.items():
            print(
                f"  - {error_type}: {metrics['final_success_count']}/{metrics['task_count']} "
                f"({metrics['final_accuracy']:.1%})"
            )
    for item in reports:
        report = item["report"]
        mark = "✅" if report["success"] else "❌"
        print(f"{mark} {item['task']['task_id']} ({item['task']['error_type']}): {len(report['steps'])} turn(s)")

    output = args.output or default_output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_config": {
                "repairer": repairer_kind,
                "model": resolve_model_name(
                    repairer_kind, getattr(args, "model", None)
                ),
                "max_turns": args.max_turns,
                "task_file": str(args.tasks.resolve()),
                "adapter_path": (
                    str(args.adapter_path.resolve())
                    if getattr(args, "adapter_path", None)
                    else None
                ),
            },
            "summary": summary.to_dict(),
            "reports": reports,
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"详细报告：{output.resolve()}")


def command_eval(args: argparse.Namespace) -> None:
    _run_evaluation(args, repairer_kind="rule")


def command_baseline(args: argparse.Namespace) -> None:
    print(f"Provider：{args.provider}")
    print(f"模型：{resolve_model_name(args.provider, args.model)}")
    _run_evaluation(args, repairer_kind=args.provider)


def command_debug(args: argparse.Namespace) -> None:
    ensure_database(args.database)
    report = make_agent(
        args.database,
        args.max_turns,
        repairer_kind=args.repairer,
        model=args.model,
        base_url=args.base_url,
        adapter_path=args.adapter_path,
        cache_dir=args.cache_dir,
    ).run(
        question=args.question,
        initial_sql=args.sql,
        reference_sql=args.reference_sql,
    )
    for step in report.steps:
        print(f"\nTurn {step.turn}\nSQL: {step.sql}\n反馈: {step.feedback}\n奖励: {step.reward:.1f}")
    print(f"\n最终状态：{'成功' if report.success else '失败'}")
    print(f"最终 SQL：{report.final_sql}")


def command_prepare_sft(args: argparse.Namespace) -> None:
    ensure_database(args.database)
    tasks = load_tasks(args.tasks)
    train_path, eval_path, train_count, eval_count = prepare_sft_data(
        tasks=tasks,
        verifier=SQLVerifier(args.database),
        database_path=args.database,
        output_dir=args.output_dir,
        eval_ratio=args.eval_ratio,
    )
    print(f"训练数据：{train_count} 条 -> {train_path.resolve()}")
    print(f"评测数据：{eval_count} 条 -> {eval_path.resolve()}")


def command_analyze_badcases(args: argparse.Namespace) -> None:
    analysis = analyze_badcases(args.report, args.json_output, args.markdown_output)
    print(f"基线失败：{analysis['failure_count']}/{analysis['task_count']}")
    print("优先改进方向：")
    for row in analysis["priorities"]:
        if row["failure_count"]:
            print(
                f"  - {row['error_type']}: 失败 {row['failure_count']} 条，"
                f"准确率 {row['accuracy']:.1%}"
            )
    print(f"JSON 报告：{args.json_output.resolve()}")
    print(f"Markdown 报告：{args.markdown_output.resolve()}")


def _assert_no_test_leakage(
    training_tasks: list[DebugTask], test_tasks: list[DebugTask]
) -> None:
    training_fingerprints = {task_fingerprint(task) for task in training_tasks}
    leaked = [
        task.task_id
        for task in test_tasks
        if task_fingerprint(task) in training_fingerprints
    ]
    if leaked:
        raise RuntimeError(f"训练数据与独立测试集重复：{', '.join(leaked)}")


def command_build_data(args: argparse.Namespace) -> None:
    database_path = create_training_database(args.database)
    test_tasks = load_tasks(args.test_tasks)
    test_fingerprints = {task_fingerprint(task) for task in test_tasks}
    raw_path, tasks = build_training_dataset(
        database_path, args.output, excluded_fingerprints=test_fingerprints
    )
    _assert_no_test_leakage(tasks, test_tasks)
    train_path, eval_path, train_count, eval_count = prepare_sft_data(
        tasks=tasks,
        verifier=SQLVerifier(database_path),
        database_path=database_path,
        output_dir=args.sft_output_dir,
        eval_ratio=args.eval_ratio,
    )
    manifest = {
        "purpose": "根据本地模型基线 Bad Case 构建金融 SQL 修复 SFT 数据",
        "database": str(database_path.resolve()),
        "raw_data": str(raw_path.resolve()),
        "independent_test_set": str(args.test_tasks.resolve()),
        "total_count": len(tasks),
        "distribution": summarize_distribution(tasks),
        "sft_train_count": train_count,
        "sft_eval_count": eval_count,
        "split_key": "template_id",
        "exact_test_leakage_count": 0,
        "validation": {
            "reference_sql_executable": True,
            "initial_sql_is_bad_case": True,
            "fingerprints_unique": True,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"已构建并经 SQLite 验证：{len(tasks)} 条 Bad Case")
    print("错误类型分布：")
    for error_type, count in summarize_distribution(tasks).items():
        print(f"  - {error_type}: {count}")
    print("独立测试集泄漏检查：通过")
    print(f"训练数据库：{database_path.resolve()}")
    print(f"原始数据：{raw_path.resolve()}")
    print(f"SFT 训练数据：{train_count} 条 -> {train_path.resolve()}")
    print(f"SFT 内部评测数据：{eval_count} 条 -> {eval_path.resolve()}")
    print(f"数据清单：{args.manifest.resolve()}")


def command_prepare_mlx(args: argparse.Namespace) -> None:
    ensure_database(args.test_database)
    manifest = prepare_mlx_dataset(
        sft_train_path=args.sft_train,
        sft_eval_path=args.sft_eval,
        test_tasks_path=args.test_tasks,
        test_database_path=args.test_database,
        output_dir=args.output_dir,
        sql_terminator=args.sql_terminator,
    )
    print("MLX-LM 数据准备完成：")
    print(f"  - train: {manifest['train_count']} 条")
    print(f"  - valid: {manifest['valid_count']} 条")
    print(f"  - test: {manifest['test_count']} 条（不参与梯度更新）")
    print(f"数据目录：{args.output_dir.resolve()}")
    print(f"清单：{manifest['manifest']}")


def command_train_sft(args: argparse.Namespace) -> None:
    ensure_database(args.test_database)
    prepare_mlx_dataset(
        sft_train_path=args.sft_train,
        sft_eval_path=args.sft_eval,
        test_tasks_path=args.test_tasks,
        test_database_path=args.test_database,
        output_dir=args.data_dir,
        sql_terminator=args.sql_terminator,
    )
    iters = args.iters if args.iters is not None else (5 if args.smoke else 50)
    adapter_path = args.adapter_path or (
        DEFAULT_SMOKE_ADAPTER if args.smoke else DEFAULT_SFT_ADAPTER
    )
    print(f"训练模式：{'smoke test' if args.smoke else '完整 SFT'}")
    print(f"模型：{args.model}")
    print(f"训练步数：{iters}")
    print(f"Adapter 输出：{adapter_path.resolve()}")
    manifest_path = run_mlx_lora_training(
        model=args.model,
        data_dir=args.data_dir,
        adapter_path=adapter_path,
        cache_dir=args.cache_dir,
        iters=iters,
        smoke=args.smoke,
        learning_rate=args.learning_rate,
        resume_adapter_file=args.resume_adapter_file,
    )
    print(f"训练成功，运行清单：{manifest_path.resolve()}")


def command_compare_runs(args: argparse.Namespace) -> None:
    comparison = compare_evaluation_reports(
        args.base_report,
        args.candidate_report,
        args.json_output,
        args.markdown_output,
        base_label=args.base_label,
        candidate_label=args.candidate_label,
    )
    print(f"{comparison['base_label']} 准确率：{comparison['base_accuracy']:.1%}")
    print(
        f"{comparison['candidate_label']} 准确率："
        f"{comparison['candidate_accuracy']:.1%}"
    )
    print(f"准确率变化：{comparison['accuracy_delta']:+.1%}")
    print(f"修复任务：{', '.join(comparison['fixed_tasks']) or '无'}")
    print(f"回退任务：{', '.join(comparison['regressed_tasks']) or '无'}")
    print(f"下一阶段：{comparison['decision']['next_stage']}")
    print(f"对比报告：{args.markdown_output.resolve()}")


def command_build_v2(args: argparse.Namespace) -> None:
    ensure_database(args.test_database)
    holdout_path, holdout_tasks = build_final_holdout(
        args.test_database, args.holdout_output
    )
    training_database = create_training_database(args.training_database)
    forbidden_tasks = (
        load_tasks(args.development_tasks)
        + load_tasks(args.v1_raw_tasks)
        + holdout_tasks
    )
    manifest = build_v2_dataset(
        database_path=training_database,
        v1_train_path=args.v1_train,
        v1_eval_path=args.v1_eval,
        raw_output_path=args.increment_output,
        output_dir=args.output_dir,
        forbidden_tasks=forbidden_tasks,
    )
    print(f"冻结最终 holdout：{len(holdout_tasks)} 条 -> {holdout_path.resolve()}")
    print(f"V2 新增针对性数据：{manifest['increment_count']} 条")
    for error_type, count in manifest["increment_distribution"].items():
        print(f"  - {error_type}: {count}")
    print(f"V2 训练集：{manifest['train_count']} 条")
    print(f"V2 内部验证集：{manifest['valid_count']} 条")
    print("开发集、最终 holdout 精确泄漏检查：通过")
    print(f"V2 数据清单：{manifest['manifest']}")


def command_summarize_experiment(args: argparse.Namespace) -> None:
    summary = summarize_sft_experiment(
        base_report_path=args.base_report,
        v1_report_path=args.v1_report,
        v2_report_path=args.v2_report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )
    print("最终留出集结果：")
    for row in summary["runs"]:
        print(
            f"  - {row['label']}: {row['success_count']}/{row['task_count']} "
            f"({row['accuracy']:.1%})"
        )
    print(f"当前最优：{summary['winner']}")
    print(f"是否进入 RL：{'是' if summary['stage_gate']['start_rl'] else '否'}")
    print(f"下一阶段：{summary['stage_gate']['next_stage']}")
    print(f"实验报告：{args.markdown_output.resolve()}")


def command_build_v3(args: argparse.Namespace) -> None:
    ensure_database(args.test_database)
    holdout_path, holdout_tasks = build_v3_holdout(args.test_database, args.holdout_output)
    training_database = create_training_database(args.training_database)
    forbidden_tasks = []
    for path in args.forbidden_tasks:
        forbidden_tasks.extend(load_tasks(path))
    forbidden_tasks.extend(holdout_tasks)
    manifest = build_v3_dataset(
        training_database,
        args.v1_train,
        args.v1_eval,
        args.increment_output,
        args.output_dir,
        forbidden_tasks,
    )
    print(f"冻结 V3 最终 holdout：{len(holdout_tasks)} 条 -> {holdout_path.resolve()}")
    print(f"V3 新增多模板数据：{manifest['increment_count']} 条")
    for error_type, count in manifest["increment_distribution"].items():
        print(f"  - {error_type}: {count}")
    print(f"V3 训练集：{manifest['train_count']} 条")
    print(f"V3 内部验证集：{manifest['valid_count']} 条")
    print("历史评测集与 V3 最终 holdout 精确泄漏检查：通过")
    print(f"V3 数据清单：{manifest['manifest']}")


def command_build_rl_data(args: argparse.Namespace) -> None:
    ensure_database(args.demo_database)
    create_training_database(args.training_database)
    manifest = build_rl_dataset(
        args.development_tasks,
        args.v3_tasks,
        args.demo_database,
        args.training_database,
        args.output_dir,
    )
    print(f"GRPO prompts：{manifest['source_task_count']} 条")
    print(f"偏好数据：{manifest['preference_pair_count']} 对")
    print(f"最小奖励间隔：{manifest['reward_margin']['minimum']:.2f}")
    print("最终留出集隔离检查：通过")
    print("当前 training_ready：false（只完成奖励审计，不启动 GRPO）")
    print(f"奖励报告：{manifest['files']['reward_audit']}")


def command_replay_rl(args: argparse.Namespace) -> None:
    ensure_database(args.demo_database)
    if not args.training_database.exists():
        create_training_database(args.training_database)

    def progress(done: int, total: int) -> None:
        print(f"Rollout 进度：{done}/{total}", flush=True)

    summary = replay_model_rollouts(
        args.prompts,
        args.demo_database,
        args.training_database,
        args.adapter_path,
        args.output_dir,
        args.model,
        args.cache_dir,
        progress=progress,
    )
    print(f"安全率：{summary['safety_rate']:.1%}")
    print(f"两库均可执行：{summary['executable_on_all_rate']:.1%}")
    print(f"两库结果均正确：{summary['correct_on_all_rate']:.1%}")
    print(f"重复原 SQL：{summary['repeated_sql_rate']:.1%}")
    print(f"奖励层是否通过：{'是' if summary['reward_gate']['reward_ready'] else '否'}")
    print("当前是否启动 GRPO：否")
    print(f"回放报告：{summary['files']['report']}")


def command_export_hard_preferences(args: argparse.Namespace) -> None:
    summary = export_hard_preferences(args.prompts, args.rollouts, args.output)
    print(f"真实失败偏好对：{summary['pair_count']} 条")
    for error_type, count in summary["error_distribution"].items():
        print(f"  - {error_type}: {count}")
    print(f"最小奖励间隔：{summary['minimum_margin']:.2f}")
    print(f"输出：{summary['output']}")


def command_build_preference_replay(args: argparse.Namespace) -> None:
    manifest = build_corrective_replay_dataset(
        args.hard_preferences,
        args.prompts,
        args.rollouts,
        args.validation,
        args.output_dir,
    )
    print("偏好纠错回放数据构建完成：")
    print(f"  - 真实失败纠错：{manifest['hard_correction_count']} 条")
    print(f"  - 强项保护：{manifest['capability_protection_count']} 条")
    print(f"  - 总训练数据：{manifest['train_count']} 条")
    print("训练方式：短步数 corrective replay SFT（不是 DPO）")
    print(f"数据清单：{manifest['files']['manifest']}")


def command_compare_replays(args: argparse.Namespace) -> None:
    comparison = compare_replay_runs(
        args.base_summary,
        args.candidate_summary,
        args.base_rollouts,
        args.candidate_rollouts,
        args.json_output,
        args.markdown_output,
    )
    print(f"双库正确数变化：{comparison['correct_delta']:+d}")
    print(f"重复 SQL 变化：{comparison['repeat_delta']:+d}")
    print(f"新修复任务：{', '.join(comparison['fixed_tasks']) or '无'}")
    print(f"回退任务：{', '.join(comparison['regressed_tasks']) or '无'}")
    print(f"是否接受候选：{'是' if comparison['decision']['accept_candidate'] else '否'}")
    print(f"当前模型：{comparison['decision']['selected_adapter']}")
    print(f"报告：{args.markdown_output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="执行反馈驱动的 SQL Debug Agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="重建演示数据库")
    init_parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    init_parser.set_defaults(func=command_init_db)

    eval_parser = subparsers.add_parser("eval", help="运行离线评测")
    eval_parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    eval_parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    eval_parser.add_argument("--max-turns", type=int, default=3)
    eval_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "eval_report.json")
    eval_parser.set_defaults(func=command_eval)

    baseline_parser = subparsers.add_parser(
        "baseline", help="使用本地或 API 模型运行独立基线评测"
    )
    baseline_parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    baseline_parser.add_argument("--tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    baseline_parser.add_argument("--max-turns", type=int, default=2)
    baseline_parser.add_argument("--limit", type=int, default=5)
    baseline_parser.add_argument(
        "--provider", choices=("ollama", "openai", "mlx"), default="ollama"
    )
    baseline_parser.add_argument("--model", default=None)
    baseline_parser.add_argument("--base-url", default=None)
    baseline_parser.add_argument("--adapter-path", type=Path, default=None)
    baseline_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MLX_CACHE)
    baseline_parser.add_argument(
        "--output", type=Path, default=DEFAULT_BASELINE_REPORT
    )
    baseline_parser.set_defaults(func=command_baseline)

    debug_parser = subparsers.add_parser("debug", help="修复一条 SQL")
    debug_parser.add_argument("--question", required=True)
    debug_parser.add_argument("--sql", required=True)
    debug_parser.add_argument("--reference-sql")
    debug_parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    debug_parser.add_argument("--max-turns", type=int, default=3)
    debug_parser.add_argument(
        "--repairer", choices=("rule", "ollama", "openai", "mlx"), default="rule"
    )
    debug_parser.add_argument("--model", default=None)
    debug_parser.add_argument("--base-url", default=None)
    debug_parser.add_argument("--adapter-path", type=Path, default=None)
    debug_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MLX_CACHE)
    debug_parser.set_defaults(func=command_debug)

    prepare_parser = subparsers.add_parser("prepare-sft", help="把原始 Bad Case 转换为 SFT 对话数据")
    prepare_parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    prepare_parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    prepare_parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "sft"
    )
    prepare_parser.add_argument("--eval-ratio", type=float, default=0.2)
    prepare_parser.set_defaults(func=command_prepare_sft)

    badcase_parser = subparsers.add_parser(
        "analyze-badcases", help="分析模型基线失败案例并确定优化优先级"
    )
    badcase_parser.add_argument("--report", type=Path, default=DEFAULT_BASELINE_REPORT)
    badcase_parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "badcase_analysis.json",
    )
    badcase_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "badcase_analysis.md",
    )
    badcase_parser.set_defaults(func=command_analyze_badcases)

    data_parser = subparsers.add_parser(
        "build-data", help="构建、验证并切分 200 条针对性 SFT 数据"
    )
    data_parser.add_argument("--database", type=Path, default=DEFAULT_TRAIN_DB)
    data_parser.add_argument("--output", type=Path, default=DEFAULT_TRAIN_TASKS)
    data_parser.add_argument("--test-tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    data_parser.add_argument(
        "--sft-output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "sft_v1"
    )
    data_parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_manifest.json",
    )
    data_parser.add_argument("--eval-ratio", type=float, default=0.2)
    data_parser.set_defaults(func=command_build_data)

    mlx_data_parser = subparsers.add_parser(
        "prepare-mlx", help="把 SFT 数据整理为 MLX-LM 所需的 chat JSONL"
    )
    mlx_data_parser.add_argument(
        "--sft-train",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_train.jsonl",
    )
    mlx_data_parser.add_argument(
        "--sft-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_eval.jsonl",
    )
    mlx_data_parser.add_argument("--test-tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    mlx_data_parser.add_argument("--test-database", type=Path, default=DEFAULT_DB)
    mlx_data_parser.add_argument("--output-dir", type=Path, default=DEFAULT_MLX_DATA)
    mlx_data_parser.add_argument("--sql-terminator", action="store_true")
    mlx_data_parser.set_defaults(func=command_prepare_mlx)

    train_parser = subparsers.add_parser(
        "train-sft", help="在 Apple Silicon 上用 MLX-LM 运行 4-bit LoRA SFT"
    )
    train_parser.add_argument("--model", default=DEFAULT_MLX_MODEL)
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--iters", type=int, default=None)
    train_parser.add_argument("--adapter-path", type=Path, default=None)
    train_parser.add_argument("--data-dir", type=Path, default=DEFAULT_MLX_DATA)
    train_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MLX_CACHE)
    train_parser.add_argument(
        "--sft-train",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_train.jsonl",
    )
    train_parser.add_argument(
        "--sft-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_eval.jsonl",
    )
    train_parser.add_argument("--test-tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    train_parser.add_argument("--test-database", type=Path, default=DEFAULT_DB)
    train_parser.add_argument("--sql-terminator", action="store_true")
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--resume-adapter-file", type=Path, default=None)
    train_parser.set_defaults(func=command_train_sft)

    compare_parser = subparsers.add_parser(
        "compare-runs", help="比较同一测试集上的 Base 与 SFT 评测报告"
    )
    compare_parser.add_argument(
        "--base-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "mlx_base_report.json",
    )
    compare_parser.add_argument(
        "--candidate-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1_eval_report.json",
    )
    compare_parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1_comparison.json",
    )
    compare_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1_comparison.md",
    )
    compare_parser.add_argument("--base-label", default="Base")
    compare_parser.add_argument("--candidate-label", default="Candidate")
    compare_parser.set_defaults(func=command_compare_runs)

    v2_parser = subparsers.add_parser(
        "build-v2", help="冻结最终 holdout 并构建 SFT V2 针对性数据"
    )
    v2_parser.add_argument("--test-database", type=Path, default=DEFAULT_DB)
    v2_parser.add_argument("--training-database", type=Path, default=DEFAULT_TRAIN_DB)
    v2_parser.add_argument("--holdout-output", type=Path, default=DEFAULT_FINAL_HOLDOUT)
    v2_parser.add_argument("--development-tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    v2_parser.add_argument("--v1-raw-tasks", type=Path, default=DEFAULT_TRAIN_TASKS)
    v2_parser.add_argument(
        "--v1-train",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_train.jsonl",
    )
    v2_parser.add_argument(
        "--v1-eval",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_eval.jsonl",
    )
    v2_parser.add_argument(
        "--increment-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft_v2_increment.jsonl",
    )
    v2_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SFT_V2_DIR)
    v2_parser.set_defaults(func=command_build_v2)

    summary_parser = subparsers.add_parser(
        "summarize-experiment", help="汇总 Base/SFT V1/SFT V2 并执行 RL 阶段门槛"
    )
    summary_parser.add_argument(
        "--base-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_holdout_base_report.json",
    )
    summary_parser.add_argument(
        "--v1-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_holdout_sft_v1_report.json",
    )
    summary_parser.add_argument(
        "--v2-report",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "final_holdout_sft_v2_report.json",
    )
    summary_parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_final_summary.json",
    )
    summary_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "sft_final_summary.md",
    )
    summary_parser.set_defaults(func=command_summarize_experiment)

    v3_parser = subparsers.add_parser(
        "build-v3", help="冻结新 holdout 并构建最小修改与 SQLite 方言 V3 数据"
    )
    v3_parser.add_argument("--test-database", type=Path, default=DEFAULT_DB)
    v3_parser.add_argument("--training-database", type=Path, default=DEFAULT_TRAIN_DB)
    v3_parser.add_argument("--holdout-output", type=Path, default=DEFAULT_FINAL_HOLDOUT_V3)
    v3_parser.add_argument(
        "--forbidden-tasks",
        type=Path,
        nargs="+",
        default=[DEFAULT_BASELINE_TASKS, DEFAULT_FINAL_HOLDOUT, DEFAULT_TRAIN_TASKS, PROJECT_ROOT / "data" / "sft_v2_increment.jsonl"],
    )
    v3_parser.add_argument("--v1-train", type=Path, default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_train.jsonl")
    v3_parser.add_argument("--v1-eval", type=Path, default=PROJECT_ROOT / "artifacts" / "sft_v1" / "sft_eval.jsonl")
    v3_parser.add_argument("--increment-output", type=Path, default=PROJECT_ROOT / "data" / "sft_v3_increment.jsonl")
    v3_parser.add_argument("--output-dir", type=Path, default=DEFAULT_SFT_V3_DIR)
    v3_parser.set_defaults(func=command_build_v3)

    rl_parser = subparsers.add_parser(
        "build-rl-data", help="构建多数据库奖励审计、偏好数据和 GRPO prompts"
    )
    rl_parser.add_argument("--development-tasks", type=Path, default=DEFAULT_BASELINE_TASKS)
    rl_parser.add_argument("--v3-tasks", type=Path, default=PROJECT_ROOT / "data" / "sft_v3_increment.jsonl")
    rl_parser.add_argument("--demo-database", type=Path, default=DEFAULT_DB)
    rl_parser.add_argument("--training-database", type=Path, default=DEFAULT_TRAIN_DB)
    rl_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1")
    rl_parser.set_defaults(func=command_build_rl_data)

    replay_parser = subparsers.add_parser(
        "replay-rl", help="用真实 MLX 模型生成 rollout 并离线回放稳健奖励"
    )
    replay_parser.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl")
    replay_parser.add_argument("--demo-database", type=Path, default=DEFAULT_DB)
    replay_parser.add_argument("--training-database", type=Path, default=DEFAULT_TRAIN_DB)
    replay_parser.add_argument("--adapter-path", type=Path, default=PROJECT_ROOT / "artifacts" / "adapters" / "sft_v3_30")
    replay_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_v3_30")
    replay_parser.add_argument("--model", default=DEFAULT_MLX_MODEL)
    replay_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_MLX_CACHE)
    replay_parser.set_defaults(func=command_replay_rl)

    hard_parser = subparsers.add_parser(
        "export-hard-preferences", help="把真实模型失败 rollout 转为偏好优化数据"
    )
    hard_parser.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl")
    hard_parser.add_argument("--rollouts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_v3_30" / "rollouts.jsonl")
    hard_parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl")
    hard_parser.set_defaults(func=command_export_hard_preferences)

    preference_parser = subparsers.add_parser(
        "build-preference-replay", help="将真实 hard pairs 转为纠错回放与强项保护数据"
    )
    preference_parser.add_argument("--hard-preferences", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "hard_preference_pairs.jsonl")
    preference_parser.add_argument("--prompts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "grpo_prompts.jsonl")
    preference_parser.add_argument("--rollouts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_v3_30" / "rollouts.jsonl")
    preference_parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "artifacts" / "sft_v3" / "sft_eval.jsonl")
    preference_parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "preference_v1")
    preference_parser.set_defaults(func=command_build_preference_replay)

    replay_compare_parser = subparsers.add_parser(
        "compare-replays", help="比较纠错回放前后的真实 rollout 指标"
    )
    replay_compare_parser.add_argument("--base-summary", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_v3_30" / "summary.json")
    replay_compare_parser.add_argument("--candidate-summary", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_preference_v1_10" / "summary.json")
    replay_compare_parser.add_argument("--base-rollouts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_v3_30" / "rollouts.jsonl")
    replay_compare_parser.add_argument("--candidate-rollouts", type=Path, default=PROJECT_ROOT / "artifacts" / "rl_v1" / "replay_preference_v1_10" / "rollouts.jsonl")
    replay_compare_parser.add_argument("--json-output", type=Path, default=PROJECT_ROOT / "artifacts" / "preference_v1_comparison.json")
    replay_compare_parser.add_argument("--markdown-output", type=Path, default=PROJECT_ROOT / "artifacts" / "preference_v1_comparison.md")
    replay_compare_parser.set_defaults(func=command_compare_replays)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ModelConfigurationError, ModelAPIError) as exc:
        print(f"模型配置或调用失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
