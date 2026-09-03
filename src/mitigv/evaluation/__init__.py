"""Evaluation utilities."""

from mitigv.evaluation.chair import (
    COCO_CLASSES,
    ChairEvaluator,
    build_ground_truth,
    evaluate_chair,
    load_chair_synonyms,
)

__all__ = [
    "COCO_CLASSES",
    "ChairEvaluator",
    "build_ground_truth",
    "evaluate_chair",
    "load_chair_synonyms",
    "parse_yes_no",
    "evaluate_discriminative",
    "evaluate_pope_subsets",
    "evaluate_amber",
    "evaluate_suite",
    "DeepSeekObjectExtractor",
    "GroundingDINOService",
    "DoubleJudgeEvaluator",
    "evaluate_double_judge",
    "analyze_length_control",
    "load_pipeline_records",
    "evaluate_pipeline_json",
    "expand_parameter_grid",
    "tune_mitigator",
    "grid_search",
    "auto_tune",
    "load_manifest",
    "pareto_frontier",
    "analyze_frontier",
    "write_report",
]

_LAZY = {
    "parse_yes_no": ("mitigv.evaluation.discriminative", "parse_yes_no"),
    "evaluate_discriminative": (
        "mitigv.evaluation.discriminative",
        "evaluate_discriminative",
    ),
    "evaluate_pope_subsets": (
        "mitigv.evaluation.discriminative",
        "evaluate_pope_subsets",
    ),
    "evaluate_amber": ("mitigv.evaluation.discriminative", "evaluate_amber"),
    "evaluate_suite": ("mitigv.evaluation.discriminative", "evaluate_suite"),
    "DeepSeekObjectExtractor": (
        "mitigv.evaluation.judge",
        "DeepSeekObjectExtractor",
    ),
    "GroundingDINOService": (
        "mitigv.evaluation.judge",
        "GroundingDINOService",
    ),
    "DoubleJudgeEvaluator": (
        "mitigv.evaluation.judge",
        "DoubleJudgeEvaluator",
    ),
    "evaluate_double_judge": (
        "mitigv.evaluation.judge",
        "evaluate_double_judge",
    ),
    "analyze_length_control": (
        "mitigv.evaluation.length_analysis",
        "analyze_length_control",
    ),
    "load_pipeline_records": ("mitigv.evaluation.pipeline", "load_pipeline_records"),
    "evaluate_pipeline_json": ("mitigv.evaluation.pipeline", "evaluate_pipeline_json"),
    "expand_parameter_grid": ("mitigv.evaluation.tuning", "expand_parameter_grid"),
    "tune_mitigator": ("mitigv.evaluation.tuning", "tune_mitigator"),
    "grid_search": ("mitigv.evaluation.tuning", "grid_search"),
    "auto_tune": ("mitigv.evaluation.tuning", "auto_tune"),
    "load_manifest": ("mitigv.evaluation.frontier", "load_manifest"),
    "pareto_frontier": ("mitigv.evaluation.frontier", "pareto_frontier"),
    "analyze_frontier": ("mitigv.evaluation.frontier", "analyze_frontier"),
    "write_report": ("mitigv.evaluation.frontier", "write_report"),
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
