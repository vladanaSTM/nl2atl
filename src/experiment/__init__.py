"""Experiment orchestration subpackage."""

__all__ = [
    "ExperimentRunner",
    "ExperimentDataManager",
    "ExperimentReporter",
]


def __getattr__(name):
    if name == "ExperimentRunner":
        from .runner import ExperimentRunner

        return ExperimentRunner
    if name == "ExperimentDataManager":
        from .data_manager import ExperimentDataManager

        return ExperimentDataManager
    if name == "ExperimentReporter":
        from .reporter import ExperimentReporter

        return ExperimentReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
