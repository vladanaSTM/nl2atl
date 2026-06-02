import json
import sys

from src.cli import main as cli_main
from src.cli.generate_eval_reports import write_reproducibility_manifest


def test_cli_dispatch_sets_sys_argv(monkeypatch):
    captured = {}

    def _handler():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(cli_main, "_load_handler", lambda command: _handler)
    monkeypatch.setattr(sys, "argv", ["nl2atl", "run", "--flag", "value"])

    cli_main.main()

    assert captured["argv"][0] == "nl2atl run"
    assert captured["argv"][1:] == ["--flag", "value"]


def test_write_reproducibility_manifest_hashes_inputs(tmp_path):
    eval_dir = tmp_path / "eval"
    predictions_dir = tmp_path / "predictions"
    evaluated_dir = eval_dir / "evaluated_datasets" / "judge"
    predictions_dir.mkdir(parents=True)
    evaluated_dir.mkdir(parents=True)
    (predictions_dir / "run.json").write_text('{"predictions": []}', encoding="utf-8")
    (evaluated_dir / "run__judge-judge.json").write_text(
        '{"detailed_results": []}', encoding="utf-8"
    )

    manifest_path = write_reproducibility_manifest(eval_dir, predictions_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["inputs"]["prediction_files"][0]["sha256"]
    assert manifest["inputs"]["evaluated_files"][0]["sha256"]
    assert manifest["limitations"]
