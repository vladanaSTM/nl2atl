import sys

from src.cli import main as cli_main


def test_cli_dispatch_sets_sys_argv(monkeypatch):
    captured = {}

    def _handler():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr(cli_main, "_load_handler", lambda command: _handler)
    monkeypatch.setattr(sys, "argv", ["nl2atl", "run-all", "--flag", "value"])

    cli_main.main()

    assert captured["argv"][0] == "nl2atl run-all"
    assert captured["argv"][1:] == ["--flag", "value"]
