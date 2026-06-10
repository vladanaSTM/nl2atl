from fastapi.testclient import TestClient

from src import api_server
from src.config import Config, ModelConfig


def _api_config(models_dir: str) -> Config:
    return Config(
        models_dir=models_dir,
        num_few_shot_examples=3,
        models={
            "qwen-3b": ModelConfig(
                name="Qwen/Qwen3-3B",
                short_name="qwen",
                provider="huggingface",
            ),
            "azure-gpt": ModelConfig(
                name="gpt-4o-mini",
                short_name="gpt-4o-mini",
                provider="azure",
            ),
        },
    )


def _reset_api_state() -> None:
    api_server._MODEL_CACHE.clear()
    cache_clear = getattr(api_server._get_config, "cache_clear", None)
    if cache_clear:
        cache_clear()


def test_generate_returns_resolved_key_and_trimmed_prompt(monkeypatch, tmp_path):
    _reset_api_state()
    config = _api_config(str(tmp_path))
    captured = {}
    load_calls = []

    def fake_load_model(model_config, for_training=False, load_adapter=None):
        load_calls.append((model_config.short_name, for_training, load_adapter))
        return object(), object()

    def fake_generate(model, tokenizer, prompt, max_new_tokens=256):
        captured["prompt"] = prompt
        captured["max_new_tokens"] = max_new_tokens
        return "  <<Agent>>F goal  "

    monkeypatch.setattr(api_server, "_get_config", lambda: config)
    monkeypatch.setattr(api_server, "load_model", fake_load_model)
    monkeypatch.setattr(api_server, "generate", fake_generate)

    response = TestClient(api_server.app).post(
        "/generate",
        json={
            "description": "  Reach goal  ",
            "model": "qwen",
            "max_new_tokens": 17,
            "return_raw": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["formula"] == "<<Agent>>F goal"
    assert body["model_key"] == "qwen-3b"
    assert body["model_name"] == "Qwen/Qwen3-3B"
    assert body["provider"] == "huggingface"
    assert body["raw_output"] == "  <<Agent>>F goal  "
    assert "Convert to ATL formula: Reach goal" in captured["prompt"]
    assert captured["max_new_tokens"] == 17
    assert load_calls == [("qwen", False, None)]


def test_generate_preserves_zero_few_shot_count(monkeypatch, tmp_path):
    _reset_api_state()
    config = _api_config(str(tmp_path))
    captured = {}

    def fake_format_prompt(
        input_text,
        output_text=None,
        few_shot=False,
        num_examples=5,
        model_type="generic",
        tokenizer=None,
    ):
        captured["input_text"] = input_text
        captured["few_shot"] = few_shot
        captured["num_examples"] = num_examples
        return "prompt"

    monkeypatch.setattr(api_server, "_get_config", lambda: config)
    monkeypatch.setattr(
        api_server,
        "load_model",
        lambda model_config, for_training=False, load_adapter=None: (
            object(),
            object(),
        ),
    )
    monkeypatch.setattr(api_server, "format_prompt", fake_format_prompt)
    monkeypatch.setattr(api_server, "generate", lambda *args, **kwargs: "<<A>>F p")

    response = TestClient(api_server.app).post(
        "/generate",
        json={
            "description": "p",
            "model": "qwen-3b",
            "few_shot": True,
            "num_few_shot": 0,
        },
    )

    assert response.status_code == 200, response.text
    assert captured == {"input_text": "p", "few_shot": True, "num_examples": 0}


def test_generate_with_adapter_does_not_load_cached_base(monkeypatch, tmp_path):
    _reset_api_state()
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    config = _api_config(str(tmp_path))
    load_adapters = []

    def fake_load_model(model_config, for_training=False, load_adapter=None):
        load_adapters.append(load_adapter)
        return object(), object()

    monkeypatch.setattr(api_server, "_get_config", lambda: config)
    monkeypatch.setattr(api_server, "load_model", fake_load_model)
    monkeypatch.setattr(api_server, "generate", lambda *args, **kwargs: "<<A>>F p")

    response = TestClient(api_server.app).post(
        "/generate",
        json={"description": "p", "model": "qwen-3b", "adapter": "adapter"},
    )

    assert response.status_code == 200, response.text
    assert load_adapters == [str(adapter_dir.resolve())]
    assert api_server._MODEL_CACHE == {}


def test_generate_rejects_unknown_model_before_loading(monkeypatch, tmp_path):
    _reset_api_state()
    config = _api_config(str(tmp_path))
    load_calls = []

    monkeypatch.setattr(api_server, "_get_config", lambda: config)
    monkeypatch.setattr(
        api_server,
        "load_model",
        lambda *args, **kwargs: load_calls.append(args),
    )

    response = TestClient(api_server.app).post(
        "/generate",
        json={"description": "p", "model": "missing"},
    )

    assert response.status_code == 400
    assert "missing" in response.json()["detail"]
    assert load_calls == []


def test_generate_rejects_blank_description():
    _reset_api_state()

    response = TestClient(api_server.app).post(
        "/generate",
        json={"description": "   "},
    )

    assert response.status_code == 422
    assert "Description cannot be empty" in response.text
