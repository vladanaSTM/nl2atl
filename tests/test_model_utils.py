from src import model_utils


def test_normalize_and_resolve():
    assert model_utils.normalize_model_token("azure-qwen-3b") == "qwen-3b"

    models = {
        "qwen-3b": {"name": "qwen-3b", "short_name": "qwen-3b"},
        "foo": {"name": "bar", "short_name": "baz"},
    }

    # exact key
    assert model_utils.resolve_model_key("qwen-3b", models) == "qwen-3b"
    # case-insensitive key match
    assert model_utils.resolve_model_key("QWEN-3B", models) == "qwen-3b"
    # short_name match
    assert model_utils.resolve_model_key("baz", models) == "foo"


def test_resolve_model_key_prefix_and_case():
    models = {
        "azure-gpt": {"name": "gpt-4", "short_name": "gpt-4"},
        "qwen-3b": {"name": "qwen-3b", "short_name": "QWEN-3B"},
    }

    assert model_utils.resolve_model_key("azure-gpt", models) == "azure-gpt"
    assert model_utils.resolve_model_key("gpt-4", models) == "azure-gpt"
    assert model_utils.resolve_model_key("qwen-3b", models) == "qwen-3b"


def test_resolve_model_key_missing_raises():
    models = {"qwen-3b": {"name": "qwen-3b", "short_name": "qwen-3b"}}
    try:
        model_utils.resolve_model_key("unknown", models)
    except KeyError as exc:
        assert str(exc).strip("'") == "unknown"
    else:
        raise AssertionError("Expected KeyError for unknown model key")
