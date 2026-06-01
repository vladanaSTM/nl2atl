import torch

from src.models import registry


class _Batch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token_id = 32000
    pad_token_id = 32000
    unk_token_id = 0

    def __init__(self):
        self.decoded_tokens = None

    def __call__(self, prompt, return_tensors="pt"):
        return _Batch({"input_ids": torch.tensor([[10, 11, 12]])})

    def convert_tokens_to_ids(self, token):
        return {"<|end|>": 32007, "<|im_end|>": 151645}.get(token, self.unk_token_id)

    def decode(self, tokens, skip_special_tokens=False):
        self.decoded_tokens = tokens.tolist()
        return "<<A>>F p<|end|>"


class _FakeConfig:
    def __init__(self, model_type="qwen2"):
        self.model_type = model_type


class _FakeModel:
    device = "cpu"

    def __init__(self, model_type="qwen2"):
        self.config = _FakeConfig(model_type)
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[42, 32007]])
        return torch.cat([input_ids, generated], dim=1)


class _RemotePhiFakeModel(_FakeModel):
    pass


_RemotePhiFakeModel.__module__ = "transformers_modules.microsoft.Phi3"


class _BuiltinPhiFakeModel(_FakeModel):
    pass


_BuiltinPhiFakeModel.__module__ = "transformers.models.phi3.modeling_phi3"


def test_generation_uses_chat_stop_tokens_and_decodes_completion_only():
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    text = registry.generate(model, tokenizer, "prompt")

    assert model.generate_kwargs["eos_token_id"] == [32000, 32007, 151645]
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["use_cache"] is True
    assert "repetition_penalty" not in model.generate_kwargs
    assert "no_repeat_ngram_size" not in model.generate_kwargs
    assert "temperature" not in model.generate_kwargs
    assert tokenizer.decoded_tokens == [42, 32007]
    assert text == "<<A>>F p<|end|>"


def test_generation_disables_cache_for_phi3_remote_code():
    tokenizer = _FakeTokenizer()
    model = _RemotePhiFakeModel(model_type="phi3")

    registry.generate(model, tokenizer, "prompt")

    assert model.generate_kwargs["use_cache"] is False


def test_generation_keeps_cache_for_builtin_phi3():
    tokenizer = _FakeTokenizer()
    model = _BuiltinPhiFakeModel(model_type="phi3")

    registry.generate(model, tokenizer, "prompt")

    assert model.generate_kwargs["use_cache"] is True


def test_phi_uses_builtin_transformers_code():
    assert registry._trust_remote_code("microsoft/Phi-3.5-mini-instruct") is False
    assert registry._trust_remote_code("Qwen/Qwen2.5-3B-Instruct") is True
