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
    model_type = "phi3"


class _FakeModel:
    device = "cpu"
    config = _FakeConfig()

    def __init__(self):
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[42, 32007]])
        return torch.cat([input_ids, generated], dim=1)


def test_generation_uses_chat_stop_tokens_and_decodes_completion_only():
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    text = registry.generate(model, tokenizer, "prompt")

    assert model.generate_kwargs["eos_token_id"] == [32000, 32007, 151645]
    assert model.generate_kwargs["repetition_penalty"] == 1.1
    assert model.generate_kwargs["no_repeat_ngram_size"] == 4
    assert model.generate_kwargs["renormalize_logits"] is True
    assert tokenizer.decoded_tokens == [42, 32007]
    assert text == "<<A>>F p<|end|>"
