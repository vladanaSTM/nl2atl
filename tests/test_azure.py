from src.infra.azure import AzureClient


def test_build_messages_splits_generic_prompt_roles():
    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")
    prompt = "System:\nRules\n\nUser:\nConvert to ATL formula: x\n\nAssistant:\n"

    messages = client._build_messages(prompt)

    assert messages == [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "Convert to ATL formula: x"},
    ]


def test_build_messages_keeps_plain_prompt_as_user_message():
    client = AzureClient(endpoint="https://example.test", api_key="key", model="m")

    assert client._build_messages("plain") == [{"role": "user", "content": "plain"}]
