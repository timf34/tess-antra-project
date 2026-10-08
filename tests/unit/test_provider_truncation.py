from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from npbench.providers import ProviderRequest
from npbench.providers.openai_compatible import OpenAICompatibleProvider
from npbench.runner.tools import ToolHost


def test_truncated_tool_call_keeps_stop_reason_and_explains_recovery(monkeypatch):
    monkeypatch.setenv("TEST_FAKE_API_KEY", "test-only")
    provider = OpenAICompatibleProvider("test-model", api_key_env="TEST_FAKE_API_KEY")
    response = ChatCompletion.model_validate(
        dict(
            id="test",
            model="test-model",
            created=0,
            object="chat.completion",
            choices=[
                dict(
                    index=0,
                    finish_reason="length",
                    message=dict(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            dict(
                                id="tool1",
                                type="function",
                                function=dict(name="write_file", arguments='{"path":'),
                            )
                        ],
                    ),
                )
            ],
            usage=dict(prompt_tokens=1, completion_tokens=10, total_tokens=11),
        )
    )
    raw = SimpleNamespace(headers={}, parse=lambda: response)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=lambda **kw: raw))
        )
    )
    result = provider.complete(ProviderRequest(system="test", messages=[], max_tokens=10))
    assert result.stop_reason == "max_tokens"
    args = result.content_blocks[0]["input"]
    assert "_raw" in args
    text, record = ToolHost(None, None, "test").call("write_file", args)
    assert "No file was written" in text
    assert record["error"] == "malformed_arguments"
