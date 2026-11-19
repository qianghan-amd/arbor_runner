from arbor_ab import registry

D1, D2 = "sha256:" + "1" * 64, "sha256:" + "2" * 64


def test_parse_ref():
    assert registry.parse_ref("vllm/vllm-openai-rocm:kimi-k31") == ("vllm/vllm-openai-rocm", "kimi-k31", None)
    assert registry.parse_ref("vllm/vllm-openai-rocm:kimi-k31@" + D1) == ("vllm/vllm-openai-rocm", "kimi-k31", D1)
    assert registry.parse_ref("lmsysorg/sglang-rocm") == ("lmsysorg/sglang-rocm", "latest", None)


def test_summarize_cases():
    # Kimi K3 (verified 2026-11-19): tag deleted upstream, pinned digest still resolves -> available
    ok, note = registry.summarize({"tag_http": 404, "tag_digest": None, "digest_http": 200}, D1)
    assert ok is True and "tag MISSING" in note
    # Kimi K3.1 Pro (verified 2026-11-19): tag now points at another digest -> available, but flagged
    ok, note = registry.summarize({"tag_http": 200, "tag_digest": D2, "digest_http": 200}, D1)
    assert ok is True and "TAG MOVED" in note
    ok, _ = registry.summarize({"tag_http": 200, "tag_digest": D1, "digest_http": 404}, D1)
    assert ok is False
    ok, note = registry.summarize({"tag_http": 200, "tag_digest": D1, "digest_http": None}, None)
    assert ok is True and "no digest pinned" in note
    ok, note = registry.summarize({"error": "URLError: timeout"}, D1)
    assert ok is None and "error" in note
