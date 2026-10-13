from blackhole.local_jobs import sanitize_remote_value


def test_sanitization_preserves_generation_token_count_and_redacts_credentials() -> None:
    payload = {
        "max_new_tokens": 128,
        "access_token": "hf_abcdefghijk",
        "api_key": "super-secret-key",
        "nested": {"token": "another-secret"},
    }

    sanitized = sanitize_remote_value(payload)

    assert sanitized["max_new_tokens"] == 128
    assert sanitized["access_token"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["token"] == "[REDACTED]"
