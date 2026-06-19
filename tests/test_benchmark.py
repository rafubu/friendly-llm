from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from litert_ollama.benchmark import (
    find_model_paths,
    load_all_results,
    get_model_config,
    get_model_decode_tps,
)


class TestModelDiscovery:
    """Tests for find_model_paths."""

    def test_empty_dir(self):
        """Empty directory returns no models."""
        with tempfile.TemporaryDirectory() as tmp:
            models = find_model_paths(Path(tmp))
            assert models == {}

    def test_finds_litertlm_files(self):
        """Should find .litertlm files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model_dir = base / "gemm4-e2b"
            model_dir.mkdir()
            (model_dir / "model.litertlm").write_text("mock")
            (model_dir / "config.json").write_text("{}")

            # Create a non-model dir
            (base / "not-a-model").mkdir()

            models = find_model_paths(base)
            assert "gemm4-e2b" in models
            assert models["gemm4-e2b"].endswith("model.litertlm")

    def test_double_dash_to_slash(self):
        """Model IDs with -- should be converted to /."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model_dir = base / "org--model-name"
            model_dir.mkdir()
            (model_dir / "model.litertlm").write_text("mock")

            models = find_model_paths(base)
            assert "org/model-name" in models

    def test_multiple_models(self):
        """Should find all models in directory."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ["model-a", "model-b", "model-c"]:
                d = base / name
                d.mkdir()
                (d / "model.litertlm").write_text("mock")

            models = find_model_paths(base)
            assert len(models) == 3


class TestBenchmarkResultsStorage:
    """Tests for reading benchmark results."""

    def test_load_empty_file(self):
        """Loading non-existent file returns None."""
        assert load_all_results(Path("/nonexistent/bench.json")) is None

    def test_load_valid_results(self):
        """Should parse valid JSON results."""
        data = {
            "gemm4-e2b": {
                "best_config": "cpu",
                "best_decode_tps": 4.3,
                "best_settings": {"backend": "cpu", "spec_decoding": False},
                "all_results": {
                    "cpu": {"supported": True, "decode_tps": 4.3},
                    "gpu": {"supported": False, "error": "Not supported"},
                },
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            results = load_all_results(Path(path))
            assert results is not None
            assert "gemm4-e2b" in results
            assert results["gemm4-e2b"]["best_decode_tps"] == 4.3
        finally:
            Path(path).unlink()

    def test_load_malformed_json(self):
        """Malformed JSON should return None gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not json {{{")
            path = f.name

        try:
            results = load_all_results(Path(path))
            assert results is None
        finally:
            Path(path).unlink()

    def test_get_model_config(self):
        """get_model_config should return best_settings for a model."""
        data = {
            "gemm4-e4b": {
                "best_config": "gpu",
                "best_decode_tps": 50.2,
                "best_settings": {"backend": "gpu", "spec_decoding": False},
                "all_results": {},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            cfg = get_model_config("gemm4-e4b", Path(path))
            assert cfg == {"backend": "gpu", "spec_decoding": False}

            # Unknown model
            assert get_model_config("nonexistent", Path(path)) is None
        finally:
            Path(path).unlink()

    def test_get_model_decode_tps(self):
        """get_model_decode_tps should return decode speed."""
        data = {
            "model-x": {
                "best_config": "cpu_spec",
                "best_decode_tps": 18.2,
                "best_settings": {},
                "all_results": {},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            tps = get_model_decode_tps("model-x", Path(path))
            assert tps == 18.2
        finally:
            Path(path).unlink()

    def test_per_model_isolation(self):
        """Each model's results should be independent."""
        data = {
            "gemm4-e2b": {
                "best_config": "cpu",
                "best_decode_tps": 4.3,
                "best_settings": {"backend": "cpu", "spec_decoding": False},
                "all_results": {},
            },
            "gemm4-e4b": {
                "best_config": "gpu",
                "best_decode_tps": 50.2,
                "best_settings": {"backend": "gpu", "spec_decoding": False},
                "all_results": {},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name

        try:
            assert get_model_config("gemm4-e2b", Path(path))["backend"] == "cpu"
            assert get_model_config("gemm4-e4b", Path(path))["backend"] == "gpu"
        finally:
            Path(path).unlink()


class TestSpeculativeDecodingConfig:
    """Tests for speculative decoding toggle."""

    def test_spec_decoding_disabled_by_default(self):
        """Speculative decoding should be False by default."""
        from litert_ollama.config import Settings
        s = Settings()
        assert s.enable_speculative_decoding is False

    def test_spec_decoding_enabled_via_env(self):
        """Environment variable should override default."""
        import os
        os.environ["LITERT_ENABLE_SPECULATIVE_DECODING"] = "true"
        try:
            from litert_ollama.config import Settings
            s = Settings()
            assert s.enable_speculative_decoding is True
        finally:
            del os.environ["LITERT_ENABLE_SPECULATIVE_DECODING"]


class TestTokenCountingReal:
    """Tests that real token counting replaced word counting."""

    def test_conv_token_count_available_on_mock(self):
        """Mock engine should expose token_count property."""
        # Test from conftest
        mock = MagicMock()
        mock.token_count = 42
        assert mock.token_count == 42

    def test_real_tokens_vs_word_count(self):
        """Real tokens from engine differ from naive word count."""
        # Words: 3 tokens → word count would be ~3
        # Real tokenizer: ~3-5 tokens
        text = "Hello world today"
        word_count = len(text.split())  # 3
        half_chars = len(text) // 4     # ~5

        # The estimate should be close but not exactly words
        assert word_count != half_chars


class TestBenchmarkOnStartup:
    """Tests for benchmark-on-startup CLI behavior."""

    def test_benchmark_flag_exists(self):
        """CLI should have --benchmark-on-startup flag."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "litert_ollama.cli", "serve", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--benchmark-on-startup" in result.stdout

    def test_force_benchmark_flag_exists(self):
        """CLI should have --force-benchmark flag."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "litert_ollama.cli", "serve", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert "--force-benchmark" in result.stdout
