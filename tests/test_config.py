"""Tests for vector_db.config."""

import pytest

from vector_db.config import QdrantConfig, _normalize_qdrant_url, _parse_bool


# ---------------------------------------------------------------------------
# _parse_bool
# ---------------------------------------------------------------------------

class TestParseBool:
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "on"])
    def test_truthy_values(self, value):
        assert _parse_bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", "", "random"])
    def test_falsy_values(self, value):
        assert _parse_bool(value) is False

    def test_strips_whitespace(self):
        assert _parse_bool("  true  ") is True


# ---------------------------------------------------------------------------
# _normalize_qdrant_url
# ---------------------------------------------------------------------------

class TestNormalizeQdrantUrl:
    def test_empty_returns_none(self):
        assert _normalize_qdrant_url("", https_hint=False) is None

    def test_whitespace_returns_none(self):
        assert _normalize_qdrant_url("   ", https_hint=False) is None

    def test_already_http(self):
        assert _normalize_qdrant_url("http://example.com", https_hint=False) == "http://example.com"

    def test_already_https(self):
        assert _normalize_qdrant_url("https://example.com", https_hint=True) == "https://example.com"

    def test_localhost_gets_http(self):
        url = _normalize_qdrant_url("localhost:6333", https_hint=False)
        assert url == "http://localhost:6333"

    def test_remote_without_scheme_gets_https(self):
        url = _normalize_qdrant_url("my-cluster.qdrant.io", https_hint=False)
        assert url.startswith("https://")


# ---------------------------------------------------------------------------
# QdrantConfig defaults
# ---------------------------------------------------------------------------

class TestQdrantConfigDefaults:
    def test_default_mode_is_server(self):
        cfg = QdrantConfig()
        assert cfg.mode == "server"

    def test_default_vector_size(self):
        assert QdrantConfig().vector_size == 768

    def test_default_hnsw_m(self):
        assert QdrantConfig().hnsw_m == 16

    def test_default_quantization_flags(self):
        cfg = QdrantConfig()
        assert cfg.int8_quantization is True
        assert cfg.quantization_always_ram is True
        assert cfg.quantile == 0.99

    def test_default_on_disk_flags(self):
        cfg = QdrantConfig()
        assert cfg.scripts_on_disk is True
        assert cfg.hnsw_on_disk is True


# ---------------------------------------------------------------------------
# QdrantConfig.from_env
# ---------------------------------------------------------------------------

class TestQdrantConfigFromEnv:
    def test_reads_vector_size(self, monkeypatch):
        monkeypatch.setenv("QDRANT_VECTOR_SIZE", "1024")
        cfg = QdrantConfig.from_env()
        assert cfg.vector_size == 1024

    def test_reads_hnsw_m(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HNSW_M", "32")
        cfg = QdrantConfig.from_env()
        assert cfg.hnsw_m == 32

    def test_hnsw_m_empty_string_becomes_none(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HNSW_M", "")
        cfg = QdrantConfig.from_env()
        assert cfg.hnsw_m is None

    def test_reads_scripts_on_disk_false(self, monkeypatch):
        monkeypatch.setenv("QDRANT_SCRIPTS_ON_DISK", "false")
        cfg = QdrantConfig.from_env()
        assert cfg.scripts_on_disk is False

    def test_reads_int8_quantization_false(self, monkeypatch):
        monkeypatch.setenv("QDRANT_INT8_QUANTIZATION", "false")
        cfg = QdrantConfig.from_env()
        assert cfg.int8_quantization is False

    def test_reads_quantile(self, monkeypatch):
        monkeypatch.setenv("QDRANT_QUANTILE", "0.95")
        cfg = QdrantConfig.from_env()
        assert cfg.quantile == pytest.approx(0.95)

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.setenv("QDRANT_MODE", "banana")
        with pytest.raises(ValueError, match="QDRANT_MODE"):
            QdrantConfig.from_env()

    def test_server_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_MODE", "server")
        monkeypatch.setenv("QDRANT_HOST", "qdrant.example.com")
        monkeypatch.setenv("QDRANT_PORT", "6334")
        cfg = QdrantConfig.from_env()
        assert cfg.mode == "server"
        assert cfg.host == "qdrant.example.com"
        assert cfg.port == 6334

    def test_local_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_MODE", "local")
        monkeypatch.setenv("QDRANT_PATH", "/tmp/qdrant")
        cfg = QdrantConfig.from_env()
        assert cfg.mode == "local"
        assert cfg.path == "/tmp/qdrant"


# ---------------------------------------------------------------------------
# QdrantConfig.validate
# ---------------------------------------------------------------------------

class TestQdrantConfigValidate:
    def test_valid_server_config(self):
        QdrantConfig(mode="server", host="localhost", port=6333).validate()

    def test_valid_local_config(self):
        QdrantConfig(mode="local", path="/tmp/qdrant").validate()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            QdrantConfig(mode="bad").validate()  # type: ignore[arg-type]

    def test_server_missing_host_raises(self):
        with pytest.raises(ValueError, match="Host is required"):
            QdrantConfig(mode="server", host="", url=None).validate()

    def test_server_invalid_port_raises(self):
        with pytest.raises(ValueError, match="Invalid port"):
            QdrantConfig(mode="server", host="localhost", port=0).validate()

    def test_local_missing_path_raises(self):
        with pytest.raises(ValueError, match="Path is required"):
            QdrantConfig(mode="local", path="").validate()

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="Timeout"):
            QdrantConfig(mode="local", path="/tmp", timeout=0).validate()

    def test_zero_vector_size_raises(self):
        with pytest.raises(ValueError, match="Vector size"):
            QdrantConfig(mode="local", path="/tmp", vector_size=0).validate()

    def test_quantile_out_of_range_raises(self):
        with pytest.raises(ValueError, match="QDRANT_QUANTILE"):
            QdrantConfig(mode="local", path="/tmp", quantile=0.0).validate()

    def test_negative_hnsw_m_raises(self):
        with pytest.raises(ValueError, match="QDRANT_HNSW_M"):
            QdrantConfig(mode="local", path="/tmp", hnsw_m=-1).validate()

    def test_none_hnsw_m_is_valid(self):
        QdrantConfig(mode="local", path="/tmp", hnsw_m=None).validate()
