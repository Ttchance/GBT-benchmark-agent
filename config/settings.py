# -*- coding: utf-8 -*-
"""
全局配置文件
"""

import os
from pathlib import Path


def _load_dotenv(path: str | Path | None = None, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency."""
    env_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()

# ─────────────────────────────────────────
# LLM 配置
# ─────────────────────────────────────────
# LLM_CONFIG = {
#     "model": "gpt-4o",          # 使用的模型名称
#     "temperature": 0.0,
#     "max_tokens": 4096,
#     "api_key": os.getenv("OPENAI_API_KEY", ""),
# }

LLM_CONFIG = {
    "model": _env_str("OPENAI_MODEL", "gpt-5"),
    "api_key": _env_str("OPENAI_API_KEY"),
    "base_url": _env_str("OPENAI_BASE_URL"),
}

# ─────────────────────────────────────────
# Azure OpenAI 配置（导师提供）
# ─────────────────────────────────────────
AZURE_LLM_CONFIG = {
    "azure_endpoint": _env_str("AZURE_OPENAI_ENDPOINT"),
    "api_key": _env_str("AZURE_OPENAI_API_KEY"),
    "api_version": _env_str("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    "model": _env_str("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"),       # Azure 部署名称
    # "model": "claude-opus-4-6",  
    "temperature": 0.0,
    "max_tokens": 16000,
}

# ─────────────────────────────────────────
# 后端选择: "proxy" 使用代理 OpenAI, "azure" 使用 Azure OpenAI
# ─────────────────────────────────────────
LLM_BACKEND = _env_str("LLM_BACKEND", "proxy")


# ─────────────────────────────────────────
# 多模态模型配置
# ─────────────────────────────────────────
MULTIMODAL_LLM_CONFIG = {
    "model": _env_str("MULTIMODAL_OPENAI_MODEL", _env_str("OPENAI_MODEL", "gpt-5")),
    "temperature": 0.0,
    "max_tokens": 4096,
    "api_key": _env_str("MULTIMODAL_OPENAI_API_KEY", _env_str("OPENAI_API_KEY")),
    "base_url": _env_str("MULTIMODAL_OPENAI_BASE_URL", _env_str("OPENAI_BASE_URL")),
}

# ─────────────────────────────────────────
# 文档解析配置
# ─────────────────────────────────────────
PARSER_CONFIG = {
    "supported_formats": [".docx", ".pdf"],
    "max_paragraph_length": 2000,
    "context_window": 2,        # 术语上下文前后各 N 句
}

# ─────────────────────────────────────────
# PDF 解析路径开关
# ─────────────────────────────────────────
# True: 使用 Docling 新解析路径；False: 使用原有 PyMuPDF/LLM/regex 路径
Docling_Is_true = _env_bool("DOCLING_IS_TRUE", False)

# ─────────────────────────────────────────
# 报告输出配置
# ─────────────────────────────────────────
REPORT_CONFIG = {
    "output_dir": "reports/output",
    "formats": ["json", "html"],
}

# ─────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────
LOG_CONFIG = {
    "level": "INFO",
    "file": "logs/gbt_parse.log",
}

# ─────────────────────────────────────────
# RAG 配置（ChromaDB + Ollama 本地 embedding）
# ─────────────────────────────────────────
RAG_CONFIG = {
    "persist_dir": os.getenv("RAG_CHROMA_DIR", "data/chroma_gbt_kb"),
    "collection_name": os.getenv("RAG_COLLECTION", "gbt_review_rules"),
    "seed_path": os.getenv("RAG_SEED_PATH", "data/rag/gbt_review_rules.jsonl"),
    "embedding_model": os.getenv("RAG_EMBEDDING_MODEL", "bge-m3:latest"),
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "top_k": int(os.getenv("RAG_TOP_K", "5")),
}
