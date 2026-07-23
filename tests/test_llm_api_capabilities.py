# -*- coding: utf-8 -*-
"""
查询当前配置的 LLM API 能拿到哪些模型和能力信息。

用法:
    python tests/test_llm_api_capabilities.py
    python tests/test_llm_api_capabilities.py --backend azure
    python tests/test_llm_api_capabilities.py --backend proxy --probe

说明:
    直接运行脚本会请求 /models 接口并打印模型能力元数据。
    加上 --probe 后，会通过 core.llm_client 中的封装探测 chat/tools/image 三类调用。
    pytest 默认只跑本地接口形状检查；若要联网检查，设置 RUN_LLM_API_TESTS=1。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import pytest
except ImportError:  # pragma: no cover - 直接作为脚本运行时不强依赖 pytest
    pytest = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import AZURE_LLM_CONFIG, LLM_CONFIG  # noqa: E402
from core.llm_client import AzureLLMClient, BaseLLMClient, OpenAILLMClient  # noqa: E402


REQUEST_TIMEOUT = 30
TEST_MESSAGES = [{"role": "user", "content": "请只回复 ok"}]
TEST_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "echo_text",
            "description": "返回输入文本",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "需要返回的文本"},
                },
                "required": ["text"],
            },
        },
    }
]
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFgwJ/"
    "l4xS5QAAAABJRU5ErkJggg=="
)


@dataclass
class ModelInfo:
    """统一后的模型信息。"""

    backend: str
    model_id: str
    raw_capabilities: dict[str, Any] = field(default_factory=dict)
    owned_by: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    probes: dict[str, str] = field(default_factory=dict)

    @property
    def capability_labels(self) -> list[str]:
        """把 API 返回的 capabilities 转成便于阅读的标签。"""
        labels: list[str] = []
        for key, value in sorted(self.raw_capabilities.items()):
            if isinstance(value, bool):
                if value:
                    labels.append(key)
            elif value not in (None, "", [], {}):
                labels.append(f"{key}={value}")
        return labels


def _request_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"请求失败: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"接口返回的不是 JSON: {body[:300]}") from exc


def list_proxy_models() -> list[ModelInfo]:
    """查询 OpenAI 兼容代理的 /models 接口。"""
    base_url = LLM_CONFIG.get("base_url", "").rstrip("/")
    api_key = LLM_CONFIG.get("api_key", "")
    if not base_url or not api_key:
        raise RuntimeError("LLM_CONFIG 缺少 base_url 或 api_key")

    payload = _request_json(
        f"{base_url}/models",
        {"Authorization": f"Bearer {api_key}"},
    )
    models = payload.get("data", [])
    return [
        ModelInfo(
            backend="proxy",
            model_id=str(item.get("id", "")),
            raw_capabilities=dict(item.get("capabilities") or {}),
            owned_by=str(item.get("owned_by", "")),
            raw=item,
        )
        for item in models
        if item.get("id")
    ]


def list_azure_models() -> list[ModelInfo]:
    """查询 Azure OpenAI 的 /openai/models 接口。"""
    endpoint = AZURE_LLM_CONFIG.get("azure_endpoint", "").rstrip("/")
    api_key = AZURE_LLM_CONFIG.get("api_key", "")
    api_version = AZURE_LLM_CONFIG.get("api_version", "")
    if not endpoint or not api_key or not api_version:
        raise RuntimeError("AZURE_LLM_CONFIG 缺少 azure_endpoint/api_key/api_version")

    payload = _request_json(
        f"{endpoint}/openai/models?api-version={api_version}",
        {"api-key": api_key},
    )
    models = payload.get("data", [])
    return [
        ModelInfo(
            backend="azure",
            model_id=str(item.get("id", "")),
            raw_capabilities=dict(item.get("capabilities") or {}),
            owned_by=str(item.get("owned_by", "")),
            raw=item,
        )
        for item in models
        if item.get("id")
    ]


def make_client(backend: str, model: str | None = None) -> BaseLLMClient:
    """按 settings.py 的配置创建 llm_client.py 中对应的客户端。"""
    if backend == "azure":
        config = dict(AZURE_LLM_CONFIG)
        if model:
            config["model"] = model
        return AzureLLMClient(config)

    config = dict(LLM_CONFIG)
    if model:
        config["model"] = model
    return OpenAILLMClient(config)


def probe_model_functions(backend: str, model: str) -> dict[str, str]:
    """
    通过 llm_client.py 的三个核心方法探测模型能力。

    返回值:
        {"chat": "ok|fail: ...", "tools": "ok|fail: ...", "image": "ok|fail: ..."}
    """
    client = make_client(backend, model)
    results: dict[str, str] = {}

    try:
        text = client.chat(TEST_MESSAGES, model=model, max_tokens=20)
        results["chat"] = "ok" if text.strip() else "fail: empty response"
    except Exception as exc:
        results["chat"] = f"fail: {exc.__class__.__name__}: {str(exc)[:120]}"

    try:
        tool_result = client.chat_with_tools(TEST_MESSAGES, TEST_TOOL, model=model, max_tokens=40)
        has_content = bool(tool_result.get("content"))
        has_tool_calls = bool(tool_result.get("tool_calls"))
        results["tools"] = "ok" if has_content or has_tool_calls else "fail: empty response"
    except Exception as exc:
        results["tools"] = f"fail: {exc.__class__.__name__}: {str(exc)[:120]}"

    try:
        text = client.chat_with_image(
            [{"role": "user", "content": "这是一张什么颜色的图片？请简短回答。"}],
            ONE_PIXEL_PNG,
            model=model,
            max_tokens=40,
        )
        results["image"] = "ok" if text.strip() else "fail: empty response"
    except Exception as exc:
        results["image"] = f"fail: {exc.__class__.__name__}: {str(exc)[:120]}"

    return results


def collect_models(backends: list[str]) -> list[ModelInfo]:
    all_models: list[ModelInfo] = []
    if "proxy" in backends:
        all_models.extend(list_proxy_models())
    if "azure" in backends:
        all_models.extend(list_azure_models())
    return sorted(all_models, key=lambda item: (item.backend, item.model_id))


def print_model_report(models: list[ModelInfo], show_raw: bool = False) -> None:
    if not models:
        print("未发现模型。")
        return

    current_backend = ""
    for item in models:
        if item.backend != current_backend:
            current_backend = item.backend
            print()
            print("=" * 80)
            print(f"{current_backend.upper()} models")
            print("=" * 80)

        caps = ", ".join(item.capability_labels) if item.capability_labels else "接口未返回 capabilities"
        print(f"- {item.model_id}")
        if item.owned_by:
            print(f"  owned_by: {item.owned_by}")
        print(f"  capabilities: {caps}")
        if item.probes:
            for name in ("chat", "tools", "image"):
                print(f"  {name}: {item.probes.get(name, 'not probed')}")
        if show_raw:
            print(f"  raw: {json.dumps(item.raw, ensure_ascii=False, sort_keys=True)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查询 LLM API 可见模型和能力信息")
    parser.add_argument(
        "--backend",
        choices=["proxy", "azure", "all"],
        default="all",
        help="要查询的后端，默认 all",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="逐个模型调用 chat/tools/image，验证 llm_client.py 封装能力",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="只探测指定模型；可重复传入。默认查询/探测全部模型",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="打印 /models 接口返回的原始模型 JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backends = ["proxy", "azure"] if args.backend == "all" else [args.backend]
    models = collect_models(backends)

    if args.model:
        wanted = set(args.model)
        models = [item for item in models if item.model_id in wanted]

    if args.probe:
        for item in models:
            item.probes = probe_model_functions(item.backend, item.model_id)

    print_model_report(models, show_raw=args.raw)
    return 0


def test_llm_client_surface_is_available() -> None:
    """本地测试：确认 BaseLLMClient 约定的三类调用都在具体实现中存在。"""
    for client_cls in (OpenAILLMClient, AzureLLMClient):
        for method_name in ("chat", "chat_with_tools", "chat_with_image", "parse_json_response"):
            assert callable(getattr(client_cls, method_name))


if pytest is not None:
    requires_live_llm_api = pytest.mark.skipif(
        os.getenv("RUN_LLM_API_TESTS") != "1",
        reason="联网查询模型列表需要显式设置 RUN_LLM_API_TESTS=1",
    )
else:
    requires_live_llm_api = lambda test_func: test_func


@requires_live_llm_api
def test_configured_model_apis_return_models() -> None:
    """联网测试：确认 settings.py 中配置的 API 至少能返回模型列表。"""
    models = collect_models(["proxy", "azure"])
    assert models
    assert all(item.backend in {"proxy", "azure"} and item.model_id for item in models)


if __name__ == "__main__":
    raise SystemExit(main())
