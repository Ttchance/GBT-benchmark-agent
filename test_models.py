# -*- coding: utf-8 -*-
"""
测试各后端可用的大模型

用法：
    python test_models.py                  # 测试两个后端的所有 chat 模型
    python test_models.py --backend azure  # 只测 Azure
    python test_models.py --backend proxy  # 只测代理
    python test_models.py --backend diagnose  # 诊断 Azure 是认证问题还是部署名问题
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.settings import LLM_CONFIG, AZURE_LLM_CONFIG

TEST_MESSAGE = [{"role": "user", "content": "请回复ok"}]


def _request_json(method: str, url: str, headers: dict, body: Optional[dict] = None, timeout: int = 30) -> Tuple[int, dict, str]:
    """使用标准库发送 JSON 请求，避免依赖 requests。"""
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = {}
            return resp.status, parsed, text
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = {}
        return exc.code, parsed, text


def _preview_text(text: str, limit: int = 200) -> str:
    cleaned = " ".join((text or "").strip().split())
    return cleaned[:limit] if cleaned else "<empty response>"


# ── 代理端（OpenAI 兼容） ─────────────────────────────────────────────────────

def list_proxy_models() -> List[str]:
    """获取代理端所有模型 ID。"""
    base_url = LLM_CONFIG["base_url"].rstrip("/")
    api_key = LLM_CONFIG["api_key"]
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is empty. Set it in .env.")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set it in .env.")

    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    status, payload, text = _request_json("GET", url, headers=headers, timeout=15)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {text[:200]}")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise RuntimeError(
            "The /models response is not OpenAI-compatible JSON. "
            "Check OPENAI_BASE_URL; it usually needs the /v1 suffix, "
            "for example https://api.vveai.com/v1. "
            f"Requested URL: {url}. Response preview: {_preview_text(text)}"
        )
    return sorted(m["id"] for m in data if isinstance(m, dict) and "id" in m)


def test_proxy_model(model: str) -> Tuple[bool, str]:
    """测试代理端某个模型是否可用，返回 (成功, 信息)。"""
    url = f"{LLM_CONFIG['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
        "Content-Type": "application/json",
    }
    body = {"model": model, "messages": TEST_MESSAGE, "max_tokens": 20}
    try:
        status, payload, text = _request_json("POST", url, headers=headers, body=body, timeout=30)
        if status == 200:
            content = payload["choices"][0]["message"]["content"]
            return True, content.strip()[:50]
        return False, f"HTTP {status}: {text[:80]}"
    except Exception as e:
        return False, str(e)[:80]


# ── Azure 端 ──────────────────────────────────────────────────────────────────

def _mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _extract_error(payload: dict, text: str) -> Tuple[str, str]:
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    code = str(error.get("code", ""))
    message = str(error.get("message", "")) or text[:200]
    return code, message

def list_azure_chat_models() -> List[str]:
    """获取 Azure 端支持 chat_completion 的模型 ID。"""
    url = (
        f"{AZURE_LLM_CONFIG['azure_endpoint']}openai/models"
        f"?api-version={AZURE_LLM_CONFIG['api_version']}"
    )
    headers = {"api-key": AZURE_LLM_CONFIG["api_key"]}
    status, payload, text = _request_json("GET", url, headers=headers, timeout=15)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {text[:200]}")
    return sorted(
        m["id"]
        for m in payload.get("data", [])
        if m.get("capabilities", {}).get("chat_completion")
    )


def test_azure_model(model: str) -> Tuple[bool, str]:
    """测试 Azure 端某个模型是否可调用，返回 (成功, 信息)。"""
    url = (
        f"{AZURE_LLM_CONFIG['azure_endpoint']}openai/deployments/{model}"
        f"/chat/completions?api-version={AZURE_LLM_CONFIG['api_version']}"
    )
    headers = {
        "api-key": AZURE_LLM_CONFIG["api_key"],
        "Content-Type": "application/json",
    }
    body = {"messages": TEST_MESSAGE, "max_tokens": 20}
    try:
        status, payload, text = _request_json("POST", url, headers=headers, body=body, timeout=30)
        if status == 200:
            content = payload["choices"][0]["message"]["content"]
            return True, content.strip()[:50]
        return False, f"HTTP {status}: {text[:80]}"
    except Exception as e:
        return False, str(e)[:80]


def diagnose_azure_api() -> None:
    """诊断 Azure 是 API/key 认证问题，还是 deployment/model 配置问题。"""
    endpoint = AZURE_LLM_CONFIG["azure_endpoint"]
    api_version = AZURE_LLM_CONFIG["api_version"]
    deployment = AZURE_LLM_CONFIG["model"]
    api_key = AZURE_LLM_CONFIG["api_key"]
    headers = {"api-key": api_key}

    checks = [
        (
            "1. 列出模型/能力",
            f"{endpoint}openai/models?api-version={api_version}",
            "GET",
            None,
        ),
        (
            f"2. 调用当前配置 deployment: {deployment}",
            f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version={api_version}",
            "POST",
            {"messages": TEST_MESSAGE, "max_tokens": 20},
        ),
        (
            "3. 调用故意不存在的 deployment",
            f"{endpoint}openai/deployments/__definitely_not_a_deployment__/chat/completions?api-version={api_version}",
            "POST",
            {"messages": TEST_MESSAGE, "max_tokens": 20},
        ),
    ]

    print("=" * 72)
    print("  Azure API 诊断")
    print("=" * 72)
    print(f"endpoint     : {endpoint}")
    print(f"api_version  : {api_version}")
    print(f"deployment   : {deployment}")
    print(f"api_key      : {_mask_secret(api_key)}")
    print()

    results = []
    for title, url, method, body in checks:
        print(title)
        try:
            status, payload, text = _request_json(method, url, headers=headers, body=body, timeout=30)
            code, message = _extract_error(payload, text)
            results.append((title, status, code, message))
            print(f"  HTTP 状态 : {status}")
            if status == 200:
                content = ""
                try:
                    content = payload["choices"][0]["message"]["content"].strip()
                except Exception:
                    pass
                print(f"  结果      : 成功 {content[:80]}")
            else:
                print(f"  错误代码  : {code or '<none>'}")
                print(f"  错误信息  : {message[:300]}")
        except Exception as exc:
            results.append((title, 0, "CLIENT_ERROR", str(exc)))
            print(f"  本地异常  : {exc}")
        print()

    statuses = [item[1] for item in results]
    codes = [item[2] for item in results]
    configured_call = results[1]

    print("-" * 72)
    print("诊断结论：")
    if "AuthenticationTypeDisabled" in codes:
        print("  API Key 认证被这个 Azure 资源禁用了。")
        print("  这不是模型名不对；请求在检查 deployment/model 之前就被认证层拒绝了。")
        print("  解决方式：开启 Key based authentication，或改用 Microsoft Entra ID 认证。")
    elif configured_call[1] == 200:
        print("  当前 API Key、endpoint、api_version、deployment 都可以正常调用。")
        print("  如果 main.py 仍失败，问题更可能在请求参数、max_tokens 或调用封装。")
    elif configured_call[1] == 404 or configured_call[2] in {"DeploymentNotFound", "ResourceNotFound"}:
        print("  API/key 能到达服务，但当前 deployment/model 名称可能不对，或部署不存在。")
        print("  请检查 config/settings.py 里的 AZURE_LLM_CONFIG['model'] 是否为 Azure 部署名，不一定是模型名。")
    elif configured_call[1] in {401, 403}:
        print("  API Key 无权限、失效，或资源访问策略拒绝。")
        print("  如果错误代码不是 AuthenticationTypeDisabled，请检查 key 是否属于该 endpoint。")
    elif configured_call[1] == 400:
        print("  请求到达服务，但请求参数或 api_version 可能不兼容。")
        print("  请重点检查 api_version、max_tokens/max_completion_tokens 和模型接口类型。")
    elif any(status == 0 for status in statuses):
        print("  本地网络或客户端请求失败，尚不能判断模型和 API 是否可用。")
    else:
        print("  返回结果不属于常见情况，请根据上面的 HTTP 状态和错误代码继续判断。")


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def run_test(backend: str):
    if backend == "diagnose":
        diagnose_azure_api()
        return

    if backend in ("proxy", "all"):
        print("=" * 60)
        print("  代理端 (proxy)  —  " + LLM_CONFIG["base_url"])
        print("=" * 60)
        models = list_proxy_models()
        print(f"共发现 {len(models)} 个模型\n")

        ok_count = 0
        for m in models:
            success, info = test_proxy_model(m)
            status = "✓" if success else "✗"
            print(f"  {status}  {m:<45s}  {info}")
            if success:
                ok_count += 1
        print(f"\n代理端可用: {ok_count}/{len(models)}\n")

    if backend in ("azure", "all"):
        print("=" * 60)
        print("  Azure 端  —  " + AZURE_LLM_CONFIG["azure_endpoint"])
        print("=" * 60)
        models = list_azure_chat_models()
        print(f"共发现 {len(models)} 个 chat 模型\n")

        ok_count = 0
        for m in models:
            success, info = test_azure_model(m)
            status = "✓" if success else "✗"
            print(f"  {status}  {m:<45s}  {info}")
            if success:
                ok_count += 1
        print(f"\nAzure 端可用: {ok_count}/{len(models)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试可用大模型")
    parser.add_argument(
        "--backend", "-b",
        choices=["proxy", "azure", "all", "diagnose"],
        default="all",
        help="测试哪个后端（默认 all）",
    )
    args = parser.parse_args()
    run_test(args.backend)
