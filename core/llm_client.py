from __future__ import annotations

# -*- coding: utf-8 -*-
"""
LLM 调用封装层
统一管理与大语言模型的交互，屏蔽底层 API 差异。
"""

from abc import ABC, abstractmethod
from typing import Any
import base64
import logging

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类"""

    _USAGE_KEYS = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")

    def _init_usage_tracking(self) -> None:
        self._usage = {key: 0 for key in self._USAGE_KEYS}

    @staticmethod
    def _usage_value(usage, key: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            value = usage.get(key, 0)
        else:
            value = getattr(usage, key, 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _record_usage(self, response) -> None:
        if not hasattr(self, "_usage"):
            self._init_usage_tracking()
        usage = getattr(response, "usage", None)
        self._usage["requests"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._usage[key] += self._usage_value(usage, key)

    def get_usage_snapshot(self) -> dict[str, int]:
        if not hasattr(self, "_usage"):
            self._init_usage_tracking()
        return dict(self._usage)

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """
        发送对话请求并返回模型回复文本。

        Args:
            messages: OpenAI 格式的消息列表
                      [{"role": "system", "content": "..."}, ...]
            **kwargs: 额外参数（temperature / max_tokens 等）

        Returns:
            模型回复的纯文本字符串
        """

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> dict:
        """
        带工具调用（Function Calling）的对话。

        Args:
            messages: OpenAI 格式的消息列表
            tools: OpenAI tools schema 列表
            **kwargs: 额外参数（temperature / max_tokens 等）

        Returns:
            {"content": str | None, "tool_calls": list | None}
            tool_calls 格式: [{"id": str, "name": str, "arguments": str}, ...]
        """

    @abstractmethod
    def chat_with_image(self, messages: list[dict], image_data: bytes, **kwargs) -> str:
        """
        多模态调用（文本 + 图片）。

        Args:
            messages: 消息列表
            image_data: 图片原始字节
            **kwargs: 额外参数

        Returns:
            模型回复的纯文本字符串
        """

    def parse_json_response(self, response: str) -> Any:
        """
        将模型返回的 JSON 字符串解析为 Python 对象。
        容错处理：去除 <think> 思维链标签、markdown 代码块包裹等。
        """
        import json, re

        clean = response.strip()

        # 去除 <think>...</think> 思维链标签（部分模型会返回）
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()

        # 去除 ```json ... ``` 包裹
        md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", clean, flags=re.DOTALL | re.IGNORECASE)
        if md_match:
            clean = md_match.group(1).strip()

        # 如果还不是 JSON，尝试提取第一个 [ ... ] 或 { ... }
        if clean and clean[0] not in ("[", "{"):
            bracket = re.search(r"(\[.*\])", clean, flags=re.DOTALL)
            brace = re.search(r"(\{.*\})", clean, flags=re.DOTALL)
            if bracket:
                clean = bracket.group(1)
            elif brace:
                clean = brace.group(1)

        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}\n原始内容: {response[:200]}")
            return {}


class AzureLLMClient(BaseLLMClient):
    """基于 Azure OpenAI 的 LLM 客户端"""

    def __init__(self, config: dict):
        self._init_usage_tracking()
        self.config = config
        self.model = config.get("model", "gpt-5-mini")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.client = None

        azure_endpoint = config.get("azure_endpoint", "")
        api_key = config.get("api_key", "")
        api_version = config.get("api_version", "2024-12-01-preview")

        if not api_key or api_key == "<your-api-key>":
            logger.warning("未配置 AZURE_OPENAI_API_KEY，Azure LLM 客户端将以降级模式运行。")
            return

        try:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        except Exception as exc:
            logger.warning("Azure OpenAI 客户端初始化失败，将以降级模式运行: %s", exc)
            self.client = None

    def chat(self, messages: list[dict], **kwargs) -> str:
        if self.client is None:
            return "{}"

        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        response = self.client.chat.completions.create(**params)
        self._record_usage(response)
        message = response.choices[0].message
        return message.content or ""

    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> dict:
        if self.client is None:
            return {"content": "{}", "tool_calls": None}

        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        self._record_usage(response)
        msg = response.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

    def chat_with_image(self, messages: list[dict], image_data: bytes, **kwargs) -> str:
        if self.client is None:
            return "{}"

        image_b64 = base64.b64encode(image_data).decode("ascii")
        prepared_messages = list(messages)
        if prepared_messages:
            last = dict(prepared_messages[-1])
            content = last.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{0}".format(image_b64),
                    },
                }
            )
            last["content"] = content
            prepared_messages[-1] = last
        else:
            prepared_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{0}".format(image_b64),
                            },
                        },
                    ],
                }
            ]

        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=prepared_messages,
            max_completion_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        self._record_usage(response)
        message = response.choices[0].message
        return message.content or ""


class OpenAILLMClient(BaseLLMClient):
    """基于 OpenAI SDK 的 LLM 客户端（占位实现）"""

    def __init__(self, config: dict):
        self._init_usage_tracking()
        self.config = config
        self.model = config.get("model", "gpt-5")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.client = None

        if not self.api_key:
            logger.warning("未配置 OPENAI_API_KEY，LLM 客户端将以降级模式运行。")
            return

        try:
            from openai import OpenAI
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)
        except Exception as exc:
            logger.warning("OpenAI 客户端初始化失败，将以降级模式运行: %s", exc)
            self.client = None

    def chat(self, messages: list[dict], **kwargs) -> str:
        if self.client is None:
            return "{}"

        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        logger.info("OpenAI request model=%s base_url=%s", params["model"], self.base_url)
        response = self.client.chat.completions.create(**params)
        self._record_usage(response)
        message = response.choices[0].message
        return message.content or ""

    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs) -> dict:
        if self.client is None:
            return {"content": "{}", "tool_calls": None}

        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        self._record_usage(response)
        msg = response.choices[0].message
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in msg.tool_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

    def chat_with_image(self, messages: list[dict], image_data: bytes, **kwargs) -> str:
        if self.client is None:
            return "{}"

        image_b64 = base64.b64encode(image_data).decode("ascii")
        prepared_messages = list(messages)
        if prepared_messages:
            last = dict(prepared_messages[-1])
            content = last.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,{0}".format(image_b64),
                    },
                }
            )
            last["content"] = content
            prepared_messages[-1] = last
        else:
            prepared_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请分析这张图片。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,{0}".format(image_b64),
                            },
                        },
                    ],
                }
            ]

        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=prepared_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        self._record_usage(response)
        message = response.choices[0].message
        return message.content or ""
