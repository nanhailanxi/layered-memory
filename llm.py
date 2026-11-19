#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSeek 接口封装。只做一件事：把 messages 发过去，拿回一句话。"""

import os


class LLMError(Exception):
    pass


def _config():
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return base, model, key


def chat(messages, temperature: float = 0.9, max_tokens: int = 800, timeout: int = 60) -> str:
    import requests

    base, model, key = _config()
    if not key:
        raise LLMError("缺少 DEEPSEEK_API_KEY")

    try:
        response = requests.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError(f"网络不通：{exc}") from exc

    if response.status_code != 200:
        raise LLMError(f"HTTP {response.status_code}：{response.text[:300]}")

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError) as exc:
        raise LLMError(f"返回格式不对：{exc}") from exc
