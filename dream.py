#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""做梦：离线拿原始对话核对记忆，合并重复、按原文改错。

设计见 docs/design.md。只在后台跑（或 /做梦 手动），主回复不等它。
原则：只改不删；事实修正照旧走版本化；合并只是把重复条目挂到保留的那条下面。
"""

import json
import os
import re
import time

import llm
import memory


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


DREAM_INTERVAL_DAYS = _env_int("MEMORY_DREAM_DAYS", 7)   # 隔多久做一次
DREAM_MIN_NEW_MESSAGES = _env_int("MEMORY_DREAM_MESSAGES", 20)  # 至少攒这么多条新消息
MAX_ACTIONS = 8
MAX_FACTS = 30
MAX_MEMORIES = 60
MAX_TRANSCRIPT_CHARS = 5000

DREAM_SYSTEM = """你在替一个角色做离线整理记忆。你只会看到：当前状态、记忆索引、最近的原始对话。
只根据原始对话核对，不要补充、不要推测。没有把握就不动：宁可漏，不要改错。

可做的动作（最多 8 条）：
1. merge：两条记忆是同一件事，保留信息更全的一条，把另一条并过去。
2. correct_memory：记忆和原始对话矛盾，按原文改写标题或内容。
3. correct_fact：状态和原始对话矛盾，按原文改写内容，subject 不变。

只处理上面给了编号的条目。置顶、封口、已合并的不在列表里。

只输出一个 JSON 对象，不要解释、不要代码块：
{"actions":[...],"summary":"一句话"}
动作格式：
{"type":"merge","keep":12,"drop":[15,18],"reason":"同一次搬家"}
{"type":"correct_memory","id":20,"title":"索引","content":"按原文改写后的一两句","reason":"原文说……"}
{"type":"correct_fact","id":3,"content":"按原文改写后的一句","reason":"原文说……"}
没有要改的就输出 {"actions":[],"summary":"没发现要改的"}。"""


def _text(value, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _clip(value, limit: int) -> str:
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("没有 JSON 对象")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON 不是对象")
    return data


def _candidate_text(store, user_id: str, now: int) -> str:
    parts = []

    facts = store.current_facts(user_id)[:MAX_FACTS]
    if facts:
        lines = []
        for fact in facts:
            when = time.strftime("%Y-%m", time.localtime(int(fact["as_of"] or now)))
            lines.append(f"F{fact['id']} {fact['subject']}：{_clip(fact['content'], 80)}（{when}）")
        parts.append("【当前状态】\n" + "\n".join(lines))

    memories = [m for m in store.memories(user_id) if not m.get("pinned")][-MAX_MEMORIES:]
    if memories:
        lines = []
        for item in memories:
            when = time.strftime("%Y-%m-%d", time.localtime(int(item["as_of"] or now)))
            lines.append(
                f"M{item['id']} {when} {_clip(item['title'], 24)}：{_clip(item['content'], 60)}"
            )
        parts.append("【记忆索引】\n" + "\n".join(lines))

    rows = store.recent_messages(user_id, limit=80)
    picked = []
    total = 0
    for row in reversed(rows):
        who = "用户" if row["role"] == "user" else "助手"
        line = f"[#{row['id']} {who}] {_clip(row['content'], 300)}"
        if picked and total + len(line) > MAX_TRANSCRIPT_CHARS:
            break
        picked.append(line)
        total += len(line)
    picked.reverse()
    if picked:
        parts.append("【最近的原始对话】\n" + "\n".join(picked))

    return "\n\n".join(parts)


def _request(prompt: str) -> dict:
    messages = [
        {"role": "system", "content": DREAM_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    raw = llm.chat(messages, temperature=0.2, max_tokens=900, timeout=60)
    return _parse_json(raw)


def apply(store, user_id: str, data: dict, now: int = None) -> list:
    """执行模型给的整理动作。只认列表里存在的编号，越界的丢掉。"""
    now = int(now or time.time())
    facts = {int(f["id"]): f for f in store.current_facts(user_id)}
    memories = {int(m["id"]): m for m in store.memories(user_id) if not m.get("pinned")}
    report = []

    actions = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(actions, list):
        return report

    for action in actions[:MAX_ACTIONS]:
        if not isinstance(action, dict):
            continue
        kind = _text(action.get("type"), 20)
        reason = _text(action.get("reason"), 120)

        if kind == "merge":
            try:
                keep = int(action.get("keep"))
                drops = [int(x) for x in (action.get("drop") or [])]
            except (TypeError, ValueError):
                continue
            if keep not in memories:
                continue
            for drop in drops:
                if drop == keep or drop not in memories:
                    continue
                if store.merge_memory(user_id, keep, drop):
                    memories.pop(drop, None)
                    report.append({
                        "type": "merge",
                        "text": f"M{drop} → M{keep}",
                        "reason": reason,
                    })

        elif kind == "correct_memory":
            try:
                item_id = int(action.get("id"))
            except (TypeError, ValueError):
                continue
            old = memories.get(item_id)
            if not old:
                continue
            content = _text(action.get("content"), 300)
            title = _text(action.get("title"), 24) or old["title"]
            if not content or content == old["content"]:
                continue
            store.update_memory(user_id, item_id, title=title, content=content)
            memories[item_id] = dict(old, title=title, content=content)
            report.append({
                "type": "correct_memory",
                "text": f"改 M{item_id}",
                "reason": reason,
            })

        elif kind == "correct_fact":
            try:
                item_id = int(action.get("id"))
            except (TypeError, ValueError):
                continue
            old = facts.get(item_id)
            if not old:
                continue
            content = _text(action.get("content"), 120)
            if not content or content == old["content"]:
                continue
            result = memory.record_fact(
                store, user_id, old["subject"], content, old["importance"], now=now
            )
            facts.pop(item_id, None)
            report.append({
                "type": "correct_fact",
                "text": f"改 F{item_id}（{result.get('action')}）",
                "reason": reason,
            })

    return report


def run(store, user_id: str, now: int = None) -> dict:
    now = int(now or time.time())
    memories = store.memories(user_id)
    facts = store.current_facts(user_id)
    if len(memories) < 3 and len(facts) < 3:
        return {"skipped": "not_enough"}

    prompt = _candidate_text(store, user_id, now)
    data = _request(prompt)
    report = apply(store, user_id, data, now=now)
    summary = _text(data.get("summary"), 120)
    store.save_dream(
        user_id,
        ran_at=now,
        message_id=store.last_message_id(user_id),
        summary=summary,
        actions=json.dumps(report, ensure_ascii=False),
    )
    return {
        "skipped": None,
        "report": report,
        "summary": summary,
        "memory_count": len(memories),
        "fact_count": len(facts),
    }


_LAST_ATTEMPT = 0


def maybe_run(store, user_id: str, now: int = None) -> dict:
    """每周、且攒够新消息才自动做一次。失败后一小时再试，别每轮都撞墙。"""
    global _LAST_ATTEMPT
    now = int(now or time.time())
    if now - _LAST_ATTEMPT < 3600:
        return None

    last = store.last_dream(user_id)
    if last and now - int(last["ran_at"]) < DREAM_INTERVAL_DAYS * 86400:
        return None

    since = int(last["message_id"] or 0) if last else 0
    if store.count_messages_after(user_id, since) < DREAM_MIN_NEW_MESSAGES:
        return None
    if len(store.memories(user_id)) < 5:
        return None

    _LAST_ATTEMPT = now
    try:
        return run(store, user_id, now)
    except Exception:
        return None


def format_report(result) -> str:
    if not result:
        return "还没到整理的时候。"
    if result.get("skipped"):
        return "记忆还太少，先攒着。"

    report = result.get("report") or []
    summary = result.get("summary") or ""
    if not report:
        return "整理完了，没发现要改的。" + (f"\n{summary}" if summary else "")

    lines = ["整理完了："]
    for item in report:
        suffix = f"（{item['reason']}）" if item.get("reason") else ""
        lines.append(f"- {item['text']}{suffix}")
    lines.append("原文没动，合并的只是重复条目。")
    return "\n".join(lines)
