#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆三层的检索与排版：权重、预算、拼上下文。

三层里这里管两层半：
- 人格层由调用方传进来（persona.md），永远全文进上下文，不检索、不衰减；
- 事实层（facts）、情景层（memories）在这里算权重、选条目；
- 当前对话（messages）按 token 预算从最新往回塞。

设计见 docs/design.md
"""

import math
import os
import random
import re
import time

# ---------- 待定参数（第七节）：先按设计稿的推荐值，可用 .env 覆盖 ----------


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


HALF_LIFE_DAYS = _env_int("MEMORY_HALF_LIFE", 60)        # 时效半衰期
FACT_TOKEN_BUDGET = _env_int("MEMORY_FACT_TOKENS", 800)   # 事实层预算
EPISODE_TOKEN_BUDGET = _env_int("MEMORY_EPISODE_TOKENS", 3000)  # 情景层预算
ASSOCIATION_GAP_DAYS = _env_int("MEMORY_ASSOC_GAP", 14)   # 主动联想间隔
RECALL_TOKEN_BUDGET = _env_int("MEMORY_RECALL_TOKENS", 600)     # 置顶 + 联想那一块

# 检索层的小上限，防一条长记忆把预算吃光
MAX_PINNED = 5
MAX_MATCHED = 6
MAX_STALE_FACTS = 2
MIN_RELEVANCE = 0.15

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-z0-9_]{2,}")
_PUNCT = re.compile(r"[^\w\u4e00-\u9fff]+")

# 高频虚词：单字（用于过滤 bigram）与双字（整词）
_STOP_CHARS = set("的了是我你他她它们在有和就不也都很太更最这那哪吧呢啊吗呀哦嗯哈么着过与及或而被把让给对从到向于为以之其此每各些个点么呗咯哇")
_STOPWORDS = set(
    "什么 怎么 为什么 因为 所以 但是 然后 现在 时候 一个 一下 可以 没有 不是 就是 自己 "
    "这个 那个 我们 你们 他们 应该 觉得 知道 感觉 真的 其实 反正 还是 已经 一直 有点 "
    "一些 这样 那样 怎样 东西 事情 问题 地方".split()
)


def estimate_tokens(text: str) -> int:
    """粗估 token：中文按 1 字 1 token，ASCII 按 4 字符 1 token，宁多勿少。"""
    text = text or ""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    other_chars = len(text) - ascii_chars
    return int(math.ceil(other_chars + ascii_chars / 4.0)) + 1


def bigrams(text: str) -> set:
    """两字一组的字面特征。不用单字，避免「的、我、你」到处误命中。"""
    text = (text or "").lower()
    grams = set()
    for run in _CJK.findall(text):
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            gram = run[i:i + 2]
            if gram in _STOPWORDS:
                continue
            if gram[0] in _STOP_CHARS and gram[1] in _STOP_CHARS:
                continue
            grams.add(gram)
    for word in _ASCII_WORD.findall(_PUNCT.sub(" ", text)):
        if word not in _STOPWORDS:
            grams.add(word)
    return grams


def match_ratio(query_grams: set, text: str) -> float:
    """问题里的特征有多少在这段文字里出现（0–1）。"""
    if not query_grams:
        return 0.0
    return len(query_grams & bigrams(text)) / float(len(query_grams))


def relevance(query_grams: set, memory: dict) -> float:
    """标题、正文、关键词各自匹配，取最高。"""
    fields = [
        memory.get("title") or "",
        memory.get("content") or "",
        memory.get("keywords") or "",
    ]
    return max((match_ratio(query_grams, f) for f in fields), default=0.0)


def freshness(last_used_at, now: int, half_life_days: int = None) -> float:
    """0.5 ^ (距 last_used_at 的天数 / 半衰期)。从最后一次被用到算。"""
    half_life = float(half_life_days or HALF_LIFE_DAYS)
    if not last_used_at:
        return 1.0
    days = max(0.0, (now - float(last_used_at)) / 86400.0)
    return 0.5 ** (days / max(half_life, 1.0))


def weight(query_grams: set, memory: dict, now: int) -> float:
    """关联度 × 时效 × 重要度。乘法：任何一项接近 0，整体就接近 0。"""
    rel = relevance(query_grams, memory)
    imp = max(1, min(5, int(memory.get("importance") or 3))) / 5.0
    return rel * freshness(memory.get("last_used_at"), now) * imp


def _age_days(memory: dict, now: int) -> float:
    # 以「这件事发生的时间」算旧不旧；读进来但一直没提的旧事也算旧
    last = memory.get("last_used_at") or memory.get("as_of") or memory.get("created_at")
    return max(0.0, (now - float(last)) / 86400.0)


def _date(ts) -> str:
    return time.strftime("%Y-%m", time.localtime(int(ts or 0)))


def _clip(text: str, limit: int = 90) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------- 事实层 ----------


def select_facts(store, user_id: str, query_grams: set, now: int) -> tuple:
    """当前有效状态全带上（按重要度），再补最多两条相关但已过时的。"""
    lines = []
    used = []
    budget = FACT_TOKEN_BUDGET

    current = store.current_facts(user_id)
    for fact in current:
        line = f"- {fact['subject']}：{_clip(fact['content'])}（{_date(fact['as_of'])}）"
        cost = estimate_tokens(line)
        if cost > budget and lines:
            break
        budget -= cost
        lines.append(line)
        used.append(int(fact["id"]))

    stale = []
    for fact in store.superseded_facts(user_id):
        if len(stale) >= MAX_STALE_FACTS:
            break
        if match_ratio(query_grams, f"{fact['subject']}{fact['content']}") <= 0:
            continue
        stale.append(fact)
    if stale:
        lines.append("（以下已过时，别当成现在）")
        for fact in stale:
            line = f"- {fact['subject']}：{_clip(fact['content'])}（已过时，{_date(fact['as_of'])}）"
            cost = estimate_tokens(line)
            if cost > budget:
                break
            budget -= cost
            lines.append(line)
            used.append(int(fact["id"]))
    return lines, used


def record_fact(store, user_id: str, subject: str, content: str,
                importance: int = 3, as_of: int = None, now: int = None) -> dict:
    """写一条状态。同一 subject 的新值不删旧的，只把旧的标成被取代。"""
    now = int(now or time.time())
    subject = " ".join((subject or "").split())[:20]
    content = " ".join((content or "").split())
    if not subject or not content:
        return {"action": "skipped"}

    existing = store.facts_for_subject(user_id, subject)
    current = [f for f in existing if not f["superseded_by"]]
    for fact in current:
        if _same_content(fact["content"], content):
            store.touch_fact(fact["id"], now)
            return {"action": "same", "id": int(fact["id"])}

    new_id = store.add_fact(user_id, subject, content, as_of or now, importance)
    for fact in current:
        store.supersede_fact(fact["id"], new_id)
    return {
        "action": "new" if not current else "superseded",
        "id": new_id,
        "superseded": [int(f["id"]) for f in current],
    }


def _same_content(a: str, b: str) -> bool:
    """同一条事实的不同说法：完全一样、一个掐掉中间一段能变成另一个、或字面高度重叠。"""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 2 and len(short) / float(len(long)) >= 0.5:
        if short in long:
            return True
        # 「住在北京朝阳区」→「住在朝阳区」：短的是长的前缀 + 后缀
        for cut in range(len(short) + 1):
            if long.startswith(short[:cut]) and long.endswith(short[cut:]):
                return True
    left = bigrams(a)
    right = bigrams(b)
    if not left or not right:
        return False
    union = left | right
    return len(left & right) / float(len(union)) >= 0.6


# ---------- 情景层 ----------


def select_episode(store, user_id: str, budget: int = None) -> list:
    """从最新往回塞，塞到预算为止；跳过已经被摘要覆盖的旧消息。"""
    budget = int(budget or EPISODE_TOKEN_BUDGET)
    covered = store.last_covered_message_id(user_id)
    rows = store.recent_messages(user_id, after_id=covered, limit=600)

    taken = []
    total = 0
    for row in reversed(rows):
        cost = estimate_tokens(row["content"]) + 4
        if taken and total + cost > budget:
            break
        taken.append(row)
        total += cost
    taken.reverse()

    # 开头若是助手的话，前面那句已经没进来了，去掉，避免上下文从半截开始
    while taken and taken[0]["role"] != "user":
        taken.pop(0)
    return [{"role": r["role"], "content": r["content"]} for r in taken]


def _memory_line(memory: dict, tag: str = "") -> str:
    prefix = f"{tag}｜" if tag else ""
    return f"- {prefix}{_date(memory.get('as_of'))} {_clip(memory.get('title'), 24)}：{_clip(memory.get('content'))}"


def select_recall(store, user_id: str, query_grams: set, now: int, rng=None) -> dict:
    """置顶的、这次相关的、突然想起的一条。"""
    rng = rng or random
    candidates = store.memories(user_id)
    budget = RECALL_TOKEN_BUDGET

    pinned = sorted(
        [m for m in candidates if m.get("pinned")],
        key=lambda m: (-int(m.get("importance") or 3), -int(m.get("as_of") or 0)),
    )[:MAX_PINNED]

    pinned_lines = []
    used = []
    for memory in pinned:
        line = _memory_line(memory, "置顶")
        cost = estimate_tokens(line)
        if cost > budget and pinned_lines:
            break
        budget -= cost
        pinned_lines.append(line)
        used.append(int(memory["id"]))

    taken_ids = set(used)
    scored = []
    for memory in candidates:
        if int(memory["id"]) in taken_ids:
            continue
        score = weight(query_grams, memory, now)
        if score <= 0:
            continue
        if relevance(query_grams, memory) < MIN_RELEVANCE:
            continue
        scored.append((score, memory))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    matched_lines = []
    for score, memory in scored[:MAX_MATCHED]:
        line = _memory_line(memory)
        cost = estimate_tokens(line)
        if cost > budget:
            break
        budget -= cost
        matched_lines.append(line)
        taken_ids.add(int(memory["id"]))
        used.append(int(memory["id"]))

    association = None
    association_line = None
    old = [
        m for m in candidates
        if int(m["id"]) not in taken_ids
        and not m.get("pinned")
        and not m.get("suppressed")
        and _age_days(m, now) >= ASSOCIATION_GAP_DAYS
    ]
    if old:
        weights = [max(1, int(m.get("importance") or 3)) for m in old]
        association = rng.choices(old, weights=weights, k=1)[0]
        line = _memory_line(association, "突然想起")
        if estimate_tokens(line) <= budget:
            association_line = line
            used.append(int(association["id"]))
        else:
            association = None

    return {
        "pinned": pinned_lines,
        "matched": matched_lines,
        "association": association_line,
        "association_id": int(association["id"]) if association else None,
        "used_ids": used,
    }


def compose(store, user_id: str, persona: str, question: str,
            now: int = None, rng=None) -> dict:
    """按第五节排版拼好：人格 → 事实 → 情景 → 置顶与联想。"""
    now = int(now or time.time())
    query_grams = bigrams(question)

    fact_lines, fact_ids = select_facts(store, user_id, query_grams, now)
    episode = select_episode(store, user_id)
    recall = select_recall(store, user_id, query_grams, now, rng=rng)

    parts = [
        (persona or "").strip(),
        f"【现在】{time.strftime('%Y-%m-%d %H:%M', time.localtime(now))}。"
        "你在手机网页里回消息，默认 1 到 3 句，别写长文。",
    ]
    if fact_lines:
        parts.append("【你记得的状态】\n" + "\n".join(fact_lines))

    system = "\n\n".join(p for p in parts if p)

    recall_parts = []
    if recall["pinned"]:
        recall_parts.append("【置顶的记忆】\n" + "\n".join(recall["pinned"]))
    if recall["matched"]:
        recall_parts.append("【这次想起来的】\n" + "\n".join(recall["matched"]))
    if recall["association"]:
        recall_parts.append("【突然想起】\n" + recall["association"])

    return {
        "system": system,
        "episode": episode,
        "recall": "\n\n".join(recall_parts) if recall_parts else None,
        "used_fact_ids": fact_ids,
        "used_memory_ids": recall["used_ids"],
        "association_id": recall["association_id"],
    }


def mark_used(store, fact_ids, memory_ids, now: int = None) -> None:
    """进过上下文的算「被用到」，时效从这一刻重新起算。"""
    now = int(now or time.time())
    for fact_id in fact_ids or []:
        store.touch_fact(int(fact_id), now)
    for memory_id in memory_ids or []:
        store.touch_memory(int(memory_id), now)


# 最近一次主动联想，供「别再提这个」用。单进程本地服务，放内存足够。
_LAST_ASSOCIATION = {}


def note_association(user_id: str, memory_id) -> None:
    if memory_id:
        _LAST_ASSOCIATION[user_id] = int(memory_id)
    else:
        _LAST_ASSOCIATION.pop(user_id, None)


def last_association(user_id: str):
    return _LAST_ASSOCIATION.get(user_id)


# ---------- /想起 用的人话列表 ----------


def format_all(store, user_id: str) -> str:
    facts = store.all_facts(user_id)
    memories = store.memories(user_id, include_suppressed=True, include_merged=True)
    if not facts and not memories:
        return "记忆还是空的。聊几轮，我慢慢攒。"

    lines = ["我记得这些："]
    if facts:
        lines.append("")
        lines.append("【状态】")
        for fact in facts[:20]:
            state = ""
            if fact["suppressed"]:
                state = "（已封口）"
            elif fact["superseded_by"]:
                state = f"（已被 F{fact['superseded_by']} 取代）"
            lines.append(
                f"F{fact['id']} {fact['subject']}：{_clip(fact['content'], 60)}{state}"
            )
    if memories:
        lines.append("")
        lines.append("【记忆】")
        for memory in memories[-30:]:
            marks = []
            if memory.get("merged_into"):
                marks.append(f"已并入 M{memory['merged_into']}")
            if memory["pinned"]:
                marks.append("置顶")
            if memory["suppressed"] and not memory.get("merged_into"):
                marks.append("不主动提")
            mark = f"（{'，'.join(marks)}）" if marks else ""
            lines.append(
                f"M{memory['id']} {_date(memory['as_of'])} {_clip(memory['title'], 24)}："
                f"{_clip(memory['content'], 60)}{mark}"
            )
    lines.append("")
    last = store.last_dream(user_id)
    if last:
        when = time.strftime("%m-%d", time.localtime(int(last["ran_at"])))
        lines.append(f"上次整理：{when}。")
    lines.append("「/重要 编号」标成置顶，「/别再提 编号」封口。")
    lines.append("「/翻 关键词」翻旧聊天，「/做梦」现在整理一次。")
    return "\n".join(lines)
