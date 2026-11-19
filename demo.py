#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线 demo：不联网、不花钱，看三层记忆怎么存、怎么拼、怎么整理。

跑法：
    python demo.py

会生成一个 demo.db（可随时删）。想接真模型，把下面 fake_chat 换掉即可。
"""

import json
import time
from pathlib import Path

import dream
import extract
import llm
import memory
import store

PERSONA = """你是一个话不多的助手。
先给结论，再补一句人味的话。不卖萌，不刷屏。"""

UID = "demo"

# 这里假装是模型抽取出来的结果，省掉一次 API 调用
FAKE_EXTRACT = {
    "我上个月从北京搬到厦门，住思明区，养了只橘猫叫饭团。": {
        "facts": [
            {"subject": "住处", "content": "住在厦门思明区", "importance": 4, "about": "user"},
            {"subject": "宠物", "content": "养了一只橘猫，叫饭团", "importance": 3, "about": "user"},
        ],
        "events": [
            {"title": "从北京搬到厦门", "content": "用户从北京搬到厦门，住思明区。",
             "keywords": ["搬家", "厦门"], "importance": 4},
        ],
        "emotion": 3,
    },
    "对了，我不吃香菜。": {
        "facts": [{"subject": "饮食", "content": "不吃香菜", "importance": 3, "about": "user"}],
        "events": [],
        "emotion": 1,
    },
}


MERGE = {"keep": 0, "drop": 0}


def fake_chat(messages, **_kwargs):
    system = messages[0]["content"]
    last = messages[-1]["content"]

    if "离线整理记忆" in system:  # 做梦
        return json.dumps({
            "actions": [
                {"type": "merge", "keep": MERGE["keep"], "drop": [MERGE["drop"]], "reason": "同一次搬家"},
            ],
            "summary": "合并了一条重复的搬家记忆",
        }, ensure_ascii=False)

    if "整理长期记忆" in system:  # 每轮抽取
        for text, data in FAKE_EXTRACT.items():
            if text in last:
                return json.dumps(data, ensure_ascii=False)
        return json.dumps({"facts": [], "events": [], "emotion": 1}, ensure_ascii=False)

    return "嗯，记下了。"


def show_messages(ctx, question):
    print("--- 实际发给模型的消息 ---")
    print("[system]")
    print(ctx["system"])
    print()
    if ctx["episode"]:
        print(f"[最近对话] 共 {len(ctx['episode'])} 条（这里省略内容）")
    if ctx["recall"]:
        print("[system]")
        print(ctx["recall"])
        print()
    print("[user]")
    print(question)
    print()


def main():
    db_path = Path("demo.db")
    db_path.unlink(missing_ok=True)
    llm.chat = fake_chat  # 全程离线

    st = store.Store(db_path)
    now = int(time.time())

    print("=" * 60)
    print("1. 聊天，抽取状态和事件")
    print("=" * 60)
    for text in FAKE_EXTRACT:
        data = extract.extract_turn(text, "嗯。")
        stats = extract.apply(st, UID, data, now=now)
        st.add_message(UID, "user", text)
        st.add_message(UID, "assistant", "嗯，记下了。")
        print(f"用户：{text}")
        print(f"  -> 新增状态 {stats['facts']} 条，事件 {stats['events']} 条\n")

    # 人为放两条重复记忆，等会儿让“做梦”合并
    first = st.add_memory(UID, "搬家", "用户从北京搬到厦门", as_of=now - 30 * 86400)
    second = st.add_memory(UID, "搬到厦门", "用户搬到厦门思明区", as_of=now - 29 * 86400)
    MERGE.update(keep=second, drop=first)
    print(f"（另放了 2 条重复记忆 M{first} / M{second}，留到第 3 步合并）\n")

    print("=" * 60)
    print("2. 拼上下文：人格 → 状态 → 最近对话 → 置顶/联想")
    print("=" * 60)
    question = "你还记得我住哪吗？"
    ctx = memory.compose(st, UID, PERSONA, question, now=now)
    show_messages(ctx, question)

    print("=" * 60)
    print("3. 做梦：合并重复、按原文改错")
    print("=" * 60)
    result = dream.run(st, UID, now=now)
    print(dream.format_report(result))
    print()

    print("=" * 60)
    print("4. 翻旧账：按字面搜原文")
    print("=" * 60)
    rows = st.search_messages(UID, "厦门", limit=5)
    for row in rows:
        who = "用户" if row["role"] == "user" else "助手"
        print(f"  #{row['id']} {who}：{row['content']}")
    print()

    print("=" * 60)
    print("5. 最后的状态和记忆")
    print("=" * 60)
    print(memory.format_all(st, UID))
    print("\n完成。demo.db 可以直接删掉。")


if __name__ == "__main__":
    main()
