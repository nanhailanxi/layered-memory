#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不联网的自检：用 unittest 跑，不用装任何东西。

    python -m unittest discover -s tests -v
"""

import json
import os
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dream  # noqa: E402
import extract  # noqa: E402
import memory  # noqa: E402
import store  # noqa: E402


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = store.Store(os.path.join(self.tmp.name, "test.db"))
        self.uid = "owner"
        self.now = int(time.time())
        self.day = 86400

    def tearDown(self):
        self.tmp.cleanup()

    def test_fact_versions(self):
        first = memory.record_fact(self.db, self.uid, "住处", "深圳", 4, now=self.now)
        self.assertEqual(first["action"], "new")

        same = memory.record_fact(self.db, self.uid, "住处", "深圳", 4, now=self.now)
        self.assertEqual(same["action"], "same")

        moved = memory.record_fact(self.db, self.uid, "住处", "北京", 4, now=self.now)
        self.assertEqual(moved["action"], "superseded")
        self.assertEqual([f["content"] for f in self.db.current_facts(self.uid)], ["北京"])
        self.assertEqual(len(self.db.superseded_facts(self.uid)), 1)

    def test_owner_guard(self):
        data = {
            "facts": [
                {"subject": "住处", "content": "搬到上海", "importance": 4, "about": "user"},
                {"subject": "小王住处", "content": "小王搬到上海", "importance": 3, "about": "小王"},
            ],
            "events": [{"title": "小王搬家", "content": "小王搬到上海", "importance": 3}],
            "emotion": 1,
        }
        stats = extract.apply(self.db, self.uid, data, now=self.now)
        self.assertEqual(stats["facts"], 1)
        self.assertEqual([f["subject"] for f in self.db.current_facts(self.uid)], ["住处"])

    def test_compose_order(self):
        memory.record_fact(self.db, self.uid, "住处", "厦门思明区", 4, now=self.now)
        pinned = self.db.add_memory(
            self.uid, "写作口味", "不喜欢 AI 腔，喜欢短句",
            as_of=self.now - 90 * self.day, importance=5, pinned=True,
        )
        matched = self.db.add_memory(
            self.uid, "看海", "说过想去看海",
            as_of=self.now - 60 * self.day, importance=4,
        )
        old = self.db.add_memory(
            self.uid, "修电脑", "帮朋友重装系统",
            as_of=self.now - 40 * self.day, importance=2,
        )
        for i in range(3):
            self.db.add_message(self.uid, "user", f"旧问题{i}")
            self.db.add_message(self.uid, "assistant", f"旧回答{i}")
        self.db.add_message(self.uid, "user", "我想去看海")

        ctx = memory.compose(
            self.db, self.uid, "PERSONA", "我想去看海",
            now=self.now, rng=random.Random(1),
        )
        self.assertTrue(ctx["system"].startswith("PERSONA"))
        self.assertIn("厦门思明区", ctx["system"])
        self.assertEqual(ctx["episode"][-1]["content"], "我想去看海")
        self.assertIn("看海", ctx["recall"])
        self.assertIn("置顶", ctx["recall"])
        self.assertIn("修电脑", ctx["recall"])
        self.assertNotEqual(matched, old)

    def test_merge_and_search(self):
        keep = self.db.add_memory(self.uid, "搬家", "从北京搬到厦门", as_of=self.now - 30 * self.day)
        drop = self.db.add_memory(self.uid, "搬到厦门", "搬到厦门思明区", as_of=self.now - 29 * self.day)
        self.assertTrue(self.db.merge_memory(self.uid, keep, drop))
        self.assertEqual(self.db.memory(self.uid, drop)["merged_into"], keep)
        self.assertNotIn(drop, [m["id"] for m in self.db.memories(self.uid)])

        self.db.add_message(self.uid, "user", "进度 100% 完成")
        self.db.add_message(self.uid, "user", "另一个 100 完成")
        hits = self.db.search_messages(self.uid, "100%")
        self.assertEqual(len(hits), 1)
        self.assertIn("100%", hits[0]["content"])

    def test_dream(self):
        keep = self.db.add_memory(self.uid, "搬家", "从北京搬到厦门", as_of=self.now - 30 * self.day)
        drop = self.db.add_memory(self.uid, "搬到厦门", "搬到厦门思明区", as_of=self.now - 29 * self.day)
        self.db.add_memory(self.uid, "猫", "养了一只橘猫叫饭团", as_of=self.now - 20 * self.day)
        self.db.add_message(self.uid, "user", "我搬到厦门了")
        fact = memory.record_fact(self.db, self.uid, "住处", "厦门", 4, now=self.now)

        actions = {
            "actions": [
                {"type": "merge", "keep": keep, "drop": [drop], "reason": "同一次搬家"},
                {"type": "correct_fact", "id": fact["id"], "content": "住在厦门思明区", "reason": "原文补了区"},
                {"type": "merge", "keep": 999, "drop": [1], "reason": "越界，应被丢掉"},
            ],
            "summary": "合并一条、修正一条",
        }
        dream.llm.chat = lambda messages, **kwargs: json.dumps(actions, ensure_ascii=False)

        result = dream.run(self.db, self.uid, now=self.now)
        self.assertEqual(len(result["report"]), 2)
        self.assertEqual(self.db.memory(self.uid, drop)["merged_into"], keep)
        self.assertEqual(self.db.current_facts(self.uid)[0]["content"], "住在厦门思明区")
        self.assertEqual(len(self.db.superseded_facts(self.uid)), 1)
        self.assertIsNotNone(self.db.last_dream(self.uid))

    def test_episode_budget(self):
        for _ in range(60):
            self.db.add_message(self.uid, "user", "长" * 100)
            self.db.add_message(self.uid, "assistant", "嗯" * 100)
        episode = memory.select_episode(self.db, self.uid)
        tokens = sum(memory.estimate_tokens(m["content"]) for m in episode)
        self.assertLessEqual(tokens, memory.EPISODE_TOKEN_BUDGET)
        self.assertEqual(episode[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
