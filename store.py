#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite 存档：聊天记录 + 记事本 + 记忆三层里的两层（事实、情景）。

设计见 docs/design.md。这里只管存取，怎么排序、怎么拼上下文在 memory.py。
"""

import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    as_of INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    superseded_by INTEGER,
    importance INTEGER DEFAULT 3,
    last_used_at INTEGER,
    use_count INTEGER DEFAULT 0,
    suppressed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    keywords TEXT,
    as_of INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    source_from INTEGER,
    source_to INTEGER,
    importance INTEGER DEFAULT 3,
    last_used_at INTEGER,
    use_count INTEGER DEFAULT 0,
    pinned INTEGER DEFAULT 0,
    suppressed INTEGER DEFAULT 0,
    merged_into INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);
CREATE INDEX IF NOT EXISTS idx_notes_user ON notes(user_id, id);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(user_id, subject);
CREATE INDEX IF NOT EXISTS idx_facts_current ON facts(user_id, superseded_by);
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, id);
CREATE TABLE IF NOT EXISTS dreams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ran_at INTEGER NOT NULL,
    message_id INTEGER,
    summary TEXT,
    actions TEXT
);
CREATE INDEX IF NOT EXISTS idx_dreams_user ON dreams(user_id, id);
"""


def _row(r):
    return dict(r) if r is not None else None


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            try:
                self._db.execute("ALTER TABLE memories ADD COLUMN merged_into INTEGER")
            except sqlite3.OperationalError:
                pass  # 新库或已经加过
            self._db.commit()

    # ---------- 情景层：messages ----------

    def add_message(self, user_id: str, role: str, content: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages(user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, role, content, int(time.time())),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def history(self, user_id: str, limit: int = 12) -> list:
        """取最近 limit 条，按时间正序返回。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def recent_messages(self, user_id: str, after_id: int = 0, limit: int = 400) -> list:
        """按时间正序返回 id 之后的原始消息，带 id / created_at。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE user_id=? AND id>? ORDER BY id ASC LIMIT ?",
                (user_id, after_id, limit),
            ).fetchall()
        return [_row(r) for r in rows]

    def last_covered_message_id(self, user_id: str) -> int:
        """已被摘要覆盖到哪条消息；没有摘要就是 0。"""
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(source_to) AS m FROM memories WHERE user_id=? AND source_to IS NOT NULL",
                (user_id,),
            ).fetchone()
        return int(row["m"] or 0) if row else 0

    def clear_history(self, user_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            self._db.commit()

    # ---------- 记事本：notes（手工的，和记忆分开） ----------

    def add_note(self, user_id: str, content: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO notes(user_id, content, created_at) VALUES (?,?,?)",
                (user_id, content, int(time.time())),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def list_notes(self, user_id: str, limit: int = 20) -> list:
        with self._lock:
            return self._db.execute(
                "SELECT id, content, created_at FROM notes WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()

    def delete_note(self, user_id: str, note_id: int) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM notes WHERE id=? AND user_id=?", (note_id, user_id)
            )
            self._db.commit()
            return cur.rowcount > 0

    # ---------- 事实层：facts（状态，版本化） ----------

    def add_fact(
        self,
        user_id: str,
        subject: str,
        content: str,
        as_of: int,
        importance: int = 3,
    ) -> int:
        now = int(time.time())
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO facts(user_id, subject, content, as_of, created_at, importance) "
                "VALUES (?,?,?,?,?,?)",
                (user_id, subject, content, int(as_of), now, int(importance)),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def current_facts(self, user_id: str) -> list:
        """当前有效（没被取代、没被压制）的状态。"""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id=? AND superseded_by IS NULL AND suppressed=0 "
                "ORDER BY importance DESC, as_of DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [_row(r) for r in rows]

    def facts_for_subject(self, user_id: str, subject: str) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id=? AND subject=? ORDER BY as_of DESC, id DESC",
                (user_id, subject),
            ).fetchall()
        return [_row(r) for r in rows]

    def superseded_facts(self, user_id: str) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id=? AND superseded_by IS NOT NULL AND suppressed=0 "
                "ORDER BY as_of DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [_row(r) for r in rows]

    def all_facts(self, user_id: str) -> list:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM facts WHERE user_id=? ORDER BY subject ASC, as_of DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [_row(r) for r in rows]

    def supersede_fact(self, fact_id: int, new_id: int) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE facts SET superseded_by=? WHERE id=? AND superseded_by IS NULL",
                (int(new_id), int(fact_id)),
            )
            self._db.commit()

    def touch_fact(self, fact_id: int, now: int = None) -> None:
        now = int(now or time.time())
        with self._lock:
            self._db.execute(
                "UPDATE facts SET last_used_at=?, use_count=use_count+1 WHERE id=?",
                (now, int(fact_id)),
            )
            self._db.commit()

    def fact(self, user_id: str, fact_id: int) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM facts WHERE user_id=? AND id=?", (user_id, int(fact_id))
            ).fetchone()
        return _row(row)

    def update_fact(self, user_id: str, fact_id: int, **fields) -> bool:
        allowed = {"importance", "suppressed", "subject", "content"}
        sets = [(k, int(v) if k in {"importance", "suppressed"} else v)
                for k, v in fields.items() if k in allowed]
        if not sets:
            return False
        clause = ", ".join(f"{k}=?" for k, _ in sets)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE facts SET {clause} WHERE id=? AND user_id=?",
                [v for _, v in sets] + [int(fact_id), user_id],
            )
            self._db.commit()
            return cur.rowcount > 0

    # ---------- 情景层：memories（事件，累积） ----------

    def add_memory(
        self,
        user_id: str,
        title: str,
        content: str,
        keywords: str = "",
        as_of: int = None,
        importance: int = 3,
        pinned: bool = False,
        source_from: int = None,
        source_to: int = None,
    ) -> int:
        now = int(time.time())
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO memories(user_id, title, content, keywords, as_of, created_at, "
                "source_from, source_to, importance, pinned) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    title.strip(),
                    content.strip(),
                    (keywords or "").strip(),
                    int(as_of or now),
                    now,
                    source_from,
                    source_to,
                    int(importance),
                    1 if pinned else 0,
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def memories(self, user_id: str, include_suppressed: bool = False,
                 include_merged: bool = False) -> list:
        sql = "SELECT * FROM memories WHERE user_id=?"
        if not include_suppressed:
            sql += " AND suppressed=0"
        if not include_merged:
            sql += " AND merged_into IS NULL"
        sql += " ORDER BY id ASC"
        with self._lock:
            rows = self._db.execute(sql, (user_id,)).fetchall()
        return [_row(r) for r in rows]

    def memory(self, user_id: str, memory_id: int) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM memories WHERE user_id=? AND id=?", (user_id, int(memory_id))
            ).fetchone()
        return _row(row)

    def touch_memory(self, memory_id: int, now: int = None) -> None:
        now = int(now or time.time())
        with self._lock:
            self._db.execute(
                "UPDATE memories SET last_used_at=?, use_count=use_count+1 WHERE id=?",
                (now, int(memory_id)),
            )
            self._db.commit()

    def update_memory(self, user_id: str, memory_id: int, **fields) -> bool:
        allowed = {"importance", "pinned", "suppressed", "title", "content", "keywords"}
        sets = [(k, int(v) if k in {"importance", "pinned", "suppressed"} else v)
                for k, v in fields.items() if k in allowed]
        if not sets:
            return False
        clause = ", ".join(f"{k}=?" for k, _ in sets)
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET {clause} WHERE id=? AND user_id=?",
                [v for _, v in sets] + [int(memory_id), user_id],
            )
            self._db.commit()
            return cur.rowcount > 0

    def merge_memory(self, user_id: str, keep_id: int, drop_id: int) -> bool:
        """把重复记忆挂到保留的那条下面。不删行。"""
        if int(keep_id) == int(drop_id):
            return False
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET merged_into=? WHERE id=? AND user_id=? AND merged_into IS NULL",
                (int(keep_id), int(drop_id), user_id),
            )
            self._db.commit()
            return cur.rowcount > 0

    def search_messages(self, user_id: str, keyword: str, limit: int = 8) -> list:
        """按字面翻旧聊天。LIKE 的 % 和 _ 都转义。"""
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            rows = self._db.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE user_id=? AND content LIKE ? ESCAPE '\\' ORDER BY id DESC LIMIT ?",
                (user_id, f"%{escaped}%", int(limit)),
            ).fetchall()
        return [_row(r) for r in rows]

    def count_messages_after(self, user_id: str, after_id: int) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE user_id=? AND id>?",
                (user_id, int(after_id)),
            ).fetchone()
        return int(row["n"])

    def last_message_id(self, user_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT MAX(id) AS m FROM messages WHERE user_id=?", (user_id,)
            ).fetchone()
        return int(row["m"] or 0)

    # ---------- 做梦记录 ----------

    def save_dream(self, user_id: str, ran_at: int, message_id: int,
                   summary: str, actions: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO dreams(user_id, ran_at, message_id, summary, actions) VALUES (?,?,?,?,?)",
                (user_id, int(ran_at), int(message_id), summary or "", actions or "[]"),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def last_dream(self, user_id: str) -> dict:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM dreams WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)
            ).fetchone()
        return _row(row)

    def clear_memory(self, user_id: str) -> None:
        """用户主动要求全忘。保留 messages 原文，只清两层记忆。"""
        with self._lock:
            self._db.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
            self._db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            self._db.execute("DELETE FROM dreams WHERE user_id=?", (user_id,))
            self._db.commit()

    def memory_count(self, user_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id=?", (user_id,)
            ).fetchone()
        return int(row["n"])
