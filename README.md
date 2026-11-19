# Layered Memory · 给聊天助手的三层记忆

一个**不依赖向量库**的本地记忆方案：人格 + 状态 + 事件，存在一个 SQLite 文件里，按「关联 × 时效 × 重要度」检索，每周离线整理一次。

从一个个人聊天助手项目里抽出来的，去掉了业务代码和人格数据，只留记忆层。适合想给自己的聊天助手加长期记忆、又不想上向量数据库的人。

## 它做什么

- **人格层**：`persona` 全文每轮进上下文，不检索、不衰减。
- **状态层**：住处、工作、当前项目这类「同一时刻只有一个值」的事，版本化保存。同一个键允许多行，旧值不删，只标「被取代」。
- **情景层**：聊天原文全留；旧的对话压成 3–5 句摘要，`source_from` / `source_to` 指回原文。
- **检索**：中文 bigram 字面匹配 × 时间半衰 × 重要度，乘法不是加法。
- **排版**：人格 → 状态 → 最近对话 → 置顶/联想，限 token 不限条数。
- **做梦**：每周拿原文核对记忆，合并重复、按原文改错；只改不删。
- **全部本地**：一个 `memory.db`，没有外部服务。

## 快速开始

不需要 API key，也不需要联网：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python demo.py                      # 离线跑一遍完整流程
python -m unittest discover -s tests -v
```

`demo.py` 会生成一个 `demo.db`，看完直接删掉即可。

## 文件

| 文件 | 管什么 |
|---|---|
| `store.py` | SQLite 存取：messages / facts / memories / dreams |
| `memory.py` | bigram 关联度、时效、权重、token 预算、拼上下文 |
| `extract.py` | 每轮抽状态与事件、情绪加权、摘要压缩 |
| `dream.py` | 每周离线核对：合并重复、按原文改错 |
| `llm.py` | 一个最小的 OpenAI 兼容接口封装 |
| `demo.py` | 离线演示 |
| `docs/design.md` | 设计稿，含取舍和已知缺陷 |

## 接自己的模型

`llm.py` 只做一件事：把 `messages` 发到 OpenAI 兼容的 `/chat/completions`。默认读这几个环境变量：

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat
```

换成别的模型，改 `llm.py` 里的请求地址和字段即可。**不要把 key 写进仓库**；`env.example` 只是模板。

抽取和做梦的提示词是中文的，换模型后建议先跑 `demo.py` 看输出，再调 `extract.py` / `dream.py` 里的规则。

## 设计

完整设计、参数和取舍见 [docs/design.md](docs/design.md)。核心是一句话：

> 时间会让记忆变淡，但「被用到」会让它重新变亮；重要的东西衰减得慢；无关的东西不该因为「老」而被想起来。

## 已知限制

- 关联度是字面匹配，同义不同词会漏。这是第一步的已知缺陷，不是 bug。
- 做梦由模型判断显著性，可能选错保留哪条，也可能把对的改错；没有回滚，需要人工复核。
- 单用户 SQLite，定位是个人助手，不是服务端多租户方案。
- 只在 DeepSeek 上跑过；其他模型的表现没验过。

## License

MIT。见 [LICENSE](LICENSE)。如果要把版权行改成你的名字，直接改第一行。

---

## English

Layered Memory is a small, self-contained memory layer for chat assistants. It keeps three kinds of state in one SQLite file:

Persona — static text, loaded in full on every turn.
Facts — single-valued state such as home city, job, or current project. Values are versioned: a new value supersedes the old one, but nothing is deleted, so the model can still say "that was before".
Episodes — the raw transcript is kept forever; older stretches are compressed into 3–5 sentence summaries that point back to the original message range.
Retrieval scores each memory with relevance × recency × importance (multiplication, not addition), so an irrelevant item cannot float up just because it is old or important. Relevance uses character bigrams, which works for Chinese without a tokenizer; recency decays from the last time the memory was used, not from when it was created. Context is assembled in a fixed order — persona, facts, recent conversation, then pinned or associated memories right before the user's message — under token budgets rather than item counts.

A weekly offline pass ("dreaming") re-reads the transcript, merges duplicate memories, and corrects notes that contradict the source. It only edits or soft-merges; it never deletes the transcript.

Quick start (offline, no API key)
bash


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
Using your own model
llm.py posts to any OpenAI-compatible /chat/completions endpoint. Set DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, and DEEPSEEK_MODEL, or replace the chat() function. The extraction and dreaming prompts are in Chinese; run demo.py first, then adjust the rules in extract.py / dream.py.

Limitations
Relevance is literal bigram matching, so synonyms are missed. The dreaming pass can pick the wrong duplicate to keep and has no rollback. Storage is single-user SQLite, not a multi-tenant service. Design notes (Chinese) are in docs/design.md.

License
MIT.
