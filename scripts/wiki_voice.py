#!/usr/bin/env python3
"""从 biligame wiki 的「胡桃语音」页面提取语音台词 + 触发标题（场景）。"""
import json
import os
from pathlib import Path
from bs4 import BeautifulSoup

DATA = Path(os.environ.get("HUTAO_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))
SRC = DATA / "wiki_voice.html"            # 需自备：胡桃语音页 wikitext/html
OUT = Path(__file__).resolve().parent.parent / "corpus" / "hutao_voice_wiki.jsonl"

soup = BeautifulSoup(SRC.read_text(encoding="utf-8"), "html.parser")

rows = []
for cell in soup.select(".voice_text_chs"):
    row = cell.parent.parent  # 文本单元格 -> 行容器
    kids = [c for c in row.find_all(recursive=False) if c.name == "div"]
    if not kids:
        continue
    title = " ".join(kids[0].get_text(" ", strip=True).split())
    line = " ".join(cell.get_text(" ", strip=True).split())
    if not title or not line:
        continue
    rows.append({"source": "voice", "title": title, "line": line})

seen, out = set(), []
for r in rows:
    k = (r["title"], r["line"])
    if k in seen:
        continue
    seen.add(k)
    out.append(r)

OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
print("voice entries:", len(rows), "-> dedup", len(out))
for r in out[:6]:
    print(f"  [{r['title']}] {r['line'][:60]}")
print("--- 标题列表 ---")
print("; ".join(r["title"] for r in out))
