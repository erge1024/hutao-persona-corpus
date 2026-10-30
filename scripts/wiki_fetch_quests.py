#!/usr/bin/env python3
"""抓取指定的 wiki 任务页，提取胡桃台词 + 上下文（限速 + 重试）。"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

BASE = "https://wiki.biligame.com/ys/"
_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "corpus" / "hutao_events_wiki.jsonl"
CACHE = Path(os.environ.get("HUTAO_CACHE_DIR", str(_ROOT / "data" / "wiki_events")))
CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
SPK = re.compile(r"^([\u4e00-\u9fff·「」（）A-Za-z0-9]{1,15})[：:]")


def fetch(title, retries=5):
    u = BASE + urllib.parse.quote(title)
    for i in range(retries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=45)
            return r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  retry {i} {type(e).__name__} {title}", flush=True)
            time.sleep(45)
    return None


def extract(html, page):
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".mw-parser-output") or soup
    items = []
    for el in main.descendants:
        if el.name in ("h2", "h3", "h4"):
            t = el.get_text(strip=True)
            if t and t != "目录":
                items.append(("h", t))
        elif el.name in ("li", "p"):
            t = " ".join(el.get_text(" ", strip=True).split())
            if t and (el.name == "li" or len(t) > 3):
                items.append(("l", t))
    rows = []
    for i, (k, t) in enumerate(items):
        if k != "l":
            continue
        m = SPK.match(t)
        if not m or m.group(1) != "胡桃":
            continue
        line = t[m.end():].strip()
        if not line:
            continue
        title = next((x[1] for x in reversed(items[:i]) if x[0] == "h"), "")
        rows.append({
            "source": "wiki_event",
            "page": page,
            "title": title,
            "line": line,
            "context_before": [x[1] for x in items[max(0, i - 3):i] if x[0] == "l"],
            "context_after": [x[1] for x in items[i + 1:i + 2] if x[0] == "l"],
        })
    return rows


def main():
    titles = sys.argv[1:]
    all_rows = []
    for t in titles:
        html = fetch(t)
        if not html:
            print("FAIL", t, flush=True)
            continue
        (CACHE / (re.sub(r'[\\/:*?"<>|（）]', "_", t) + ".html")).write_text(html, encoding="utf-8")
        rows = extract(html, t)
        all_rows.extend(rows)
        print(f"{t}: 胡桃 {len(rows)} 句", flush=True)
        time.sleep(30)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n", encoding="utf-8")
    print("TOTAL:", len(all_rows), "->", OUT, flush=True)


if __name__ == "__main__":
    main()
