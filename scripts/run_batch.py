#!/usr/bin/env python3
"""任务 C：串行抓取 wikitext（限速/冷却/止损）并解析出胡桃台词。"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from parse_wikitext import extract

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("HUTAO_DATA_DIR", str(ROOT / "data" / "wikitext")))
CORPUS = ROOT / "corpus" / "corpus_v3.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
API = "https://wiki.biligame.com/ys/api.php?"


def fetch_wikitext(page, retries=3):
    p = {"action": "parse", "page": page, "prop": "wikitext", "format": "json", "formatversion": "2"}
    url = API + urllib.parse.urlencode(p)
    for i in range(retries):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read().decode("utf-8", "replace")
            d = json.loads(raw)
            if "error" in d:
                return None, "apierror:" + d["error"].get("code", "?")
            return d.get("parse", {}).get("wikitext", ""), "ok"
        except urllib.error.HTTPError as e:
            if e.code in (567, 514, 429):
                print(f"  BLOCK {e.code} {page}, cool 40s", flush=True)
                time.sleep(40)
                continue
            return None, f"http:{e.code}"
        except Exception as e:
            print(f"  ERR {type(e).__name__} {page}", flush=True)
            time.sleep(40)
    return None, "blocked"


def norm(s):
    return " ".join((s or "").split())


def main():
    targets = [l.split("\t") for l in (OUT / "target_pages.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    known = {(norm(r["line"]), norm(r.get("ctx", "") or r.get("context", ""))) for r in base}
    known_lines = {norm(r["line"]) for r in base}

    rows, stats = [], {"ok": 0, "blocked": 0, "empty": 0, "parsed": 0, "zero": 0}
    consecutive_blocks = 0
    for parts in targets:
        page = parts[0]
        conf = parts[2] if len(parts) > 2 else ""
        safe = re.sub(r'[\\/:*?"<>|（）]', "_", page)
        fp = OUT / (safe + ".wikitext")
        if fp.exists():
            wt = fp.read_text(encoding="utf-8")
            status = "cached"
        else:
            wt, status = fetch_wikitext(page)
            time.sleep(2.2)
            if wt is None:
                stats["blocked"] += 1
                consecutive_blocks += 1
                print(f"SKIP {page} ({status})", flush=True)
                if consecutive_blocks >= 5:
                    print("连续 5 页被拦截，停止", flush=True)
                    break
                continue
            consecutive_blocks = 0
            fp.write_text(wt, encoding="utf-8")
        if not wt.strip():
            stats["empty"] += 1
            print(f"EMPTY {page}", flush=True)
            continue
        if status == "cached":
            consecutive_blocks = 0
        stats["ok"] += 1
        got = extract(wt, page)
        stats["parsed"] += 1
        new = 0
        for r in got:
            key = (norm(r["line"]), norm(r["ctx_before"]))
            if key in known:
                continue
            dup = norm(r["line"]) in known_lines
            r["duplicate_line"] = dup
            rows.append(r)
            new += 1
        stats["zero"] += (1 if not got else 0)
        print(f"{page}: 胡桃 {len(got)} 句, 新增 {new}", flush=True)

    (OUT / "hutao_wiki_expanded.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    before = sum(1 for r in rows if r["ctx_before"])
    after = sum(1 for r in rows if r["ctx_after"])
    dup = sum(1 for r in rows if r["duplicate_line"])
    print(f"\nSTATS ok={stats['ok']} blocked={stats['blocked']} empty={stats['empty']} parsed={stats['parsed']}", flush=True)
    print(f"expanded: {len(rows)} | duplicate_line={dup} | ctx_before非空={before} ({before/max(1,len(rows)):.1%}) | ctx_after非空={after}", flush=True)
    (OUT / "run_stats.json").write_text(json.dumps(stats | {"expanded": len(rows), "dup": dup, "before": before, "after": after}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
