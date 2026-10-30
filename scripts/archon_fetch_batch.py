#!/usr/bin/env python3
"""魔神任务：用单次多 titles 的方式批量取 wikitext（避免逐页请求触发 567），再解析。"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from parse_wikitext import extract

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("HUTAO_DATA_DIR", str(ROOT / "data" / "wikitext")))
ARCH = OUT / "archon"
ARCH.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
API = "https://wiki.biligame.com/ys/api.php?"
BATCH = 20


def api(p, retries=4, wait=60):
    p = dict(p, format="json", formatversion="2")
    url = API + urllib.parse.urlencode(p)
    for i in range(retries):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
            return json.loads(raw), "ok"
        except urllib.error.HTTPError as e:
            if e.code in (567, 514, 429):
                print(f"  BLOCK {e.code}, cool {wait}s", flush=True)
                time.sleep(wait)
            else:
                return None, f"http:{e.code}"
        except Exception as e:
            print(f"  ERR {type(e).__name__}", flush=True)
            time.sleep(wait)
    return None, "blocked"


def norm(s):
    return " ".join((s or "").split())


def fetch_batch(titles):
    """返回 {title: wikitext}；用 rvslots=main。"""
    p = {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main",
         "titles": "|".join(titles), "redirects": "1"}
    d, st = api(p)
    out = {}
    if not d:
        return out, st
    pages = d.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    for pg in pages:
        t = pg.get("title")
        if pg.get("missing"):
            out[t] = None
            continue
        revs = pg.get("revisions") or []
        if revs:
            slot = revs[0].get("slots", {}).get("main", {}) or {}
            content = slot.get("content") or slot.get("*") or ""
            out[t] = content
    return out, st


def main():
    pages = [l.strip() for l in (OUT / "archon_pages.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    print("archon pages:", len(pages), flush=True)
    got = {}
    ok_req = 0
    for i in range(0, len(pages), BATCH):
        batch = pages[i:i + BATCH]
        res, st = fetch_batch(batch)
        time.sleep(8)
        if st != "ok":
            print(f"batch {i} failed: {st}", flush=True)
            continue
        ok_req += 1
        for t, wt in res.items():
            if wt:
                got[t] = wt
        print(f"batch {i//BATCH+1}/{(len(pages)+BATCH-1)//BATCH}: {len(res)} pages, {sum(len(v or '') for v in res.values())} chars", flush=True)

    # 落盘
    for t, wt in got.items():
        safe = re.sub(r'[\\/:*?"<>|（）]', "_", t)
        (ARCH / (safe + ".wikitext")).write_text(wt, encoding="utf-8")
    print("saved:", len(got), flush=True)

    # 解析 + 去重
    base = [json.loads(l) for l in (WS / "hutao_pipeline" / "corpus_v3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    prev = [json.loads(l) for l in (OUT / "hutao_wiki_expanded.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    known = {(norm(r["line"]), norm(r.get("ctx", "") or "")) for r in base}
    known |= {(norm(r["line"]), norm(r.get("ctx_before", ""))) for r in prev}
    known_lines = {norm(r["line"]) for r in base} | {norm(r["line"]) for r in prev}

    rows, hits = [], 0
    for page, wt in got.items():
        got_rows = extract(wt, page)
        hits += len(got_rows)
        new = 0
        for r in got_rows:
            key = (norm(r["line"]), norm(r["ctx_before"]))
            if key in known:
                continue
            r["duplicate_line"] = norm(r["line"]) in known_lines
            rows.append(r)
            new += 1
        if got_rows:
            print(f"{page}: 胡桃 {len(got_rows)} 句, 新增 {new}", flush=True)

    (OUT / "hutao_wiki_expanded_archon.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    before = sum(1 for r in rows if r["ctx_before"])
    print(f"\nSTATS pages={len(pages)} fetched={len(got)} req_ok={ok_req} 胡桃句={hits} 新增={len(rows)} "
          f"dup={sum(1 for r in rows if r['duplicate_line'])} ctx_before={before}", flush=True)
    (OUT / "archon_run_stats.json").write_text(json.dumps({"pages": len(pages), "fetched": len(got),
        "req_ok": ok_req, "hutao": hits, "new": len(rows),
        "dup": sum(1 for r in rows if r["duplicate_line"]), "ctx_before": before}, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
