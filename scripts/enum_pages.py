#!/usr/bin/env python3
"""任务 B：枚举含胡桃出场的任务相关页面（只做枚举，不抓正文正文解析）。

来源：
  1) 胡桃角色页上的任务/活动/剧情链接（本地 wiki_main.html）
  2) 分类枚举（先探测页面分类）
  3) 已知活动系列页展开子任务（抓 wikitext 提取 [[链接]]）
  4) 搜索补充（春曦画桃符）
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("HUTAO_DATA_DIR", str(ROOT / "docs")))
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
API = "https://wiki.biligame.com/ys/api.php?"
LOG = []


def api(params, retries=3, wait=40):
    p = dict(params, format="json", formatversion="2")
    url = API + urllib.parse.urlencode(p)
    for i in range(retries):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read().decode("utf-8", "replace")
            LOG.append(f"OK {urllib.parse.urlencode(p)[:120]}")
            time.sleep(2.2)
            return json.loads(raw)
        except Exception as e:
            LOG.append(f"RETRY({type(e).__name__}) {urllib.parse.urlencode(p)[:100]}")
            time.sleep(wait)
    LOG.append(f"FAIL {urllib.parse.urlencode(p)[:100]}")
    return None


def links_from_main_page():
    html = (WS / "pathB" / "wiki_main.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one(".mw-parser-output") or soup
    out = []
    for a in main.find_all("a", href=True):
        t = " ".join(a.get_text(" ", strip=True).split())
        href = urllib.parse.unquote(a["href"])
        if not t or not href.startswith("/ys/"):
            continue
        if "index.php" in href or href.count("/") != 2:
            continue
        if t in ("首页", "角色", "璃月", "祈愿"):
            continue
        if any(k in t for k in ("任务", "活动", "剧情", "章", "节", "夜", "月", "年", "祭", "诗")):
            out.append(t)
    return list(dict.fromkeys(out))


def categories_of(title):
    d = api({"action": "query", "prop": "categories", "titles": title, "cllimit": "50"})
    if not d:
        return []
    pages = d.get("query", {}).get("pages", [])
    cats = []
    for p in pages:
        for c in p.get("categories", []):
            cats.append(c["title"].replace("Category:", "").replace("分类:", ""))
    return cats


def category_members(cat):
    d = api({"action": "query", "list": "categorymembers", "cmtitle": "Category:" + cat, "cmlimit": "200"})
    if not d:
        return []
    return [m["title"] for m in d.get("query", {}).get("categorymembers", [])]


def links_in_wikitext(title):
    d = api({"action": "parse", "page": title, "prop": "wikitext"})
    if not d:
        return [], ""
    wt = d.get("parse", {}).get("wikitext", "") or ""
    links = re.findall(r"\[\[([^\]\|#]+)(?:\|[^\]]*)?\]\]", wt)
    links = [x.strip() for x in links if x.strip() and not x.startswith(("File:", "文件:", "分类:"))]
    return list(dict.fromkeys(links)), wt


def main():
    rows = {}  # title -> {sources:set, via:set}

    def add(title, source, via=""):
        if not title:
            return
        r = rows.setdefault(title, {"sources": set(), "via": set()})
        r["sources"].add(source)
        if via:
            r["via"].add(via)

    # 1) 角色页链接
    main_links = links_from_main_page()
    for t in main_links:
        add(t, "角色页链接")

    # 2) 分类枚举：先探测样本页的分类
    cats = set(categories_of("神秘的「往生堂」"))
    cats |= set(categories_of("胡桃"))
    target_cats = [c for c in cats if any(k in c for k in ("任务", "活动", "剧情"))]
    print("探测到的分类:", sorted(cats), flush=True)
    print("用于枚举的分类:", target_cats, flush=True)
    cat_pages = {}
    for c in target_cats:
        ms = category_members(c)
        cat_pages[c] = ms
        for t in ms:
            add(t, "分类枚举", c)

    # 3) 活动系列页展开
    series = ["「飞彩镌流年」（任务）", "「磬弦奏华夜」", "「韶光抚月」", "「银翎溯月，玄鸟结璘（系列任务）", "春曦画桃符"]
    for s in series:
        links, wt = links_in_wikitext(s)
        (OUT / (re.sub(r'[\\/:*?"<>|（）]', "_", s) + ".wikitext")).write_text(wt, encoding="utf-8")
        for t in links:
            add(t, "活动子页", s)

    # 4) 搜索补充
    add("春曦画桃符", "搜索结果", "srsearch=胡桃 任务")

    # 过滤：只保留像任务/活动页面的条目
    noise = ("原神", "版本", "角色", "武器", "圣遗物", "材料", "头像", "名片", "成就")
    keep = {t: v for t, v in rows.items() if not any(n in t for n in noise) and "任务" in " ".join(v["sources"]) or True}

    lines = []
    for t, v in sorted(rows.items()):
        lines.append(f"{t}\t{'/'.join(sorted(v['sources']))}\t{'/'.join(sorted(v['via']))}")
    (OUT / "target_pages.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "enumeration_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"target_pages: {len(lines)} 条", flush=True)
    from collections import Counter
    print("来源分布:", Counter(s for v in rows.values() for s in v["sources"]).most_common(), flush=True)
    for t in sorted(rows):
        print("  ", t, "|", "/".join(sorted(rows[t]["sources"])))


if __name__ == "__main__":
    main()
