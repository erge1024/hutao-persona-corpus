#!/usr/bin/env python3
"""回填传说任务 211 条的 ctx_after（从 subpages HTML 重算），不改动 ctx_before。"""
import json
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_context import walk, classify  # noqa: E402

WS = Path(__file__).resolve().parent.parent
SUB = Path(os.environ.get("HUTAO_SUBPAGES_DIR", str(WS / "data" / "subpages")))
SAMPLES = WS / "corpus" / "hutao_context_samples.jsonl"
CORPUS = WS / "corpus" / "hutao_corpus_441.jsonl"
ACTS = [("HuTao_act1_神秘的往生堂.html", "第一幕"), ("HuTao_act2_寻找不存在的人.html", "第二幕"),
        ("HuTao_act3_友情，愿望与遗憾.html", "第三幕"), ("HuTao_act4_为了圆满的送别.html", "第四幕")]
ACTS = [(a, b) for a, b in ACTS if (SUB / a).exists()]


def fmt(kind, val):
    if kind == "dialogue":
        sp, line = val
        return f"{sp}：{line}"
    if kind == "option_group":
        return "（选项）" + val
    return val  # narration


def build_items(html_path):
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    main = soup.select_one(".mw-parser-output")
    items = []
    walk(main, items)
    return items


def main():
    # 真·文件名（第二幕页面名与最初抓取时不一致，按实际文件匹配）
    files = sorted(SUB.glob("HuTao_act*.html"))
    acts = []
    for i, f in enumerate(files, 1):
        name = re.search(r"HuTao_act(\d+)", f.name)
        acts.append((f, f"第{'一二三四五六七八九'[int(name.group(1))-1]}幕"))
    print("acts:", [(f.name, a) for f, a in acts])

    after_by_act = {}
    for f, act in acts:
        items = build_items(f)
        rows = []
        for i, (kind, val) in enumerate(items):
            if kind != "dialogue" or val[0] != "胡桃":
                continue
            parts = []
            for j in range(i + 1, min(i + 3, len(items))):
                k, v = items[j]
                if k == "dialogue" and v[0] == "胡桃":
                    break  # 不把胡桃自己的下一句当上下文
                parts.append(fmt(k, v))
                if k == "dialogue":
                    break
            rows.append("\n".join(parts))
        after_by_act[act] = rows
        print(f"{f.name} -> {len(rows)} 句胡桃")

    # 按幕内顺序回填
    corpus = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    idx = {a: 0 for a in after_by_act}
    filled = changed = 0
    for r in corpus:
        if r["source"] != "传说":
            continue
        act = r.get("scene", "")
        i = idx.get(act, 0)
        rows = after_by_act.get(act, [])
        if i < len(rows):
            r["ctx_after"] = rows[i]
            idx[act] = i + 1
            filled += 1
            if rows[i]:
                changed += 1
    print("filled:", filled, "| ctx_after 非空:", changed)

    CORPUS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in corpus) + "\n", encoding="utf-8")
    print("corpus updated ->", CORPUS)

    # 统计
    SPK = re.compile(r"^([^\n：]{1,15})：", re.M)
    from collections import Counter
    by = Counter()
    both = Counter()
    for r in corpus:
        s = r["source"]
        by[s] += 1
        if SPK.search(r["ctx_before"] or "") and SPK.search(r["ctx_after"] or ""):
            both[s] += 1
    print("total:", len(corpus))
    for s in ["传说", "语音", "活动", "wiki_api"]:
        print(f"  {s}: {by[s]} | 两侧都有={both[s]}")
    print("两侧都有合计:", sum(both.values()))


if __name__ == "__main__":
    main()
