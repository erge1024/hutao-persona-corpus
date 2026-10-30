#!/usr/bin/env python3
"""二次提取：从本地 subpages HTML 中还原胡桃台词的情境上下文。

只读本地文件，不下载、不生成任何内容。
"""
import json
import os
import re
import random
from pathlib import Path
from bs4 import BeautifulSoup

_ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("HUTAO_SUBPAGES_DIR", str(_ROOT / "data" / "subpages")))   # 需自备：4 个分幕 HTML
OUT = _ROOT / "corpus" / "hutao_context_samples.jsonl"

ACTS = [
    ("HuTao_act1_神秘的往生堂.html", "第一幕"),
    ("HuTao_act2_寻找不存在之人.html", "第二幕"),
    ("HuTao_act3_友情，愿望与遗憾.html", "第三幕"),
    ("HuTao_act4_为了圆满的送别.html", "第四幕"),
]

SPEAKER_RE = re.compile(r"^([\u4e00-\u9fff·「」（）A-Za-z0-9]{1,15})[：:]")


def classify(text):
    m = SPEAKER_RE.match(text)
    if m:
        return ("dialogue", (m.group(1), text[m.end():].strip()))
    return ("narration", text)


def split_lines(el):
    """把带 <br> 的元素拆成多行文本。"""
    lines = []
    buf = []
    for child in el.children:
        if getattr(child, "name", None) is None:
            buf.append(str(child))
        elif child.name == "br":
            lines.append("".join(buf))
            buf = []
        else:
            buf.append(child.get_text(" ", strip=True))
    lines.append("".join(buf))
    return [" ".join(x.split()) for x in lines if " ".join(x.split())]


def push(items, kind, val):
    if items and items[-1] == (kind, val):
        return
    items.append((kind, val))


def emit_text(el, items):
    for line in split_lines(el):
        push(items, *classify(line))


def expand_plot_frame(el, items):
    """plotFrame = 玩家选项组 + 各分支回应文本（contentA/content 内容常重复，去重）。"""
    opts = [o.get_text(" ", strip=True) for o in el.select(".plotOptions")]
    opts = [o for o in opts if o]
    if opts:
        push(items, "option_group", " / ".join(opts))
    seen = set()
    for c in el.select(".content"):
        text = c.get_text(" ", strip=True)
        if not text or text in seen:
            continue
        seen.add(text)
        emit_text(c, items)


def walk(el, items):
    for child in el.children:
        if getattr(child, "name", None) is None:
            continue
        cls = child.get("class") or []
        name = child.name
        if name in ("h2", "h3", "h4"):
            text = child.get_text(strip=True)
            if text and text != "目录":
                items.append(("heading", text))
            continue
        if name == "li":
            if any(c.startswith("toclevel") for c in cls):
                continue
            emit_text(child, items)
            continue
        if name == "div" and "plotFrame" in cls:
            expand_plot_frame(child, items)
            continue
        if name == "div" and "ys-collapse-content" in cls:
            for sub in child.contents:
                if getattr(sub, "name", None) is None:
                    text = " ".join(str(sub).split())
                    if text:
                        push(items, *classify(text))
                elif sub.name in ("p", "font", "li"):
                    emit_text(sub, items)
                elif sub.name in ("div", "ul"):
                    walk(sub, items)
            continue
        if name in ("p", "font"):
            emit_text(child, items)
            continue
        walk(child, items)


def collect_side(items, idx, step):
    parts = []
    j = idx + step
    chars = 0
    while 0 <= j < len(items) and len(parts) < 4:
        kind, val = items[j]
        if kind in ("dialogue", "heading"):
            break
        text = f"（玩家选项）{val}" if kind == "option_group" else val
        if chars + len(text) > 400:
            break
        parts.append(text)
        chars += len(text)
        j += step
    if step < 0:
        parts.reverse()
    return "\n".join(parts)


def main():
    samples = []
    for fname, act in ACTS:
        soup = BeautifulSoup((BASE / fname).read_text(encoding="utf-8"), "html.parser")
        main_el = soup.select_one(".mw-parser-output")
        items = []
        walk(main_el, items)
        try:
            cut = items.index(("heading", "任务剧情"))
            items = items[cut + 1:]
        except ValueError:
            pass

        dialogue_idx = [i for i, (k, _) in enumerate(items) if k == "dialogue"]
        per_act = 0
        for i, (kind, val) in enumerate(items):
            if kind != "dialogue":
                continue
            per_act += 1
            speaker, line = val
            if speaker != "胡桃":
                continue
            d = dialogue_idx.index(i)
            prev_d = next_d = None
            prev_other = next_other = None
            if d > 0:
                s, l = items[dialogue_idx[d - 1]][1]
                prev_d = f"{s}：{l}"
            if d + 1 < len(dialogue_idx):
                s, l = items[dialogue_idx[d + 1]][1]
                next_d = f"{s}：{l}"
            # 最近的“非胡桃”前/后台词（跨过胡桃连续独白），用于构造对话上下文
            for j in range(d - 1, -1, -1):
                s, l = items[dialogue_idx[j]][1]
                if s != "胡桃":
                    prev_other = f"{s}：{l}"
                    break
            for j in range(d + 1, len(dialogue_idx)):
                s, l = items[dialogue_idx[j]][1]
                if s != "胡桃":
                    next_other = f"{s}：{l}"
                    break
            samples.append({
                "act": act,
                "speaker": speaker,
                "line": line,
                "context_before": collect_side(items, i, -1),
                "context_after": collect_side(items, i, +1),
                "turn_index": per_act,
                "_prev_dialogue": prev_d,
                "_next_dialogue": next_d,
                "_prev_other": prev_other,
                "_next_other": next_other,
            })
        print(f"{act}: {per_act} 句对话, 胡桃 {sum(1 for s in samples if s['act'] == act)} 条")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n = len(samples)
    before = sum(1 for s in samples if s["context_before"])
    after = sum(1 for s in samples if s["context_after"])
    both_empty = sum(1 for s in samples if not s["context_before"] and not s["context_after"])
    ext = sum(1 for s in samples if s["_prev_dialogue"] or s["_next_dialogue"])
    print(f"\nsamples: {n} -> {OUT}")
    print(f"context_before 非空: {before} ({before/n:.1%})")
    print(f"context_after  非空: {after} ({after/n:.1%})")
    print(f"两者都为空: {both_empty} ({both_empty/n:.1%})")
    print(f"至少一侧非空: {n-both_empty} ({(n-both_empty)/n:.1%})")
    print(f"含相邻对话(扩展字段): {ext} ({ext/n:.1%})")


if __name__ == "__main__":
    main()
