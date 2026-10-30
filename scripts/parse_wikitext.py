#!/usr/bin/env python3
"""wikitext 解析器：从任务页 wikitext 提取台词/旁白/选项，筛胡桃发言并带上下文。"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("HUTAO_DATA_DIR", str(ROOT / "data" / "wikitext")))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"}
API = "https://wiki.biligame.com/ys/api.php?"
TARGET = "胡桃"

LINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:\|([^\]]*))?\]\]")
TAG_RE = re.compile(r"<[^>]+>")
TPL_ICON_RE = re.compile(r"\{\{图标\|[^}]*\}\}")
SPEAKER_RE = re.compile(r"^([^：]{1,20})：")


def strip_markup(s):
    s = LINK_RE.sub(lambda m: (m.group(2) or m.group(1)).strip(), s)
    s = TPL_ICON_RE.sub("", s)
    s = TAG_RE.sub("", s)
    s = s.replace("'''", "").replace("''", "")
    return " ".join(s.split())


def parse_speaker(line):
    """返回 (speaker, text) 或 None。按全角冒号最后一个作分隔，带合法性校验。"""
    if "：" not in line:
        return None
    head, text = line.rsplit("：", 1)
    head = head.strip()
    ok = head and len(head) <= 20 and not re.search(r"[。，！？；、]", head)
    if not ok:
        # 回退：第一个冒号
        i = line.find("：")
        head, text = line[:i].strip(), line[i + 1:].strip()
        ok = head and len(head) <= 20 and not re.search(r"[。，！？；、]", head)
    if not ok or not text.strip():
        return None
    return head, text.strip()


def split_sections(wt):
    """按 ==/=== 标题切段，返回 [(heading, body)]。"""
    parts = re.split(r"^(={2,4})\s*(.*?)\s*\1\s*$", wt, flags=re.M)
    secs = []
    cur_head, buf = "", []
    i = 1
    while i < len(parts):
        level, title, body = parts[i], parts[i + 1], parts[i + 2]
        secs.append((cur_head, "\n".join(buf)))
        cur_head = title.strip()
        buf = [body]
        i += 3
    secs.append((cur_head, "\n".join(buf)))
    return secs


def parse_body(body, in_story):
    """返回 item 列表：dialogue / narration / option。"""
    items = []
    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        # 多行模板：{{剧情选项 ... }}
        if stripped.startswith("{{剧情选项"):
            block = [stripped]
            while "}}" not in block[-1] and i + 1 < len(lines):
                i += 1
                block.append(lines[i].strip())
            text = "\n".join(block)
            body_inner = text[len("{{剧情选项"):].split("}}")[0]
            params = re.split(r"\n?\s*\|", body_inner)
            opts, branches = [], []
            for p in params:
                if not p.strip():
                    continue
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                k, v = k.strip(), v.strip()
                if k.startswith("选项"):
                    opts.append(strip_markup(v))
                elif k.startswith("剧情"):
                    for bl in re.split(r"<br\s*/?>", v):
                        bl = strip_markup(bl)
                        if not bl:
                            continue
                        sp = parse_speaker(bl)
                        if sp:
                            branches.append(("dialogue", sp[0], sp[1]))
                        else:
                            branches.append(("narration", "", bl))
            if opts:
                items.append(("option", "", "；".join(opts)))
            for b in branches:
                items.append(b)
            i += 1
            continue

        # 多行模板：{{任务描述|文本|地点}}
        if stripped.startswith("{{任务描述"):
            block = [stripped]
            while "}}" not in block[-1] and i + 1 < len(lines):
                i += 1
                block.append(lines[i].strip())
            text = "\n".join(block)
            inner = text[len("{{任务描述"):].split("}}")[0]
            parts = [p.strip() for p in inner.split("|")]
            desc = strip_markup(parts[0].lstrip("|"))
            place = strip_markup(parts[1]) if len(parts) > 1 else ""
            if desc:
                items.append(("narration", "", desc + (f"（{place}）" if place else "")))
            i += 1
            continue

        # 列表行
        if stripped.startswith("*"):
            depth = len(stripped) - len(stripped.lstrip("*"))
            content = strip_markup(stripped[depth:].strip())
            if content:
                sp = parse_speaker(content)
                if sp:
                    items.append(("dialogue", sp[0], sp[1]))
                elif in_story:
                    items.append(("narration", "", content))
                else:
                    items.append(("outside", "", content))
            i += 1
            continue

        # 普通段落
        if stripped and not stripped.startswith(("{", "|", "}", "[[File:", "[[文件:", "[[分类:")):
            content = strip_markup(stripped)
            if content:
                sp = parse_speaker(content)
                if sp:
                    items.append(("dialogue", sp[0], sp[1]))
                elif in_story:
                    items.append(("narration", "", content))
                else:
                    items.append(("outside", "", content))
        i += 1
    return items


def extract(wt, page):
    sections = split_sections(wt)
    rows = []
    in_story = False
    all_items = []  # (section, kind, speaker, text)
    for head, body in sections:
        if head == "任务剧情":
            in_story = True
        for kind, spk, txt in parse_body(body, in_story):
            if kind == "outside":
                continue
            all_items.append((head, kind, spk, txt))

    for i, (head, kind, spk, txt) in enumerate(all_items):
        if kind != "dialogue" or spk != TARGET:
            continue
        before = all_items[max(0, i - 3):i]
        after = all_items[i + 1:i + 2]

        def fmt(x):
            h, k, s, t = x
            if k == "option":
                return "（选项）" + t
            if k == "narration":
                return t
            return f"{s}：{t}"

        rows.append({
            "source": "wiki_api",
            "page": page,
            "scene": (head + " · " + page) if head else page,
            "line": txt,
            "ctx_before": "\n".join(fmt(x) for x in before),
            "ctx_after": "\n".join(fmt(x) for x in after),
            "act_or_section": head,
        })
    return rows


def test_mode():
    for name in ["_sample1.wikitext", "_sample2.wikitext"]:
        wt = (OUT / name).read_text(encoding="utf-8")
        rows = extract(wt, name)
        print(f"{name}: 胡桃 {len(rows)} 句")
        for r in rows[:6]:
            print(f"  [{r['act_or_section']}] {r['line'][:60]}")
            print(f"     before={r['ctx_before'][:80]!r} after={r['ctx_after'][:60]!r}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode()
    else:
        print("use --test")
