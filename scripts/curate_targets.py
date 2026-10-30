#!/usr/bin/env python3
"""任务 B 收尾：整理高/中置信目标页面清单。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("HUTAO_DATA_DIR", str(ROOT / "docs")))
RAW = OUT / "target_pages.txt"

raw = [l.split("\t") for l in RAW.read_text(encoding="utf-8").splitlines() if l.strip()]

# 高置信：角色页直接链接的任务/活动页 + 飞彩鎸流年子页 + 搜索命中
high = {
    "引蝶之章", "「磬弦奏华夜」", "「韶光抚月」", "「银翎溯月，玄鸟结璘」", "「银翎溯月，玄鸟结璘（系列任务）」",
    "「飞彩镌流年」", "「游逸旅闻·诗汇之章」", "飞彩镌流年", "磬弦奏华夜", "韶光抚月",
    "吉星似火，福禄成喜", "品星观月，遍踏山河", "飞星聚彩，愿刻华年", "飞花照船，海上捞月", "今夜窗明，花灯藏影",
    "春曦画桃符",
}
# 中置信：活动系列页里展开出的任务页
mid = {
    "一曲瑶琴与谁听", "人前人后", "光焰蹈跃", "千里寻音迹难觅", "幕前二三言", "幕后人", "灯后勤绩", "烟海梭巡",
    "风起鹤归", "一筷食二轮烹三餐尽欢四海扬名五味俱全", "六情通七仙贺八方飘香九霄得闻十分完满",
    "幽客之章", "迫近的客星",
}
# 噪声：其他角色的传说任务/无关页
drop = {"夜兰", "悬壶之章", "「游水酝诗籍」", "游水酝诗籍", "「医心」", "「水的女儿」", "「狮之血」", "「审判之时」"}

rows = []
seen = set()
for parts in raw:
    t = parts[0].strip()
    src = parts[1] if len(parts) > 1 else ""
    via = parts[2] if len(parts) > 2 else ""
    if t in seen:
        continue
    seen.add(t)
    if t in high:
        rows.append((t, src, "高", via))
    elif t in mid:
        rows.append((t, src, "中", via))
    elif t in drop:
        continue
    elif "分类枚举" in src:
        continue
    else:
        rows.append((t, src, "中", via))

lines = [f"{t}\t{src}\t{conf}\t{via}" for t, src, conf, via in sorted(rows)]
(OUT / "target_pages.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
from collections import Counter
print("target_pages:", len(lines))
print("来源:", Counter(r[1] for r in rows).most_common())
print("置信:", Counter(r[2] for r in rows).most_common())
for t, s, c, v in sorted(rows):
    print(f"  [{c}] {t}  ({s}{' / ' + v if v else ''})")
