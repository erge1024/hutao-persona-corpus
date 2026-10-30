# 胡桃人格语料（Hu Tao Persona Corpus）

《原神》角色「胡桃」的对话语料与配套处理脚本，服务于**角色人格学习 / 粉丝研究**目的（非官方、非商用）。

## 内容概览

| 目录 | 说明 |
|---|---|
| `corpus/` | 结构化语料（JSONL）。主文件 `hutao_corpus_441.jsonl` 为合并后的 441 条；其余为分片来源 |
| `scripts/` | 语料采集与实验脚本（Wiki 抓取、wikitext 解析、QLoRA 训练/评估、RAG 对照） |
| `docs/` | 报告、目标页清单、训练日志等复现材料 |

## 语料规模（441 条）

| 来源 | 条数 |
|---|---|
| 传说任务 | 211 |
| 角色语音 | 77 |
| 活动剧情 | 153（其中 wiki 抓取新增 136） |
| 魔神任务 / 世界任务 / 邀约任务 | 0 |

- 全部 441 条带场景标签（幕名 / 语音触发标题 / 活动小节）
- 243 条（55.1%）两侧都有上下文（“情境 → 反应”多轮样本）
- 字段：`source` / `scene` / `line` / `ctx_before` / `ctx_after` / `page` / `act_or_section` / `duplicate_line`

详见 `docs/REPORT.md`。

## 数据字段说明

```jsonc
{
  "source": "传说 | 语音 | 活动 | wiki_api",
  "scene": "幕名 / 语音触发标题 / 活动小节",
  "line": "胡桃台词原文",
  "ctx_before": "前 1–3 轮对话或旁白（纯文本）",
  "ctx_after": "后 1 轮对话或旁白",
  "page": "来源页面名（wiki 来源）",
  "act_or_section": "幕名或章节名",
  "duplicate_line": false
}
```

## 目录与路径约定

脚本已改为**相对仓库根目录**运行，不再含任何机器绝对路径。默认布局：

```
hutao-persona-corpus/
├── corpus/     # 语料 jsonl（入库）
├── scripts/    # 处理/训练/评估脚本
├── docs/       # 报告、目标页清单、日志
└── data/       # （不入库）外部输入放这里：wikitext / subpages HTML / 语音页缓存
```

`data/`、`models/`、`out/` 都不进仓库。可用环境变量覆盖这些位置：

| 环境变量 | 含义 | 默认 |
|---|---|---|
| `HU_TAO_MODEL` | Qwen2.5-7B 基座目录（4-bit 预量化或 fp16） | `<repo>/models/Qwen2.5-7B-Instruct-bnb-4bit` |
| `HUTAO_DATA_DIR` | wikitext / 语音页等原始数据目录 | `<repo>/data/wikitext` |
| `HUTAO_SUBPAGES_DIR` | 传说任务分幕 HTML 目录 | `<repo>/data/subpages` |
| `HUTAO_CACHE_DIR` | 活动页抓取缓存 | `<repo>/data/wiki_events` |

## 复现与使用

1. 依赖（版本为此项目实测可用组合，见 `requirements.txt`）：`torch 2.9.1+cu126`、`transformers 4.46.3`、`peft 0.13.2`、`bitsandbytes 0.45.0`。
2. 基座模型需自行下载：`Qwen/Qwen2.5-7B-Instruct`（或 `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` 预量化版），放到 `models/` 或用 `HU_TAO_MODEL` 指定。
3. **端到端复现说明**：本仓库只收录胡桃相关成果（语料 + 脚本 + 报告）。完整重跑数据管线还需要外部输入——`data/` 下的 wikitext / 分幕 HTML / 语音页缓存，以及解包数据源（`AnimeGameData` 的 ExcelBinOutput / TextMap）。这些属游戏文本衍生缓存，未随仓库分发；可按 `scripts/` 中各抓取脚本从对应来源重新生成，或自备。仅使用现成 441 条语料则无需上述资源。
4. 训练结论：在 305 句规模下，小数据 QLoRA 微调产生**负收益**（详见 `docs/REPORT.md` 与 `docs/rag_results.json`）；本项目当前以“检索 + 少样本”为更可靠的路线，语料库本身是核心资产。

## 许可与声明

- 语料与脚本采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可（见 `LICENSE`）。
- 语料文本源自《原神》（© miHoYo / HoYoverse）及其玩家社区 Wiki（Bilibili Wiki，其内容遵循 CC BY-NC-SA）。
- 本项目**仅供学习与粉丝研究，禁止任何商业用途**；使用本仓库内容时请同时遵守源游戏及源站条款，并保留署名。
- 本项目与 miHoYo / HoYoverse 无任何关联，亦非官方内容。