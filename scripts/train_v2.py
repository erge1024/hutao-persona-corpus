#!/usr/bin/env python3
"""路径 B v2：合并 传说任务/语音/活动 语料并训练 QLoRA（2-3 epoch）。"""
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WS = Path(__file__).resolve().parent.parent
QUEST = WS / "corpus" / "hutao_context_samples.jsonl"
VOICE = WS / "corpus" / "hutao_voice_wiki.jsonl"
EVENTS = WS / "corpus" / "hutao_events_wiki.jsonl"
MODEL_DIR = Path(os.environ["HU_TAO_MODEL"]) if os.environ.get("HU_TAO_MODEL") else (WS / "models" / "Qwen2.5-7B-Instruct-bnb-4bit")
OUT = WS / "out" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是胡桃，往生堂第七十七代堂主。请以胡桃的身份说话，保持她的语气、性格和说话方式。"
MAX_LEN = 512
EPOCHS = int(os.environ.get("EPOCHS", "2"))
GRAD_ACCUM = 8
LR = float(os.environ.get("LR", "1e-4"))
HOLDOUT = 15
STYLE_RATIO_CAP = 0.5
SEED = 20261025


def load_samples():
    samples = []
    # 1) 传说任务（已有上下文）
    for ln in QUEST.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        prev = r.get("context_before", "").strip()
        prev_other = (r.get("_prev_other") or "").strip()
        parts = []
        if prev:
            parts.append("（场景）\n" + prev)
        if prev_other:
            parts.append("（上文）\n" + prev_other)
        samples.append({
            "source": "legend_quest",
            "scene": r.get("act", ""),
            "line": r["line"],
            "context": "\n".join(parts),
            "situational": bool(parts),
        })
    # 2) 语音（标题即触发场景）
    if VOICE.exists():
        for ln in VOICE.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            samples.append({
                "source": "voice",
                "scene": r["title"],
                "line": r["line"],
                "context": "（场景）语音触发：" + r["title"],
                "situational": True,
            })
    # 3) 活动剧情
    if EVENTS.exists():
        for ln in EVENTS.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            parts = []
            if r.get("context_before"):
                parts.append("（场景）\n" + "\n".join(r["context_before"][-3:]))
            samples.append({
                "source": "event",
                "scene": r.get("title") or r.get("page", ""),
                "line": r["line"],
                "context": "\n".join(parts),
                "situational": bool(parts),
            })
    return samples


def build(train_rows):
    rng = random.Random(SEED)
    idx = list(range(len(train_rows)))
    rng.shuffle(idx)
    hold = set(idx[:HOLDOUT])
    train, evals = [], []
    for i, r in enumerate(train_rows):
        (evals if i in hold else train).append(r)
    # 均衡：非情境样本不超过 50%
    sit = [r for r in train if r["situational"]]
    nos = [r for r in train if not r["situational"]]
    cap = int(len(sit) / (1 - STYLE_RATIO_CAP)) - len(sit) if sit else 0
    if len(nos) > cap:
        rng.shuffle(nos)
        nos = nos[:cap]
    return sit + nos, evals


def encode(tok, r):
    user = r["context"] if r["context"] else None
    msgs = [{"role": "system", "content": SYSTEM}]
    if user:
        msgs.append({"role": "user", "content": user})
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = r["line"] + "<|im_end|>\n"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_LEN]
    labels = ([-100] * len(p) + a)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels, "len": len(ids)}


def main():
    t0 = time.time()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), quantization_config=bnb,
                                                device_map={"": 0}, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    samples = load_samples()
    train_rows, eval_rows = build(samples)
    print(f"corpus={len(samples)} train={len(train_rows)} eval={len(eval_rows)}", flush=True)
    by_src = {}
    for r in train_rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("train by source:", by_src, flush=True)
    print("situational ratio:", f"{sum(1 for r in train_rows if r['situational'])/len(train_rows):.1%}", flush=True)

    enc = [encode(tok, r) for r in train_rows]
    enc = [x for x in enc if x["len"] > 8]
    rng = random.Random(SEED)
    rng.shuffle(enc)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    total_steps = max(1, (len(enc) + GRAD_ACCUM - 1) // GRAD_ACCUM) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps, pct_start=0.1)

    model.train()
    log, step = [], 0
    for ep in range(EPOCHS):
        for i, s in enumerate(enc):
            ids = torch.tensor([s["input_ids"]], device="cuda")
            labels = torch.tensor([s["labels"]], device="cuda")
            loss = model(input_ids=ids, labels=labels).loss
            (loss / GRAD_ACCUM).backward()
            if (i + 1) % GRAD_ACCUM == 0 or i == len(enc) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "epoch": ep, "loss": loss.item()})
                if step % 5 == 0 or step <= 2:
                    print(f"ep{ep} step{step}/{total_steps} loss={loss.item():.4f} mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"TRAIN DONE {time.time()-t0:.0f}s peak={peak:.2f}GB", flush=True)
    model.save_pretrained(str(OUT / "lora"))
    tok.save_pretrained(str(OUT / "lora"))
    (OUT / "train_log.json").write_text(json.dumps({"log": log, "peak_gb": peak, "steps": total_steps}, ensure_ascii=False), encoding="utf-8")
    json.dump(eval_rows, (OUT / "eval_rows.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
