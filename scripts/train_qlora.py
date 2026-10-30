#!/usr/bin/env python3
"""胡桃 QLoRA 冒烟测试：数据构造 + 训练（1-2 epoch，诊断用）。"""
import json
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WS = Path(__file__).resolve().parent.parent
DATA = WS / "corpus" / "hutao_context_samples.jsonl"
MODEL_DIR = Path(os.environ.get("HU_TAO_MODEL", str(WS / "models" / "Qwen2.5-7B-Instruct-bnb-4bit")))
OUT = WS / "out" / "v1"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "你是胡桃，往生堂第七十七代堂主。请以胡桃的身份说话，保持她的语气、性格和说话方式。"
MAX_LEN = 512
EPOCHS = int(os.environ.get("EPOCHS", "1"))
GRAD_ACCUM = 8
LR = 2e-4
HOLDOUT = 15
SEED = 20261025


def build_samples():
    rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    hold = set(idx[:HOLDOUT])
    train, eval_ = [], []
    for i, r in enumerate(rows):
        ctx = r["context_before"].strip()
        prev_other = (r.get("_prev_other") or "").strip()
        parts = []
        kinds = []
        if ctx:
            parts.append("（场景）\n" + ctx)
            kinds.append("situational")
        if prev_other:
            parts.append("（上文）\n" + prev_other)
            kinds.append("dialogue_context")
        user = "\n".join(parts) if parts else None
        kind = "+".join(kinds) if kinds else "style_only"
        item = {
            "messages": ([{"role": "user", "content": user}] if user else []) + [{"role": "assistant", "content": r["line"]}],
            "kind": kind,
            "act": r["act"],
            "line": r["line"],
            "user": user,
        }
        (eval_ if i in hold else train).append(item)
    return train, eval_


def encode(tok, item):
    msgs = [{"role": "system", "content": SYSTEM}] + item["messages"]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    answer = msgs[-1]["content"] + "<|im_end|>\n"
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    ids = (p_ids + a_ids)[:MAX_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_LEN]
    return {"input_ids": ids, "labels": labels, "len": len(ids)}


def main():
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    # 不用 prepare_model_for_kbit_training：它会把 embedding/lm_head 转 fp32（多 4GB），8GB 显存放不下。
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train, eval_ = build_samples()
    print(f"train={len(train)} eval={len(eval_)}", flush=True)
    kinds = {}
    for it in train + eval_:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print("kinds:", kinds, flush=True)

    train = [encode(tok, it) for it in train]
    train = [x for x in train if x["len"] > 8]
    rng = random.Random(SEED)
    rng.shuffle(train)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR,
        betas=(0.9, 0.999), weight_decay=0.0,
    )
    total_steps = max(1, (len(train) + GRAD_ACCUM - 1) // GRAD_ACCUM) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total_steps, pct_start=0.1)

    model.train()
    log = []
    step = 0
    for ep in range(EPOCHS):
        for i, sample in enumerate(train):
            ids = torch.tensor([sample["input_ids"]], device="cuda")
            labels = torch.tensor([sample["labels"]], device="cuda")
            out = model(input_ids=ids, labels=labels)
            loss = out.loss
            (loss / GRAD_ACCUM).backward()
            if (i + 1) % GRAD_ACCUM == 0 or i == len(train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                cur = loss.item()
                log.append({"step": step, "loss": cur, "lr": sched.get_last_lr()[0]})
                if step % 3 == 0 or step <= 3:
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(f"ep{ep} step{step}/{total_steps} loss={cur:.4f} lr={sched.get_last_lr()[0]:.2e} peak_mem={mem:.2f}GB", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"TRAIN DONE {time.time()-t0:.0f}s peak={peak:.2f}GB", flush=True)
    model.save_pretrained(str(OUT / "lora"))
    tok.save_pretrained(str(OUT / "lora"))
    (OUT / "train_log.json").write_text(json.dumps({"log": log, "peak_gb": peak, "steps": total_steps}, ensure_ascii=False), encoding="utf-8")
    json.dump(eval_, (OUT / "eval_items.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
