#!/usr/bin/env python3
"""RAG few-shot 对照：基座 zero-shot / 基座 + 检索台词 few-shot / 现有 LoRA。"""
import json
import os
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WS = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ["HU_TAO_MODEL"]) if os.environ.get("HU_TAO_MODEL") else (WS / "models" / "Qwen2.5-7B-Instruct-bnb-4bit")
OUT = WS / "out" / "v2"
CORPUS = WS / "corpus" / "corpus_v3.jsonl"
SYSTEM = "你是胡桃，往生堂第七十七代堂主。请以胡桃的身份说话，保持她的语气、性格和说话方式。"
FEATURES = {
    "语气词": r"[呀啦喔咯嘛嘿哈哟]",
    "生死/往生": r"(死|葬|往生|送别|魂|鬼|烧|冥|黄泉)",
    "客人": r"客人",
    "堂主/本堂主": r"本堂主|堂主",
}


def score(text):
    return {k: bool(re.search(v, text)) for k, v in FEATURES.items()}


def build_corpus():
    rows = []
    q = WS / "genshin_bg" / "legend_quests" / "hutao_context_samples.jsonl"
    for ln in q.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            parts = [r.get("context_before", "").strip(), (r.get("_prev_other") or "").strip()]
            rows.append({"source": "传说", "scene": r.get("act", ""), "line": r["line"],
                         "ctx": "\n".join(p for p in parts if p)})
    v = WS / "pathB" / "hutao_voice_wiki.jsonl"
    for ln in v.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows.append({"source": "语音", "scene": r["title"], "line": r["line"],
                         "ctx": "语音触发：" + r["title"]})
    e = WS / "pathB" / "hutao_events_wiki.jsonl"
    for ln in e.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows.append({"source": "活动", "scene": r.get("title") or r.get("page", ""), "line": r["line"],
                         "ctx": "\n".join((r.get("context_before") or [])[-3:])})
    return rows


def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def retrieve(query, corpus, k=4, exclude_line=None):
    qg = bigrams(query)
    scored = []
    for c in corpus:
        if exclude_line and c["line"] == exclude_line:
            continue
        scene_hit = len(bigrams(c["scene"]) & qg) if c["scene"] else 0
        ctx_hit = len(bigrams(c["scene"] + c["ctx"]) & qg)
        s = 2.0 * scene_hit + 0.6 * ctx_hit + 0.05 * len(bigrams(c["line"]) & qg)
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:k]]


def main():
    corpus = build_corpus()
    CORPUS.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in corpus) + "\n", encoding="utf-8")
    print(f"检索库: {len(corpus)} 条（传说 {sum(1 for c in corpus if c['source']=='传说')} / 语音 "
          f"{sum(1 for c in corpus if c['source']=='语音')} / 活动 {sum(1 for c in corpus if c['source']=='活动')}）", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), quantization_config=bnb,
                                                device_map={"": 0}, torch_dtype=torch.bfloat16)
    base.eval()

    def gen(model, msgs, max_new=80):
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                                 top_p=0.9, repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = json.load((OUT / "eval_rows.json").open(encoding="utf-8"))
    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        user = r["context"] or None
        # A: base zero-shot
        msgs = [{"role": "system", "content": SYSTEM}]
        if user:
            msgs.append({"role": "user", "content": user})
        a = gen(base, msgs)
        # B: base + few-shot
        shots = retrieve(user or r["scene"], corpus, k=4, exclude_line=r["line"])
        msgs2 = [{"role": "system", "content": SYSTEM}]
        for c in shots:
            msgs2.append({"role": "user", "content": f"（场景）{c['scene']}\n{c['ctx']}".strip()})
            msgs2.append({"role": "assistant", "content": c["line"]})
        msgs2.append({"role": "user", "content": user or "请以胡桃的口吻说一句话。"})
        b = gen(base, msgs2)
        results.append({"idx": i, "source": r["source"], "scene": r["scene"], "ref": r["line"],
                        "user": user, "base": a, "rag": b, "shots": [c["line"] for c in shots]})
        print(f"[{i}] {r['source']} {r['scene'][:20]}... {time.time()-t0:.0f}s", flush=True)

    # C: 现有 LoRA（zero-shot）
    peft_model = PeftModel.from_pretrained(base, str(OUT / "lora"))
    peft_model.eval()
    for i, item in enumerate(results):
        r = rows[item["idx"]]
        user = r["context"] or None
        msgs = [{"role": "system", "content": SYSTEM}]
        if user:
            msgs.append({"role": "user", "content": user})
        item["lora"] = gen(peft_model, msgs)

    base_hits = rag_hits = lora_hits = 0
    print("\n" + "=" * 100)
    for it in results:
        sb, sr, sl = score(it["base"]), score(it["rag"]), score(it["lora"])
        base_hits += sum(sb.values()); rag_hits += sum(sr.values()); lora_hits += sum(sl.values())
        print(f"\n[{it['idx']}] {it['source']}·{it['scene'][:24]} | REF: {it['ref'][:60]}")
        print(f"  BASE : {it['base'][:90]}")
        print(f"  RAG  : {it['rag'][:90]}")
        print(f"  LORA : {it['lora'][:90]}")
        print(f"  命中: base={sum(sb.values())} rag={sum(sr.values())} lora={sum(sl.values())}")
    print(f"\n特征命中合计（{len(results)} 条，每条最多 4 项）: base={base_hits} rag={rag_hits} lora={lora_hits}")
    (OUT / "rag_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
