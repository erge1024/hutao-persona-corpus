#!/usr/bin/env python3
"""v2 冒烟评估：base vs LoRA 对比 + 胡桃特征命中打分。"""
import json
import os
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ["HU_TAO_MODEL"]) if os.environ.get("HU_TAO_MODEL") else (WS / "models" / "Qwen2.5-7B-Instruct-bnb-4bit")
OUT = WS / "out" / "v2"
SYSTEM = "你是胡桃，往生堂第七十七代堂主。请以胡桃的身份说话，保持她的语气、性格和说话方式。"

FEATURES = {
    "语气词": r"[呀啦喔咯嘛嘿哈哟]",
    "生死/往生": r"(死|葬|往生|送别|魂|鬼|烧|冥|黄泉)",
    "客人": r"客人",
    "堂主/本堂主": r"本堂主|堂主",
}


def score(text):
    return {k: bool(re.search(v, text)) for k, v in FEATURES.items()}


def main():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), quantization_config=bnb,
                                                device_map={"": 0}, torch_dtype=torch.bfloat16)
    tuned = PeftModel.from_pretrained(base, str(OUT / "lora"))
    tuned.eval()

    def gen(model, user, greedy=False):
        msgs = [{"role": "system", "content": SYSTEM}]
        if user:
            msgs.append({"role": "user", "content": user})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=80, do_sample=not greedy,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = json.load((OUT / "eval_rows.json").open(encoding="utf-8"))
    tot = {"base": 0, "tuned": 0}
    for i, r in enumerate(rows):
        user = r["context"] or None
        b = gen(base, user)
        t = gen(tuned, user)
        sb, st = score(b), score(t)
        tot["base"] += sum(sb.values())
        tot["tuned"] += sum(st.values())
        print(f"\n[{i}] {r['source']} · {r['scene']} | USER: {(user or '(无上文)').replace(chr(10),' | ')[:100]}")
        print(f"  REF  : {r['line'][:80]}")
        print(f"  BASE : {b[:100]}  {sb}")
        print(f"  TUNED: {t[:100]}  {st}")
    print(f"\n特征命中合计: base={tot['base']} tuned={tot['tuned']} (每条最多 4 项)")


if __name__ == "__main__":
    main()
