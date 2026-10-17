"""Jev relevance scoring บนเครื่องตัวเอง — autotrust/JEV-9B (open-weight student ของ TypeSafe Jev 1.13)

รันใน Linux/WSL ที่มี GPU (ทดสอบบน RX 7800 XT 16GB + ROCm) แล้วให้ improvedRAG.py เรียกผ่าน HTTP:
  hf download autotrust/JEV-9B --local-dir ~/models/JEV-9B --include "*.json" "*.jinja" "model-*.safetensors" "head.safetensors" "adapter/*"
  python jev_local_server.py ~/models/JEV-9B      # POST /score {"query": "...", "documents": ["...", ...]}
  RERANKER=~/models/bge-reranker-v2-m3 ...       # + POST /rerank (bge-reranker-v2-m3 fp16 บน GPU ตัวเดียวกัน)

ทำแบบเดียวกับ jev-reranker (hotchpotch) โหมด pointwise: ถาม Jev ทีละเอกสารด้วย instruction + criteria ชุดเดียวกับ
ไลบรารี แล้วใช้ P(true) เป็นคะแนน 0-1 (ฝั่งเรียกใช้กรองที่ threshold 0.2 เหมือนค่า default ของไลบรารี)
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from accelerate.hooks import remove_hook_from_module
from safetensors import safe_open
from safetensors.torch import load_file
import transformers.modeling_utils
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

# คัดลอกจาก jev_reranker/instructions.py (POINTWISE_RELEVANCE_INSTRUCTION)
INSTRUCTION = ("How useful is the document for answering the query? Score this document independently using only the "
               "query and this document. Match the intended subject and requested information, not just words or a "
               "similar name. A direct answer, a relevant partial fact, or a concrete intermediate identification or "
               "relationship can be useful; a passage need not answer the entire question by itself.")
CRITERIA = {
    "true": "Contains specific information that answers or supports answering the query, including partial facts "
            "and necessary intermediate identification or relationships.",
    "false": "Unrelated, merely tangential, about a different referent, or lacking facts that help answer the query.",
}

path = Path(sys.argv[1]).expanduser()
tok = AutoTokenizer.from_pretrained(path)
# bf16 ตามต้นฉบับ (18 GB) บนการ์ด 16 GB: ชั้น transformer ทั้ง 32 ชั้น (~13.8 GB) อยู่บน GPU
# embed_tokens (2 GB) ทำ lookup เองบน CPU, lm_head (2 GB) ไม่ได้ใช้ (System 1 อ่าน hidden state ไปเข้า head เอง)
# ถอด hook ของ accelerate ออก เพราะ kernel ของ flash-linear-attention (Triton) รับ tensor บน CPU ไม่ได้
# (ลอง nf4 แล้ว — bitsandbytes บน ROCm พักน้ำหนักเต็มไว้ใน RAM ระหว่างโหลด จน RAM ไม่พอ)
device_map = {"model.embed_tokens": "cpu", "lm_head": "cpu", "model.norm": 0, "model.rotary_emb": 0,
              **{f"model.layers.{i}": 0 for i in range(32)}}
# transformers จองหน่วยความจำ GPU ทีละก้อนใหญ่ (~13 GB) ก่อนโหลด — บน ROCm/WSL การจองก้อนเดียวขนาดนั้นค้าง
# ปิดไว้ให้จองทีละ tensor แทน (โหลดช้าลงนิดหน่อย)
transformers.modeling_utils.caching_allocator_warmup = lambda *a, **k: None
base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device_map)
backbone = PeftModel.from_pretrained(base, path, subfolder="adapter",
                                     offload_dir="/tmp/jev_offload").get_base_model().model
remove_hook_from_module(backbone, recurse=True)
for m in (backbone.layers, backbone.norm, backbone.rotary_emb):  # รวม LoRA ที่ peft อาจวางไว้บน CPU
    m.cuda()
backbone.eval()
index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(path / index["model.embed_tokens.weight"], "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight")  # LoRA ไม่ได้แตะ embed_tokens
head = load_file(path / "head.safetensors")
W, b = head["proj.weight"].cuda().float(), head["proj.bias"].cuda().float()
cfg = json.loads((path / "judge_config.json").read_text())
T = json.loads((path / "calibration.json").read_text())["per_kind"]["noul"]
s, e = cfg["slots"]["ranges"]["noul"]  # slot 0 = false, 1 = true


@torch.no_grad()
def p_true(query, document):
    state = f"Query: {query}\n\nDocument:\n{document[:1500]}"
    question = f"{INSTRUCTION}\nTrue: {CRITERIA['true']}\nFalse: {CRITERIA['false']}"
    text = f"[kind] noul\n[state] {state}\n[question] {question}\n[options]\nfalse\ntrue\n[decision]:"
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    h = backbone(inputs_embeds=E[ids].cuda()).last_hidden_state[0, -1].float()
    return torch.softmax(((W @ h + b) / T)[s:e], 0)[1].item()


# bge-reranker บน GPU (fp16 ~1.1 GB ลงที่ว่างที่เหลือจาก JEV-9B ได้) — บน CPU ใช้ ~9 วินาที/20 คู่ บน GPU ไม่ถึงวินาที
# ponytail: fp16 ให้คะแนนต่างจาก fp32 ในทศนิยมท้าย ๆ; ลำดับแทบไม่เปลี่ยน
rr_path = os.environ.get("RERANKER")
if rr_path:
    rr_tok = AutoTokenizer.from_pretrained(rr_path)
    rr = AutoModelForSequenceClassification.from_pretrained(rr_path, dtype=torch.float16).cuda().eval()


@torch.no_grad()
def rerank(query, documents):
    # รูปร่างคงที่ (5 คู่ × 512 token): ROCm คอมไพล์ kernel ใหม่ทุกครั้งที่เจอรูปร่างใหม่ (ครั้งแรก ~10-20 วินาที)
    # และทีละ 5 เพื่อให้ activation ลงที่ว่างที่เหลือข้าง JEV-9B ได้ (20 คู่ในก้อนเดียว OOM)
    out = []
    for i in range(0, len(documents), 5):
        docs = documents[i:i + 5]
        docs += [""] * (5 - len(docs))
        x = rr_tok([query] * 5, docs, padding="max_length", truncation=True, max_length=512,
                   return_tensors="pt").to("cuda")
        out += rr(**x).logits.view(-1).float().tolist()[:len(documents) - i]
    return out


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/rerank" and rr_path:
            scores = rerank(req["query"], req["documents"])
        else:
            scores = [p_true(req["query"], d) for d in req["documents"]]
        body = json.dumps({"scores": scores}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    if rr_path:
        rerank("warm up", ["x"] * 5)  # คอมไพล์ kernel ของรูปร่างเดียวที่ใช้ไว้ก่อน
    print("smoke:", round(p_true("หัดเยอรมันเกิดจากเชื้ออะไร", "โรคหัดเยอรมันเกิดจากการติดเชื้อไวรัสหัดเยอรมัน (Rubella virus)"), 3),
          round(p_true("หัดเยอรมันเกิดจากเชื้ออะไร", "ต้อกระจกคือภาวะที่เลนส์ตาขุ่นมัว"), 3), flush=True)
    HTTPServer(("0.0.0.0", 8766), Handler).serve_forever()
