"""Demo: เทียบ 3 ระบบ บนคำถามเดียวกัน — ① เดิม (Part 1) ② ใหม่ (thai analyzer + chunk ใหม่ + bge-reranker) ③ ② + Jev filter

  python demo/server.py build [q25,q26,...]   # คำนวณผลล่วงหน้าทุกคำถาม -> demo/demo_data.json (คำตอบ LLM เฉพาะ showcase)
  python demo/server.py                       # FastAPI: เสิร์ฟ UI (demo/web/dist) + live API ที่ :8080

ENV: OS_HOST (default localhost), JEV_URL (ดู part2_improved/improvedRAG.py), GEMINI_API_KEY
UI (React, demo/web) ใช้ได้ทั้งแบบมี server (โหมดสด) และเป็นไฟล์ static ล้วน (อ่าน demo_data.json) เช่นบน GitHub Pages
"""
import contextlib
import io
import json
import os
import sys
import threading
from typing import Literal
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "part1_baseline"), str(ROOT / "part2_improved")]
from improvedRAG import CORPUS, ImprovedRAG  # noqa: E402
from onlysearchAuthenticRAG import AuthenticSearchRAG  # noqa: E402

DEMO = ROOT / "demo"
DATA = DEMO / "web" / "public" / "demo_data.json"  # Vite คัดลอก public/ เข้า dist/ ให้เอง
OS_HOST = os.environ.get("OS_HOST", "localhost")
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
TEXTS = {f: (ROOT / "corpus_input" / f).read_text(encoding="utf-8") for f in CORPUS}
TITLES = {f: t.splitlines()[0].lstrip("# ").split(" อาการ")[0].split(" (")[0] for f, t in TEXTS.items()}
SYSTEMS = ("baseline", "improved", "jev")

base = imp = None
lock = threading.Lock()  # ค้นทีละคำถาม (โมเดลบน CPU/GPU ชุดเดียว)
llm_slots = threading.BoundedSemaphore(2)  # Gemma free tier ~16k token/นาที — ตอบพร้อมกันได้ทีละ 2
waiting = {"search": 0, "answer": 0}  # ponytail: นับแบบหยาบ ๆ ไว้โชว์คิวบนหน้าเว็บ ไม่ได้ใช้ตัดสินอะไร
cache, answers = {}, {}


@contextlib.contextmanager
def queued(kind, gate):
    waiting[kind] += 1
    try:
        gate.acquire()
    finally:
        waiting[kind] -= 1
    try:
        yield
    finally:
        gate.release()


def source_of(text):  # index ของระบบเดิมไม่ได้เก็บชื่อไฟล์ — หาจากเนื้อหา
    return next((TITLES[f] for f, t in TEXTS.items() if text[:120] in t), "?")


def card(text, source, score, ev, crumb=""):
    return {"source": source, "crumb": crumb, "text": text, "chars": len(text), "score": round(float(score), 4),
            "hit": any(e in text for e in ev) if ev else None}


def compare(q, ev=()):
    """ผลค้นหาของ 3 ระบบ (ไม่รวมคำตอบ LLM) + hits ที่ใช้สร้างคำตอบ"""
    if q in cache:
        return cache[q]
    with queued("search", lock):
        if q in cache:  # มีคนถามข้อเดียวกันขณะรอคิว
            return cache[q]
        with contextlib.redirect_stdout(io.StringIO()):
            b = base.hybrid_search(q, 5)
        fused = imp.rrf([imp.bm25(q), imp.dense(q)])[:20]
        rr = imp.rerank_scores(q, fused)
        top = sorted(zip(rr, fused), key=lambda x: -x[0])[:5]
        try:
            jev = [round(p, 4) for p in imp.jev_scores(q, [c for _, c in top])]
        except Exception as e:  # GPU/WSL ดับ -> คอลัมน์ Jev ขึ้นว่า offline แทนการพังทั้งหน้า
            print("jev unavailable:", e, flush=True)
            jev = None
    row = {"question": q, "evidence": list(ev),
           "baseline": [card(s.get("content", ""), source_of(s.get("content", "")), sc, ev) for _, sc, s in b],
           "improved": [card(s["content"], TITLES[s["source"]], sc, ev, s["breadcrumb"]) for sc, (_, s) in top],
           "jev": jev}
    kept = [c for p, (_, c) in sorted(zip(jev, top), key=lambda x: -x[0]) if p >= 0.2] if jev else None
    result = (row, [c for _, c in top], kept)
    if jev is not None:  # ไม่ cache ตอน Jev ดับ จะได้ลองใหม่เมื่อกลับมา
        cache[q] = result
    return result


def answer(q, system):
    if (q, system) not in answers:
        _, top, kept = compare(q)
        if system == "jev" and kept is None:
            return "Jev offline — ยังเชื่อมต่อ JEV-9B ไม่ได้"
        with queued("answer", llm_slots):
            if (q, system) not in answers:
                answers[(q, system)] = _answer(q, system, top, kept)
    return answers[(q, system)]


def _answer(q, system, top, kept):
    if system == "baseline":
        with contextlib.redirect_stdout(io.StringIO()):
            return base.search_for_question(q, k=5)["answer"]
    return imp.generate(q, top if system == "improved" else kept)


def build(showcase):
    old = {r["id"]: r for r in json.loads(DATA.read_text(encoding="utf-8"))["questions"]} if DATA.exists() else {}
    out = []
    for q in QUESTIONS:
        row = dict(compare(q["question"], q["evidence"])[0], id=q["id"], type=q["type"],
                   file=TITLES[q["file"]], reference=q["reference_answer"])
        if q["id"] in showcase:
            row["answers"] = old.get(q["id"], {}).get("answers") or {s: answer(q["question"], s) for s in SYSTEMS}
        out.append(row)
        print(q["id"], [c["hit"] for c in row["baseline"]], [c["hit"] for c in row["improved"]], row["jev"], flush=True)
        DATA.write_text(json.dumps({"questions": out}, ensure_ascii=False, indent=1), encoding="utf-8")


app = FastAPI(title="KDAI NLP RAG demo")


class Ask(BaseModel):
    q: str = Field(min_length=1, max_length=300)  # คำถามจากผู้ชมทั่วไป — จำกัดความยาว
    system: Literal["baseline", "improved", "jev"] = "jev"


def evidence_for(q):
    return next((x["evidence"] for x in QUESTIONS if x["question"] == q), [])


@app.get("/api/health")
def health():
    return {"live": True}


@app.get("/api/status")
def status():
    return waiting


@app.post("/api/compare")
def api_compare(req: Ask):
    q = req.q.strip()
    return compare(q, evidence_for(q))[0]


@app.post("/api/answer")
def api_answer(req: Ask):
    return {"answer": answer(req.q.strip(), req.system)}


if __name__ == "__main__":
    base, imp = AuthenticSearchRAG(opensearch_host=OS_HOST), ImprovedRAG(host=OS_HOST)
    if sys.argv[1:2] == ["build"]:
        build(sys.argv[2].split(",") if len(sys.argv) > 2 else ["q25", "q26", "q27", "q28", "q29"])
    else:
        if DATA.exists():  # คำตอบที่คำนวณไว้แล้ว ไม่ต้องเรียก LLM ซ้ำ
            for r in json.loads(DATA.read_text(encoding="utf-8"))["questions"]:
                answers.update({(r["question"], s): a for s, a in r.get("answers", {}).items()})
        app.mount("/", StaticFiles(directory=DEMO / "web" / "dist", html=True), name="web")
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
