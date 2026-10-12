"""วัดผล retrieval ของ Part 1 vs Part 2 ด้วยชุดคำถาม eval/questions.json

chunk ถือว่า "ถูก" ถ้ามีข้อความ evidence (คัดลอกตรงจากเอกสาร) อยู่ใน content ของ chunk
— วิธีนี้ไม่ผูกกับวิธี chunking จึงเทียบสองระบบที่ตัด chunk ต่างกันได้ยุติธรรม

  python eval/evaluate.py            # retrieval metrics -> results/retrieval_metrics.md
  python eval/evaluate.py judge      # ให้ LLM ตัวเดิมให้คะแนนคำตอบ 5 ข้อของครู -> results/answer_judgement.md
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "part1_baseline"), str(ROOT / "part2_improved")]
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
K = 5


def score(ranked_contents, evidence):
    """คืน (rank แรกที่เจอ evidence หรือ None)"""
    for i, text in enumerate(ranked_contents[:K], 1):
        if any(e in text for e in evidence):
            return i
    return None


def metrics(ranks, precs, sizes):
    n = len(ranks)
    return {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
            "MRR@5": sum(1 / r for r in ranks if r) / n,
            "Precision@5": sum(precs) / n,  # สัดส่วนท่อนที่ส่งให้ LLM ที่มีคำตอบจริง (ส่ง 0 ท่อน = 0)
            "avg chars to LLM": sum(sizes) / len(sizes)}


def retrieval():
    from improvedRAG import ImprovedRAG
    from onlysearchAuthenticRAG import AuthenticSearchRAG

    base, imp = AuthenticSearchRAG(), ImprovedRAG()
    systems = {
        "P1 baseline: BM25 (standard analyzer)": lambda q: [s["content"] for _, _, s in base.sparse_search(q, K)],
        "P1 baseline: dense": lambda q: [s["content"] for _, _, s in base.dense_search(q, K)],
        "P1 baseline: hybrid RRF (ระบบเดิม)": lambda q: [s["content"] for _, _, s in base.hybrid_search(q, K)],
        "P2: BM25 (thai analyzer)": lambda q: [s["content"] for _, s in imp.search(q, "bm25")],
        "P2: dense": lambda q: [s["content"] for _, s in imp.search(q, "dense")],
        "P2: hybrid RRF": lambda q: [s["content"] for _, s in imp.search(q, "hybrid")],
        "P2: hybrid + bge-reranker": lambda q: [s["content"] for _, s in imp.search(q, "rerank")],
        "P2: hybrid + reranker + Gemma filter": lambda q: [s["content"] for _, s in imp.search(q, "gemma")],
        "P2: hybrid -> JEV-9B rerank+filter (top-20)": lambda q: [s["content"] for _, s in imp.search(q, "jev20")],
        "P2: hybrid + reranker + JEV-9B filter (ระบบใหม่)": lambda q: [s["content"] for _, s in imp.search(q, "jev")],
    }
    # 3 ระบบหลักที่โชว์ในหน้า demo
    keys = {"P1 baseline: hybrid RRF (ระบบเดิม)": "baseline", "P2: hybrid + bge-reranker": "improved",
            "P2: hybrid + reranker + JEV-9B filter (ระบบใหม่)": "jev"}
    labelled = [q for q in QUESTIONS if q["evidence"]]
    rows, per_q = [], {q["id"]: {} for q in QUESTIONS}
    for name, run in systems.items():
        ranks, precs, sizes, by_type = [], [], [], {}
        for q in QUESTIONS:
            with contextlib.redirect_stdout(io.StringIO()):
                got = run(q["question"])
            sizes.append(sum(len(t) for t in got))
            if not q["evidence"]:
                per_q[q["id"]][name] = f"{len(got)} chunks"
                continue
            r = score(got, q["evidence"])
            ranks.append(r)
            precs.append(sum(any(e in t for e in q["evidence"]) for t in got[:K]) / len(got[:K]) if got else 0)
            by_type.setdefault(q["type"], []).append(r)
            per_q[q["id"]][name] = r or "-"
        m = metrics(ranks, precs, sizes)
        m.update({f"Hit@5 {t}": sum(r is not None for r in v) / len(v) for t, v in sorted(by_type.items())})
        rows.append((name, m))
        print(name, {k: round(v, 3) for k, v in m.items()})

    cols = list(rows[0][1])
    md = [f"# Retrieval metrics ({len(labelled)} labelled questions, top-{K})\n",
          "| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in rows:
        md.append(f"| {name} | " + " | ".join(f"{v:.0f}" if "chars" in k else f"{v:.2f}" for k, v in m.items()) + " |")
    md += ["\n## rank ของ chunk ที่ถูกต้องรายข้อ ('-' = ไม่อยู่ใน top-5)\n",
           "| id | type | question | " + " | ".join(n.split(":")[0] + ":" + n.split(":")[1][:22] for n in systems) + " |",
           "|---" * (len(systems) + 3) + "|"]
    for q in QUESTIONS:
        md.append(f"| {q['id']} | {q['type']} | {q['question']} | " + " | ".join(str(per_q[q['id']][n]) for n in systems) + " |")
    (ROOT / "results" / "retrieval_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_metrics_json(len(labelled), rows, keys)


def write_metrics_json(n, rows, keys):
    """ตัวเลขสำหรับ scoreboard ในหน้า demo"""
    (ROOT / "demo" / "web" / "public" / "metrics.json").write_text(json.dumps({"n": n, "rows": [
        {"key": keys.get(name), "label": name, "hit1": m["Hit@1"], "hit5": m["Hit@5"], "mrr": m["MRR@5"],
         "precision": m["Precision@5"], "chars": m["avg chars to LLM"]} for name, m in rows]},
        ensure_ascii=False, indent=1), encoding="utf-8")


def judge():
    """ให้ LLM ตัวเดิมให้คะแนนคำตอบของ 3 ระบบ (คำถาม showcase ใน demo/web/public/demo_data.json) เทียบคำตอบอ้างอิง"""
    import llm
    rows = [r for r in json.loads((ROOT / "demo" / "web" / "public" / "demo_data.json").read_text(encoding="utf-8"))["questions"]
            if r.get("answers")]
    names = {"baseline": "① ระบบเดิม", "improved": "② ระบบใหม่", "jev": "③ ใหม่ + Jev"}
    md = ["# LLM-as-judge: คะแนนคำตอบ 1-5 (judge = Gemma 4 31B ตัวเดียวกัน, เทียบคำตอบอ้างอิงจากเอกสาร)\n",
          "| id | question | " + " | ".join(names.values()) + " |", "|---" * 5 + "|"]
    totals = {k: [] for k in names}
    for r in rows:
        cells = []
        for k in names:
            out = llm.chat(f"""Grade the answer against the reference (from the source documents).
Score 1-5: 5 = correct, complete and stays within the reference facts; 1 = wrong or mostly invented.
If the reference says the documents do not contain the answer, reward answers that say so and penalise invented facts.
Question: {r["question"]}
Reference: {r["reference"]}
Answer: {r["answers"][k]}
Reply JSON only: {{"score": <1-5>, "reason": "<one short Thai sentence>"}}""", max_tokens=200, temperature=0)
            try:
                j = json.loads(re.search(r"\{.*\}", out, re.S).group())
                totals[k].append(float(j["score"]))
                cells.append(f"**{j['score']}** — {j['reason']}")
            except Exception:
                cells.append(out.replace("\n", " ")[:200])
        md.append(f"| {r['id']} | {r['question']} | " + " | ".join(cells) + " |")
    md.append("| | **เฉลี่ย** | " + " | ".join(f"**{sum(v) / len(v):.2f}**" if v else "-" for v in totals.values()) + " |")
    (ROOT / "results" / "answer_judgement.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    judge() if sys.argv[1:] == ["judge"] else retrieval()
