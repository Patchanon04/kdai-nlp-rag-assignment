"""Part 2: Authentic RAG ฉบับปรับปรุง (LLM ตัวเดิม — Gemma 4 31B ผ่าน llm.py)

สิ่งที่ปรับ เทียบกับ part1_baseline/ (รายละเอียด + ผลวัดใน README.md):
  1. Chunking   : แบ่งตามหัวข้อ markdown + จำกัดขนาด ~800 ตัวอักษร + overlap 1 บรรทัด + ติด breadcrumb หัวข้อ
  2. Context    : LLM เห็น "สารบัญของเอกสาร + section แม่" จริง (baseline ส่ง chunk ตัวเองแทนเอกสารเต็ม)
  3. BM25       : ใช้ analyzer "thai" ตัดคำไทย (baseline ใช้ standard ซึ่งมองทั้งวลีไทยเป็น token เดียว)
  4. Fusion     : RRF ที่ rank เริ่ม 1 + ดึง candidate กว้างขึ้น (20 ต่อฝั่ง)
  5. Reranking  : cross-encoder BAAI/bge-reranker-v2-m3 (รองรับภาษาไทย)
  6. Filtering  : Jev relevance filter (autotrust/JEV-9B รันบนเครื่อง, jev_local_server.py) ตัด chunk ที่ไม่เกี่ยวทิ้ง
                  + ทางเลือกที่ไม่ต้องใช้ GPU: ใช้ LLM ตัวเดิมทำ filter แนวเดียวกัน (mode="gemma")
  7. Generation : prompt ระบุแหล่งที่มา [n] + บังคับตอบจาก context เท่านั้น

ใช้งาน:
  python improvedRAG.py ingest      # สร้าง index (cache context ไว้ที่ results/improved_contexts.json)
  python improvedRAG.py ask         # ตอบ 5 คำถามของครู -> results/improved_answers.json
"""
import json
import re
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from opensearchpy import OpenSearch, helpers
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import llm  # noqa: E402

INDEX = "improved-rag-index"
CORPUS = ["1.md", "2.md", "44.md", "5555.md"]
CONTEXT_CACHE = ROOT / "results" / "improved_contexts.json"
MAX_CHARS = 800
JEV_URL = os.environ.get("JEV_URL", "http://localhost:8766/score")


# ---------------------------------------------------------------- 1. chunking
def chunk_markdown(text, source):
    """แบ่งตามหัวข้อ markdown ก่อน แล้วค่อยตัดแต่ละ section ให้ไม่เกิน MAX_CHARS"""
    title, chunks, stack, lines = "", [], [], []

    def flush():
        body = [l for l in lines if l.strip()]
        if not body:
            return
        crumb = " > ".join(stack)
        buf = []
        for line in body:
            # บรรทัดที่ยาวเกินไปตัดที่ช่องว่าง (ภาษาไทยเว้นวรรคระหว่างประโยค)
            while len(line) > MAX_CHARS:
                cut = line.rfind(" ", 0, MAX_CHARS)
                cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
                body_piece, line = line[:cut], line[cut:].strip()
                buf = _emit(chunks, buf, body_piece, crumb, source, title, body)
            buf = _emit(chunks, buf, line, crumb, source, title, body)
        if buf:
            chunks.append(_chunk(buf, crumb, source, title, body))

    for raw in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", raw)
        if m:
            flush()
            lines = []
            level, head = len(m.group(1)), m.group(2).strip()
            title = title or head
            stack = stack[: level - 1] + [head]
        else:
            lines.append(raw)
    flush()
    return chunks


def _emit(chunks, buf, line, crumb, source, title, section):
    if buf and sum(len(b) for b in buf) + len(line) > MAX_CHARS:
        chunks.append(_chunk(buf, crumb, source, title, section))
        buf = buf[-1:] if len(buf[-1]) < 300 else []  # overlap 1 บรรทัด
    return buf + [line]


def _chunk(buf, crumb, source, title, section):
    return {"source": source, "title": title, "breadcrumb": crumb,
            "content": "\n".join(buf), "section": "\n".join(section)}


# ---------------------------------------------------------------- 2. context
CONTEXT_PROMPT = """<document_title>{title}</document_title>
<document_outline>
{outline}
</document_outline>
<section path="{crumb}">
{section}
</section>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences, in Thai) to situate this chunk within the overall document for the purposes of improving search retrieval. Mention the disease name and what aspect this chunk covers (e.g. cause, symptom, treatment), and key synonyms a reader might search with. Answer only with the succinct context and nothing else."""


def contextualize(chunks, outlines):
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8")) if CONTEXT_CACHE.exists() else {}

    def one(c):
        key = c["source"] + "|" + c["content"][:200]
        if key not in cache:
            # ponytail: ส่ง outline + section แทนเอกสารเต็ม เพราะ free tier จำกัด 16k token/นาที
            cache[key] = llm.chat(CONTEXT_PROMPT.format(
                title=c["title"], outline=outlines[c["source"]], crumb=c["breadcrumb"],
                section=c["section"][:3000], chunk=c["content"]), max_tokens=200, temperature=0.1)
        c["context"] = cache[key]

    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(one, chunks), 1):
            print(f"context {i}/{len(chunks)}", end="\r")
            CONTEXT_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 3-7. RAG
class ImprovedRAG:
    def __init__(self, host="localhost", port=9200):
        self.os = OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False, timeout=60)
        self.embed = SentenceTransformer("BAAI/bge-m3")
        self._reranker = None
        self._jev_cache = {}

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        return self._reranker

    def ingest(self):
        chunks, outlines = [], {}
        for f in CORPUS:
            text = (ROOT / "corpus_input" / f).read_text(encoding="utf-8")
            outlines[f] = "\n".join(l for l in text.splitlines() if l.startswith("#"))
            chunks += chunk_markdown(text, f)
        print(f"{len(chunks)} chunks")
        contextualize(chunks, outlines)

        if self.os.indices.exists(index=INDEX):
            self.os.indices.delete(index=INDEX)
        thai = {"type": "text", "analyzer": "thai"}
        self.os.indices.create(index=INDEX, body={
            "settings": {"index.knn": True},
            "mappings": {"properties": {
                "embedding": {"type": "knn_vector", "dimension": self.embed.get_sentence_embedding_dimension(),
                              "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
                "content": thai, "context": thai, "breadcrumb": thai,
                "source": {"type": "keyword"}}}})
        texts = [f"{c['breadcrumb']}\n{c['context']}\n{c['content']}" for c in chunks]
        vecs = self.embed.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
        helpers.bulk(self.os, ({"_index": INDEX, "_id": f"c{i}", "embedding": v.tolist(),
                                **{k: c[k] for k in ("source", "breadcrumb", "content", "context")}}
                               for i, (c, v) in enumerate(zip(chunks, vecs))))
        self.os.indices.refresh(index=INDEX)
        print(f"indexed {len(chunks)} chunks into {INDEX}")

    # --- retrieval
    def bm25(self, q, k=20):
        r = self.os.search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]}, "query": {
            "multi_match": {"query": q, "fields": ["content", "context", "breadcrumb^2"], "type": "most_fields"}}})
        return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]

    def dense(self, q, k=20):
        v = self.embed.encode(q, normalize_embeddings=True).tolist()
        r = self.os.search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]},
                                              "query": {"knn": {"embedding": {"vector": v, "k": k}}}})
        return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]

    @staticmethod
    def rrf(rankings, k=60):
        scores, docs = {}, {}
        for ranking in rankings:
            for rank, (doc_id, src) in enumerate(ranking, 1):  # rank เริ่มที่ 1 ตามสูตร RRF
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
                docs[doc_id] = src
        return [(d, docs[d]) for d in sorted(scores, key=scores.get, reverse=True)]

    def rerank(self, q, cands):
        scores = self.reranker.predict([(q, f"{s['breadcrumb']}\n{s['content']}") for _, s in cands])
        return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]

    def llm_filter(self, q, cands, threshold=0.2):
        """Jev-style: ให้ LLM ตัดสินความเกี่ยวข้องตาม instruction แล้วตัดที่ต่ำกว่า threshold ทิ้ง"""
        listing = "\n\n".join(f"[{i}] ({s['breadcrumb']})\n{s['content'][:600]}" for i, (_, s) in enumerate(cands))
        out = llm.chat(f"""Instruction: Judge whether each passage contains information that helps answer the question.
A passage about a different disease, or only loosely on the topic, is NOT relevant.
Question: {q}

{listing}

Reply with JSON only: {{"<passage number>": <relevance 0.0-1.0>, ...}}""", max_tokens=200, temperature=0)
        try:
            rel = {int(k): float(v) for k, v in json.loads(re.search(r"\{.*\}", out, re.S).group()).items()}
        except Exception:
            return cands  # parse ไม่ได้ -> ไม่กรอง ดีกว่าทิ้งข้อมูล
        kept = [c for i, c in enumerate(cands) if rel.get(i, 1.0) >= threshold]
        return kept or cands[:1]

    def jev_scores(self, q, cands):
        """P(เกี่ยวข้อง) จาก JEV-9B ที่รันบนเครื่อง (jev_local_server.py) — cache ไว้ เพราะหลายโหมดถามซ้ำ"""
        todo = [(d, s) for d, s in cands if (q, d) not in self._jev_cache]
        if todo:
            req = urllib.request.Request(JEV_URL, data=json.dumps(
                {"query": q, "documents": [f"{s['breadcrumb']}\n{s['content']}" for _, s in todo]}).encode(),
                headers={"Content-Type": "application/json"})
            scores = json.loads(urllib.request.urlopen(req, timeout=600).read())["scores"]
            self._jev_cache.update({(q, d): p for (d, _), p in zip(todo, scores)})
        return [self._jev_cache[(q, d)] for d, _ in cands]

    def jev_filter(self, q, cands, threshold=0.2):
        """แบบ jev-reranker relevance_rerank(): เรียงตาม P(เกี่ยวข้อง) แล้วตัดที่ต่ำกว่า threshold ทิ้ง"""
        scored = sorted(zip(self.jev_scores(q, cands), cands), key=lambda x: -x[0])
        return [c for p, c in scored if p >= threshold]

    def search(self, q, mode="jev", k=5):
        if mode == "bm25":
            return self.bm25(q)[:k]
        if mode == "dense":
            return self.dense(q)[:k]
        fused = self.rrf([self.bm25(q), self.dense(q)])
        if mode == "hybrid":
            return fused[:k]
        if mode == "jev20":  # Jev จัดอันดับ top-20 เองแทน bge (แบบในบทความ)
            return self.jev_filter(q, fused[:20])[:k]
        ranked = self.rerank(q, fused[:20])
        if mode == "rerank":
            return ranked[:k]
        if mode == "jev":  # ระบบใหม่ตัวเต็ม: bge-reranker top-5 -> Jev คัดทิ้ง
            return self.jev_filter(q, ranked[:k])
        return self.llm_filter(q, ranked[:k])  # mode == "gemma": แนวคิด Jev ทำด้วย LLM ตัวเดิม

    # --- generation
    @staticmethod
    def generate(q, hits):
        ctx = "\n\n".join(f"[{i}] แหล่ง: {s['source']} > {s['breadcrumb']}\n{s['content']}"
                          for i, (_, s) in enumerate(hits, 1))
        return llm.chat(f"""You are a question answering assistant. Answer in Thai using ONLY the context below.
Cite sources like [1]. If the context does not contain the answer, say so clearly instead of guessing.

Context:
{ctx or "(no relevant passages were found)"}

Question: {q}""", max_tokens=1000, temperature=0.3)

    def answer(self, q, mode="jev", k=5):
        hits = self.search(q, mode, k)
        ans = self.generate(q, hits)
        return {"question": q, "answer": ans,
                "results": [{"doc_id": d, "source": s["source"], "breadcrumb": s["breadcrumb"],
                             "content": s["content"], "context": s["context"]} for d, s in hits]}


OFFICIAL_QUESTIONS = [
    "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
    "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
    "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?",
]

if __name__ == "__main__":
    rag = ImprovedRAG()
    if sys.argv[1:] == ["ingest"]:
        rag.ingest()
    else:
        out = [rag.answer(q) for q in OFFICIAL_QUESTIONS]
        for r in out:
            print(f"\nQ: {r['question']}\nA: {r['answer']}")
        (ROOT / "results" / "improved_answers.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
