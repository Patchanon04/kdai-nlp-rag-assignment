"""สร้าง index ทั้งสองระบบใน OpenSearch ถ้ายังไม่มี — service `init` ใน docker-compose.yml รันตัวนี้ตอนเปิดครั้งแรก

context ของทุก chunk ถูก cache ไว้ใน repo แล้ว (results/llm_cache.json สำหรับ Part 1, results/improved_contexts.json
สำหรับ Part 2) จึงสร้าง index ได้ผลเดิมโดยไม่ต้องเรียก LLM และไม่ต้องมี GEMINI_API_KEY
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "part1_baseline"), str(ROOT / "part2_improved")]

from opensearchpy import OpenSearch  # noqa: E402

HOST = os.environ.get("OS_HOST", "localhost")
client = OpenSearch(hosts=[{"host": HOST, "port": 9200}], use_ssl=False, timeout=60)


def count(index):
    return client.count(index=index)["count"] if client.indices.exists(index=index) else 0


if count("anthropic-bm25-index") == 0 or count("anthropic-vector-index") == 0:
    print("Part 1: สร้าง index ของระบบเดิม...", flush=True)
    from authenticRAG import AnthropicStyleContextualRAG
    for index in ("anthropic-bm25-index", "anthropic-vector-index"):  # ลบของที่สร้างค้างไว้ครึ่งทาง
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
    rag = AnthropicStyleContextualRAG(opensearch_host=HOST)
    rag.add_documents_with_context(rag.load_documents(
        [str(ROOT / "corpus_input" / f) for f in ("1.md", "2.md", "44.md", "5555.md")]))
else:
    print("Part 1: มี index แล้ว", flush=True)

if count("improved-rag-index") == 0:
    print("Part 2: สร้าง index ของระบบใหม่...", flush=True)
    from improvedRAG import ImprovedRAG
    ImprovedRAG(host=HOST).ingest()
else:
    print("Part 2: มี index แล้ว", flush=True)

print("init เสร็จ:", {i: count(i) for i in ("anthropic-bm25-index", "anthropic-vector-index", "improved-rag-index")},
      flush=True)
